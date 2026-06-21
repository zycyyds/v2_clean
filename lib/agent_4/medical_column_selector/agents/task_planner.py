"""
Task Planner Agent for Medical Column Selector.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, Optional

try:
    from agentscope.agent import AgentBase
except Exception:  # pragma: no cover - 兼容未安装 agentscope 的环境
    class AgentBase:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

from ..config_manager import get_llm_api_config


def _normalize_model_name(model_name: Optional[str]) -> str:
    """保留配置里的真实模型名，仅去掉多余的 openai/ 前缀。"""
    raw = str(model_name or "").strip()
    if not raw:
        return "gpt-4.1-mini"
    if raw.lower().startswith("openai/"):
        raw = raw.split("/", 1)[1].strip()
    return raw


class TaskPlannerAgent(AgentBase):
    """把自然语言任务转成结构化任务规格。"""

    def __init__(self, model_name: Optional[str] = None, allow_fallback: bool = False):
        super().__init__()
        self.sys_prompt = self._get_system_prompt()
        self.model_name = _normalize_model_name(model_name)
        self.allow_fallback = allow_fallback
        self.model = None
        self.init_error: Optional[str] = None
        self.last_call_mode = "uninitialized"
        self.last_call_error: Optional[str] = None
        self.last_raw_response: Optional[str] = None

        llm_cfg = get_llm_api_config()
        api_key = llm_cfg.get("api_key")
        if not api_key:
            self.init_error = "Missing API key for OpenAI model. Expected environment variable: OPENAI_API_KEY"
            return

        try:
            from agentscope.model import OpenAIChatModel

            self.model = OpenAIChatModel(
                model_name=self.model_name,
                api_key=api_key,
                client_kwargs={
                    "base_url": llm_cfg.get("base_url", "https://api.openai.com/v1"),
                    "timeout": int(llm_cfg.get("timeout", 120)),
                },
                generate_kwargs={"temperature": float(llm_cfg.get("temperature", 0.2))},
            )
        except Exception as exc:
            self.init_error = str(exc)

    def ensure_llm_ready(self) -> None:
        """在真正跑任务前显式确认模型已可用。"""
        if self.model is None:
            self.last_call_mode = "init_failed"
            self.last_call_error = self.init_error or "LLM unavailable"
            raise RuntimeError(self.last_call_error)

    def _get_system_prompt(self) -> str:
        """系统提示词只负责约束输出格式。"""
        return """
You are a medical data task planner. Your job is to analyze a task description for medical data analysis and generate a structured JSON specification.

Based on the user's task description, you will produce a JSON object with the following fields:
- task_type: Default to "cohort_selection" unless clearly specified otherwise
- need_modalities: An array of required data modalities from the following options:
  * Identifiers/Keys: Patient IDs, MRNs, SSNs, etc.
  * Time: Admission dates, procedure times, timestamps, etc.
  * Demographics: Age, gender, race, ethnicity, etc.
  * Encounter: Visit types, admission methods, discharge dispositions, etc.
  * Diagnosis: ICD codes, diagnosis descriptions, conditions, etc.
  * Medication: Drug names, dosages, prescriptions, etc.
  * Labs: Lab values, test results, biomarkers, etc.
  * Vitals: Blood pressure, heart rate, temperature, etc.
  * Procedures: CPT codes, procedure names, interventions, etc.
  * Outcomes: Discharge status, mortality, readmissions, etc.
  * ClinicalNotes: Physician notes, discharge summaries, etc.

- index_event: The main clinical event the study is centered around (optional)
- time_window: The time window of interest relative to the index event (optional)
- uncertainty: A confidence score (0-1) indicating how certain you are about the task requirements

Respond ONLY with valid JSON, no other text. Do not use markdown code blocks.
"""

    def generate_task_spec(self, task_text: str) -> Dict[str, Any]:
        """兼容旧调用方式。"""
        return self.__call__(task_text)

    def __call__(self, task_text: str) -> Dict[str, Any]:
        """优先调用 LLM，失败时回退到规则模板。"""
        if self.model is None:
            self.last_call_error = self.init_error or "LLM unavailable"
            if self.allow_fallback:
                print(f"Warning: Could not process with LLM, using template: {self.last_call_error}")
                self.last_call_mode = "fallback"
                return self._parse_task_spec(self._create_mock_response(task_text))
            self.last_call_mode = "error"
            raise RuntimeError(self.last_call_error)

        prompt = f"""
Analyze this medical task and generate the structured JSON specification:

Task: {task_text}

Remember to respond ONLY with valid JSON, no other text.
"""

        llm_response = self._try_llm(prompt)
        if llm_response is not None:
            self.last_call_mode = "llm"
            self.last_call_error = None
            return self._parse_task_spec(llm_response)

        if self.last_call_error and self.allow_fallback:
            print(f"Warning: Could not process with LLM, using template: {self.last_call_error}")
            self.last_call_mode = "fallback"
            return self._parse_task_spec(self._create_mock_response(task_text))

        self.last_call_mode = "error"
        raise RuntimeError(self.last_call_error or "Task planner LLM call failed")

    def _try_llm(self, prompt: str) -> Optional[str]:
        """调用模型；失败时只记录错误，不在这里做复杂分支。"""
        if self.model is None:
            self.last_call_error = self.init_error or "LLM unavailable"
            return None

        try:
            messages = self._build_messages(prompt)
            response_content = self._invoke_model(messages)
            self.last_raw_response = response_content
            return self._extract_json_from_response(response_content)
        except Exception as exc:
            self.last_call_error = str(exc)
            if not self.allow_fallback:
                raise RuntimeError(f"Task planner LLM call failed: {exc}") from exc
            return None

    def _build_messages(self, prompt: str) -> Any:
        """构造 AgentScope 所需的 system/user 消息。"""
        from agentscope.message import Msg

        return [
            Msg(name="system", content=self.sys_prompt, role="system").to_dict(),
            Msg(name="user", content=prompt, role="user").to_dict(),
        ]

    def _invoke_model(self, messages: Any) -> str:
        """兼容同步环境调用异步模型。"""

        async def _call() -> str:
            response = await self.model(messages)
            if hasattr(response, "__aiter__"):
                merged_text = ""
                async for chunk in response:
                    text = self._extract_text_from_chunk(chunk)
                    if text:
                        merged_text = self._merge_stream_text(merged_text, text)
                return merged_text
            return self._extract_text_from_chunk(response)

        return self._run_coro_sync(_call())

    def _run_coro_sync(self, coro: Any) -> Any:
        """在脚本和已有事件循环两种场景下都能拿到结果。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Dict[str, Any] = {}
        error: Dict[str, Exception] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:
                error["value"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "value" in error:
            raise error["value"]
        return result.get("value", "")

    def _extract_text_from_chunk(self, chunk: Any) -> str:
        """把不同响应结构统一抽成字符串，并过滤 thinking 内容。"""
        if chunk is None:
            return ""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            if str(chunk.get("type") or "") == "thinking":
                return ""
            if isinstance(chunk.get("text"), str):
                return chunk["text"]
            if isinstance(chunk.get("content"), str):
                return chunk["content"]

        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if str(item.get("type") or "") == "thinking":
                        continue
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        if hasattr(content, "text") and isinstance(content.text, str):
            return content.text
        if hasattr(chunk, "text") and isinstance(chunk.text, str):
            return chunk.text
        return str(chunk)

    def _merge_stream_text(self, current: str, incoming: str) -> str:
        """兼容累计式和增量式流式响应。"""
        if not current:
            return incoming
        if not incoming:
            return current
        if incoming.startswith(current):
            return incoming
        if current.startswith(incoming):
            return current

        max_overlap = min(len(current), len(incoming))
        for overlap in range(max_overlap, 0, -1):
            if current.endswith(incoming[:overlap]):
                return current + incoming[overlap:]
        return current + incoming

    def _create_mock_response(self, task_text: str) -> str:
        """LLM 不可用时，用简单规则给出任务规格。"""
        task_lower = (task_text or "").lower()
        vestibular_task = any(
            word in task_lower
            for word in [
                "vestibular", "vertigo", "dizziness", "nystagmus",
                "eye movement", "oculomotor", "balance",
                "前庭", "眩晕", "头晕", "眼震", "眼动", "平衡",
            ]
        )

        if any(word in task_lower for word in ["predict", "outcome", "风险", "预测"]):
            task_type = "prediction"
        elif any(word in task_lower for word in ["compare", "association", "相关", "比较"]):
            task_type = "association_study"
        else:
            task_type = "cohort_selection"

        need_modalities = ["Demographics"]
        keyword_rules = {
            "Diagnosis": ["diagnosis", "disease", "condition", "诊断", "疾病", "异常"],
            "Medication": ["medication", "drug", "therapy", "药", "治疗"],
            "Labs": ["lab", "test", "result", "glucose", "hba1c", "检验", "检查", "眼震"],
            "Vitals": ["vital", "blood pressure", "heart rate", "生命体征"],
            "Procedures": ["procedure", "intervention", "exam", "试验", "查体", "hit"],
            "Time": ["time", "date", "admission", "discharge", "时间", "日期"],
            "Outcomes": ["outcome", "survival", "mortality", "readmission", "结局"],
        }

        for modality, keywords in keyword_rules.items():
            if any(word in task_lower for word in keywords):
                need_modalities.append(modality)

        if vestibular_task:
            need_modalities.extend(["Diagnosis", "Labs", "Procedures"])
        if len(need_modalities) == 1:
            need_modalities.extend(["Diagnosis", "Labs"])

        mock_spec = {
            "task_type": task_type,
            "need_modalities": list(dict.fromkeys(need_modalities)),
            "uncertainty": 0.25,
            "index_event": "",
            "time_window": "",
        }
        return json.dumps(mock_spec, ensure_ascii=False)

    def _extract_json_from_response(self, response_content: str) -> str:
        """从模型输出中提取 JSON；兼容代码块和多余文本。"""
        text = (response_content or "").strip()
        if not text:
            raise ValueError("Empty response from task planner")

        try:
            json.loads(text)
            return text
        except Exception:
            pass

        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

        for start in [match.start() for match in re.finditer(r"\{", text)]:
            depth = 0
            for idx in range(start, len(text)):
                char = text[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : idx + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            break

        raise ValueError(f"No valid JSON object found in response: {text[:200]}")

    def _parse_task_spec(self, json_str: str) -> Dict[str, Any]:
        """做轻量归一化，不再做一长串硬校验。"""
        task_spec = json.loads(json_str)

        modalities = task_spec.get("need_modalities") or ["Demographics", "Diagnosis", "Labs"]
        if not isinstance(modalities, list):
            modalities = [str(modalities)]

        uncertainty = task_spec.get("uncertainty", 0.5)
        try:
            uncertainty = float(uncertainty)
        except (TypeError, ValueError):
            uncertainty = 0.5

        return {
            "task_type": str(task_spec.get("task_type") or "cohort_selection"),
            "need_modalities": list(dict.fromkeys(str(item) for item in modalities if item)),
            "uncertainty": min(1.0, max(0.0, uncertainty)),
            "index_event": str(task_spec.get("index_event") or ""),
            "time_window": str(task_spec.get("time_window") or ""),
        }
