"""Task-specific medical term planner for column recall."""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, Optional

try:
    from agentscope.agent import AgentBase
except Exception:  # pragma: no cover
    class AgentBase:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

from ..config_manager import get_llm_api_config


def _normalize_model_name(model_name: Optional[str]) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return "gpt-4.1-mini"
    if raw.lower().startswith("openai/"):
        raw = raw.split("/", 1)[1].strip()
    return raw


class TermPlannerAgent(AgentBase):
    """Generate task-specific medical terms for deterministic column recall."""

    REQUIRED_LIST_KEYS = [
        "trigger_terms",
        "column_terms",
        "direct_diagnosis_terms",
        "supportive_evidence_terms",
        "background_disease_terms",
        "imaging_finding_terms",
        "lab_marker_terms",
        "treatment_context_terms",
        "short_terms",
        "negative_terms",
    ]

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
                generate_kwargs={"temperature": float(llm_cfg.get("temperature", 0.1))},
            )
        except Exception as exc:
            self.init_error = str(exc)

    def ensure_llm_ready(self) -> None:
        if self.model is None:
            self.last_call_mode = "init_failed"
            self.last_call_error = self.init_error or "LLM unavailable"
            raise RuntimeError(self.last_call_error)

    def _get_system_prompt(self) -> str:
        return """
You are a medical terminology planner for task-driven column selection.

Given a clinical task, generate a compact JSON object of medical terms that
should be used to recall relevant column names from a wide medical table.

Output exactly this JSON shape:
{
  "task_domain": "short_snake_case_domain",
  "trigger_terms": ["terms from or closely naming the user task"],
  "column_terms": ["backward-compatible broad column terms"],
  "direct_diagnosis_terms": ["disease names, synonyms, ICD/problem-list phrases"],
  "supportive_evidence_terms": ["clinical evidence terms used to support diagnosis"],
  "background_disease_terms": ["risk factors, precursor diseases, common comorbid context"],
  "imaging_finding_terms": ["radiology/pathology/anatomic finding terms"],
  "lab_marker_terms": ["lab tests, biomarkers, physiologic measurements"],
  "treatment_context_terms": ["procedures, medications, or treatment context useful for this task"],
  "short_terms": ["short biomarkers/acronyms that need token-boundary matching"],
  "negative_terms": ["terms that are common false positives for the short terms"],
  "reason": "one short explanation"
}

Rules:
- Generate terms for column-name matching, not patient diagnosis.
- Include English synonyms/acronyms likely to appear in MIMIC-derived columns.
- For diagnosis tasks, always include both direct diagnosis terms and supporting evidence.
- Do not specialize to one disease template; infer organ system and evidence categories from the task.
- Put short ambiguous terms such as AST, ALT, INR, AFP in short_terms.
- Put false-positive column fragments in negative_terms when useful.
- Avoid generic standalone terms such as status, value, result, disease, tumor, cancer, mass.
  If they matter, combine them with task context, for example liver mass or pulmonary edema.
- Avoid generic treatment words such as transplant by themselves. If useful, include task-specific
  phrases such as liver transplant, kidney transplant, valve replacement, or chemotherapy.
- Respond ONLY with valid JSON. Do not use markdown.
"""

    def generate_term_profile(self, task_text: str, task_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.__call__(task_text=task_text, task_spec=task_spec)

    def __call__(self, task_text: str, task_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.model is None:
            self.last_call_error = self.init_error or "LLM unavailable"
            if self.allow_fallback:
                self.last_call_mode = "fallback"
                return self._fallback_profile(task_text)
            self.last_call_mode = "error"
            raise RuntimeError(self.last_call_error)

        prompt = f"""
Task text:
{task_text}

Task spec:
{json.dumps(task_spec or {}, ensure_ascii=False, indent=2)}
"""
        llm_response = self._try_llm(prompt)
        if llm_response is not None:
            self.last_call_mode = "llm"
            self.last_call_error = None
            return self._parse_term_profile(llm_response)

        if self.allow_fallback:
            self.last_call_mode = "fallback"
            return self._fallback_profile(task_text)

        self.last_call_mode = "error"
        raise RuntimeError(self.last_call_error or "Term planner LLM call failed")

    def _fallback_profile(self, task_text: str) -> Dict[str, Any]:
        terms = []
        for token in re.findall(r"[a-z][a-z0-9_]{1,}|[\u4e00-\u9fff]{2,}", (task_text or "").lower()):
            if token not in terms:
                terms.append(token)
        return {
            "task_domain": "task_text_keywords",
            "trigger_terms": terms,
            "column_terms": terms,
            "direct_diagnosis_terms": [],
            "supportive_evidence_terms": [],
            "background_disease_terms": [],
            "imaging_finding_terms": [],
            "lab_marker_terms": [],
            "treatment_context_terms": [],
            "short_terms": [],
            "negative_terms": [],
            "reason": "Fallback profile from task text only; no LLM-generated medical expansion.",
        }

    def _try_llm(self, prompt: str) -> Optional[str]:
        try:
            messages = self._build_messages(prompt)
            response_content = self._invoke_model(messages)
            self.last_raw_response = response_content
            return self._extract_json_from_response(response_content)
        except Exception as exc:
            self.last_call_error = str(exc)
            if not self.allow_fallback:
                raise RuntimeError(f"Term planner LLM call failed: {exc}") from exc
            return None

    def _build_messages(self, prompt: str) -> Any:
        from agentscope.message import Msg

        return [
            Msg(name="system", content=self.sys_prompt, role="system").to_dict(),
            Msg(name="user", content=prompt, role="user").to_dict(),
        ]

    def _invoke_model(self, messages: Any) -> str:
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

    def _extract_json_from_response(self, response_content: str) -> str:
        text = (response_content or "").strip()
        if not text:
            raise ValueError("Empty response from term planner")

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
                        json.loads(candidate)
                        return candidate
        raise ValueError(f"No valid JSON object found in response: {text[:200]}")

    def _parse_term_profile(self, json_str: str) -> Dict[str, Any]:
        data = json.loads(json_str)
        profile = {
            "task_domain": str(data.get("task_domain") or "unknown_task"),
            "reason": str(data.get("reason") or ""),
        }
        for key in self.REQUIRED_LIST_KEYS:
            profile[key] = self._normalize_terms(data.get(key))
        return profile

    def _normalize_terms(self, values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            raw_values = re.split(r"[\n,，;；]+", values)
        elif isinstance(values, list):
            raw_values = values
        else:
            return []

        normalized = []
        for value in raw_values:
            term = str(value).strip().lower()
            if term and term not in normalized:
                normalized.append(term)
        return normalized
