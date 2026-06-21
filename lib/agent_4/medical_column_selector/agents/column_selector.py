"""
Column Selector Agent for Medical Column Selector.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, Iterable, List, Optional

from ..components.schema_profiler import VALID_MODALITIES

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


class ColumnSelectorAgent(AgentBase):
    """根据任务规格和列画像挑选最终列。"""

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
        self.last_normalization_info: Dict[str, Any] = {}

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
        """在真正筛列前显式确认模型已可用。"""
        if self.model is None:
            self.last_call_mode = "init_failed"
            self.last_call_error = self.init_error or "LLM unavailable"
            raise RuntimeError(self.last_call_error)

    def _get_system_prompt(self) -> str:
        """系统提示词主要约束输出结构。"""
        modalities_str = ", ".join(VALID_MODALITIES)
        return f"""
You are a medical data column selector. Your job is to analyze a task specification and the schema of a medical dataset to identify relevant columns.

Given:
- task_spec: The structured task specification including task type and required modalities
- schema_profile: A profile of the dataset's columns with metadata (data type, missing rate, examples, etc.)

Your output should be a strict JSON object with the following structure:
{{
  "must_have_columns": ["list", "of", "essential", "columns"],
  "useful_columns": ["list", "of", "columns", "that", "are", "likely", "relevant"],
  "maybe_columns": ["list", "of", "columns", "with", "uncertain", "relevance"],
  "drop_columns": ["list", "of", "columns", "to", "exclude"],
  "reason_by_column": {{
    "column_name": "Brief reason why this column was categorized as it was"
  }},
  "warnings": ["Potential issues such as PHI risk or missing key information"],
  "transform_suggestions": {{
    "column_name": "Suggested transformation (e.g., DOB->age, dates->relative days)"
  }}
}}

Available modalities: {modalities_str}

Important selection guidance:
- Keep task-supporting identifiers such as patient_id, subject_id, encounter_id, visit_id, and record_id when available.
- These identifier/key columns are usually useful for joins, deduplication, grouping, or traceability, even if the task is mainly about labs or demographics.
- Only exclude identifier columns when they are clearly direct PHI (for example MRN, SSN, insurance/account numbers) or plainly irrelevant metadata.
- Every column name you output must be copied exactly from the provided candidate column list. Do not translate, rename, abbreviate, or invent column names.

Respond ONLY with valid JSON, no other text. Do not use markdown code blocks.
"""

    def select_columns(self, task_spec: Dict[str, Any], schema_profile: Dict[str, Any]) -> Dict[str, Any]:
        """兼容旧调用方式。"""
        return self.__call__(task_spec, schema_profile)

    def __call__(self, task_spec: Dict[str, Any], schema_profile: Dict[str, Any]) -> Dict[str, Any]:
        """优先使用 LLM，失败时退回规则筛选。"""
        if self.model is None:
            self.last_call_error = self.init_error or "LLM unavailable"
            if self.allow_fallback:
                print(f"Warning: Could not call LLM, using template response: {self.last_call_error}")
                self.last_call_mode = "fallback"
                result = self._parse_selection_result(
                    self._create_mock_response(task_spec, schema_profile),
                    schema_profile,
                )
                result["task_spec"] = task_spec
                return result
            self.last_call_mode = "error"
            raise RuntimeError(self.last_call_error)

        prompt = f"""
Based on this task specification and schema profile, please select the relevant columns:

Task Specification:
{json.dumps(task_spec, indent=2, ensure_ascii=False)}

Schema Profile:
{json.dumps(schema_profile, indent=2, ensure_ascii=False)}

Candidate Column Names (copy exactly from this list when populating any *_columns field):
{json.dumps(list(schema_profile.get("columns", {}).keys()), ensure_ascii=False)}

Remember to respond ONLY with valid JSON containing:
must_have_columns, useful_columns, maybe_columns, drop_columns, reason_by_column, warnings, and transform_suggestions.
"""

        llm_response = self._try_llm(prompt)
        if llm_response is not None:
            self.last_call_mode = "llm"
            self.last_call_error = None
            result = self._parse_selection_result(llm_response, schema_profile)
            result["task_spec"] = task_spec
            return result

        if self.last_call_error and self.allow_fallback:
            print(f"Warning: Could not call LLM, using template response: {self.last_call_error}")
            self.last_call_mode = "fallback"
            result = self._parse_selection_result(
                self._create_mock_response(task_spec, schema_profile),
                schema_profile,
            )
            result["task_spec"] = task_spec
            return result

        self.last_call_mode = "error"
        raise RuntimeError(self.last_call_error or "Column selector LLM call failed")

    def _try_llm(self, prompt: str) -> Optional[str]:
        """调用模型；失败时由上层决定是否 fallback。"""
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
                raise RuntimeError(f"Column selector LLM call failed: {exc}") from exc
            return None

    def _build_messages(self, prompt: str) -> Any:
        """构造 AgentScope 所需消息。"""
        from agentscope.message import Msg

        return [
            Msg(name="system", content=self.sys_prompt, role="system").to_dict(),
            Msg(name="user", content=prompt, role="user").to_dict(),
        ]

    def _invoke_model(self, messages: Any) -> str:
        """兼容同步上下文调用异步模型。"""

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
        """让脚本环境和已有事件循环都能复用同一套调用。"""
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
        """统一提取文本，并过滤 thinking 内容。"""
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

    def _create_mock_response(self, task_spec: Dict[str, Any], schema_profile: Dict[str, Any]) -> str:
        """fallback 规则专注于给出稳定且可解释的基础结果。"""
        need_modalities = task_spec.get("need_modalities", [])
        privacy_mode = task_spec.get("privacy_mode", "research")
        dataset_info = schema_profile.get("dataset_info", {})
        task_keywords = list(dataset_info.get("task_keywords_used", []))
        for term in dataset_info.get("expanded_task_terms_used", []):
            if term not in task_keywords:
                task_keywords.append(term)
        task_context = self._build_task_context(task_spec, task_keywords)
        term_profile = dataset_info.get("task_terms_used", {})
        task_context["negative_terms"].update(str(term).lower() for term in term_profile.get("negative_terms", []))
        task_context["short_terms"].update(str(term).lower() for term in term_profile.get("short_terms", []))

        must_have: List[str] = []
        useful: List[str] = []
        maybe: List[str] = []
        drop: List[str] = []
        reasons: Dict[str, str] = {}
        warnings: List[str] = []

        for col_name, col_info in schema_profile.get("columns", {}).items():
            col_modality = col_info.get("inferred_modality")
            col_is_phi = bool(col_info.get("is_potential_phi", False))
            relevance_score, relevance_reason = self._score_column_relevance(
                col_name=col_name,
                col_modality=col_modality,
                task_context=task_context,
            )

            if col_modality == "Identifiers/Keys" and not col_is_phi:
                useful.append(col_name)
                reasons[col_name] = "Task-supporting identifier for joins, grouping, deduplication, or traceability"
                continue

            if col_is_phi:
                if privacy_mode == "strict":
                    drop.append(col_name)
                    reasons[col_name] = "PHI protection in strict mode"
                else:
                    maybe.append(col_name)
                    reasons[col_name] = "Contains potential PHI"
                    warnings.append(f"Column {col_name} may contain PHI")
                continue

            if col_modality in need_modalities:
                if relevance_score >= 4:
                    must_have.append(col_name)
                    reasons[col_name] = relevance_reason or f"High task relevance within modality: {col_modality}"
                elif relevance_score >= 2 or col_modality in {"Demographics", "Time", "Vitals", "Encounter", "Procedures"}:
                    useful.append(col_name)
                    reasons[col_name] = relevance_reason or f"Supports required modality: {col_modality}"
                else:
                    drop.append(col_name)
                    reasons[col_name] = f"Weak task relevance despite matching modality: {col_modality}"
            elif relevance_score >= 3:
                maybe.append(col_name)
                reasons[col_name] = relevance_reason or f"Task-related despite outside primary modalities: {col_modality}"
            else:
                drop.append(col_name)
                reasons[col_name] = relevance_reason or f"Not task-specific enough for required modalities: {need_modalities}"

        if "Time" in need_modalities:
            has_time = any(
                token in col.lower()
                for col in must_have + useful + maybe
                for token in ["date", "time", "admission"]
            )
            if not has_time:
                warnings.append("Task requires temporal analysis but no obvious time columns identified")

        mock_response = {
            "must_have_columns": must_have,
            "useful_columns": useful,
            "maybe_columns": maybe,
            "drop_columns": drop,
            "reason_by_column": reasons,
            "warnings": warnings,
            "transform_suggestions": {},
        }
        return json.dumps(mock_response, ensure_ascii=False)

    def _build_task_context(self, task_spec: Dict[str, Any], task_keywords: List[str]) -> Dict[str, Any]:
        """构造 fallback 规则用的轻量任务上下文。"""
        keywords = [str(keyword).lower() for keyword in task_keywords]
        keyword_text = " ".join(keywords)
        vestibular_task = any(
            token in keyword_text
            for token in ["vestibular", "eye", "movement", "nystagmus", "balance", "vertigo", "dizziness"]
        )

        expanded_terms = set(keywords)
        if vestibular_task:
            expanded_terms.update({
                "vestibular", "前庭", "眩晕", "头晕", "眼震", "眼动", "眼球", "视物",
                "平衡", "步态", "hit试验", "dix-hallpike", "roll-test", "冷热试验",
                "位置试验", "踏步试验", "角速度", "偏差", "标准差", "翻滚角", "异常", "减低", "分型",
            })

        negative_terms = set()
        if vestibular_task:
            negative_terms.update({
                "自身免疫抗体", "抗体", "c3", "c4", "rf", "anca", "抗ccp", "抗dsdna",
                "抗sm", "抗ssa", "抗ssb", "抗ena", "抗核小体", "抗rnp", "抗ro", "抗scl-70", "抗aca",
            })

        return {
            "need_modalities": set(task_spec.get("need_modalities", [])),
            "expanded_terms": expanded_terms,
            "negative_terms": negative_terms,
            "short_terms": set(),
            "vestibular_task": vestibular_task,
        }

    def _score_column_relevance(
        self,
        col_name: str,
        col_modality: Optional[str],
        task_context: Dict[str, Any],
    ) -> tuple[int, str]:
        """给 fallback 规则一个简单的相关性分数。"""
        col_lower = (col_name or "").lower()
        col_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", col_lower)
            if token
        }
        score = 0
        reasons: List[str] = []

        if col_modality in task_context["need_modalities"]:
            score += 1
            reasons.append(f"matches modality {col_modality}")

        positive_hits = []
        for term in task_context["expanded_terms"]:
            if not term:
                continue
            if len(term) <= 3 and term.isascii():
                if term in col_tokens:
                    positive_hits.append(term)
            elif term in col_lower:
                positive_hits.append(term)
        if positive_hits:
            score += max(3, min(4, len(positive_hits)))
            reasons.append(f"matched task terms: {positive_hits[:4]}")

        negative_hits = [term for term in task_context["negative_terms"] if term and term in col_lower]
        if negative_hits:
            score -= min(4, len(negative_hits) + 1)
            reasons.append(f"matched weakly related terms: {negative_hits[:3]}")

        if task_context["vestibular_task"]:
            if col_lower.startswith("table_症状学-本次发作"):
                score += 2
                reasons.append("acute vestibular symptom section")
            elif col_lower.startswith(("table_初步诊断", "table_第一次复诊")):
                score += 2
                reasons.append("diagnostic section")
            elif col_lower.startswith("table_床旁查体"):
                score += 2
                reasons.append("bedside exam section")
            elif col_lower.startswith("table_辅助检查-眼震视图等检查"):
                score += 3
                reasons.append("oculomotor/vestibular testing section")
            elif col_lower.startswith("table_辅助检查-实验室检查"):
                score -= 2
                reasons.append("generic lab section not central to vestibular task")

        return score, "; ".join(reasons)

    def _extract_json_from_response(self, response_content: str) -> str:
        """从模型输出里提取第一个可解析的 JSON。"""
        text = (response_content or "").strip()
        if not text:
            raise ValueError("Empty response from column selector")

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

    def _parse_selection_result(self, json_str: str, schema_profile: Dict[str, Any]) -> Dict[str, Any]:
        """只做必要归一化：补默认值、去重、过滤未知列。"""
        result = json.loads(json_str)
        known_columns = list(schema_profile.get("columns", {}).keys())

        must_have_columns, must_have_unmatched = self._normalize_columns(
            result.get("must_have_columns"),
            known_columns,
        )
        useful_columns, useful_unmatched = self._normalize_columns(
            result.get("useful_columns"),
            known_columns,
        )
        maybe_columns, maybe_unmatched = self._normalize_columns(
            result.get("maybe_columns"),
            known_columns,
        )
        drop_columns, drop_unmatched = self._normalize_columns(
            result.get("drop_columns"),
            known_columns,
        )

        reason_by_column, reason_unmatched = self._normalize_mapping_keys(
            result.get("reason_by_column"),
            known_columns,
        )
        transform_suggestions, transform_unmatched = self._normalize_mapping_keys(
            result.get("transform_suggestions"),
            known_columns,
        )

        normalization_info = {
            "must_have_unmatched": must_have_unmatched,
            "useful_unmatched": useful_unmatched,
            "maybe_unmatched": maybe_unmatched,
            "drop_unmatched": drop_unmatched,
            "reason_unmatched": reason_unmatched,
            "transform_unmatched": transform_unmatched,
        }
        self.last_normalization_info = normalization_info

        parsed = {
            "must_have_columns": must_have_columns,
            "useful_columns": useful_columns,
            "maybe_columns": maybe_columns,
            "drop_columns": drop_columns,
            "reason_by_column": reason_by_column,
            "warnings": self._normalize_strings(result.get("warnings")),
            "transform_suggestions": transform_suggestions,
            "normalization_info": normalization_info,
        }

        matched_count = sum(
            len(parsed[key])
            for key in ["must_have_columns", "useful_columns", "maybe_columns", "drop_columns"]
        )
        referenced_count = sum(
            len(list(self._iter_column_candidates(result.get(key))))
            for key in ["must_have_columns", "useful_columns", "maybe_columns", "drop_columns"]
        )

        if matched_count == 0:
            if referenced_count > 0:
                raise ValueError(
                    "Column selector returned column names that did not match the schema. "
                    f"Unmatched samples: {self._first_unmatched_sample(normalization_info)}"
                )
            raise ValueError("Column selector returned no usable columns")
        return parsed

    def _normalize_columns(self, values: Any, known_columns: List[str]) -> tuple[List[str], List[str]]:
        """列名统一去重，并尽量映射回当前 schema 中的真实列名。"""
        normalized: List[str] = []
        unmatched: List[str] = []

        for candidate in self._iter_column_candidates(values):
            matched = self._match_known_column(candidate, known_columns)
            if matched is None:
                if candidate not in unmatched:
                    unmatched.append(candidate)
                continue
            if matched not in normalized:
                normalized.append(matched)
        return normalized, unmatched

    def _normalize_strings(self, values: Any) -> List[str]:
        """把 warning 这类字段统一整理成字符串列表。"""
        if not isinstance(values, list):
            return []
        return [str(value) for value in values if value not in (None, "")]

    def _normalize_mapping_keys(self, values: Any, known_columns: List[str]) -> tuple[Dict[str, Any], List[str]]:
        """把 reason_by_column / transform_suggestions 的 key 尽量映射回真实列名。"""
        if not isinstance(values, dict):
            return {}, []

        normalized: Dict[str, Any] = {}
        unmatched: List[str] = []
        for raw_key, raw_value in values.items():
            matched = self._match_known_column(raw_key, known_columns)
            if matched is None:
                if raw_key not in unmatched:
                    unmatched.append(str(raw_key))
                continue
            normalized[matched] = raw_value
        return normalized, unmatched

    def _iter_column_candidates(self, values: Any) -> Iterable[str]:
        """从模型输出中提取可能的列名表示。"""
        if values is None:
            return
        if isinstance(values, str):
            for item in re.split(r"[\n,，;；]+", values):
                cleaned = self._clean_candidate_name(item)
                if cleaned:
                    yield cleaned
            return
        if isinstance(values, list):
            for item in values:
                yield from self._iter_column_candidates(item)
            return
        if isinstance(values, dict):
            for key in ["column", "column_name", "name", "col"]:
                if key in values:
                    yield from self._iter_column_candidates(values[key])
                    return
            return

    def _clean_candidate_name(self, value: Any) -> str:
        """清理模型返回的列名，去掉常见包裹符号。"""
        text = str(value).strip()
        text = re.sub(r"^[`'\"“”‘’\-\*\s]+", "", text)
        text = re.sub(r"[`'\"“”‘’\s]+$", "", text)
        return text.strip()

    def _canonicalize_column_name(self, value: str) -> str:
        """为宽松匹配生成稳定 key。"""
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())

    def _match_known_column(self, value: Any, known_columns: List[str]) -> Optional[str]:
        """把模型返回的列名尽量匹配回真实 schema。"""
        candidate = self._clean_candidate_name(value)
        if not candidate:
            return None

        exact_map = {column: column for column in known_columns}
        if candidate in exact_map:
            return candidate

        lower_candidate = candidate.lower()
        for column in known_columns:
            if column.lower() == lower_candidate:
                return column

        canonical_candidate = self._canonicalize_column_name(candidate)
        if not canonical_candidate:
            return None

        canonical_matches = [
            column for column in known_columns
            if self._canonicalize_column_name(column) == canonical_candidate
        ]
        if len(canonical_matches) == 1:
            return canonical_matches[0]

        substring_matches = [
            column for column in known_columns
            if canonical_candidate in self._canonicalize_column_name(column)
            or self._canonicalize_column_name(column) in canonical_candidate
        ]
        if len(substring_matches) == 1:
            return substring_matches[0]
        return None

    def _first_unmatched_sample(self, normalization_info: Dict[str, Any]) -> List[str]:
        """挑一小部分未匹配列名用于错误提示。"""
        samples: List[str] = []
        for key in [
            "must_have_unmatched",
            "useful_unmatched",
            "maybe_unmatched",
            "drop_unmatched",
            "reason_unmatched",
            "transform_unmatched",
        ]:
            for item in normalization_info.get(key, []):
                if item not in samples:
                    samples.append(item)
                if len(samples) >= 5:
                    return samples
        return samples
