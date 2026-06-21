"""项目主流程：读取数据、生成任务规格、筛列并导出结果。"""
import argparse
import os
import json
import sys
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
import glob

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from .agents.task_planner import TaskPlannerAgent
    from .agents.column_selector import ColumnSelectorAgent
    from .agents.term_planner import TermPlannerAgent
    from .components.schema_profiler import SchemaProfiler
    from .components.hard_gates import HardGates
    from .config_manager import get_default_model_name, get_llm_api_config
except ImportError:
    from medical_column_selector.agents.task_planner import TaskPlannerAgent
    from medical_column_selector.agents.column_selector import ColumnSelectorAgent
    from medical_column_selector.agents.term_planner import TermPlannerAgent
    from medical_column_selector.components.schema_profiler import SchemaProfiler
    from medical_column_selector.components.hard_gates import HardGates
    from medical_column_selector.config_manager import get_default_model_name, get_llm_api_config


MIMIC_CORE_COLUMNS = {
    "subject_id",
    "hadm_id",
    "gender",
    "anchor_age",
    "dod",
    "admittime",
    "dischtime",
    "deathtime",
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "diagnoses_count",
    "diagnoses_icd_codes",
    "labevents_count",
    "labevents_abnormal_count",
    "prescriptions_count",
    "prescriptions_drugs",
    "icu_count",
    "icu_careunit",
    "impressions",
}

LAYERED_TERM_KEYS = [
    "direct_diagnosis_terms",
    "supportive_evidence_terms",
    "background_disease_terms",
    "imaging_finding_terms",
    "lab_marker_terms",
    "treatment_context_terms",
]

GENERIC_COLUMN_RECALL_TERMS = {
    "diagnosis",
    "diagnoses",
    "finding",
    "findings",
    "status",
    "value",
    "values",
    "result",
    "results",
    "procedure",
    "procedures",
    "radiology",
    "discharge",
    "report",
    "clinical",
    "patient",
    "patients",
    "disease",
    "cancer",
    "tumor",
    "mass",
    "symptom",
    "symptoms",
    "onset",
    "trend",
    "etiology",
    "severity",
    "transplant",
    "transplantation",
}

STRUCTURED_KEEP_MODALITIES = {
    "Identifiers/Keys",
}

IDENTIFIER_COLUMN_TERMS = {
    "id",
    "patient_id",
    "subject_id",
    "hadm_id",
    "encounter_id",
    "visit_id",
    "record_id",
    "study_id",
    "case_id",
    "pid",
    "eid",
    "mrn",
    "患者id",
    "患者_id",
    "病历号",
    "住院号",
    "就诊号",
    "病例号",
}

BASIC_GENERAL_INFO_TERMS = {
    "性别",
    "年龄",
    "年龄单位",
    "出生年月",
    "民族",
    "文化程度",
    "工作性质",
    "职业",
    "婚姻",
    "婚育",
    "身高",
    "体重",
    "bmi",
    "sex",
    "gender",
    "age",
    "birth",
    "dob",
    "race",
    "ethnicity",
    "language",
    "marital",
    "occupation",
    "education",
}

BASIC_TIME_OR_ENCOUNTER_TERMS = {
    "visit_date",
    "visit_time",
    "encounter_date",
    "encounter_time",
    "admission_date",
    "admission_time",
    "discharge_date",
    "discharge_time",
    "就诊时间",
    "就诊日期",
    "入院时间",
    "入院日期",
    "出院时间",
    "出院日期",
}


def get_agent_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_root() -> str:
    return os.path.dirname(get_agent_root())


def get_default_input_dir() -> str:
    return os.path.join(get_agent_root(), "data_input")


def get_default_output_dir() -> str:
    return os.path.join(get_project_root(), "program", "output", "step4_results")


def get_default_task_text() -> str:
    llm_cfg = get_llm_api_config()
    return os.environ.get(
        "AGENT4_TASK_TEXT",
        os.environ.get(
            "AGENT5_TASK_TEXT",
            llm_cfg.get("task_text") or (
                "Analyze vestibular function test results to identify patients with "
                "abnormal eye movement patterns and nystagmus-related findings."
            ),
        ),
    )


def resolve_input_csv(input_path: str) -> str:
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        if not input_path.lower().endswith((".csv", ".xlsx", ".xls")):
            raise FileNotFoundError(f"输入文件不是支持的表格文件: {input_path}")
        return input_path
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    candidates: List[str] = []
    for root, _, files in os.walk(input_path):
        for name in files:
            if name.lower().endswith((".csv", ".xlsx", ".xls")):
                candidates.append(os.path.join(root, name))

    if not candidates:
        raise FileNotFoundError(f"在目录 {input_path} 中未找到 CSV/XLSX/XLS 文件。")

    basename_to_paths: Dict[str, List[str]] = {}
    for path in candidates:
        basename_to_paths.setdefault(os.path.basename(path).lower(), []).append(path)

    for preferred_name in ("input.csv", "filtered.csv", "input.xlsx", "input.xls"):
        if preferred_name in basename_to_paths:
            paths = basename_to_paths[preferred_name]
            paths.sort(key=os.path.getmtime, reverse=True)
            return paths[0]

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


class TaskDrivenColumnSelector:
    """任务驱动的医疗表格筛列器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化主流程对象，并准备两个 agent。"""
        # AgentScope 只需要初始化一次最小配置即可，不在这里重复解析模型配置。
        try:
            import agentscope

            agentscope.init(
                logging_level="INFO",
                project="medical_column_selector",
                name="medical_column_selector_main"
            )
        except Exception as e:
            print(f"Warning: Could not initialize agentscope properly: {e}")

        self.config = config or {}
        self.default_config = {
            "privacy_mode": "research",
            "keep_free_text": False,
            "max_rows_profile": 1000,
            "max_final_columns": None,
            "sampling_method": "random",
            "random_seed": 42,
            "input_sheet_name": None,
            "model_name": self._get_default_model_name(),
            "require_llm": True,
            "selector_input_mode": "heuristic_prefilter",
            "wide_schema_column_threshold": 1000,
            "force_keep_structured_fields": True,
            "structured_keep_modalities": sorted(STRUCTURED_KEEP_MODALITIES),
        }

        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        model_name = self.config.get("model_name", self._get_default_model_name())
        allow_fallback = not self.config.get("require_llm", True)
        self.task_planner_agent = TaskPlannerAgent(model_name=model_name, allow_fallback=allow_fallback)
        self.term_planner_agent = TermPlannerAgent(model_name=model_name, allow_fallback=allow_fallback)
        self.column_selector_agent = ColumnSelectorAgent(model_name=model_name, allow_fallback=allow_fallback)

        if self.config.get("require_llm", True):
            self.task_planner_agent.ensure_llm_ready()
            self.term_planner_agent.ensure_llm_ready()
            self.column_selector_agent.ensure_llm_ready()

    def _get_default_model_name(self) -> str:
        """从配置文件里拿默认模型名。"""
        return get_default_model_name()

    def _extract_task_keywords(self, task_text: str) -> List[str]:
        """把任务文本压成一组简短关键词，用于候选列召回。"""
        text = (task_text or "").lower()
        tokens = re.findall(r"[a-z][a-z0-9_]{1,}|[\u4e00-\u9fff]{2,}", text)
        stopwords = {
            "identify", "analyze", "analysis", "study", "patient", "patients",
            "with", "for", "from", "into", "using", "based", "task", "medical",
            "data", "and", "the", "of", "to", "in", "on", "a", "an",
            "进行", "分析", "患者", "数据", "研究", "筛选", "识别", "用于", "相关",
        }

        deduped = []
        for token in tokens:
            if token in stopwords:
                continue
            if token not in deduped:
                deduped.append(token)
            if len(deduped) >= 25:
                break
        return deduped

    def _is_mimic_core_column(self, col_name: str) -> bool:
        """当前 Step2 聚合宽表中必须保留的 MIMIC structured/core 字段。"""
        return (col_name or "").lower() in MIMIC_CORE_COLUMNS

    def _is_protected_structured_column(self, col_name: str, col_profile: Dict[str, Any]) -> bool:
        """判断是否应跨数据集稳定保留结构化字段。

        这里的保护对象是基础结构化字段：主键、MIMIC core、一般资料/人口学字段。
        症状学、辅助检查、复诊、床旁查体等临床字段仍需通过任务词表或模型选择进入结果。
        strict PHI 和全空列仍交给 prefilter / hard gates 处理。
        """
        if not self.config.get("force_keep_structured_fields", True):
            return False
        if not col_name:
            return False
        if col_profile.get("missing_rate", 0) == 1.0:
            return False
        if self.config.get("privacy_mode", "research") == "strict" and col_profile.get("is_potential_phi", False):
            return False

        if self._is_mimic_core_column(col_name):
            return True

        if self._is_identifier_column_name(col_name):
            return True

        if self._is_general_info_column(col_name):
            return True

        return self._is_basic_demographic_or_encounter_column(col_name)

    def _is_identifier_column_name(self, col_name: str) -> bool:
        """只按明确主键名保护 ID，避免 mastoid 这类包含 id 字符串的医学词误判。"""
        normalized = (col_name or "").strip().lower()
        tokens = {
            token
            for token in re.split(r"[^a-z0-9_\u4e00-\u9fff]+", normalized)
            if token
        }
        return normalized in IDENTIFIER_COLUMN_TERMS or bool(tokens & IDENTIFIER_COLUMN_TERMS)

    def _is_general_info_column(self, col_name: str) -> bool:
        """当前中文数据里，一般资料相当于基础结构化字段。"""
        normalized = (col_name or "").strip().lower()
        if normalized.startswith("table_"):
            normalized = normalized[len("table_"):]
        return normalized.startswith("一般资料")

    def _is_basic_demographic_or_encounter_column(self, col_name: str) -> bool:
        """保护跨数据集通用的人口学/基础就诊字段，但不扩到症状和检查大类。"""
        normalized = (col_name or "").strip().lower()
        tail = re.split(r"[-_/：:]+", normalized)[-1].strip()
        candidates = {normalized, tail}
        return bool(candidates & BASIC_GENERAL_INFO_TERMS or candidates & BASIC_TIME_OR_ENCOUNTER_TERMS)

    def _tokenize_column_name(self, col_name: str) -> set[str]:
        """把列名切成 token，用于 AST/ALT/INR 等短词边界匹配。"""
        return {
            token
            for token in re.split(r"[^a-z0-9]+", (col_name or "").lower())
            if token
        }

    def _normalize_term_list(self, values: Any) -> List[str]:
        """把词表字段统一成小写去重列表。"""
        if values is None:
            return []
        if isinstance(values, str):
            raw_values = re.split(r"[\n,，;；]+", values)
        elif isinstance(values, list):
            raw_values = values
        else:
            return []

        normalized: List[str] = []
        for value in raw_values:
            term = str(value).strip().lower()
            if term and term not in normalized:
                normalized.append(term)
        return normalized

    def _normalize_term_profile(self, term_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """保证 term_profile 的结构稳定，方便报告和匹配复用。"""
        term_profile = term_profile or {}
        normalized = {
            "task_domain": str(term_profile.get("task_domain") or "unknown_task"),
            "trigger_terms": self._filter_generic_recall_terms(
                self._normalize_term_list(term_profile.get("trigger_terms"))
            ),
            "column_terms": self._filter_generic_recall_terms(
                self._normalize_term_list(term_profile.get("column_terms"))
            ),
            "short_terms": self._filter_generic_recall_terms(
                self._normalize_term_list(term_profile.get("short_terms"))
            ),
            "negative_terms": self._normalize_term_list(term_profile.get("negative_terms")),
            "reason": str(term_profile.get("reason") or ""),
        }
        for key in LAYERED_TERM_KEYS:
            normalized[key] = self._filter_generic_recall_terms(
                self._normalize_term_list(term_profile.get(key))
            )

        # Backward compatibility: old callers may only provide column_terms.
        # Keep those usable without pretending they belong to a specific evidence layer.
        return normalized

    def _filter_generic_recall_terms(self, terms: List[str]) -> List[str]:
        """去掉会在宽表里召回整类列的通用词。"""
        return [
            term for term in terms
            if term not in GENERIC_COLUMN_RECALL_TERMS
        ]

    def _is_wide_schema(self, schema_profile: Dict[str, Any]) -> bool:
        """超宽表必须用更严格的候选召回，不能按 modality 全量保留。"""
        total_columns = schema_profile.get("dataset_info", {}).get("total_columns")
        if total_columns is None:
            total_columns = len(schema_profile.get("columns", {}))
        threshold = int(self.config.get("wide_schema_column_threshold", 1000) or 1000)
        return int(total_columns) >= threshold

    def _term_profile_matches_column(self, col_name: str, term_profile: Optional[Dict[str, Any]]) -> bool:
        """用动态医学词表召回列名，短词必须 token 边界命中。"""
        return bool(self._matching_term_layers(col_name, term_profile))

    def _matching_term_layers(
        self,
        col_name: str,
        term_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """返回列名命中的词表层，供候选召回和报告统计复用。"""
        profile = self._normalize_term_profile(term_profile)
        col_lower = (col_name or "").lower()
        tokens = self._tokenize_column_name(col_name)
        matches: Dict[str, List[str]] = {}

        if self._term_profile_blocks_column(col_name, profile):
            return matches

        keys = ["trigger_terms", "column_terms"] + LAYERED_TERM_KEYS
        for key in keys:
            for term in profile.get(key, []):
                if not term:
                    continue
                term_tokens = self._tokenize_column_name(term)
                if len(term) <= 3 and term.isascii():
                    matched = term in tokens
                elif len(term_tokens) > 1 and term.isascii():
                    matched = all(token in tokens for token in term_tokens)
                else:
                    matched = term in col_lower
                if matched:
                    matches.setdefault(key, []).append(term)

        for term in profile.get("short_terms", []):
            if term and term in tokens:
                matches.setdefault("short_terms", []).append(term)

        return matches

    def _term_profile_blocks_column(self, col_name: str, term_profile: Optional[Dict[str, Any]]) -> bool:
        """negative_terms 命中的列不进入任务词表候选兜底。"""
        profile = self._normalize_term_profile(term_profile)
        col_lower = (col_name or "").lower()
        return any(term and term in col_lower for term in profile.get("negative_terms", []))

    def _task_text_with_memory_context(self, task_text: str, memory_context: str | None) -> str:
        """给词表规划阶段补充历史经验，但不改变原始任务本身。"""
        context = str(memory_context or "").strip()
        if not context:
            return task_text
        clipped = context[:5000]
        return (
            f"{task_text}\n\n"
            "[Step4 memory context]\n"
            "Use these prior Step4 term/profile/final-column experiences only when they are relevant. "
            "Do not override the current task or current schema.\n"
            f"{clipped}"
        )

    def _force_keep_structured_columns(
        self,
        selection_result: Dict[str, Any],
        schema_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把通用结构化字段加入 must-have，避免被动态词表或模型漏掉。"""
        columns = schema_profile.get("columns", {})
        must_have = selection_result.setdefault("must_have_columns", [])
        drop_columns = selection_result.setdefault("drop_columns", [])
        reasons = selection_result.setdefault("reason_by_column", {})
        overrides = selection_result.setdefault("structured_keep_overrides", [])
        legacy_mimic_overrides = selection_result.setdefault("mimic_core_overrides", [])

        for col_name, col_profile in columns.items():
            if not self._is_protected_structured_column(col_name, col_profile):
                continue
            for key in ["useful_columns", "maybe_columns", "drop_columns"]:
                if col_name in selection_result.get(key, []):
                    selection_result[key] = [col for col in selection_result[key] if col != col_name]
            if col_name not in must_have:
                must_have.append(col_name)
            if col_name in drop_columns:
                drop_columns.remove(col_name)
            reasons[col_name] = (
                "Forced keep: baseline structured field needed for patient identity, "
                "demographics, general information, encounter time, or MIMIC core traceability."
            )
            action = "forced_keep_mimic_core" if self._is_mimic_core_column(col_name) else "forced_keep_structured_field"
            reason = "MIMIC structured/core column" if self._is_mimic_core_column(col_name) else "Baseline structured field"
            item = {
                "column": col_name,
                "action": action,
                "reason": reason,
                "inferred_modality": col_profile.get("inferred_modality"),
            }
            overrides.append(item)
            if self._is_mimic_core_column(col_name):
                legacy_mimic_overrides.append(item)

        return selection_result

    def _read_input_dataframe(self, input_path: str) -> pd.DataFrame:
        """统一读取 CSV / Excel，避免分析和导出逻辑不一致。"""
        return SchemaProfiler()._read_tabular_file(
            input_path,
            sheet_name=self.config.get("input_sheet_name"),
        )

    def _structured_table_modalities(self, col_name: str) -> List[str]:
        """给中文结构化表头做稳定 modality 映射。"""
        col_lower = (col_name or "").lower()
        modalities: List[str] = []

        if col_lower.startswith("table_一般资料"):
            modalities.append("Demographics")
            if any(token in col_lower for token in ["出生年月", "年龄", "日期"]):
                modalities.append("Time")

        if col_lower.startswith("table_症状学"):
            modalities.append("Diagnosis")

        if col_lower.startswith("table_床旁查体"):
            modalities.append("Procedures")

        if col_lower.startswith("table_初步诊断") or col_lower.startswith("table_第一次复诊"):
            modalities.append("Diagnosis")

        if col_lower.startswith("table_辅助检查"):
            modalities.append("Labs")
            if any(token in col_lower for token in ["查体", "hit试验", "dix-hallpike", "roll-test", "位置试验"]):
                modalities.append("Procedures")
            if any(token in col_lower for token in ["诊断日期", "持续时间"]):
                modalities.append("Time")

        deduped: List[str] = []
        for modality in modalities:
            if modality not in deduped:
                deduped.append(modality)
        return deduped

    def _is_structured_clinical_table_column(self, col_name: str) -> bool:
        """判断是否属于需要稳定召回的中文结构化表字段。"""
        col_lower = (col_name or "").lower()
        return col_lower.startswith((
            "table_一般资料",
            "table_症状学",
            "table_辅助检查",
            "table_初步诊断",
            "table_第一次复诊",
            "table_床旁查体",
        ))

    def _is_task_relevant_structured_column(
        self,
        col_name: str,
        task_keywords: List[str],
        need_modalities: set[str],
    ) -> bool:
        """对中文结构化字段做补充召回，避免重要列被漏掉。"""
        structured_modalities = set(self._structured_table_modalities(col_name))
        if structured_modalities & need_modalities:
            return True

        col_lower = (col_name or "").lower()
        vestibular_tokens = [
            "眩晕", "头晕", "前庭", "眼震", "眼动", "平衡", "步态",
            "hit试验", "dix-hallpike", "roll-test", "冷热试验",
            "耳鸣", "复视", "姿势", "踏步试验"
        ]
        if any(token in col_lower for token in vestibular_tokens):
            return True

        return any(keyword in col_lower for keyword in task_keywords)

    def _build_llm_input_profile(
        self,
        schema_profile: Dict[str, Any],
        task_spec: Dict[str, Any],
        task_text: str,
        term_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """给列筛选阶段准备 schema，默认把完整候选集交给 LLM。"""
        all_columns = schema_profile.get("columns", {})
        privacy_mode = self.config.get("privacy_mode", "research")
        task_keywords = self._extract_task_keywords(task_text)
        normalized_term_profile = self._normalize_term_profile(term_profile)
        task_terms = (
            normalized_term_profile["trigger_terms"] +
            normalized_term_profile["column_terms"] +
            [
                term
                for key in LAYERED_TERM_KEYS
                for term in normalized_term_profile[key]
            ] +
            normalized_term_profile["short_terms"]
        )
        need_modalities = set(task_spec.get("need_modalities", []))
        selector_input_mode = self.config.get("selector_input_mode", "full_schema")
        wide_schema_mode = self._is_wide_schema(schema_profile)

        prefiltered_cols: Dict[str, Dict[str, Any]] = {}
        dropped_constant_or_missing: List[str] = []
        dropped_phi_in_strict: List[str] = []

        for col_name, col_profile in all_columns.items():
            is_all_missing = col_profile.get("missing_rate", 0) == 1.0
            is_constant = col_profile.get("is_constant", False)
            is_identifier = col_profile.get("inferred_modality") == "Identifiers/Keys"

            if is_all_missing:
                dropped_constant_or_missing.append(col_name)
                continue

            if is_constant and not is_identifier:
                dropped_constant_or_missing.append(col_name)
                continue

            if privacy_mode == "strict" and col_profile.get("is_potential_phi", False):
                dropped_phi_in_strict.append(col_name)
                continue

            prefiltered_cols[col_name] = col_profile

        if not prefiltered_cols:
            prefiltered_cols = dict(all_columns)

        candidate_cols: List[str]
        if selector_input_mode == "heuristic_prefilter":
            candidate_cols = []
            term_recall_hits: Dict[str, List[str]] = {
                key: [] for key in ["trigger_terms", "column_terms"] + LAYERED_TERM_KEYS + ["short_terms"]
            }
            for col_name, col_profile in prefiltered_cols.items():
                col_lower = col_name.lower()
                modality = col_profile.get("inferred_modality")
                mimic_core_hit = self._is_mimic_core_column(col_name)

                if not mimic_core_hit and self._term_profile_blocks_column(col_name, normalized_term_profile):
                    continue

                modality_hit = modality in need_modalities and not wide_schema_mode
                keyword_hit = any(keyword in col_lower for keyword in task_keywords)
                matched_layers = self._matching_term_layers(col_name, normalized_term_profile)
                term_hit = bool(matched_layers)
                baseline_hit = modality in {"Identifiers/Keys", "Time"} and not wide_schema_mode
                structured_hit = self._is_structured_clinical_table_column(col_name) and self._is_task_relevant_structured_column(
                    col_name,
                    task_keywords=task_keywords,
                    need_modalities=need_modalities,
                )

                if mimic_core_hit or term_hit or modality_hit or keyword_hit or baseline_hit or structured_hit:
                    candidate_cols.append(col_name)
                    for key in matched_layers:
                        term_recall_hits.setdefault(key, []).append(col_name)

            min_candidates = 0 if wide_schema_mode else min(12, len(prefiltered_cols))
            if len(candidate_cols) < min_candidates:
                for col_name, col_profile in prefiltered_cols.items():
                    if col_name in candidate_cols:
                        continue
                    if self._term_profile_blocks_column(col_name, normalized_term_profile):
                        continue
                    if col_profile.get("inferred_modality") in {
                        "Demographics", "Encounter", "Outcomes", "Vitals",
                        "Diagnosis", "Labs", "Medication", "Procedures"
                    }:
                        candidate_cols.append(col_name)
                    if len(candidate_cols) >= min_candidates:
                        break

            if not candidate_cols:
                candidate_cols = list(prefiltered_cols.keys())
        else:
            candidate_cols = list(prefiltered_cols.keys())
            term_recall_hits = {
                key: [] for key in ["trigger_terms", "column_terms"] + LAYERED_TERM_KEYS + ["short_terms"]
            }

        candidate_set = set(candidate_cols)
        candidate_schema_columns = {
            col_name: col_profile
            for col_name, col_profile in prefiltered_cols.items()
            if col_name in candidate_set
        }
        term_recall_stats = {
            key: {
                "count": len(list(dict.fromkeys(columns))),
                "sample_columns": list(dict.fromkeys(columns))[:20],
            }
            for key, columns in term_recall_hits.items()
        }

        llm_input_profile = {
            "dataset_info": {
                **schema_profile.get("dataset_info", {}),
                "total_columns_before_prefilter": len(all_columns),
                "columns_after_prefilter": len(prefiltered_cols),
                "columns_passed_to_selector": len(candidate_schema_columns),
                "dropped_constant_or_all_missing_count": len(dropped_constant_or_missing),
                "dropped_phi_in_strict_count": len(dropped_phi_in_strict),
                "task_keywords_used": task_keywords,
                "task_terms_used": normalized_term_profile,
                "expanded_task_terms_used": list(dict.fromkeys(task_terms)),
                "term_recall_stats": term_recall_stats,
                "selector_input_mode": selector_input_mode,
                "wide_schema_mode": wide_schema_mode,
            },
            "columns": candidate_schema_columns,
        }
        return llm_input_profile

    def run(
        self,
        input_csv_path: str,
        task_text: str,
        output_dir: Optional[str] = None,
        memory_context: str | None = None,
    ) -> Dict[str, str]:
        """执行完整筛列流程，并导出 CSV 和报告。"""
        print(f"Starting task-driven column selection for: {task_text}")

        if output_dir is None:
            output_dir = os.path.dirname(input_csv_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        # 1. 先分析原始表结构。
        print("Step 1: Running Schema Profiler...")
        profiler = SchemaProfiler(
            max_rows=self.config.get("max_rows_profile", 1000),
            sampling_method=self.config.get("sampling_method", "random"),
            random_seed=self.config.get("random_seed", 42)
        )
        schema_profile = profiler.analyze_file(
            input_csv_path,
            sheet_name=self.config.get("input_sheet_name"),
        )

        # 2. 把自然语言任务转成结构化任务规格。
        print("Step 2: Running Task Planner Agent...")
        task_spec = self.task_planner_agent.generate_task_spec(task_text)
        task_spec["privacy_mode"] = self.config.get("privacy_mode", "research")

        # 3. 由 LLM 生成任务医学词表，用于超宽表的候选列召回。
        print("Step 3: Running Term Planner Agent...")
        term_profile = self.term_planner_agent.generate_term_profile(
            task_text=self._task_text_with_memory_context(task_text, memory_context),
            task_spec=task_spec,
        )
        term_profile = self._normalize_term_profile(term_profile)

        # 4. 先缩小候选列范围，避免把整张表都送入后续筛选。
        print("Step 4: Building candidate columns for selector...")
        llm_input_profile = self._build_llm_input_profile(
            schema_profile=schema_profile,
            task_spec=task_spec,
            task_text=task_text,
            term_profile=term_profile,
        )
        candidate_count = llm_input_profile["dataset_info"]["columns_passed_to_selector"]
        print(f"  - Candidate columns passed to selector: {candidate_count}")

        # 5. 让列筛选器输出 must/useful/maybe/drop。
        print("Step 5: Running Column Selector Agent...")
        selection_result = self.column_selector_agent.select_columns(
            task_spec=task_spec,
            schema_profile=llm_input_profile
        )
        selection_result = self._force_keep_structured_columns(
            selection_result=selection_result,
            schema_profile=schema_profile,
        )

        # 6. 再套一层硬规则，处理 PHI、自由文本、常量列等。
        print("Step 6: Applying Hard Gates...")
        hard_gates = HardGates(
            privacy_mode=self.config.get("privacy_mode", "research"),
            keep_free_text=self.config.get("keep_free_text", False),
            max_final_columns=self.config.get("max_final_columns")
        )
        final_selection = hard_gates.apply_gates(
            selection_result=selection_result,
            schema_profile=schema_profile
        )

        # 7. 导出筛选后的数据和完整审计报告。
        print("Step 7: Exporting results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_csv_path))[0]

        filtered_csv_path = os.path.join(output_dir, f"{base_name}_filtered_{timestamp}.csv")
        selection_report_path = os.path.join(output_dir, f"{base_name}_selection_report_{timestamp}.json")
        final_columns = final_selection["final_columns"]

        selection_report = {
            "timestamp": datetime.now().isoformat(),
            "task_text": task_text,
            "task_spec": task_spec,
            "term_profile": term_profile,
            "memory_context_used": bool(str(memory_context or "").strip()),
            "column_counts": {
                "original_column_count": len(schema_profile.get("columns", {})),
                "prefiltered_column_count": llm_input_profile["dataset_info"].get("columns_after_prefilter", 0),
                "selector_candidate_column_count": llm_input_profile["dataset_info"].get("columns_passed_to_selector", 0),
                "final_column_count": len(final_columns),
            },
            "agent_execution": {
                "task_planner_mode": self.task_planner_agent.last_call_mode,
                "task_planner_error": self.task_planner_agent.last_call_error,
                "task_planner_raw_response": self.task_planner_agent.last_raw_response,
                "term_planner_mode": self.term_planner_agent.last_call_mode,
                "term_planner_error": self.term_planner_agent.last_call_error,
                "term_planner_raw_response": self.term_planner_agent.last_raw_response,
                "column_selector_mode": self.column_selector_agent.last_call_mode,
                "column_selector_error": self.column_selector_agent.last_call_error,
                "column_selector_raw_response": self.column_selector_agent.last_raw_response,
                "column_selector_normalization_info": self.column_selector_agent.last_normalization_info,
            },
            "schema_profile": schema_profile,
            "llm_input_profile": llm_input_profile,
            "agent_decisions": selection_result,
            "hard_gate_overrides": final_selection.get("overrides", {}),
            "final_columns": final_columns,
            "warnings": final_selection.get("warnings", []),
            "config_used": self.config
        }

        with open(selection_report_path, 'w', encoding='utf-8') as f:
            json.dump(selection_report, f, indent=2, ensure_ascii=False)

        if not final_columns:
            raise RuntimeError(
                "Column selector produced no final columns. "
                f"See selection report for diagnostics: {selection_report_path}"
            )

        df = self._read_input_dataframe(input_csv_path)
        filtered_df = df[final_columns]
        filtered_df.to_csv(filtered_csv_path, index=False)

        column_counts = selection_report["column_counts"]
        print(
            "  - Column counts: "
            f"{column_counts['original_column_count']} -> "
            f"{column_counts['prefiltered_column_count']} -> "
            f"{column_counts['selector_candidate_column_count']} -> "
            f"{column_counts['final_column_count']}"
        )
        print(f"Results saved:\n  - Filtered CSV: {filtered_csv_path}\n  - Selection Report: {selection_report_path}")

        return {
            "filtered_csv_path": filtered_csv_path,
            "selection_report_path": selection_report_path
        }

    def run_on_directory_csv(self, directory_path: str, csv_pattern: str, task_text: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Execute the workflow on a specific CSV file in a directory

        Args:
            directory_path: Path to the directory containing CSV files
            csv_pattern: Pattern to match the CSV file (e.g., "input_cleaned_38.csv")
            task_text: Medical research/diagnosis/task text description
            output_dir: Output directory for results (optional)

        Returns:
            Dict with paths to filtered CSV and selection report
        """
        # Find the specific CSV file
        target_path = os.path.join(directory_path, csv_pattern)

        if not os.path.exists(target_path):
            # If exact match not found, try glob pattern matching
            matching_files = glob.glob(os.path.join(directory_path, csv_pattern))

            if not matching_files:
                raise FileNotFoundError(f"No CSV file matching pattern '{csv_pattern}' found in directory: {directory_path}")

            # Use the first matching file
            target_path = matching_files[0]
            print(f"Found matching file: {target_path}")

        return self.run(target_path, task_text, output_dir)


def run_example():
    """Example usage of the TaskDrivenColumnSelector"""

    # Create sample CSV data for demonstration
    sample_data = {
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],
        "last_name": ["Doe", "Smith", "Johnson", "Brown", "Wilson"],
        "dob": ["1980-01-15", "1990-05-20", "1975-11-10", "1985-03-25", "1992-07-30"],
        "gender": ["M", "F", "M", "F", "M"],
        "admission_date": ["2023-01-10", "2023-01-15", "2023-01-20", "2023-01-25", "2023-02-01"],
        "discharge_date": ["2023-01-15", "2023-01-20", "2023-01-25", "2023-01-30", "2023-02-05"],
        "diagnosis_code": ["J45.909", "E11.9", "I10", "J44.1", "E10.9"],
        "diagnosis_description": ["Asthma", "Type 2 DM", "Hypertension", "COPD", "Type 1 DM"],
        "medication": ["Albuterol", "Metformin", "Lisinopril", "Tiotropium", "Insulin"],
        "lab_glucose": [180, 220, 160, 190, 250],
        "lab_hba1c": [7.2, 8.1, 6.8, 7.5, 9.0],
        "vital_bp_sys": [140, 135, 150, 145, 155],
        "vital_bp_dia": [90, 85, 95, 90, 100],
        "procedure_count": [2, 1, 3, 1, 2],
        "outcome_days": [5, 5, 5, 5, 4],
        "free_text_notes": ["Patient responded well", "Required follow-up", "Stable condition", "Improved significantly", "Monitoring needed"]
    }

    import tempfile

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_csv:
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(temp_csv.name, index=False)
        temp_csv_path = temp_csv.name

    try:
        # Define the task
        task_text = "Identify patients with Type 2 diabetes and analyze glucose control factors"

        # Initialize selector with configuration
        config = {
            "privacy_mode": "strict",
            "keep_free_text": False,
            "max_rows_profile": 100,
            "max_final_columns": 50
        }

        selector = TaskDrivenColumnSelector(config=config)

        # Run the selection process
        result_paths = selector.run(
            input_csv_path=temp_csv_path,
            task_text=task_text
        )

        print("\nExample completed!")
        print(f"Filtered CSV: {result_paths['filtered_csv_path']}")
        print(f"Selection Report: {result_paths['selection_report_path']}")

        # Display the content of the selection report
        with open(result_paths['selection_report_path'], 'r') as f:
            report = json.load(f)
            print("\nSelection Report Summary:")
            print(f"- Final columns: {len(report['final_columns'])}")
            print(f"- Task type: {report['task_spec']['task_type']}")
            print(f"- Needed modalities: {report['task_spec']['need_modalities']}")
            if report['warnings']:
                print(f"- Warnings: {len(report['warnings'])}")

        return result_paths

    finally:
        # Clean up temporary file
        os.unlink(temp_csv_path)


def run_directory_example():
    """Example usage for processing CSV files from a directory"""
    import tempfile

    # Create sample CSV data
    sample_data = {
        "patient_id": ["P001", "P002", "P003"],
        "age": [45, 67, 32],
        "gender": ["M", "F", "F"],
        "diagnosis": ["Diabetes", "Hypertension", "Asthma"],
        "lab_value": [6.5, 140, 42],
        "medication": ["Metformin", "Lisinopril", "Albuterol"],
        "name": ["John Doe", "Jane Smith", "Alice Brown"]  # This will be flagged as PHI
    }

    # Create temporary directory and CSV file
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "input_cleaned_38.csv")
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(csv_path, index=False)

        print(f"Created sample CSV: {csv_path}")

        # Process the file from directory
        task_text = "Analyze patient data for diabetes-related factors"
        config = {"privacy_mode": "strict"}

        selector = TaskDrivenColumnSelector(config=config)

        result_paths = selector.run_on_directory_csv(
            directory_path=temp_dir,
            csv_pattern="input_cleaned_38.csv",
            task_text=task_text
        )

        print(f"\nDirectory example completed!")
        print(f"Filtered CSV: {result_paths['filtered_csv_path']}")
        print(f"Selection Report: {result_paths['selection_report_path']}")

        # Show results
        filtered_df = pd.read_csv(result_paths['filtered_csv_path'])
        print(f"Final columns: {list(filtered_df.columns)}")

        return result_paths


def run_current_liver_dataset():
    """Run the selector against the current liver dataset in the project root."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    input_csv_path = os.path.join(project_root, "liver_notes_extracted.csv")

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Current liver dataset not found: {input_csv_path}")

    task_text = (
        "Identify factors associated with mortality risk and length of stay "
        "among hospitalized patients with cirrhosis or hepatic failure."
    )
    config = {
        "privacy_mode": "strict",
        "keep_free_text": False,
        "max_rows_profile": 1000,
        "max_final_columns": 100,
    }

    print(f"Using input dataset: {input_csv_path}")
    print(f"Task: {task_text}")

    selector = TaskDrivenColumnSelector(config=config)
    result_paths = selector.run(
        input_csv_path=input_csv_path,
        task_text=task_text,
        output_dir=project_root,
    )

    print("\nCurrent liver dataset run completed!")
    print(f"Filtered CSV: {result_paths['filtered_csv_path']}")
    print(f"Selection Report: {result_paths['selection_report_path']}")
    return result_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent_4 task-driven column selection.")
    parser.add_argument(
        "--input",
        "-i",
        default=get_default_input_dir(),
        help=f"输入 CSV/XLSX 文件或目录（默认: {get_default_input_dir()}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=get_default_output_dir(),
        help=f"输出目录（默认: {get_default_output_dir()}）",
    )
    parser.add_argument(
        "--task",
        "-t",
        default=get_default_task_text(),
        help="任务文本；也可通过 AGENT4_TASK_TEXT 环境变量设置。",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_csv_path = resolve_input_csv(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using input dataset: {input_csv_path}")
    print(f"Task: {args.task}")
    print(f"Output dir: {output_dir}")

    selector = TaskDrivenColumnSelector()
    result_paths = selector.run(
        input_csv_path=input_csv_path,
        task_text=args.task,
        output_dir=output_dir,
    )

    print("\nAgent 5 standalone run completed!")
    print(f"Filtered CSV: {result_paths['filtered_csv_path']}")
    print(f"Selection Report: {result_paths['selection_report_path']}")


if __name__ == "__main__":
    main()
