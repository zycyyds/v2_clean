"""硬规则层：在 agent 结果之外，再做一次强制过滤。"""
import copy
from typing import Dict, Any, List, Optional


class HardGates:
    """处理 PHI、自由文本、常量列、全空列等硬规则。"""

    def __init__(
        self,
        privacy_mode: str = "research",
        keep_free_text: bool = False,
        max_final_columns: Optional[int] = None
    ):
        """初始化硬规则配置。"""
        self.privacy_mode = privacy_mode
        self.keep_free_text = keep_free_text
        self.max_final_columns = max_final_columns

    def apply_gates(self, selection_result: Dict[str, Any], schema_profile: Dict[str, Any]) -> Dict[str, Any]:
        """把硬规则应用到 agent 筛选结果上。"""
        final_result = copy.deepcopy(selection_result)
        overrides = []
        warnings = []
        phi_columns = [
            col_name
            for col_name, col_profile in schema_profile["columns"].items()
            if col_profile.get("is_potential_phi", False)
        ]

        if self.privacy_mode == "strict":
            for col in phi_columns:
                self._move_column(
                    final_result,
                    col,
                    source_keys=["must_have_columns", "useful_columns"],
                    target_key="drop_columns",
                    overrides=overrides,
                    reason="PHI protection in strict mode",
                    source_actions={
                        "must_have_columns": "removed_from_must_have",
                        "useful_columns": "removed_from_useful",
                    },
                )

        if not self.keep_free_text:
            for col in self._get_text_columns(schema_profile):
                self._move_column(
                    final_result,
                    col,
                    source_keys=["must_have_columns", "useful_columns"],
                    target_key="maybe_columns",
                    overrides=overrides,
                    reason="Free text in non-keep mode",
                    source_actions={
                        "must_have_columns": "downgraded_to_maybe",
                        "useful_columns": "downgraded_to_maybe",
                    },
                )

        for col_name, col_profile in schema_profile["columns"].items():
            if col_profile.get("is_constant", False):
                if col_name in final_result["must_have_columns"]:
                    col_modality = col_profile.get("inferred_modality", "")
                    if col_modality != "Identifiers/Keys":
                        final_result["must_have_columns"].remove(col_name)
                        final_result["drop_columns"].append(col_name)
                        overrides.append({
                            "column": col_name,
                            "action": "removed_constant",
                            "reason": "Constant value column",
                        })
                    else:
                        warnings.append(f"Keeping constant identifier column: {col_name} (might be needed for joins)")
            elif col_profile.get("missing_rate", 0) == 1.0:
                removed = self._remove_selected_column(final_result, col_name)
                if removed:
                    final_result["drop_columns"].append(col_name)
                    overrides.append({
                        "column": col_name,
                        "action": "removed_all_missing",
                        "reason": "All missing values column",
                    })

        time_needed = self._check_time_needed(selection_result.get("task_spec", {}))
        if time_needed:
            if not self._get_time_columns(schema_profile):
                warnings.append("Task requires temporal analysis but no time columns detected in data")

        selected_columns = set(
            final_result["must_have_columns"] +
            final_result["useful_columns"] +
            final_result["maybe_columns"]
        )
        final_columns = [
            col_name for col_name in schema_profile["columns"].keys()
            if col_name in selected_columns
        ]

        if isinstance(self.max_final_columns, int) and self.max_final_columns > 0 and len(final_columns) > self.max_final_columns:
            protected_columns = set(final_result["must_have_columns"])
            protected_final_columns = [
                col_name for col_name in final_columns
                if col_name in protected_columns
            ]
            remaining_slots = max(0, self.max_final_columns - len(protected_final_columns))
            non_protected_columns = [
                col_name for col_name in final_columns
                if col_name not in protected_columns
            ]
            capped_final_columns = protected_final_columns + non_protected_columns[:remaining_slots]
            dropped_for_cap = [
                col_name for col_name in final_columns
                if col_name not in set(capped_final_columns)
            ]
            final_columns = capped_final_columns

            for col in dropped_for_cap:
                if col in final_result["must_have_columns"]:
                    final_result["must_have_columns"].remove(col)
                if col in final_result["useful_columns"]:
                    final_result["useful_columns"].remove(col)
                if col in final_result["maybe_columns"]:
                    final_result["maybe_columns"].remove(col)
                final_result["drop_columns"].append(col)
                overrides.append({
                    "column": col,
                    "action": "truncated_by_max_final_columns",
                    "reason": f"Exceeded max_final_columns={self.max_final_columns}"
                })

            if len(protected_final_columns) > self.max_final_columns:
                warnings.append(
                    f"Final columns exceed max_final_columns={self.max_final_columns} "
                    f"because {len(protected_final_columns)} must-have columns were protected"
                )
            else:
                warnings.append(
                    f"Trimmed final columns from {len(final_columns) + len(dropped_for_cap)} "
                    f"to {self.max_final_columns} due to max_final_columns setting"
                )

        final_result["must_have_columns"] = self._dedupe(final_result["must_have_columns"])
        final_result["useful_columns"] = self._dedupe(final_result["useful_columns"])
        final_result["maybe_columns"] = self._dedupe(final_result["maybe_columns"])
        final_result["drop_columns"] = self._dedupe(final_result["drop_columns"])
        final_columns = self._dedupe(final_columns)

        final_result["final_columns"] = final_columns
        final_result["overrides"] = overrides
        final_result["warnings"] = self._dedupe(warnings + selection_result.get("warnings", []))

        return final_result

    def _dedupe(self, items: List[Any]) -> List[Any]:
        """列表去重，同时保留原始顺序。"""
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _move_column(
        self,
        result: Dict[str, Any],
        column: str,
        source_keys: List[str],
        target_key: str,
        overrides: List[Dict[str, Any]],
        reason: str,
        source_actions: Dict[str, str],
    ) -> None:
        """把列从一个层级移动到另一个层级，用于统一处理 override。"""
        for source_key in source_keys:
            if column in result[source_key]:
                result[source_key].remove(column)
                result[target_key].append(column)
                overrides.append({
                    "column": column,
                    "action": source_actions[source_key],
                    "reason": reason,
                })
                return

    def _remove_selected_column(self, result: Dict[str, Any], column: str) -> bool:
        """把列从 must/useful/maybe 中全部移除，并返回是否真的移除了内容。"""
        removed = False
        for key in ["must_have_columns", "useful_columns", "maybe_columns"]:
            if column in result[key]:
                result[key].remove(column)
                removed = True
        return removed

    def _get_text_columns(self, schema_profile: Dict[str, Any]) -> List[str]:
        """根据 dtype 识别文本列。"""
        text_cols = []
        for col_name, profile in schema_profile["columns"].items():
            dtype = profile.get("dtype_guess", "")
            if any(text_type in dtype.lower() for text_type in ["object", "string", "str"]):
                modality = profile.get("inferred_modality", "")
                if modality != "Identifiers/Keys":
                    text_cols.append(col_name)
        return text_cols

    def _get_time_columns(self, schema_profile: Dict[str, Any]) -> List[str]:
        """识别时间相关列。"""
        time_cols = []
        for col_name, profile in schema_profile["columns"].items():
            col_lower = col_name.lower()
            dtype = profile.get("dtype_guess", "")
            if any(time_type in dtype.lower() for time_type in ["datetime", "timestamp", "date", "time"]):
                time_cols.append(col_name)
            elif any(keyword in col_lower for keyword in [
                "date", "time", "datetime", "timestamp", "admission", "discharge",
                "visit", "encounter", "onset", "death", "event", "follow_up",
                "birth", "created", "modified", "updated", "started", "ended",
                "day", "month", "year"
            ]):
                time_cols.append(col_name)
        return time_cols

    def _check_time_needed(self, task_spec: Dict[str, Any]) -> bool:
        """判断任务是否需要时间维度。"""
        need_modalities = task_spec.get("need_modalities", [])
        if "Time" in need_modalities:
            return True

        task_type = task_spec.get("task_type", "")
        task_desc = task_type.lower()
        time_sensitive_tasks = [
            "cohort_selection", "longitudinal", "follow_up", "outcomes",
            "time_to_event", "survival", "temporal", "prognosis", "prediction"
        ]

        for time_task in time_sensitive_tasks:
            if time_task in task_desc:
                return True

        return False
