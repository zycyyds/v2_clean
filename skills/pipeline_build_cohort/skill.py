from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import (
    maybe_copy,
    mimic_root_from_spec,
    raw_sources,
    run_pipeline_step,
    table_info,
    task_spec_from_spec,
)
from workflow.mimic_pipeline import apply_case_key_filter, build_outcome_label, build_visit_base


SOURCE_PIPELINE_FILES = [
    "preprocessing/day_intervals_preproc/day_intervals_cohort_v3.py",
    "mimic4_preprocess_util.py",
]
EXPECTED_RAW_FILES = [
    "hosp/patients.csv",
    "hosp/admissions.csv",
    "icu/icustays.csv",
]


def pipeline_build_cohort_tool(spec_json: str):
    """Build a MIMIC cohort from train/raw or validation/raw.

    spec_json fields:
    - mimic_root/raw_root: path containing hosp/ and optionally icu/.
    - task_spec: care_setting, outcome_type, time_window_days, record_grain.
    - case_keys_path: optional split keys CSV to filter the cohort.
    - include_label: optional, default true.
    """

    def runner(spec, output: Path):
        root = mimic_root_from_spec(spec)
        task_spec = task_spec_from_spec(spec)
        visit = build_visit_base(root, output / "visit_base", task_spec)
        current = visit["visit_base"]
        label_result = {}
        if spec.get("include_label", True):
            label_result = build_outcome_label(current, output / "label", task_spec)
            current = label_result["cohort"]
        if spec.get("case_keys_path"):
            filtered = apply_case_key_filter(
                current,
                spec["case_keys_path"],
                output / "split_filter",
                preferred_key=str(task_spec.get("record_grain") or ""),
            )
            current = filtered["cohort"]
            label_result = {**label_result, **filtered}
        target = maybe_copy(current, output / "cohort.csv")
        rows, columns = table_info(target)
        return {
            "status": "SUCCESS",
            "cohort": str(target),
            "visit_base": visit.get("visit_base", ""),
            "row_counts": {"cohort.csv": rows},
            "columns": {"cohort.csv": columns},
            "source_files": raw_sources(root, EXPECTED_RAW_FILES),
            "record_grain": task_spec.get("record_grain", ""),
            "label_distribution": label_result.get("label_distribution", {}),
            "split_key": label_result.get("split_key", ""),
        }

    return run_pipeline_step(
        "pipeline_build_cohort",
        spec_json,
        runner,
        summary="Built cohort using parameterized teacher-pipeline cohort logic.",
        artifact_keys=("cohort", "visit_base"),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_build_cohort",
    "layer": "process",
    "description": "Pipeline 化 cohort 构建能力：按 spec_json 中的 ICU/Non-ICU、outcome、record_grain 和 split keys 构建 MIMIC cohort。CodexAgent 在写 cohort 大脚本前必须优先尝试。",
    "when_to_use": "reference-guided 任务需要从 MIMIC train/raw 或 validation/raw 生成 cohort.csv，或需要复用 pipeline 的 visit/outcome 逻辑时使用。",
    "when_to_skip": "输入不是 MIMIC-IV 3.1 结构化 raw，或 reference 不是 cohort/visit 粒度结果时跳过并说明原因。",
    "capability_types": ["pipeline_cohort", "mimic_visit_base", "mimic_outcome_label"],
    "inputs": ["spec_json"],
    "outputs": ["cohort.csv", "visit_base.csv(optional)", "row_counts", "columns", "source_files"],
    "prerequisites": ["mimic_raw_root_available", "task_or_reference_shape_inferred"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": [
        "workflow.mimic_pipeline.build_visit_base",
        "workflow.mimic_pipeline.build_outcome_label",
        "workflow.mimic_pipeline.apply_case_key_filter",
    ],
    "source_pipeline_files": [
        "preprocessing/day_intervals_preproc/day_intervals_cohort_v3.py",
        "mimic4_preprocess_util.py",
    ],
    "expected_raw_files": [
        "hosp/patients.csv",
        "hosp/admissions.csv",
        "icu/icustays.csv",
    ],
    "output_contract": ["cohort CSV must contain a split key such as stay_id, hadm_id, or subject_id"],
    "common_failure_modes": ["missing admissions/patients/icustays", "case_keys_path has no shared key"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_build_cohort_tool)
