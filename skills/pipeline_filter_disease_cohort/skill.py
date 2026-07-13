from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import mimic_root_from_spec, raw_sources, run_pipeline_step, table_info
from workflow.mimic_pipeline import filter_disease_cohort, resolve_disease_icd


SOURCE_PIPELINE_FILES = ["preprocessing/day_intervals_preproc/disease_cohort.py"]
EXPECTED_RAW_FILES = ["hosp/diagnoses_icd.csv", "hosp/d_icd_diagnoses.csv"]


def pipeline_filter_disease_cohort_tool(spec_json: str):
    """Filter an existing cohort by disease ICD rules.

    spec_json fields:
    - mimic_root/raw_root.
    - cohort_path.
    - disease_rule with icd_prefixes, or disease_text to resolve from d_icd_diagnoses.
    - primary_only: optional boolean.
    """

    def runner(spec, output: Path):
        root = mimic_root_from_spec(spec)
        cohort_path = spec.get("cohort_path")
        if not cohort_path:
            raise ValueError("spec_json requires cohort_path")
        disease_rule = spec.get("disease_rule") or {}
        if not disease_rule and spec.get("disease_text"):
            disease_rule = resolve_disease_icd(root, str(spec["disease_text"]))
        result = filter_disease_cohort(
            cohort_path,
            root,
            output,
            disease_rule,
            primary_only=bool(spec.get("primary_only", False)),
        )
        rows, columns = table_info(result["cohort"])
        return {
            **result,
            "row_counts": {"disease_filtered_cohort.csv": rows},
            "columns": {"disease_filtered_cohort.csv": columns},
            "source_files": raw_sources(root, EXPECTED_RAW_FILES),
            "disease_rule": disease_rule,
        }

    return run_pipeline_step(
        "pipeline_filter_disease_cohort",
        spec_json,
        runner,
        summary="Filtered cohort using teacher-pipeline disease ICD logic.",
        artifact_keys=("cohort",),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_filter_disease_cohort",
    "layer": "process",
    "description": "Pipeline 化疾病 cohort 筛选能力：按 ICD prefix 或疾病文本解析结果筛选 cohort。CodexAgent 遇到 disease/admitted_due_to cohort 规则时优先使用。",
    "when_to_use": "reference 或任务要求按疾病、ICD、入院原因筛选 cohort，且已有 cohort.csv 时使用。",
    "when_to_skip": "reference 没有疾病 cohort 条件，或 diagnoses 字典缺失且疾病规则无法解析时跳过。",
    "capability_types": ["pipeline_disease_filter", "mimic_disease_cohort"],
    "inputs": ["spec_json"],
    "outputs": ["disease_filtered_cohort.csv", "matched_hadm_count", "icd_prefixes"],
    "prerequisites": ["cohort_available", "mimic_diagnoses_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": [
        "workflow.mimic_pipeline.resolve_disease_icd",
        "workflow.mimic_pipeline.filter_disease_cohort",
    ],
    "source_pipeline_files": ["preprocessing/day_intervals_preproc/disease_cohort.py"],
    "expected_raw_files": ["hosp/diagnoses_icd.csv", "hosp/d_icd_diagnoses.csv"],
    "output_contract": ["filtered cohort keeps original cohort columns and row grain"],
    "common_failure_modes": ["no ICD prefixes", "disease text maps to ambiguous ICD candidates"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_filter_disease_cohort_tool)
