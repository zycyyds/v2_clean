from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import mimic_root_from_spec, raw_sources, run_pipeline_step, table_info
from workflow.mimic_pipeline import extract_lab_features


SOURCE_PIPELINE_FILES = [
    "preprocessing/hosp_module_preproc/feature_selection_icu.py",
    "utils/labs_preprocess_util.py",
    "utils/outlier_removal.py",
    "utils/uom_conversion.py",
]
EXPECTED_RAW_FILES = ["hosp/labevents.csv", "hosp/d_labitems.csv"]


def pipeline_extract_lab_features_tool(spec_json: str):
    """Extract lab features for a cohort.

    spec_json fields: mimic_root/raw_root, cohort_path, optional top_n/chunksize.
    """

    def runner(spec, output: Path):
        root = mimic_root_from_spec(spec)
        cohort_path = spec.get("cohort_path")
        if not cohort_path:
            raise ValueError("spec_json requires cohort_path")
        result = extract_lab_features(
            root,
            cohort_path,
            output,
            top_n=int(spec.get("top_n") or 30),
            chunksize=int(spec.get("chunksize") or 250_000),
        )
        rows, columns = table_info(result["features"])
        return {
            **result,
            "row_counts": {"lab_features.csv": rows},
            "columns": {"lab_features.csv": columns},
            "source_files": raw_sources(root, EXPECTED_RAW_FILES),
        }

    return run_pipeline_step(
        "pipeline_extract_lab_features",
        spec_json,
        runner,
        summary="Extracted lab features using teacher-pipeline lab logic.",
        artifact_keys=("features",),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_extract_lab_features",
    "layer": "process",
    "description": "Pipeline 化实验室指标能力：分块读取 labevents，连接 d_labitems，生成 count/mean/min/max 等 lab 特征。CodexAgent 写 lab 大脚本前必须优先尝试。",
    "when_to_use": "reference 或任务包含 lab/labevents/d_labitems/检验指标特征，且已有 cohort.csv 时使用。",
    "when_to_skip": "reference 不需要实验室指标，或 labevents 缺失时跳过并说明。",
    "capability_types": ["pipeline_lab_features", "mimic_lab_features"],
    "inputs": ["spec_json"],
    "outputs": ["lab_features.csv"],
    "prerequisites": ["cohort_available", "labevents_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.mimic_pipeline.extract_lab_features"],
    "source_pipeline_files": [
        "preprocessing/hosp_module_preproc/feature_selection_icu.py",
        "utils/labs_preprocess_util.py",
        "utils/outlier_removal.py",
        "utils/uom_conversion.py",
    ],
    "expected_raw_files": ["hosp/labevents.csv", "hosp/d_labitems.csv"],
    "output_contract": ["CSV keyed by hadm_id with lab feature columns"],
    "common_failure_modes": ["cohort lacks hadm_id", "labevents very large and needs chunksize"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_extract_lab_features_tool)
