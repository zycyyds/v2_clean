from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import mimic_root_from_spec, raw_sources, run_pipeline_step, table_info
from workflow.mimic_pipeline import extract_medication_features


SOURCE_PIPELINE_FILES = [
    "preprocessing/hosp_module_preproc/feature_selection_icu.py",
    "utils/hosp_preprocess_util.py",
]
EXPECTED_RAW_FILES = ["hosp/prescriptions.csv", "icu/inputevents.csv"]


def pipeline_extract_med_features_tool(spec_json: str):
    """Extract medication features for a cohort.

    spec_json fields: mimic_root/raw_root, cohort_path, optional top_n/chunksize.
    """

    def runner(spec, output: Path):
        root = mimic_root_from_spec(spec)
        cohort_path = spec.get("cohort_path")
        if not cohort_path:
            raise ValueError("spec_json requires cohort_path")
        result = extract_medication_features(
            root,
            cohort_path,
            output,
            top_n=int(spec.get("top_n") or 30),
            chunksize=int(spec.get("chunksize") or 250_000),
        )
        rows, columns = table_info(result["features"])
        return {
            **result,
            "row_counts": {"medication_features.csv": rows},
            "columns": {"medication_features.csv": columns},
            "source_files": raw_sources(root, EXPECTED_RAW_FILES),
        }

    return run_pipeline_step(
        "pipeline_extract_med_features",
        spec_json,
        runner,
        summary="Extracted medication features using teacher-pipeline medication logic.",
        artifact_keys=("features",),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_extract_med_features",
    "layer": "process",
    "description": "Pipeline 化用药特征能力：从 prescriptions 和可扩展 ICU inputevents 生成药物 order/count/top-drug 特征。CodexAgent 写 medication 脚本前优先使用。",
    "when_to_use": "reference 或任务包含 prescriptions、drug、medication、inputevents 用药特征，且已有 cohort.csv 时使用。",
    "when_to_skip": "reference 不需要用药特征，或 prescriptions/inputevents 均不可用时跳过。",
    "capability_types": ["pipeline_medication_features", "mimic_medication_features"],
    "inputs": ["spec_json"],
    "outputs": ["medication_features.csv"],
    "prerequisites": ["cohort_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.mimic_pipeline.extract_medication_features"],
    "source_pipeline_files": [
        "preprocessing/hosp_module_preproc/feature_selection_icu.py",
        "utils/hosp_preprocess_util.py",
    ],
    "expected_raw_files": ["hosp/prescriptions.csv", "icu/inputevents.csv"],
    "output_contract": ["CSV keyed by hadm_id with medication feature columns"],
    "common_failure_modes": ["drug column missing", "inputevents-only medication needs adapter"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_extract_med_features_tool)
