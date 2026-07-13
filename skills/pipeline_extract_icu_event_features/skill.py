from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import mimic_root_from_spec, raw_sources, run_pipeline_step, table_info
from workflow.mimic_pipeline import extract_icu_event_features


SOURCE_PIPELINE_FILES = [
    "preprocessing/hosp_module_preproc/feature_selection_icu.py",
    "utils/icu_preprocess_util.py",
    "utils/outlier_removal.py",
    "utils/uom_conversion.py",
]
EXPECTED_RAW_FILES = [
    "icu/icustays.csv",
    "icu/chartevents.csv",
    "icu/d_items.csv",
    "icu/outputevents.csv",
    "icu/procedureevents.csv",
    "icu/inputevents.csv",
]


def pipeline_extract_icu_event_features_tool(spec_json: str):
    """Extract ICU event/chart features for a cohort.

    spec_json fields: mimic_root/raw_root, cohort_path, optional top_n/chunksize.
    """

    def runner(spec, output: Path):
        root = mimic_root_from_spec(spec)
        cohort_path = spec.get("cohort_path")
        if not cohort_path:
            raise ValueError("spec_json requires cohort_path")
        result = extract_icu_event_features(
            root,
            cohort_path,
            output,
            top_n=int(spec.get("top_n") or 20),
            chunksize=int(spec.get("chunksize") or 250_000),
        )
        rows, columns = table_info(result["features"])
        return {
            **result,
            "row_counts": {"icu_features.csv": rows},
            "columns": {"icu_features.csv": columns},
            "source_files": raw_sources(root, EXPECTED_RAW_FILES),
        }

    return run_pipeline_step(
        "pipeline_extract_icu_event_features",
        spec_json,
        runner,
        summary="Extracted ICU event features using teacher-pipeline ICU event logic.",
        artifact_keys=("features",),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_extract_icu_event_features",
    "layer": "process",
    "description": "Pipeline 化 ICU event/chart 特征能力：从 icustays、chartevents、d_items 等生成 ICU stay/count/vital/event 特征。CodexAgent 写 ICU event 大脚本前必须优先尝试。",
    "when_to_use": "reference 或任务包含 ICU stay、chartevents、d_items、vitals、outputevents、procedureevents、inputevents 特征时使用。",
    "when_to_skip": "Non-ICU reference 且没有 ICU event 特征，或 icustays 缺失时跳过。",
    "capability_types": ["pipeline_icu_event_features", "mimic_icu_event_features"],
    "inputs": ["spec_json"],
    "outputs": ["icu_features.csv"],
    "prerequisites": ["cohort_available", "icustays_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.mimic_pipeline.extract_icu_event_features"],
    "source_pipeline_files": [
        "preprocessing/hosp_module_preproc/feature_selection_icu.py",
        "utils/icu_preprocess_util.py",
        "utils/outlier_removal.py",
        "utils/uom_conversion.py",
    ],
    "expected_raw_files": [
        "icu/icustays.csv",
        "icu/chartevents.csv",
        "icu/d_items.csv",
        "icu/outputevents.csv",
        "icu/procedureevents.csv",
        "icu/inputevents.csv",
    ],
    "output_contract": ["CSV keyed by hadm_id with ICU event feature columns"],
    "common_failure_modes": ["cohort lacks hadm_id", "chartevents missing or too large without chunksize"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_extract_icu_event_features_tool)
