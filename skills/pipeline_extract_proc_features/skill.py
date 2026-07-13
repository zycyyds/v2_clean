from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import mimic_root_from_spec, raw_sources, run_pipeline_step, table_info
from workflow.mimic_pipeline import extract_procedure_features


SOURCE_PIPELINE_FILES = [
    "preprocessing/hosp_module_preproc/feature_selection_icu.py",
    "utils/icu_preprocess_util.py",
    "utils/hosp_preprocess_util.py",
]
EXPECTED_RAW_FILES = [
    "hosp/procedures_icd.csv",
    "hosp/d_icd_procedures.csv",
    "icu/procedureevents.csv",
]


def pipeline_extract_proc_features_tool(spec_json: str):
    """Extract procedure features for a cohort.

    spec_json fields: mimic_root/raw_root, cohort_path, optional top_n.
    """

    def runner(spec, output: Path):
        root = mimic_root_from_spec(spec)
        cohort_path = spec.get("cohort_path")
        if not cohort_path:
            raise ValueError("spec_json requires cohort_path")
        result = extract_procedure_features(root, cohort_path, output, top_n=int(spec.get("top_n") or 20))
        rows, columns = table_info(result["features"])
        return {
            **result,
            "row_counts": {"procedure_features.csv": rows},
            "columns": {"procedure_features.csv": columns},
            "source_files": raw_sources(root, EXPECTED_RAW_FILES),
        }

    return run_pipeline_step(
        "pipeline_extract_proc_features",
        spec_json,
        runner,
        summary="Extracted procedure features using teacher-pipeline procedure logic.",
        artifact_keys=("features",),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_extract_proc_features",
    "layer": "process",
    "description": "Pipeline 化手术/操作特征能力：从 procedures_icd、d_icd_procedures 及可用 ICU procedure events 生成操作特征。CodexAgent 写 procedure 脚本前优先使用。",
    "when_to_use": "reference 或任务包含 procedures、procedure titles、procedure counts、ICU procedure event 特征时使用。",
    "when_to_skip": "reference 不需要手术/操作特征，或 cohort 没有 hadm_id 且不能回连时跳过。",
    "capability_types": ["pipeline_procedure_features", "mimic_procedure_features"],
    "inputs": ["spec_json"],
    "outputs": ["procedure_features.csv"],
    "prerequisites": ["cohort_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.mimic_pipeline.extract_procedure_features"],
    "source_pipeline_files": [
        "preprocessing/hosp_module_preproc/feature_selection_icu.py",
        "utils/icu_preprocess_util.py",
        "utils/hosp_preprocess_util.py",
    ],
    "expected_raw_files": [
        "hosp/procedures_icd.csv",
        "hosp/d_icd_procedures.csv",
        "icu/procedureevents.csv",
    ],
    "output_contract": ["CSV keyed by hadm_id with procedure feature columns"],
    "common_failure_modes": ["procedures_icd missing", "procedure dictionary missing"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_extract_proc_features_tool)
