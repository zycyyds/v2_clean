from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import mimic_root_from_spec, raw_sources, run_pipeline_step, table_info
from workflow.mimic_pipeline import extract_diagnosis_features


SOURCE_PIPELINE_FILES = [
    "preprocessing/hosp_module_preproc/feature_selection_icu.py",
    "utils/hosp_preprocess_util.py",
]
EXPECTED_RAW_FILES = ["hosp/diagnoses_icd.csv", "hosp/d_icd_diagnoses.csv"]


def pipeline_extract_diag_features_tool(spec_json: str):
    """Extract diagnosis ICD features for a cohort.

    spec_json fields: mimic_root/raw_root, cohort_path, optional top_n.
    """

    def runner(spec, output: Path):
        root = mimic_root_from_spec(spec)
        cohort_path = spec.get("cohort_path")
        if not cohort_path:
            raise ValueError("spec_json requires cohort_path")
        result = extract_diagnosis_features(root, cohort_path, output, top_n=int(spec.get("top_n") or 20))
        rows, columns = table_info(result["features"])
        return {
            **result,
            "row_counts": {"diagnosis_features.csv": rows},
            "columns": {"diagnosis_features.csv": columns},
            "source_files": raw_sources(root, EXPECTED_RAW_FILES),
        }

    return run_pipeline_step(
        "pipeline_extract_diag_features",
        spec_json,
        runner,
        summary="Extracted diagnosis features using teacher-pipeline ICD logic.",
        artifact_keys=("features",),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_extract_diag_features",
    "layer": "process",
    "description": "Pipeline 化诊断特征能力：从 diagnoses_icd 和 d_icd_diagnoses 生成 ICD count/list/title/top-prefix 特征。CodexAgent 写诊断特征脚本前必须优先尝试。",
    "when_to_use": "train/reference 或目标 package 包含诊断 ICD、诊断标题、主诊断或诊断前缀特征，且已有 cohort.csv 时使用。",
    "when_to_skip": "reference 不需要诊断特征，或 diagnoses_icd 缺失时跳过并记录 unsupported。",
    "capability_types": ["pipeline_diagnosis_features", "mimic_diagnosis_features"],
    "inputs": ["spec_json"],
    "outputs": ["diagnosis_features.csv"],
    "prerequisites": ["cohort_available", "diagnoses_icd_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.mimic_pipeline.extract_diagnosis_features"],
    "source_pipeline_files": [
        "preprocessing/hosp_module_preproc/feature_selection_icu.py",
        "utils/hosp_preprocess_util.py",
    ],
    "expected_raw_files": ["hosp/diagnoses_icd.csv", "hosp/d_icd_diagnoses.csv"],
    "output_contract": ["CSV keyed by hadm_id with diagnosis feature columns"],
    "common_failure_modes": ["cohort lacks hadm_id", "dictionary join keys missing"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_extract_diag_features_tool)
