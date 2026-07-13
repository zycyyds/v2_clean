from __future__ import annotations

from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import run_pipeline_step, table_info
from workflow.mimic_pipeline import clean_feature_table


SOURCE_PIPELINE_FILES = ["utils/outlier_removal.py", "utils/uom_conversion.py"]
EXPECTED_RAW_FILES: list[str] = []


def pipeline_clean_feature_table_tool(spec_json: str):
    """Merge cohort and feature CSVs into a cleaned wide table.

    spec_json fields:
    - cohort_path.
    - feature_paths: list of CSV feature files.
    - record_grain: optional, default hadm_id.
    """

    def runner(spec, output: Path):
        cohort_path = spec.get("cohort_path")
        if not cohort_path:
            raise ValueError("spec_json requires cohort_path")
        feature_paths = spec.get("feature_paths") or []
        if not isinstance(feature_paths, list):
            raise ValueError("spec_json.feature_paths must be a list")
        result = clean_feature_table(
            cohort_path,
            feature_paths,
            output,
            record_grain=str(spec.get("record_grain") or "hadm_id"),
        )
        rows, columns = table_info(result["features_wide"])
        return {
            **result,
            "row_counts": {"features_wide.csv": rows},
            "columns": {"features_wide.csv": columns},
            "source_files": [str(Path(path).expanduser().resolve()) for path in [cohort_path, *feature_paths] if path],
        }

    return run_pipeline_step(
        "pipeline_clean_feature_table",
        spec_json,
        runner,
        summary="Merged and normalized feature table using teacher-pipeline cleaning logic.",
        artifact_keys=("features_wide",),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_clean_feature_table",
    "layer": "process",
    "description": "Pipeline 化特征表清洗/合并能力：按 record_grain 合并 cohort 与多个 feature CSV，做类型规范化和基础清洗。CodexAgent 组装宽表前必须优先尝试。",
    "when_to_use": "已经生成 cohort.csv 和一个或多个 feature CSV，需要得到 features_wide.csv 或 final_dataset.csv 前使用。",
    "when_to_skip": "reference 是多文件明细包且不需要合并宽表时，可以跳过并直接 package assembly。",
    "capability_types": ["pipeline_feature_cleaning", "mimic_clean_feature_table"],
    "inputs": ["spec_json"],
    "outputs": ["features_wide.csv"],
    "prerequisites": ["cohort_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.mimic_pipeline.clean_feature_table"],
    "source_pipeline_files": ["utils/outlier_removal.py", "utils/uom_conversion.py"],
    "expected_raw_files": [],
    "output_contract": ["CSV keyed by record_grain with cohort columns plus feature columns"],
    "common_failure_modes": ["feature files lack record_grain", "duplicate feature columns are dropped by merge policy"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_clean_feature_table_tool)
