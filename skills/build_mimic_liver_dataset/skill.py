from __future__ import annotations

import json
from pathlib import Path

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

from agent_artifacts import begin_step, get_phase_context, record_step
from skills import _common  # noqa: F401
from mimic_dataset_builder import build_mimic_liver_dataset

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = V2_DIR / "input" / "mimic-mini"
DEFAULT_OUTPUT_ROOT = V2_DIR / "output" / "code_outputs"


def build_mimic_liver_dataset_tool(task_text: str, input_root: str = "", output_root: str = "") -> ToolResponse:
    # Reserve step slot first to avoid step_index mismatch with record_step.
    step_ctx = begin_step("build_mimic_liver_dataset")
    if step_ctx is not None:
        resolved_output_root = output_root or step_ctx["step_dir"]
    else:
        resolved_output_root = output_root or str(DEFAULT_OUTPUT_ROOT)
        Path(resolved_output_root).mkdir(parents=True, exist_ok=True)
    report = build_mimic_liver_dataset(
        input_root=input_root or str(DEFAULT_INPUT_ROOT),
        output_root=resolved_output_root,
        task_text=task_text,
    )
    report_path = report.get("report_path", "")
    dataset_path = report.get("dataset_path", "")
    manifest_record = record_step(
        "build_mimic_liver_dataset",
        "已生成增强版 MIMIC 肝病诊断训练集。",
        [path for path in (dataset_path, report_path) if path],
        handoffs={
            "analysis_report": report_path,
            "ml_dataset_csv": dataset_path,
        },
        metadata={
            "row_count": report.get("row_count"),
            "feature_count": report.get("feature_count"),
            "label_distribution": report.get("label_distribution"),
        },
        _step_ctx=step_ctx,
    )
    payload = {
        "status": "SUCCESS",
        "summary": "已生成增强版 MIMIC 肝病诊断训练集。",
        "artifacts": {
            **report,
            "manifest_path": (manifest_record or {}).get("manifest_path", ""),
            "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", resolved_output_root),
            "handoffs": (manifest_record or {}).get("handoffs", {}),
        },
        "issues": [],
    }
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


SKILL = {
    "name": "build_mimic_liver_dataset",
    "layer": "process",
    "description": "为 MIMIC 结构化数据直接构造增强版肝病诊断训练集，保留 ICD、分项 lab、top 药物和轻量文本特征，并发布标准 handoff 产物。",
    "when_to_use": "仅当任务明确使用 MIMIC 或 mimic-mini 结构化表，并要求构造肝病诊断建模数据集时使用。",
    "when_to_skip": "非 MIMIC 数据集、非肝病建模任务、Gold-Guided 字段恢复任务，或只需执行通用字段规则时必须跳过。",
    "capability_types": ["mimic_liver_dataset_build", "mimic_feature_engineering"],
    "inputs": ["task_text", "input_root?", "output_root?"],
    "outputs": ["dataset_path", "report_path", "row_count", "feature_count", "label_distribution", "manifest_path"],
    "prerequisites": ["mimic_schema_available", "liver_prediction_target_confirmed"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": False,
    "lib_entrypoints": ["mimic_dataset_builder.build_mimic_liver_dataset"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(build_mimic_liver_dataset_tool)
