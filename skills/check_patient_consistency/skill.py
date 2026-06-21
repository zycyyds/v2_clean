from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step6_tools import load_data_overview, analyze_all_patients_consistency

SKILL = {
    "name": "check_patient_consistency",
    "layer": "process",
    "description": "跨表检查患者数据一致性，识别记录缺失、时间矛盾、ID 不匹配等问题。",
    "when_to_use": "数据来自多张表（admissions、icustays、labevents 等）需要合并时，先检查一致性。",
    "when_to_skip": "输入不是旧 Step6 患者数据布局、缺少其全局上下文，或任务只处理单表时跳过。",
    "capability_types": ["patient_consistency_check", "cross_table_integrity"],
    "inputs": [],
    "outputs": ["passed_patients_path", "consistency_report"],
    "prerequisites": ["legacy_step6_context_available"],
    "adapter_policy": "adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": [
        "step6_tools.load_data_overview",
        "step6_tools.analyze_all_patients_consistency",
    ],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(load_data_overview)
    toolkit.register_tool_function(analyze_all_patients_consistency)
