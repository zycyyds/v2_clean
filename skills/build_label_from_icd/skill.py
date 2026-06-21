from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step7_runtime import create_icd_binary_label

SKILL = {
    "name": "build_label_from_icd",
    "layer": "label",
    "description": "基于 ICD 诊断码列构造二分类标签（患有某疾病=1，否则=0）。",
    "when_to_use": "任务目标是疾病诊断（如肝癌、肺炎），且数据中有 ICD 诊断码列时。",
    "when_to_skip": "任务未要求疾病标签、没有 ICD 码列，或目标 ICD-9/ICD-10 范围未明确时跳过。",
    "capability_types": ["label_from_icd", "diagnosis_label_build"],
    "inputs": ["icd_column", "target_icd_codes", "label_name?"],
    "outputs": ["label_series_id", "positive_count", "negative_count"],
    "prerequisites": ["label_target_confirmed", "icd_code_rule_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step7_runtime.create_icd_binary_label"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(create_icd_binary_label)
