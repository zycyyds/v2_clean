from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step7_runtime import propose_label_mapping, build_label_series

SKILL = {
    "name": "build_label_from_column",
    "layer": "label",
    "description": "从分类列的取值映射构造二分类标签，LLM 自动提出映射方案（如 discharge_location: 'DIED'→1）。",
    "when_to_use": "任务目标可以从某个分类列的值推断（如出院去向、治疗结果），且没有直接的 ICD 码时。",
    "when_to_skip": "任务未要求预测标签、标签不是二分类，或没有经过确认的分类值映射时跳过。",
    "capability_types": ["label_from_column", "binary_label_build"],
    "inputs": ["column_name", "task_text?", "mapping?"],
    "outputs": ["label_series_id", "mapping_used", "positive_count"],
    "prerequisites": ["label_target_confirmed", "categorical_column_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": [
        "step7_runtime.propose_label_mapping",
        "step7_runtime.build_label_series",
    ],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(propose_label_mapping)
    toolkit.register_tool_function(build_label_series)
