from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step7_runtime import discover_diagnosis_fields, get_column_distribution

SKILL = {
    "name": "discover_label_candidates",
    "layer": "label",
    "description": "自动发现数据中可用作标签的列：ICD 诊断码列、事件列、分类列，并返回各列的取值分布。",
    "when_to_use": "构建标签前，先调用此 skill 了解哪些列可以作为预测目标。",
    "when_to_skip": "任务不需要预测标签，或标签来源和定义已经由用户明确指定时跳过。",
    "capability_types": ["label_candidate_discovery", "target_column_profile"],
    "inputs": ["input_csv_path?", "task_text?"],
    "outputs": ["diagnosis_fields", "event_fields", "column_distributions"],
    "prerequisites": ["feature_table_available"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": [
        "step7_runtime.discover_diagnosis_fields",
        "step7_runtime.get_column_distribution",
    ],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(discover_diagnosis_fields)
    toolkit.register_tool_function(get_column_distribution)
