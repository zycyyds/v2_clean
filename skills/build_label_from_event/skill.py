from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step7_runtime import create_notnull_binary_label

SKILL = {
    "name": "build_label_from_event",
    "layer": "label",
    "description": "基于事件是否发生构造二分类标签（如 deathtime 非空=死亡=1，hospital_expire_flag=1=死亡）。",
    "when_to_use": "任务目标是预测某事件是否发生（ICU 死亡、再入院、手术等），数据中有对应的事件列时。",
    "when_to_skip": "任务未要求事件预测、事件列不存在，或事件定义和观察窗尚未确认时跳过。",
    "capability_types": ["label_from_event", "binary_label_build"],
    "inputs": ["event_column", "label_name?"],
    "outputs": ["label_series_id", "positive_count", "negative_count"],
    "prerequisites": ["label_target_confirmed", "event_column_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step7_runtime.create_notnull_binary_label"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(create_notnull_binary_label)
