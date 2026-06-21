from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step23_tools import scan_step23_input_tool

SKILL = {
    "name": "scan_patients",
    "layer": "explore",
    "description": "统计数据目录中的患者数、图片数、文本文件数、表格数。",
    "when_to_use": "数据包含多模态（图像+文本+表格）时，用于了解各模态的规模。",
    "when_to_skip": "数据不是按患者目录组织，或只包含单一结构化表且无需患者级模态统计时跳过。",
    "capability_types": ["case_inventory", "modality_count"],
    "inputs": ["input_path"],
    "outputs": ["patient_count", "image_count", "text_count", "table_count"],
    "prerequisites": ["input_path_accessible"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step23_tools.scan_step23_input_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(scan_step23_input_tool)
