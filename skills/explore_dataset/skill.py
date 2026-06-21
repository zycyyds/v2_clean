from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step1_tools import scan_source_files_tool

SKILL = {
    "name": "explore_dataset",
    "layer": "explore",
    "description": "扫描数据目录，识别文件类型（CSV/图片/文本/PDF）、表结构、行数、列名。",
    "when_to_use": "拿到任何新数据集时第一步调用，了解数据全貌后再规划处理路径。",
    "when_to_skip": "已经存在经过校验的文件清单和 schema 报告，且输入目录没有变化时跳过。",
    "capability_types": ["input_layout_discovery", "modality_discovery"],
    "inputs": ["input_path", "max_sample_tasks?"],
    "outputs": ["file_list", "modalities", "sample_tasks"],
    "prerequisites": ["input_path_accessible"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step1_tools.scan_source_files_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(scan_source_files_tool)
