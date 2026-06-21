from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step1_tools import build_records_parallel_tool, run_reorganize_from_records_tool

SKILL = {
    "name": "reorganize_files",
    "layer": "process",
    "description": "把原始数据目录重组为按患者/模态组织的标准结构，生成 records.json 并执行文件重组。",
    "when_to_use": "原始数据文件散乱、未按患者组织时，作为处理的第一步。如果数据已经是结构化 CSV，可跳过。",
    "when_to_skip": "输入已按 case/患者稳定组织，或原始数据只需原地读取而不应复制重组时跳过。",
    "capability_types": ["case_file_reorganization", "record_manifest_build"],
    "inputs": ["input_path", "output_root?", "records_path?", "max_workers?"],
    "outputs": ["records_path", "output_root", "records_count"],
    "prerequisites": ["input_layout_scanned"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": [
        "step1_tools.build_records_parallel_tool",
        "step1_tools.run_reorganize_from_records_tool",
    ],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(build_records_parallel_tool)
    toolkit.register_tool_function(run_reorganize_from_records_tool)
