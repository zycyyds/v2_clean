from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step5_tools import profile_step5_table_tool

SKILL = {
    "name": "profile_table",
    "layer": "process",
    "description": "对指定 CSV 做列画像：分布、缺失率、数据类型、样本值。",
    "when_to_use": "在 clean_data 或 select_features 之前，了解表的数据质量和列特征。",
    "when_to_skip": "输入不是 CSV，或已有与当前文件指纹一致的列画像报告时跳过。",
    "capability_types": ["table_profile", "column_statistics"],
    "inputs": ["input_csv_path", "output_root?", "workers?"],
    "outputs": ["profile_report_path", "column_stats"],
    "prerequisites": ["csv_available"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step5_tools.profile_step5_table_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(profile_step5_table_tool)
