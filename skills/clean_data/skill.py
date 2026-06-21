from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step5_tools import run_step5_data_cleaning_tool

SKILL = {
    "name": "clean_data",
    "layer": "process",
    "description": "对 CSV 执行数据清洗：缺失值填充、类型转换、异常值修复，输出 cleaned.csv。",
    "when_to_use": "profile_table 或 assess_data_quality 发现数据质量问题后调用。",
    "when_to_skip": "没有质量问题、必须保留原始值，或验证阶段禁止填补和不可逆修复时跳过。",
    "capability_types": ["data_cleaning", "missing_value_imputation", "type_conversion"],
    "inputs": ["input_csv_path", "column_risk_report_path?", "output_root?", "workers?", "enable_llm?"],
    "outputs": ["cleaned_csv_path", "cleaning_report_path"],
    "prerequisites": ["cleaning_policy_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": False,
    "lib_entrypoints": ["step5_tools.run_step5_data_cleaning_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(run_step5_data_cleaning_tool)
