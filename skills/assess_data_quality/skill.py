from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step5_tools import classify_step5_column_risks_tool

SKILL = {
    "name": "assess_data_quality",
    "layer": "process",
    "description": "对 CSV 各列评估数据质量风险（低/中/高），决定哪些列需要清洗。",
    "when_to_use": "profile_table 之后，clean_data 之前，用于制定清洗策略。",
    "when_to_skip": "没有表格画像，或当前任务只做原值抽取、不执行清洗时跳过。",
    "capability_types": ["data_quality_assessment", "cleaning_risk_classification"],
    "inputs": ["input_csv_path", "profile_report_path?", "output_root?"],
    "outputs": ["column_risk_report_path", "risk_summary"],
    "prerequisites": ["table_profile_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step5_tools.classify_step5_column_risks_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(classify_step5_column_risks_tool)
