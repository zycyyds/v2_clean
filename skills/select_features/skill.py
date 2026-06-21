from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step4_tools import run_step4_column_selection_tool

SKILL = {
    "name": "select_features",
    "layer": "process",
    "description": "根据任务目标，用 LLM 从宽表中筛选与任务相关的列，生成 filtered.csv 和选列报告。",
    "when_to_use": "表格列数较多（>10列）或需要根据任务目标裁剪特征时。",
    "when_to_skip": "Gold-Guided 字段恢复要求保留全部目标字段，或没有建模特征裁剪需求时跳过。",
    "capability_types": ["feature_selection", "column_relevance_filter"],
    "inputs": ["input_csv_path", "task_text", "output_root?", "memory_context?"],
    "outputs": ["filtered_csv_path", "selection_report_path"],
    "prerequisites": ["wide_feature_table_available", "task_target_defined"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step4_tools.run_step4_column_selection_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(run_step4_column_selection_tool)
