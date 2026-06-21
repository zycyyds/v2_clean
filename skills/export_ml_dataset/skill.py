from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step7_runtime import format_and_save_ml_dataset

SKILL = {
    "name": "export_ml_dataset",
    "layer": "label",
    "description": "将特征表和标签合并，输出最终 ML 训练集（CSV/JSON/JSONL）及模型配置文件。",
    "when_to_use": "所有特征和标签准备好后，作为最后一步调用。",
    "when_to_skip": "任务只做 Gold 字段恢复、没有标签，或要求输出动态多 Sheet 工作簿而非旧 ML 数据集格式时跳过。",
    "capability_types": ["ml_dataset_export", "feature_label_merge"],
    "inputs": ["label_series_id", "feature_csv_path?", "output_root?", "task_text?"],
    "outputs": ["ml_dataset_csv", "ml_dataset_jsonl", "model_config_path"],
    "prerequisites": ["feature_table_available", "label_series_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step7_runtime.format_and_save_ml_dataset"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(format_and_save_ml_dataset)
