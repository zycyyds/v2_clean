from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from agentscope.tool import Toolkit

from skills._rule_tools import record, response, step_output
from workflow.rule_execution import aggregate_rule_values


def aggregate_records_tool(
    input_path: str,
    group_keys_json: str,
    aggregations_json: str,
):
    """按显式分组键跨行聚合；aggregations_json格式为{"列名":["list","mean"]}。"""
    step, output = step_output("aggregate_records")
    try:
        group_keys = json.loads(group_keys_json)
        aggregations = json.loads(aggregations_json)
        if not isinstance(group_keys, list) or not all(isinstance(value, str) for value in group_keys):
            raise ValueError("group_keys_json must be a JSON list of strings")
        if not isinstance(aggregations, dict):
            raise ValueError("aggregations_json must be a JSON object")
        if not all(
            isinstance(column, str)
            and isinstance(operations, list)
            and operations
            and all(isinstance(operation, str) for operation in operations)
            for column, operations in aggregations.items()
        ):
            raise ValueError(
                'aggregations_json must map source columns to operation lists, '
                'for example {"value":["count","mean","list"]}',
            )
        path = aggregate_rule_values(
            input_path,
            output / "aggregated_records.csv",
            group_keys=group_keys,
            aggregations=aggregations,
        )
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError("Aggregation produced zero rows")
        manifest = record(
            "aggregate_records",
            "Aggregated repeated source records.",
            [path],
            step=step,
            metadata={"row_count": int(len(frame)), "column_count": int(len(frame.columns))},
        )
        return response(
            "SUCCESS",
            "Repeated records aggregated.",
            {
                "output_path": path,
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "manifest_path": (manifest or {}).get("manifest_path", ""),
            },
        )
    except Exception as exc:
        return response("NEEDS_REPAIR", "Record aggregation failed.", issues=[str(exc)])


SKILL = {
    "name": "aggregate_records",
    "layer": "process",
    "description": "按记录主键和可选分组列执行count/mean/min/max/sum/first/last/list跨行聚合。",
    "when_to_use": "extraction_task_plan中capability_type=aggregate，且来源列和分组键已经明确时使用。",
    "when_to_skip": "来源未确定、需要文本语义抽取或需要先跨表join时跳过。",
    "capability_types": ["aggregate", "groupby_aggregate", "occurrence_collection"],
    "inputs": ["input_path", "group_keys_json", "aggregations_json"],
    "input_contract": {
        "group_keys_json": "JSON string containing a non-empty list of source column names",
        "aggregations_json": "JSON string mapping each source column to a list of count/mean/min/max/sum/first/last/list operations",
    },
    "input_example": {
        "group_keys_json": '["case_id", "item"]',
        "aggregations_json": '{"value": ["count", "mean", "min", "max", "list"]}',
    },
    "outputs": ["aggregated_records.csv"],
    "prerequisites": ["source_columns_available", "group_keys_known"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.rule_execution.aggregate_rule_values"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(aggregate_records_tool)
