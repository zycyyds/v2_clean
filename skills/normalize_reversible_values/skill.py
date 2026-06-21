from __future__ import annotations

import json

from agentscope.tool import Toolkit

from skills._rule_tools import record, response, step_output
from workflow.rule_execution import normalize_reversible_table


def normalize_reversible_values_tool(input_path: str, type_schema_json: str = "{}"):
    """只执行可逆或可审计的值规范化，不做插补。"""
    step, output = step_output("normalize_reversible_values")
    try:
        schema = json.loads(type_schema_json)
        if not isinstance(schema, dict):
            raise ValueError("type_schema_json must be a JSON object")
        path = normalize_reversible_table(input_path, output / "normalized_values.csv", schema=schema)
        manifest = record("normalize_reversible_values", "Normalized values without imputation.", [path], step=step, metadata={"typed_columns": sorted(schema)})
        return response("SUCCESS", "Values normalized without imputation.", {"output_path": path, "manifest_path": (manifest or {}).get("manifest_path", "")})
    except Exception as exc:
        return response("NEEDS_REPAIR", "Reversible normalization failed.", issues=[str(exc)])


SKILL = {
    "name": "normalize_reversible_values",
    "layer": "process",
    "description": "规范空值标记、字符串空白、显式数值/日期/布尔类型，不插补、不裁剪、不修改临床数值。",
    "when_to_use": "验证集或测试集抽取完成后，需要统一类型、日期和空值表示时使用。",
    "when_to_skip": "任务要求不可逆插补、异常值修正或业务编码映射时跳过，必须由显式规则和独立 Skill 处理。",
    "capability_types": ["reversible_cleaning", "type_normalization"],
    "inputs": ["input_path", "type_schema_json?"],
    "outputs": ["normalized_values.csv"],
    "prerequisites": ["input_artifact_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.rule_execution.normalize_reversible_table"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(normalize_reversible_values_tool)
