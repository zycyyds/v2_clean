from __future__ import annotations

import json

from agentscope.tool import Toolkit

from skills._rule_tools import record, response, step_output
from workflow.rule_execution import derive_target_fields


def derive_target_fields_tool(input_path: str, derivations_json: str):
    """执行规则声明的可解释字段派生。"""
    step, output = step_output("derive_target_fields")
    try:
        derivations = json.loads(derivations_json)
        if not isinstance(derivations, list):
            raise ValueError("derivations_json must be a JSON list")
        path = derive_target_fields(input_path, derivations, output / "derived_fields.csv")
        manifest = record("derive_target_fields", "Derived target fields.", [path], step=step, metadata={"derivation_count": len(derivations)})
        return response("SUCCESS", "Target fields derived.", {"output_path": path, "manifest_path": (manifest or {}).get("manifest_path", "")})
    except Exception as exc:
        return response("NEEDS_REPAIR", "Target field derivation failed.", issues=[str(exc)])


SKILL = {
    "name": "derive_target_fields",
    "layer": "process",
    "description": "按字段规则执行 copy、coalesce、concat、数值行聚合和非空计数等可解释派生。",
    "when_to_use": "规则 derivation_logic 已明确输入列和支持的确定性 operation 时使用。",
    "when_to_skip": "派生逻辑未定义、需要模型语义判断或需要跨行时间聚合时跳过并适配其他 Skill。",
    "capability_types": ["field_derivation", "deterministic_transform"],
    "inputs": ["input_path", "derivations_json"],
    "outputs": ["derived_fields.csv"],
    "prerequisites": ["derivation_logic_known", "input_columns_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.rule_execution.derive_target_fields"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(derive_target_fields_tool)
