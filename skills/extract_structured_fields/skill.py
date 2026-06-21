from __future__ import annotations

import json

from agentscope.tool import Toolkit

from skills._rule_tools import record, response, step_output
from workflow.rule_execution import extract_structured_rule_values


def extract_structured_fields_tool(
    rules_path: str,
    input_root: str,
    rule_ids_json: str = "[]",
):
    """按 target_field_id 执行来源明确的结构化 direct 规则。"""
    step, output = step_output("extract_structured_fields")
    try:
        rule_ids = json.loads(rule_ids_json)
        if not isinstance(rule_ids, list) or not all(isinstance(value, str) for value in rule_ids):
            raise ValueError("rule_ids_json must be a JSON list of strings")
        result = extract_structured_rule_values(rules_path, input_root, output, rule_ids=rule_ids)
        failed = result["value_count"] == 0 or (
            bool(rule_ids) and result["unsupported_rule_count"] >= len(rule_ids)
        )
        issues = ["No structured values were extracted; use the capability-specific Skill or repair the rules."] if failed else []
        manifest = record(
            "extract_structured_fields",
            "Executed direct structured field rules.",
            [result["field_values"], result["unsupported"]],
            step=step,
            metadata={"value_count": result["value_count"], "unsupported_rule_count": result["unsupported_rule_count"]},
            status="NEEDS_REPAIR" if failed else "SUCCESS",
            issues=issues,
        )
        return response(
            "NEEDS_REPAIR" if failed else "SUCCESS",
            "Structured field extraction produced no usable values." if failed else "Structured field rules executed.",
            {**result, "manifest_path": (manifest or {}).get("manifest_path", "")},
            issues,
        )
    except Exception as exc:
        return response("NEEDS_REPAIR", "Structured field extraction failed.", issues=[str(exc)])


SKILL = {
    "name": "extract_structured_fields",
    "layer": "process",
    "description": "执行来源文件、来源列和 direct 逻辑明确的字段规则，输出规范化 field values；不写死数据集或主键。",
    "when_to_use": "extraction_task_plan 中 capability_type=structured_extract，且规则已有单一来源文件和来源列时使用。",
    "when_to_skip": "规则需要文本抽取、聚合、复杂派生，或来源仍为 ambiguous/unsupported 时跳过并交给对应 Skill。",
    "capability_types": ["structured_extract", "direct_field_extract"],
    "inputs": ["rules_path", "input_root", "rule_ids_json?"],
    "outputs": ["field_values.jsonl", "unsupported_rules.json"],
    "prerequisites": ["field_extraction_rules_available", "source_schema_known"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.rule_execution.extract_structured_rule_values"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(extract_structured_fields_tool)
