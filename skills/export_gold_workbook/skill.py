from __future__ import annotations

import json
from pathlib import Path

from agentscope.tool import Toolkit

from skills._rule_tools import record, response, step_output
from workflow.workbook import export_gold_workbook


def export_gold_workbook_tool(
    cases_json: str,
    field_values_path: str,
    provenance_json: str = "[]",
    unsupported_json: str = "[]",
    target_categories_json: str = "[]",
):
    """将规范化字段值导出为动态类别 Sheet 的 final_dataset.xlsx。"""
    step, output = step_output("export_gold_workbook")
    try:
        cases = json.loads(cases_json)
        provenance = json.loads(provenance_json)
        unsupported = json.loads(unsupported_json)
        target_categories = json.loads(target_categories_json)
        if not all(isinstance(value, list) for value in (cases, provenance, unsupported, target_categories)):
            raise ValueError("cases, provenance, unsupported, and target categories must be JSON lists")
        values = [
            json.loads(line)
            for line in Path(field_values_path).expanduser().resolve().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = export_gold_workbook(
            output,
            cases=cases,
            field_values=values,
            provenance=provenance,
            unsupported=unsupported,
            target_categories=target_categories,
        )
        manifest = record(
            "export_gold_workbook",
            "Exported dynamic Gold-guided workbook result package.",
            list(result.values()),
            step=step,
            metadata={"case_count": len(cases), "field_value_count": len(values)},
        )
        return response("SUCCESS", "Gold-guided workbook exported.", {**result, "manifest_path": (manifest or {}).get("manifest_path", "")})
    except Exception as exc:
        return response("NEEDS_REPAIR", "Gold-guided workbook export failed.", issues=[str(exc)])


SKILL = {
    "name": "export_gold_workbook",
    "layer": "util",
    "description": "把规范化字段值同时导出为每个case一行的 final_dataset.csv 和分类明细 final_dataset.xlsx，并附manifest和mapping。",
    "when_to_use": "required extraction tasks 已完成，需要生成本轮统一结果包时使用。",
    "when_to_skip": "字段值尚未按 target_field_id 规范化，或 required task 仍未完成时跳过。",
    "capability_types": ["gold_workbook_export", "result_package_build"],
    "inputs": ["cases_json", "field_values_path", "provenance_json?", "unsupported_json?", "target_categories_json?"],
    "outputs": ["final_dataset.csv", "final_dataset.xlsx", "result_manifest.json", "target_field_mapping.json"],
    "prerequisites": ["normalized_field_values_available", "case_ids_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.workbook.export_gold_workbook"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(export_gold_workbook_tool)
