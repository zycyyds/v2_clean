from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXECUTABLE_STATUSES = {"supported", "active", "frozen"}


def compile_extraction_task_plan(
    rules: Iterable[dict[str, Any]],
    *,
    record_grain: str,
) -> dict[str, Any]:
    """Group executable field rules into deterministic category/capability tasks."""
    all_rules = list(rules)
    executable = [rule for rule in all_rules if rule.get("status") in EXECUTABLE_STATUSES]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rule in executable:
        category = _target_category(str(rule["target_field_path"]))
        capability = str(rule.get("capability_type") or "structured_extract")
        grouped[(category, capability)].append(rule)

    tasks = [
        _build_task(category, capability, grouped_rules, record_grain)
        for (category, capability), grouped_rules in sorted(grouped.items())
    ]
    assigned_ids = [rule_id for task in tasks for rule_id in task["rule_ids"]]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("An executable field rule was assigned to more than one extraction task")

    executable_ids = {str(rule["target_field_id"]) for rule in executable}
    if set(assigned_ids) != executable_ids:
        raise ValueError("Extraction task plan does not cover every executable field rule")

    return {
        "schema_version": 1,
        "record_grain": record_grain,
        "task_count": len(tasks),
        "executable_rule_count": len(executable),
        "tasks": tasks,
        "unassigned_rule_ids": [
            str(rule["target_field_id"])
            for rule in all_rules
            if rule.get("status") not in EXECUTABLE_STATUSES
        ],
    }


def _build_task(
    category: str,
    capability: str,
    rules: list[dict[str, Any]],
    record_grain: str,
) -> dict[str, Any]:
    ordered = sorted(rules, key=lambda rule: str(rule["target_field_path"]))
    identity = f"{category}\0{capability}"
    source_files = sorted(
        {
            str(source_file)
            for rule in ordered
            for source_file in rule.get("source_files", [])
        }
    )
    cardinalities = {str(rule.get("cardinality", "one")) for rule in ordered}
    output_grain = "occurrence" if "many" in cardinalities else record_grain
    return {
        "task_id": "task_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "stage": _stage_for_capability(capability),
        "capability_type": capability,
        "target_category": category,
        "rule_ids": [str(rule["target_field_id"]) for rule in ordered],
        "required": True,
        "input_artifacts": [
            {"alias": _artifact_alias(path, index), "path": path}
            for index, path in enumerate(source_files, start=1)
        ],
        "output_sheet": category,
        "output_grain": output_grain,
        "output_contract": {
            "target_field_ids": [str(rule["target_field_id"]) for rule in ordered],
            "target_field_paths": [str(rule["target_field_path"]) for rule in ordered],
            "case_id_required": True,
            "occurrence_id_required": output_grain == "occurrence",
        },
    }


def _target_category(field_path: str) -> str:
    return field_path.split(".", 1)[0].replace("[]", "")


def _stage_for_capability(capability: str) -> str:
    if capability == "text_extract":
        return "text_extraction"
    if capability == "aggregate":
        return "aggregation"
    return "structured_extraction"


def _artifact_alias(path: str, index: int) -> str:
    stem = Path(path).stem.strip()
    return stem or f"source_{index}"
