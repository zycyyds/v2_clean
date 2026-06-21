from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_tools.context import EngineerToolContext
from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context
from workflow.engineer_fallback import _extract_rule_values, complete_engineer_result_package
from workflow.task_compiler import compile_extraction_task_plan


SKILL_NAMES = {
    "aggregate_records",
    "derive_target_fields",
    "export_gold_workbook",
    "extract_structured_fields",
    "join_source_records",
    "normalize_reversible_values",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _rule(
    field_id: str,
    field_path: str,
    capability: str,
    source_files: list[str],
    source_columns: list[str],
    *,
    operation: str,
    cardinality: str = "one",
) -> dict:
    return {
        "target_field_id": field_id,
        "target_field_path": field_path,
        "target_type": "string",
        "cardinality": cardinality,
        "record_grain": "case",
        "source_files": source_files,
        "source_columns": source_columns,
        "join_keys": ["subject_id"],
        "filters": [],
        "derivation_logic": {"operation": operation},
        "capability_type": capability,
        "evaluation_policy": "canonical",
        "evidence": {"constant_value": "events"} if operation == "constant" else {},
        "confidence": 1.0,
        "status": "supported",
    }


def test_deterministic_fallback_finishes_audited_result_package(tmp_path: Path) -> None:
    validation = tmp_path / "validation_raw"
    structured = validation / "case_a" / "raw_mimic" / "structured"
    structured.mkdir(parents=True)
    pd.DataFrame(
        [{
            "subject_id": "10",
            "hadm_id": "20",
            "admittime": "2025-01-03 00:00:00",
            "dischtime": "2025-01-08 00:00:00",
            "deathtime": None,
            "admission_type": "URGENT",
        }]
    ).to_csv(structured / "admissions.csv", index=False)
    pd.DataFrame(
        [{"subject_id": "10", "anchor_age": "40", "anchor_year": "2020", "dod": None}]
    ).to_csv(structured / "patients.csv", index=False)
    pd.DataFrame(
        [
            {"subject_id": "10", "hadm_id": "20", "row_id": "1", "value": "7.1"},
            {"subject_id": "10", "hadm_id": "20", "row_id": "2", "value": "8.2"},
        ]
    ).to_csv(structured / "events.csv", index=False)

    rules = [
        _rule(
            "field_direct",
            "admission.admission_type",
            "structured_extract",
            ["structured/admissions.csv"],
            ["admission_type"],
            operation="direct",
        ),
        _rule(
            "field_values",
            "labs[].value",
            "aggregate",
            ["structured/events.csv"],
            ["value"],
            operation="aggregate",
            cardinality="many",
        ),
        _rule(
            "field_table",
            "labs[].source.table",
            "constant",
            ["structured/events.csv"],
            [],
            operation="constant",
            cardinality="many",
        ),
        _rule(
            "field_los",
            "stay.length_of_stay_days",
            "field_derivation",
            ["structured/admissions.csv"],
            ["admittime", "dischtime"],
            operation="field_derivation",
        ),
        _rule(
            "field_age",
            "stay.admission_age",
            "field_derivation",
            ["structured/admissions.csv", "structured/patients.csv"],
            ["anchor_age", "anchor_year", "admittime"],
            operation="field_derivation",
        ),
    ]
    reports = tmp_path / "reports"
    rules_path = reports / "field_extraction_rules.json"
    rules_payload = {
        "schema_version": 2,
        "record_grain": "case",
        "field_count": len(rules),
        "target_categories": ["admission", "labs", "stay"],
        "rules": rules,
    }
    _write_json(rules_path, rules_payload)
    task_plan = compile_extraction_task_plan(rules, record_grain="case")
    task_plan["target_categories"] = rules_payload["target_categories"]
    task_plan_path = reports / "extraction_task_plan.json"
    _write_json(task_plan_path, task_plan)
    analysis_path = reports / "data_analysis_report.json"
    record_path = reports / "explorer_run_record.json"
    _write_json(analysis_path, {"field_count": len(rules)})
    _write_json(record_path, {"status": "SUCCESS"})

    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "engineer")
    set_phase_context(run_root.name, run_root, "engineer", phase["phase_root"])
    context = EngineerToolContext(
        engineer_phase_root=phase["phase_root"],
        read_roots=[validation, reports, Path(__file__).resolve().parents[1] / "skills"],
        require_skill_plan=True,
        split_mode="validation",
        required_report_paths={
            "analysis_report": str(analysis_path),
            "field_extraction_rules": str(rules_path),
            "extraction_task_plan": str(task_plan_path),
            "explorer_run_record": str(record_path),
        },
    )
    try:
        result = complete_engineer_result_package(
            validation_raw=validation,
            rules_path=rules_path,
            task_plan_path=task_plan_path,
            tool_context=context,
            skill_names=SKILL_NAMES,
            skills_root=Path(__file__).resolve().parents[1] / "skills",
        )
    finally:
        clear_phase_context()

    output = pd.read_csv(result["csv"], dtype=object)
    assert output["case_id"].tolist() == ["case_a"]
    assert set(pd.ExcelFile(result["workbook"]).sheet_names) == {
        "_cases",
        "_provenance",
        "_unsupported",
        "admission",
        "labs",
        "stay",
    }
    mapping = json.loads(result["target_field_mapping"].read_text(encoding="utf-8"))
    assert {item["target_field_id"] for item in mapping["mappings"]} == {
        "field_direct",
        "field_values",
        "field_table",
        "field_los",
        "field_age",
    }
    audit = json.loads(result["skill_usage_report"].read_text(encoding="utf-8"))
    assert audit["status"] == "SUCCESS"
    executions = json.loads(result["task_execution"].read_text(encoding="utf-8"))
    assert len(executions["tasks"]) == task_plan["task_count"]


def test_rule_fallback_applies_equals_derivation(tmp_path: Path) -> None:
    raw = tmp_path / "case_a" / "raw_mimic"
    source = raw / "structured" / "diagnoses.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame([{"seq_num": 1}, {"seq_num": 2}]).to_csv(source, index=False)
    rule = _rule(
        "field_primary",
        "diagnoses[].is_primary",
        "aggregate",
        ["structured/diagnoses.csv"],
        ["seq_num"],
        operation="equals",
        cardinality="many",
    )
    rule["derivation_logic"]["value"] = 1

    values, unsupported = _extract_rule_values({"case_a": raw}, [rule])

    assert unsupported == []
    assert [item["value"] for item in values] == [True, False]
