from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context
from skills.extract_structured_fields.skill import extract_structured_fields_tool
from skills.aggregate_records.skill import SKILL as AGGREGATE_SKILL
from workflow.rule_execution import (
    aggregate_rule_values,
    derive_target_fields,
    extract_structured_rule_values,
    join_source_records,
    normalize_reversible_table,
)


def test_extracts_direct_rules_without_dataset_specific_names(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        [
            {"encounter": "a", "temperature": 37.1},
            {"encounter": "b", "temperature": 38.2},
        ]
    ).to_csv(raw / "observations.csv", index=False)
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "record_grain": "encounter",
                "rules": [
                    {
                        "target_field_id": "field_temperature",
                        "target_field_path": "vitals.temperature",
                        "status": "supported",
                        "capability_type": "structured_extract",
                        "source_files": ["observations.csv"],
                        "source_columns": ["temperature"],
                        "join_keys": ["encounter"],
                        "derivation_logic": {"operation": "direct"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = extract_structured_rule_values(rules, raw, tmp_path / "out")

    values = pd.read_json(result["field_values"], lines=True).to_dict(orient="records")
    assert [row["case_id"] for row in values] == ["a", "b"]
    assert {row["target_field_id"] for row in values} == {"field_temperature"}
    assert result["unsupported_rule_count"] == 0


def test_join_derive_and_reversible_normalization_are_composable(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame([{"id": " a ", "x": 2}, {"id": "b", "x": 4}]).to_csv(left, index=False)
    pd.DataFrame([{"id": "a", "y": 3}, {"id": "b", "y": 5}]).to_csv(right, index=False)

    normalized = normalize_reversible_table(
        left,
        tmp_path / "normalized.csv",
        schema={"id": "string", "x": "number"},
    )
    joined = join_source_records(
        [{"path": normalized}, {"path": str(right)}],
        ["id"],
        tmp_path / "joined.csv",
    )
    derived = derive_target_fields(
        joined,
        [{"target": "sum_xy", "operation": "sum", "columns": ["x", "y"]}],
        tmp_path / "derived.csv",
    )

    frame = pd.read_csv(derived)
    assert frame["id"].tolist() == ["a", "b"]
    assert frame["sum_xy"].tolist() == [5, 9]
    assert frame["x"].tolist() == [2, 4]


def test_generic_date_and_adjusted_age_derivations(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame([{
        "start": "2020-01-01",
        "end": "2020-01-04",
        "anchor_age": 50,
        "anchor_year": 2018,
        "event_time": "2020-06-01",
    }]).to_csv(source, index=False)

    output = derive_target_fields(
        source,
        [
            {"target": "duration_days", "operation": "datetime_difference_days", "columns": ["start", "end"]},
            {"target": "event_age", "operation": "adjusted_age", "columns": ["anchor_age", "anchor_year", "event_time"]},
        ],
        tmp_path / "derived_dates.csv",
    )

    frame = pd.read_csv(output)
    assert frame.loc[0, "duration_days"] == 3.0
    assert frame.loc[0, "event_age"] == 52


def test_aggregate_rule_values_groups_repeated_records_by_case(tmp_path: Path) -> None:
    source = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"case_id": "a", "item": "ALT", "value": 10.0},
            {"case_id": "a", "item": "ALT", "value": 14.0},
            {"case_id": "b", "item": "ALT", "value": 8.0},
        ]
    ).to_csv(source, index=False)

    output = aggregate_rule_values(
        source,
        tmp_path / "aggregated.csv",
        group_keys=["case_id", "item"],
        aggregations={"value": ["count", "mean", "min", "max", "list"]},
    )

    frame = pd.read_csv(output)
    row = frame[(frame["case_id"] == "a") & (frame["item"] == "ALT")].iloc[0]
    assert row["value_count"] == 2
    assert row["value_mean"] == 12.0
    assert json.loads(row["value_list"]) == [10.0, 14.0]


def test_aggregate_skill_declares_machine_readable_parameter_example() -> None:
    example = AGGREGATE_SKILL["input_example"]

    assert example["group_keys_json"] == '["case_id", "item"]'
    assert json.loads(example["aggregations_json"]) == {
        "value": ["count", "mean", "min", "max", "list"],
    }


def test_structured_skill_rejects_zero_values_and_all_unsupported_rules(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([{"case_id": "a", "value": 1}]).to_csv(raw / "events.csv", index=False)
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "target_field_id": "many_values",
                        "target_field_path": "events[].value",
                        "status": "supported",
                        "capability_type": "aggregate",
                        "source_files": ["events.csv"],
                        "source_columns": ["value"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "engineer")
    set_phase_context(run_root.name, run_root, "engineer", phase["phase_root"])
    try:
        response = extract_structured_fields_tool(str(rules), str(raw))
    finally:
        clear_phase_context()
    payload = json.loads(response.content[0]["text"])

    assert payload["status"] == "NEEDS_REPAIR"
    assert payload["artifacts"]["value_count"] == 0
    assert payload["artifacts"]["unsupported_rule_count"] == 1
