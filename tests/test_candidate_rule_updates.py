from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow.rule_updates import apply_candidate_rule_updates


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_adds_only_validation_reported_fields_and_updates_existing_rules(tmp_path: Path) -> None:
    rules = tmp_path / "rules.json"
    _write(
        rules,
        {
            "field_count": 1,
            "rules": [
                {
                    "target_field_id": "known",
                    "target_field_path": "known",
                    "source_files": [],
                    "source_columns": [],
                    "status": "unsupported",
                }
            ],
        },
    )
    feedback = tmp_path / "feedback.json"
    _write(
        feedback,
        {
            "field_feedback": [
                {
                    "target_field_id": "field_new",
                    "target_field_path": "validation_only",
                    "error_types": {"missing_rule": 2},
                }
            ]
        },
    )
    updates = tmp_path / "updates.json"
    _write(
        updates,
        {
            "updates": [
                {
                    "target_field_id": "known",
                    "source_files": ["known.csv"],
                    "source_columns": ["known"],
                    "status": "supported",
                },
                {
                    "target_field_id": "field_new",
                    "target_field_path": "validation_only",
                    "target_type": "string",
                    "cardinality": "one",
                    "record_grain": "record_id",
                    "source_files": ["new.csv"],
                    "source_columns": ["validation_only"],
                    "join_keys": ["record_id"],
                    "filters": [],
                    "derivation_logic": {"operation": "direct"},
                    "capability_type": "structured_extract",
                    "evaluation_policy": "canonical",
                    "evidence": {"from_validation_feedback": True},
                    "confidence": 0.8,
                    "status": "supported",
                },
            ]
        },
    )

    result = apply_candidate_rule_updates(rules, updates, feedback)
    payload = json.loads(rules.read_text())

    assert result == {"updated": 1, "added": 1, "field_count": 2}
    assert payload["field_count"] == 2
    assert {rule["target_field_id"] for rule in payload["rules"]} == {"known", "field_new"}


def test_rejects_new_field_not_disclosed_by_public_feedback(tmp_path: Path) -> None:
    rules = tmp_path / "rules.json"
    feedback = tmp_path / "feedback.json"
    updates = tmp_path / "updates.json"
    _write(rules, {"field_count": 0, "rules": []})
    _write(feedback, {"field_feedback": []})
    _write(
        updates,
        {
            "updates": [
                {
                    "target_field_id": "invented",
                    "target_field_path": "invented",
                    "source_files": ["x.csv"],
                    "source_columns": ["x"],
                    "derivation_logic": {"operation": "direct"},
                    "evaluation_policy": "canonical",
                    "capability_type": "structured_extract",
                    "status": "supported",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="not authorized"):
        apply_candidate_rule_updates(rules, updates, feedback)

