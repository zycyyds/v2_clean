from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MUTABLE_RULE_FIELDS = {
    "source_files",
    "source_columns",
    "join_keys",
    "filters",
    "derivation_logic",
    "capability_type",
    "evaluation_policy",
    "evidence",
    "confidence",
    "status",
}
NEW_RULE_REQUIRED = {
    "target_field_id",
    "target_field_path",
    "source_files",
    "source_columns",
    "derivation_logic",
    "capability_type",
    "evaluation_policy",
    "status",
}


def apply_candidate_rule_updates(
    rules_path: str | Path,
    updates_path: str | Path,
    public_feedback_path: str | Path | None,
) -> dict[str, int]:
    rules_file = Path(rules_path).expanduser().resolve()
    updates_file = Path(updates_path).expanduser().resolve()
    rules_payload = _load_json(rules_file)
    updates_payload = _load_json(updates_file)
    rules = rules_payload.get("rules")
    updates = updates_payload.get("updates")
    if not isinstance(rules, list) or not isinstance(updates, list):
        raise ValueError("Rules and updates must contain lists")
    authorized_new = _authorized_new_fields(public_feedback_path)
    by_id = {
        str(rule.get("target_field_id")): rule
        for rule in rules
        if isinstance(rule, dict) and rule.get("target_field_id")
    }
    if len(by_id) != len(rules):
        raise ValueError("Current rules contain missing or duplicate target_field_id")
    updated = 0
    added = 0
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError("Each candidate rule update must be an object")
        field_id = str(update.get("target_field_id") or "")
        if field_id in by_id:
            forbidden = set(update) - MUTABLE_RULE_FIELDS - {"target_field_id"}
            if forbidden:
                raise ValueError(f"Existing field update contains immutable keys: {sorted(forbidden)}")
            by_id[field_id].update({key: update[key] for key in MUTABLE_RULE_FIELDS if key in update})
            updated += 1
            continue
        expected_path = authorized_new.get(field_id)
        if expected_path is None:
            raise ValueError(f"New target_field_id is not authorized by public feedback: {field_id}")
        missing = sorted(NEW_RULE_REQUIRED - set(update))
        if missing:
            raise ValueError(f"New rule {field_id} is missing required keys: {', '.join(missing)}")
        if str(update.get("target_field_path") or "") != expected_path:
            raise ValueError(f"New rule path does not match public feedback for {field_id}")
        rule = dict(update)
        rules.append(rule)
        by_id[field_id] = rule
        added += 1
    rules_payload["field_count"] = len(rules)
    rules_payload.setdefault("validation_updates", []).append(
        {
            "updates_path": str(updates_file),
            "updated": updated,
            "added": added,
        }
    )
    _write_json(rules_file, rules_payload)
    return {"updated": updated, "added": added, "field_count": len(rules)}


def _authorized_new_fields(public_feedback_path: str | Path | None) -> dict[str, str]:
    if public_feedback_path is None:
        return {}
    path = Path(public_feedback_path).expanduser().resolve()
    if not path.is_file():
        return {}
    payload = _load_json(path)
    authorized = {}
    for item in payload.get("field_feedback", []):
        if not isinstance(item, dict):
            continue
        errors = item.get("error_types") if isinstance(item.get("error_types"), dict) else {}
        if "missing_rule" not in errors:
            continue
        field_id = str(item.get("target_field_id") or "")
        field_path = str(item.get("target_field_path") or "")
        if field_id and field_path:
            authorized[field_id] = field_path
    return authorized


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

