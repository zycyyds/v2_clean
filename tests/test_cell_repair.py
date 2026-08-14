from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from graph.cell_repair import (
    CellRepairError,
    _extract_correction_source,
    _static_rule_issues,
    apply_repair_plan,
    build_field_pairs,
    build_repair_targets,
    evaluate_repairs,
    recover_raw_from_graph,
    run_frozen_rules,
    synthesize_fcorr,
    validate_fcorr,
)


AMOUNT_RULE = '''def Correction(input_string):
    if re.fullmatch(r"[1-5]00", input_string):
        return input_string[:-2]
    return input_string
'''

STATUS_RULE = '''def Correction(input_string):
    match = re.fullmatch(r"BAD([0-4])", input_string)
    if match:
        return "active" + match.group(1)
    return input_string
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> dict[str, Path | dict[str, dict[str, int]]]:
    graph = tmp_path / "graph"
    supervision = tmp_path / "private"
    raw = tmp_path / "raw"
    graph.mkdir()
    supervision.mkdir()
    raw.mkdir()
    raw_rows: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    supervised_indices: list[int] = []
    labels: list[int] = []
    folds: list[int] = []

    def add_cell(
        *,
        fold: int,
        column: str,
        current: str,
        label: int,
        clean: str = "",
        subtype: str = "",
    ) -> int:
        row_number = len(raw_rows) + 1
        row = {
            "stay_id": str(70_000_000 + row_number),
            "amount": "",
            "status": "",
            "note": f"fold-private-{fold}",
        }
        row[column] = current
        raw_rows.append(row)
        observation_index = -1
        for row_column, raw_value in row.items():
            index = len(observations)
            observations.append({
                "row_id": row_number - 1,
                "table": "icu/events",
                "row_number": row_number,
                "column": row_column,
                "raw_value": raw_value,
                "value_node_id": 1000 + index,
            })
            if row_column == column:
                observation_index = index
        assert observation_index >= 0
        supervised_indices.append(observation_index)
        labels.append(label)
        folds.append(fold)
        if label:
            log_rows.append({
                "raw_file": "icu/events.csv",
                "raw_row_index": row_number - 1,
                "column": column,
                "clean_value": clean,
                "dirty_value": current,
                "error_class": "synthetic",
                "error_subtype": subtype,
                "canonical_stay_id": 30_000_000 + fold,
                "injection_seed": 666,
            })
        return observation_index

    indices: dict[int, dict[str, int]] = {}
    for fold in range(5):
        indices[fold] = {
            "dirty_amount": add_cell(
                fold=fold,
                column="amount",
                current=f"{fold + 1}00",
                clean=str(fold + 1),
                label=1,
                subtype="unit_scale_error",
            ),
            "dirty_status": add_cell(
                fold=fold,
                column="status",
                current=f"BAD{fold}",
                clean=f"active{fold}",
                label=1,
                subtype="enum_violation",
            ),
            "dirty_locator": add_cell(
                fold=fold,
                column="stay_id",
                current=str(99_000_000 + fold),
                clean=str(33_000_000 + fold),
                label=1,
                subtype="orphan_feature_stay_id",
            ),
            "clean_amount": add_cell(
                fold=fold,
                column="amount",
                current="500" if fold == 4 else str(5 + fold),
                label=0,
            ),
            "clean_status": add_cell(
                fold=fold,
                column="status",
                current=f"valid{fold}",
                label=0,
            ),
        }

    _write_csv(raw / "icu" / "events.csv", raw_rows)
    (raw / "unchanged.txt").write_text("untouched\n", encoding="utf-8")
    (graph / "cell_observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8"
    )
    (graph / "graph_manifest.json").write_text(
        json.dumps({
            "raw_root_name": "raw",
            "table_counts": {"icu/events": len(raw_rows)},
            "observation_count": len(observations),
        }) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        supervision / "supervision_masks.npz",
        cell_indices=np.asarray(supervised_indices, dtype=np.int64),
        cell_labels=np.asarray(labels, dtype=np.int8),
        cell_folds=np.asarray(folds, dtype=np.int8),
        cell_source=np.asarray([2 if value else 0 for value in labels], dtype=np.int8),
    )
    log = supervision / "merged_injection_log.csv"
    _write_csv(log, log_rows)

    predictions = tmp_path / "predictions.csv"
    prediction_rows: list[dict[str, object]] = []
    for split, fold in (("validation", 3), ("internal_test", 4)):
        split_predictions = {
            "dirty_amount": 1,
            "dirty_status": 0,
            "dirty_locator": 0,
            "clean_amount": 1,
            "clean_status": 0,
        }
        for name, observation_index in indices[fold].items():
            prediction_rows.append({
                "split": split,
                "observation_index": observation_index,
                "dirty_score": 0.99 if split_predictions[name] else 0.01,
                "threshold": 0.67,
                "prediction": split_predictions[name],
                "label": int(name.startswith("dirty_")),
                "source": "synthetic_dirty" if name.startswith("dirty_") else "sampled_clean",
                "fold": fold,
            })
    _write_csv(predictions, prediction_rows)
    return {
        "graph": graph,
        "supervision": supervision,
        "raw": raw,
        "log": log,
        "predictions": predictions,
        "counts": {
            "train": {"dirty": 9, "clean": 6},
            "validation": {"dirty": 3, "clean": 2},
            "internal_test": {"dirty": 3, "clean": 2},
        },
    }


def _build_pairs(paths: dict[str, object], output: Path) -> Path:
    build_field_pairs(
        graph_dir=paths["graph"],
        supervision_dir=paths["supervision"],
        raw_dir=paths["raw"],
        paired_log=paths["log"],
        output_dir=output,
        expected_counts=paths["counts"],
    )
    return output


def _good_completion(messages, evidence, attempt):
    del messages, attempt
    source = AMOUNT_RULE if evidence["column"] == "amount" else STATUS_RULE
    return f"General pattern.\n```python\n{source}```", "MiniMax-M3"


def _synthesize(paths: dict[str, object], tmp_path: Path) -> Path:
    evidence = _build_pairs(paths, tmp_path / "pairs")
    rules = tmp_path / "rules"
    synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=rules,
        completion=_good_completion,
    )
    return rules


def test_recover_raw_from_graph_round_trips_and_builds_pairs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    recovered = tmp_path / "recovered"
    report = recover_raw_from_graph(
        graph_dir=paths["graph"],
        output_dir=recovered,
    )

    assert report["status"] == "SUCCESS"
    assert report["table_count"] == 1
    assert report["row_count"] == 25
    assert report["observation_count"] == 100
    recovered_table = recovered / "icu" / "events.csv"
    assert recovered_table.read_bytes() == (Path(paths["raw"]) / "icu" / "events.csv").read_bytes()
    assert report["tables"]["icu/events"]["sha256"] == _sha256(recovered_table)

    pair_report = build_field_pairs(
        graph_dir=paths["graph"],
        supervision_dir=paths["supervision"],
        raw_dir=recovered,
        paired_log=paths["log"],
        output_dir=tmp_path / "recovered-pairs",
        expected_counts=paths["counts"],
    )
    assert pair_report["split_counts"] == paths["counts"]
    assert pair_report["exported_train_dirty_pair_count"] == 6


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("column_order", "columns changed"),
        ("non_contiguous_row", "non-contiguous row numbers"),
        ("duplicate_column", "duplicate Cell observation column"),
    ],
)
def test_recover_raw_from_graph_rejects_inconsistent_observations(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _fixture(tmp_path)
    observations_path = Path(paths["graph"]) / "cell_observations.jsonl"
    observations = [json.loads(line) for line in observations_path.read_text().splitlines()]
    if mutation == "column_order":
        observations[4], observations[5] = observations[5], observations[4]
    elif mutation == "non_contiguous_row":
        for observation in observations[4:8]:
            observation["row_number"] = 3
    else:
        observations[1]["column"] = observations[0]["column"]
    observations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in observations),
        encoding="utf-8",
    )

    output = tmp_path / f"bad-recovery-{mutation}"
    with pytest.raises(CellRepairError, match=message):
        recover_raw_from_graph(graph_dir=paths["graph"], output_dir=output)
    assert not output.exists()


def test_build_field_pairs_is_complete_train_only_and_redacted(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = _build_pairs(paths, tmp_path / "pairs")
    manifest = json.loads((output / "field_pairs_manifest.json").read_text())

    assert manifest["split_counts"] == paths["counts"]
    assert manifest["exported_train_dirty_pair_count"] == 6
    assert manifest["exported_train_clean_example_count"] == 6
    assert manifest["excluded_locator_pair_count"] == 3
    assert {entry["column"] for entry in manifest["fields"]} == {"amount", "status"}
    body = "\n".join(
        (output / entry["path"]).read_text() for entry in manifest["fields"]
    )
    for forbidden in (
        "BAD3",
        "BAD4",
        "active3",
        "active4",
        "unit_scale_error",
        "injection_seed",
        "fold",
        "label",
        "row_number",
        "observation_index",
        "99000000",
    ):
        assert forbidden not in body
    for entry in manifest["fields"]:
        assert entry["sha256"] == _sha256(output / entry["path"])
        assert entry["serialized_bytes"] == (output / entry["path"]).stat().st_size


def test_build_field_pairs_without_raw_uses_graph_values_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    checked = _build_pairs(paths, tmp_path / "checked-pairs")
    unchecked = tmp_path / "graph-pairs"
    manifest = build_field_pairs(
        graph_dir=paths["graph"],
        supervision_dir=paths["supervision"],
        paired_log=paths["log"],
        output_dir=unchecked,
        expected_counts=paths["counts"],
    )

    assert manifest["status"] == "SUCCESS"
    assert manifest["current_value_source"] == "cell_observations"
    assert manifest["raw_cross_check_enabled"] is False
    assert manifest["inputs"]["raw_table_sha256"] == {}
    for entry in manifest["fields"]:
        assert (unchecked / entry["path"]).read_bytes() == (
            checked / entry["path"]
        ).read_bytes()


def test_build_field_pairs_fails_on_counts_and_dirty_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(CellRepairError, match="frozen protocol"):
        build_field_pairs(
            graph_dir=paths["graph"],
            supervision_dir=paths["supervision"],
            raw_dir=paths["raw"],
            paired_log=paths["log"],
            output_dir=tmp_path / "bad-counts",
            expected_counts={"train": {"dirty": 1, "clean": 1}},
        )

    rows = list(csv.DictReader(Path(paths["log"]).open(encoding="utf-8")))
    rows[0]["dirty_value"] = "not-current"
    bad_log = tmp_path / "bad-log.csv"
    _write_csv(bad_log, rows)
    with pytest.raises(CellRepairError, match="dirty value mismatch"):
        build_field_pairs(
            graph_dir=paths["graph"],
            supervision_dir=paths["supervision"],
            raw_dir=paths["raw"],
            paired_log=bad_log,
            output_dir=tmp_path / "bad-log-output",
            expected_counts=paths["counts"],
        )


def test_validate_fcorr_covers_regex_mapping_missing_conflict_and_non_string() -> None:
    evidence = {
        "table": "t",
        "column": "v",
        "dirty_clean_pairs": [
            {"dirty": "100", "clean": "1"},
            {"dirty": "BAD", "clean": "good"},
            {"dirty": "NULL", "clean": ""},
            {"dirty": "2118/11/27", "clean": "2118-11-27"},
        ],
        "clean_examples": ["5", "good", ""],
    }
    source = '''def Correction(input_string):
    mapping = {"BAD": "good", "NULL": ""}
    match = re.fullmatch(r"(\\d)00", input_string)
    if match:
        return match.groups()[0]
    if re.fullmatch(r"\\d{4}/\\d{2}/\\d{2}", input_string):
        return input_string.replace("/", "-")
    return mapping.get(input_string, input_string)
'''
    report = validate_fcorr(source, evidence)
    assert report["status"] == "SUCCESS"
    assert report["metrics"]["exact_match_accuracy"] == 1.0
    assert report["metrics"]["clean_preservation_rate"] == 1.0

    conflict = {**evidence, "dirty_clean_pairs": [
        {"dirty": "X", "clean": "A"},
        {"dirty": "X", "clean": "B"},
    ]}
    report = validate_fcorr("def Correction(input_string):\n    return 'A'\n", conflict)
    assert report["status"] == "FAILED"
    assert report["metrics"]["exact_match_accuracy"] == 0.5

    report = validate_fcorr("def Correction(input_string):\n    return 1\n", evidence)
    assert report["status"] == "FAILED"
    assert report["metrics"]["clean_runtime_issue_count"] == 3

    changes_clean = validate_fcorr(
        '''def Correction(input_string):
    if input_string == "5":
        return "changed"
    mapping = {"100": "1", "BAD": "good", "NULL": "", "2118/11/27": "2118-11-27"}
    return mapping.get(input_string, input_string)
''',
        evidence,
    )
    assert changes_clean["status"] == "SUCCESS"
    assert changes_clean["metrics"]["clean_preservation_rate"] == pytest.approx(2 / 3)


def test_static_sandbox_rejects_import_ids_dynamic_calls_and_multiple_functions() -> None:
    assert any("imports are forbidden" in issue for issue in _static_rule_issues(
        "def Correction(input_string):\n    import os\n    return input_string\n"
    ))
    assert any("suspicious hard-coded identifier" in issue for issue in _static_rule_issues(
        "def Correction(input_string):\n    return '33976251'\n"
    ))
    assert any("forbidden call 'open'" in issue for issue in _static_rule_issues(
        "def Correction(input_string):\n    return open(input_string)\n"
    ))
    response = (
        "```python\ndef Correction(input_string):\n    return input_string\n```\n"
        "```python\ndef Correction(input_string):\n    return input_string.strip()\n```"
    )
    with pytest.raises(CellRepairError, match="exactly one"):
        _extract_correction_source(response)
    with pytest.raises(CellRepairError, match="exactly one"):
        _extract_correction_source("def Correction(:\n    pass")


def test_synthesis_uses_field_isolated_history_and_counterexample_retry(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    evidence = _build_pairs(paths, tmp_path / "pairs")
    calls: list[tuple[str, int, list[str], str]] = []

    def completion(messages, field_evidence, attempt):
        calls.append((
            field_evidence["column"],
            attempt,
            [message["role"] for message in messages],
            messages[-1]["content"],
        ))
        if field_evidence["column"] == "amount" and attempt == 1:
            return "def Correction(input_string):\n    return input_string\n", "MiniMax-M3"
        return _good_completion(messages, field_evidence, attempt)

    output = tmp_path / "rules"
    manifest = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=output,
        completion=completion,
    )
    assert manifest["status_counts"] == {"FROZEN": 2}
    amount_calls = [call for call in calls if call[0] == "amount"]
    assert amount_calls[1][2] == ["system", "user", "assistant", "user"]
    feedback = json.loads(amount_calls[1][3].split("\n\n", 1)[1])
    assert len(feedback["wrongly_corrected_pairs"]) == 3
    assert feedback["previous_function"].startswith("def Correction")
    status_first = next(call for call in calls if call[0] == "status")
    assert status_first[2] == ["system", "user"]
    assert "BAD3" not in status_first[3] and "BAD4" not in status_first[3]
    assert manifest["registry"]["rule_count"] == 2
    assert manifest["registry"]["test_time_llm_access"] is False


def test_synthesis_stops_at_twelve_or_context_limit_and_rejects_wrong_model(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    evidence = _build_pairs(paths, tmp_path / "pairs")
    attempts: list[int] = []

    def never_correct(messages, field_evidence, attempt):
        del messages, field_evidence
        attempts.append(attempt)
        return "def Correction(input_string):\n    return input_string\n", "MiniMax-M3"

    rejected = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=tmp_path / "rejected",
        fields=["icu/events.amount"],
        max_attempts=12,
        completion=never_correct,
    )
    assert attempts == list(range(1, 13))
    assert rejected["status_counts"] == {"FCORR_REJECTED": 1}
    assert rejected["registry"]["rule_count"] == 0

    def too_large(messages, field_evidence, attempt):
        del messages, field_evidence, attempt
        raise RuntimeError("maximum context length exceeded")

    limited = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=tmp_path / "too-large",
        fields=["icu/events.amount"],
        completion=too_large,
    )
    assert limited["status_counts"] == {"CONTEXT_TOO_LARGE": 1}

    wrong_model = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=tmp_path / "wrong-model",
        fields=["icu/events.amount"],
        max_attempts=1,
        completion=lambda messages, field, attempt: (AMOUNT_RULE, "other-model"),
    )
    assert wrong_model["status_counts"] == {"FCORR_REJECTED": 1}


@pytest.mark.parametrize("tampered", ["source", "evidence", "manifest"])
def test_frozen_registry_rejects_tampering(tmp_path: Path, tampered: str) -> None:
    paths = _fixture(tmp_path)
    rules = _synthesize(paths, tmp_path)
    copied = tmp_path / f"rules-{tampered}"
    shutil.copytree(rules, copied)
    registry = json.loads((copied / "rule_registry.json").read_text())
    entry = next(iter(registry["rules"].values()))
    target = {
        "source": copied / entry["source_path"],
        "evidence": copied / entry["evidence_path"],
        "manifest": copied / "synthesis_manifest.json",
    }[tampered]
    target.write_text(target.read_text() + "\n", encoding="utf-8")
    targets_dir = tmp_path / f"targets-{tampered}"
    build_repair_targets(
        predictions=paths["predictions"],
        graph_dir=paths["graph"],
        raw_dir=paths["raw"],
        split="internal_test",
        output_dir=targets_dir,
    )
    with pytest.raises(CellRepairError, match="hash mismatch"):
        run_frozen_rules(
            targets=targets_dir / "repair_targets.csv",
            rule_dir=copied,
            output_dir=tmp_path / f"run-{tampered}",
        )


def test_offline_target_run_apply_and_private_joint_evaluation(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rules = _synthesize(paths, tmp_path)
    targets = tmp_path / "targets"
    target_manifest = build_repair_targets(
        predictions=paths["predictions"],
        graph_dir=paths["graph"],
        raw_dir=paths["raw"],
        split="internal_test",
        output_dir=targets,
    )
    assert target_manifest["target_count"] == 2

    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    first_manifest = run_frozen_rules(
        targets=targets / "repair_targets.csv", rule_dir=rules, output_dir=first
    )
    run_frozen_rules(
        targets=targets / "repair_targets.csv", rule_dir=rules, output_dir=second
    )
    assert first_manifest["action_counts"] == {"replace": 2}
    assert first_manifest["llm_called"] is False
    assert (first / "repair_values.jsonl").read_bytes() == (
        second / "repair_values.jsonl"
    ).read_bytes()
    assert (first / "repair_plan.jsonl").read_bytes() == (
        second / "repair_plan.jsonl"
    ).read_bytes()

    original_hash = _sha256(Path(paths["raw"]) / "icu" / "events.csv")
    repaired = tmp_path / "repaired"
    apply_report = apply_repair_plan(
        raw_dir=paths["raw"],
        repair_plan=first / "repair_plan.jsonl",
        output_dir=repaired,
    )
    assert apply_report["applied_replacement_count"] == 2
    assert _sha256(Path(paths["raw"]) / "icu" / "events.csv") == original_hash
    assert (repaired / "unchanged.txt").read_text() == "untouched\n"

    evaluation = evaluate_repairs(
        repair_values=first / "repair_values.jsonl",
        repair_plan=first / "repair_plan.jsonl",
        injection_log=paths["log"],
        predictions=paths["predictions"],
        graph_dir=paths["graph"],
        output_dir=tmp_path / "evaluation",
    )
    metrics = evaluation["metrics"]
    assert evaluation["evaluation_role"] == "development_audit"
    assert metrics["detector_confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 2, "tn": 1}
    assert metrics["exact_repairs"] == 1
    assert metrics["incorrect_repairs"] == 0
    assert metrics["rule_only_exact_accuracy"] == 1.0
    assert metrics["clean_preservation_rate"] == 0.5
    assert metrics["correction_precision"] == 0.5
    assert metrics["correction_recall"] == pytest.approx(1 / 3)


def test_apply_supports_gzip_and_changes_only_declared_cell(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "table.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["a", "b"], lineterminator="\n")
        writer.writeheader()
        writer.writerows([{"a": "x", "b": "1"}, {"a": "y", "b": "2"}])
    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps({
        "action": "replace",
        "table": "table",
        "raw_row_index": 1,
        "column": "b",
        "current_value": "2",
        "replacement_value": "20",
    }) + "\n")
    output = tmp_path / "output"
    apply_repair_plan(raw_dir=raw, repair_plan=plan, output_dir=output)
    with gzip.open(output / "table.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "x", "b": "1"}, {"a": "y", "b": "20"}]
