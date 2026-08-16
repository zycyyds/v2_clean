from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from workflow.pi_repair_audit import (
    DELETE_ROW,
    PiRepairAuditError,
    build_public_train_view,
    load_repair_candidates,
    load_repair_submission,
    render_repair_replay_argv,
    score_repair_candidates,
    score_repair_output,
    validate_declared_repair_output,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(
    *,
    table: str,
    row_index: int,
    column: str,
    original: str,
    values: list[str],
    selected: str,
) -> dict:
    return {
        "table": table,
        "row_index": row_index,
        "column": column,
        "original_value": original,
        "candidates": [
            {"value": value, "rule_id": f"rule_{index}", "evidence": "train pattern"}
            for index, value in enumerate(values)
        ],
        "selected_value": selected,
    }


def test_build_public_train_view_exports_all_supervision_and_masks_locators(
    tmp_path: Path,
) -> None:
    dirty = tmp_path / "private/dirty"
    clean = tmp_path / "private/clean"
    rows_dirty = [
        {"subject_id": "100", "value": "BAD"},
        {"subject_id": "200", "value": "B"},
    ]
    rows_clean = [
        {"subject_id": "100", "value": "A"},
        {"subject_id": "200", "value": "B"},
    ]
    _write_csv(dirty / "hosp/example.csv", ["subject_id", "value"], rows_dirty)
    _write_csv(clean / "hosp/example.csv", ["subject_id", "value"], rows_clean)

    graph = tmp_path / "private/graph"
    observations = [
        {"table": "hosp/example", "row_number": 1, "column": "subject_id", "raw_value": "100"},
        {"table": "hosp/example", "row_number": 1, "column": "value", "raw_value": "BAD"},
        {"table": "hosp/example", "row_number": 2, "column": "subject_id", "raw_value": "200"},
        {"table": "hosp/example", "row_number": 2, "column": "value", "raw_value": "B"},
    ]
    _write_jsonl(graph / "cell_observations.jsonl", observations)
    (graph / "graph_manifest.json").write_text("{}\n", encoding="utf-8")
    supervision = tmp_path / "private/supervision"
    supervision.mkdir(parents=True)
    np.savez(
        supervision / "supervision_masks.npz",
        cell_indices=np.asarray([1, 3], dtype=np.int64),
        cell_labels=np.asarray([1, 0], dtype=np.int8),
    )
    paired = supervision / "merged_injection_log.csv"
    _write_csv(
        paired,
        ["raw_file", "raw_row_index", "column", "clean_value", "dirty_value"],
        [
            {
                "raw_file": "hosp/example.csv",
                "raw_row_index": "0",
                "column": "value",
                "clean_value": "A",
                "dirty_value": "BAD",
            }
        ],
    )

    output = tmp_path / "public"
    manifest = build_public_train_view(
        dirty_raw=dirty,
        clean_raw=clean,
        graph_dir=graph,
        supervision_dir=supervision,
        paired_log=paired,
        output_dir=output,
        expected_dirty=1,
        expected_clean=1,
        expected_table_count=1,
    )

    assert manifest["supervision"] == {
        "dirty_count": 1,
        "clean_count": 1,
        "field_count": 1,
    }
    assert {path.name for path in output.iterdir()} == {
        "clean_raw",
        "dirty_raw",
        "evidence",
        "public_train_manifest.json",
    }
    assert "reference" not in manifest
    exported = (output / "dirty_raw/hosp/example.csv").read_text(encoding="utf-8")
    assert "100" not in exported and "200" not in exported
    evidence_manifest = json.loads(
        (output / "evidence/evidence_manifest.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (output / "evidence" / evidence_manifest["fields"][0]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["dirty_clean_pairs"][0]["dirty"] == "BAD"
    assert evidence["dirty_clean_pairs"][0]["clean"] == "A"
    assert evidence["dirty_clean_pairs"][0]["row_context"]["subject_id"] not in {
        "100",
        "200",
    }

    second = tmp_path / "public-second"
    build_public_train_view(
        dirty_raw=dirty,
        clean_raw=clean,
        graph_dir=graph,
        supervision_dir=supervision,
        paired_log=paired,
        output_dir=second,
        expected_dirty=1,
        expected_clean=1,
        expected_table_count=1,
    )
    first_files = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_submission_contract_and_placeholder_rendering(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "repair_submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "current_output_root": "outputs/train",
                "outputs": {
                    "corrected_raw": "corrected_raw",
                    "repair_candidates": "repair_candidates.jsonl",
                },
                "replay": {
                    "argv": [
                        "python",
                        "pipeline/run.py",
                        "--raw-root",
                        "{raw_root}",
                        "--output-dir",
                        "{output_dir}",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    submission = load_repair_submission(workdir)

    assert submission.corrected_raw == workdir / "outputs/train/corrected_raw"
    rendered = render_repair_replay_argv(
        submission,
        raw_root="/public/raw",
        output_dir="/tmp/output",
        workdir="/tmp/workdir",
    )
    assert rendered[-2] == "--output-dir"
    assert Path(rendered[-1]) == Path("/tmp/output").resolve()


def test_submission_rejects_static_absolute_replay_paths(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "repair_submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "outputs": {
                    "corrected_raw": "corrected_raw",
                    "repair_candidates": "repair_candidates.jsonl",
                },
                "replay": {
                    "argv": [
                        "python",
                        "/private/host/pipeline.py",
                        "--raw",
                        "{raw_root}",
                        "--out",
                        "{output_dir}",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PiRepairAuditError, match="static absolute"):
        load_repair_submission(workdir)


def test_candidate_report_is_strict_and_declared_changes_match_corrected_raw(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    corrected = tmp_path / "corrected"
    _write_csv(
        raw / "table.csv",
        ["id", "value"],
        [{"id": "1", "value": "BAD"}, {"id": "2", "value": "DROP"}],
    )
    _write_csv(corrected / "table.csv", ["id", "value"], [{"id": "1", "value": "GOOD"}])
    rows = [
        _candidate(
            table="table.csv",
            row_index=0,
            column="value",
            original="BAD",
            values=["GOOD", "OK"],
            selected="GOOD",
        ),
        _candidate(
            table="table.csv",
            row_index=1,
            column="",
            original="",
            values=[DELETE_ROW],
            selected=DELETE_ROW,
        ),
    ]
    report = tmp_path / "repair_candidates.jsonl"
    _write_jsonl(report, rows)

    result = validate_declared_repair_output(
        raw_root=raw,
        corrected_raw=corrected,
        candidate_report=report,
    )

    assert result["declared_repair_count"] == 2
    assert len(load_repair_candidates(report)) == 2

    _write_csv(corrected / "table.csv", ["id", "value"], [{"id": "1", "value": "OTHER"}])
    with pytest.raises(PiRepairAuditError, match="undeclared or incorrect"):
        validate_declared_repair_output(
            raw_root=raw,
            corrected_raw=corrected,
            candidate_report=report,
        )


def test_raw_repair_scorer_reports_detection_candidates_exact_repair_and_classes(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    _write_csv(
        raw / "table.csv",
        ["a", "b"],
        [
            {"a": "bad1", "b": "ok"},
            {"a": "bad2", "b": "ok"},
            {"a": "clean", "b": "ok"},
        ],
    )
    gold = tmp_path / "gold.csv"
    _write_csv(
        gold,
        [
            "error_class",
            "operation",
            "raw_file",
            "raw_row_index",
            "column",
            "clean_value",
            "dirty_value",
        ],
        [
            {
                "error_class": "class1_intra_table",
                "operation": "cell_update",
                "raw_file": "table.csv",
                "raw_row_index": "0",
                "column": "a",
                "clean_value": "clean1",
                "dirty_value": "bad1",
            },
            {
                "error_class": "class3_cross_table",
                "operation": "cell_update",
                "raw_file": "table.csv",
                "raw_row_index": "1",
                "column": "a",
                "clean_value": "clean2",
                "dirty_value": "bad2",
            },
        ],
    )
    report = tmp_path / "candidates.jsonl"
    _write_jsonl(
        report,
        [
            _candidate(
                table="table.csv",
                row_index=0,
                column="a",
                original="bad1",
                values=["clean1"],
                selected="clean1",
            ),
            _candidate(
                table="table.csv",
                row_index=1,
                column="a",
                original="bad2",
                values=["wrong", "clean2"],
                selected="wrong",
            ),
            _candidate(
                table="table.csv",
                row_index=2,
                column="a",
                original="clean",
                values=["changed"],
                selected="changed",
            ),
        ],
    )

    result = score_repair_candidates(
        candidate_report=report,
        gold_log=gold,
        raw_root=raw,
    )

    assert result["counts"]["detected_gold"] == 2
    assert result["counts"]["exact_repairs"] == 1
    assert result["counts"]["false_positive_repairs"] == 1
    assert result["metrics"]["candidate_recall_at_1"] == 0.5
    assert result["metrics"]["candidate_recall_at_3"] == 1.0
    assert result["metrics"]["exact_repair_precision"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["by_error_class"]["class3_cross_table"]["exact_repair_count"] == 0
    assert all("row_index" not in item for item in result["failure_cases"])


def test_raw_repair_scorer_collapses_sequential_gold_events_per_cell(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    _write_csv(raw / "table.csv", ["value"], [{"value": "bad2"}])
    gold = tmp_path / "gold.csv"
    _write_csv(
        gold,
        [
            "error_class",
            "operation",
            "raw_file",
            "raw_row_index",
            "column",
            "clean_value",
            "dirty_value",
        ],
        [
            {
                "error_class": "class1_intra_table",
                "operation": "cell_update",
                "raw_file": "table.csv",
                "raw_row_index": "0",
                "column": "value",
                "clean_value": "clean",
                "dirty_value": "bad1",
            },
            {
                "error_class": "class1_intra_table",
                "operation": "cell_update",
                "raw_file": "table.csv",
                "raw_row_index": "0",
                "column": "value",
                "clean_value": "bad1",
                "dirty_value": "bad2",
            },
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            _candidate(
                table="table.csv",
                row_index=0,
                column="value",
                original="bad2",
                values=["clean"],
                selected="clean",
            )
        ],
    )

    result = score_repair_candidates(
        candidate_report=candidates,
        gold_log=gold,
        raw_root=raw,
    )

    assert result["counts"]["gold_log_events"] == 2
    assert result["counts"]["overlapping_gold_events"] == 1
    assert result["counts"]["gold_repairs"] == 1
    assert result["counts"]["exact_repairs"] == 1


def test_raw_repair_scorer_rejects_branching_gold_values(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_csv(raw / "table.csv", ["value"], [{"value": "bad"}])
    gold = tmp_path / "gold.csv"
    _write_csv(
        gold,
        [
            "error_class",
            "operation",
            "raw_file",
            "raw_row_index",
            "column",
            "clean_value",
            "dirty_value",
        ],
        [
            {
                "error_class": "class1_intra_table",
                "operation": "cell_update",
                "raw_file": "table.csv",
                "raw_row_index": "0",
                "column": "value",
                "clean_value": "clean1",
                "dirty_value": "bad",
            },
            {
                "error_class": "class1_intra_table",
                "operation": "cell_update",
                "raw_file": "table.csv",
                "raw_row_index": "0",
                "column": "value",
                "clean_value": "clean2",
                "dirty_value": "bad",
            },
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [])

    with pytest.raises(PiRepairAuditError, match="ambiguous original clean"):
        score_repair_candidates(
            candidate_report=candidates,
            gold_log=gold,
            raw_root=raw,
        )


def test_row_insert_failure_case_hashes_the_private_row_payload(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_csv(raw / "table.csv", ["subject_id", "value"], [{"subject_id": "123", "value": "x"}])
    gold = tmp_path / "gold.csv"
    private_row = json.dumps({"subject_id": "123", "stay_id": "456", "value": "x"})
    _write_csv(
        gold,
        [
            "error_class",
            "operation",
            "raw_file",
            "raw_row_index",
            "column",
            "clean_value",
            "dirty_value",
        ],
        [
            {
                "error_class": "class4_task_oriented",
                "operation": "row_insert",
                "raw_file": "table.csv",
                "raw_row_index": "0",
                "column": "",
                "clean_value": "",
                "dirty_value": private_row,
            }
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [])

    result = score_repair_candidates(
        candidate_report=candidates,
        gold_log=gold,
        raw_root=raw,
    )

    failure = result["failure_cases"][0]
    assert failure["dirty_value"].startswith("row_sha256:")
    assert "123" not in json.dumps(failure)
    assert "456" not in json.dumps(failure)
    assert failure["expected_value"] == DELETE_ROW


def test_failure_cases_pseudonymize_locator_values(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_csv(raw / "table.csv", ["stay_id"], [{"stay_id": "123456"}])
    gold = tmp_path / "gold.csv"
    _write_csv(
        gold,
        [
            "error_class",
            "operation",
            "raw_file",
            "raw_row_index",
            "column",
            "clean_value",
            "dirty_value",
        ],
        [
            {
                "error_class": "class2_entity_alignment",
                "operation": "cell_update",
                "raw_file": "table.csv",
                "raw_row_index": "0",
                "column": "stay_id",
                "clean_value": "654321",
                "dirty_value": "123456",
            }
        ],
    )
    report = tmp_path / "candidates.jsonl"
    _write_jsonl(
        report,
        [
            _candidate(
                table="table.csv",
                row_index=0,
                column="stay_id",
                original="123456",
                values=["111111"],
                selected="111111",
            )
        ],
    )

    result = score_repair_candidates(
        candidate_report=report,
        gold_log=gold,
        raw_root=raw,
    )

    serialized = json.dumps(result["failure_cases"])
    assert "123456" not in serialized
    assert "654321" not in serialized
    assert "111111" not in serialized


def test_output_score_includes_all_raw_repair_metrics(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    _write_csv(raw / "table.csv", ["id", "value"], [{"id": "1", "value": "bad"}])
    _write_csv(
        output / "corrected_raw/table.csv",
        ["id", "value"],
        [{"id": "1", "value": "good"}],
    )
    _write_jsonl(
        output / "repair_candidates.jsonl",
        [
            _candidate(
                table="table.csv",
                row_index=0,
                column="value",
                original="bad",
                values=["good"],
                selected="good",
            )
        ],
    )
    gold = tmp_path / "gold.csv"
    _write_csv(
        gold,
        [
            "error_class",
            "operation",
            "raw_file",
            "raw_row_index",
            "column",
            "clean_value",
            "dirty_value",
        ],
        [
            {
                "error_class": "class1_intra_table",
                "operation": "cell_update",
                "raw_file": "table.csv",
                "raw_row_index": "0",
                "column": "value",
                "clean_value": "good",
                "dirty_value": "bad",
            }
        ],
    )
    result = score_repair_output(
        output_root=output,
        raw_root=raw,
        gold_log=gold,
    )

    assert result["metrics"]["exact_repair_f1"] == 1.0
    assert result["metrics"]["candidate_recall_at_5"] == 1.0
    assert result["metrics"]["detection_f1"] == 1.0
    assert "downstream_composite_score" not in result["metrics"]
