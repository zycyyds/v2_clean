from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from workflow.pi_harness_evaluation import (
    load_evaluation_manifest,
    public_feedback_from_equal_weight_report,
    score_equal_weight_reference_directory,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, files: dict[str, list[str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    name: {"key_columns": keys}
                    for name, keys in files.items()
                },
            },
        ),
        encoding="utf-8",
    )


def test_equal_weight_scorer_discovers_gold_and_scores_missing_file_zero(
    tmp_path: Path,
) -> None:
    gold = tmp_path / "gold"
    result = tmp_path / "result"
    _write_csv(gold / "a/one.csv", [{"id": 1, "value": "A"}])
    _write_csv(gold / "b/two.csv", [{"id": 2, "value": "B"}])
    _write_csv(result / "moved/one.csv", [{"id": 1, "value": "A"}])
    _write_csv(result / "extra.csv", [{"id": 9, "value": "X"}])
    manifest_path = tmp_path / "evaluation_manifest.json"
    _write_manifest(
        manifest_path,
        {"a/one.csv": ["id"], "b/two.csv": ["id"]},
    )

    report = score_equal_weight_reference_directory(
        result,
        gold,
        load_evaluation_manifest(manifest_path, gold),
    )

    assert report["metrics"]["composite_score"] == pytest.approx(0.5)
    assert report["metrics"]["file_coverage"] == pytest.approx(0.5)
    by_path = {item["relative_path"]: item for item in report["file_reports"]}
    assert by_path["a/one.csv"]["match_mode"] == "basename"
    assert by_path["a/one.csv"]["metrics"]["file_score"] == pytest.approx(1.0)
    assert by_path["b/two.csv"]["status"] == "missing_file"
    assert by_path["b/two.csv"]["metrics"]["file_score"] == 0.0
    assert report["extra_result_files"] == ["extra.csv"]


def test_equal_weight_scorer_prefers_same_relative_path_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    gold = tmp_path / "gold"
    result = tmp_path / "result"
    rows = [{"id": 1, "value": "A"}]
    _write_csv(gold / "a/one.csv", rows)
    _write_csv(result / "a/one.csv", rows)
    _write_csv(result / "other/one.csv", rows)
    manifest_path = tmp_path / "evaluation_manifest.json"
    _write_manifest(manifest_path, {"a/one.csv": ["id"]})

    report = score_equal_weight_reference_directory(
        result,
        gold,
        load_evaluation_manifest(manifest_path, gold),
    )
    assert report["file_reports"][0]["match_mode"] == "relative_path"
    assert report["metrics"]["composite_score"] == 1.0

    (result / "a/one.csv").unlink()
    _write_csv(result / "another/one.csv", rows)
    report = score_equal_weight_reference_directory(
        result,
        gold,
        load_evaluation_manifest(manifest_path, gold),
    )
    assert report["file_reports"][0]["status"] == "ambiguous_file"
    assert report["metrics"]["composite_score"] == 0.0


def test_manifest_must_cover_every_gold_file_and_reference_real_columns(
    tmp_path: Path,
) -> None:
    gold = tmp_path / "gold"
    _write_csv(gold / "one.csv", [{"id": 1, "value": "A"}])
    manifest_path = tmp_path / "evaluation_manifest.json"
    _write_manifest(manifest_path, {})

    with pytest.raises(ValueError, match="missing key configuration"):
        load_evaluation_manifest(manifest_path, gold)

    _write_manifest(manifest_path, {"one.csv": ["missing_id"]})
    with pytest.raises(ValueError, match="unknown key columns"):
        load_evaluation_manifest(manifest_path, gold)


def test_public_feedback_exposes_only_scores_counts_and_status() -> None:
    report = {
        "metrics": {"composite_score": 0.75},
        "file_reports": [
            {
                "relative_path": "secret/one.csv",
                "status": "compared",
                "reference_file": "/private/gold/secret/one.csv",
                "result_file": "/work/result/one.csv",
                "reference_rows": 10,
                "result_rows": 12,
                "metrics": {"file_score": 0.75, "schema_f1": 1.0},
            },
        ],
    }

    feedback = public_feedback_from_equal_weight_report(
        report,
        round_index=2,
        best_score=0.8,
        promoted=False,
        restored_best=True,
    )

    assert feedback == {
        "schema_version": 1,
        "round": 2,
        "current_score": 0.75,
        "best_score": 0.8,
        "promoted": False,
        "restored_best": True,
        "files": [
            {
                "filename": "one.csv",
                "file_score": 0.75,
                "gold_rows": 10,
                "result_rows": 12,
                "status": "compared",
            },
        ],
    }
    serialized = json.dumps(feedback)
    assert "/private/gold" not in serialized
    assert "schema_f1" not in serialized


def test_result_symlink_is_rejected_before_host_scoring(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    result = tmp_path / "result"
    _write_csv(gold / "one.csv", [{"id": 1, "value": "secret"}])
    result.mkdir()
    (result / "one.csv").symlink_to(gold / "one.csv")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {"one.csv": ["id"]})

    with pytest.raises(ValueError, match="symbolic link"):
        score_equal_weight_reference_directory(
            result,
            gold,
            load_evaluation_manifest(manifest, gold),
        )


def test_one_result_file_cannot_score_two_same_named_gold_files(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    result = tmp_path / "result"
    rows = [{"id": 1, "value": "same"}]
    _write_csv(gold / "a/one.csv", rows)
    _write_csv(gold / "b/one.csv", rows)
    _write_csv(result / "a/one.csv", rows)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {"a/one.csv": ["id"], "b/one.csv": ["id"]})

    report = score_equal_weight_reference_directory(
        result,
        gold,
        load_evaluation_manifest(manifest, gold),
    )

    assert report["metrics"]["composite_score"] == pytest.approx(0.5)
    assert report["detail_metrics"]["matched_file_count"] == 1


def test_row_reordering_does_not_change_file_score(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    result = tmp_path / "result"
    rows = [
        {"id": 1, "value": "A", "amount": 10},
        {"id": 2, "value": "B", "amount": 20},
        {"id": 3, "value": "C", "amount": 30},
    ]
    _write_csv(gold / "data.csv", rows)
    _write_csv(result / "data.csv", list(reversed(rows)))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {"data.csv": ["id"]})

    report = score_equal_weight_reference_directory(
        result,
        gold,
        load_evaluation_manifest(manifest, gold),
    )

    assert report["metrics"]["composite_score"] == 1.0
    assert report["file_reports"][0]["metrics"]["exact_row_f1"] == 1.0


def test_duplicate_keys_use_best_row_alignment(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    result = tmp_path / "result"
    gold_rows = [
        {"id": 1, "value": "A", "amount": 10},
        {"id": 1, "value": "B", "amount": 20},
    ]
    _write_csv(gold / "data.csv", gold_rows)
    _write_csv(result / "data.csv", list(reversed(gold_rows)))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {"data.csv": ["id"]})

    report = score_equal_weight_reference_directory(
        result,
        gold,
        load_evaluation_manifest(manifest, gold),
    )

    file_report = report["file_reports"][0]
    assert file_report["matched_key_rows"] == 2
    assert file_report["exact_row_matches"] == 2
    assert file_report["metrics"]["file_score"] == 1.0
