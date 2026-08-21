from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from graph.candidate_ranking import (
    CandidateRankingError,
    RankTrainingConfig,
    exact_candidate_position,
    export_train_validation_gold,
    load_candidate_artifact,
    train_complex_ranker,
    value_embedding_text,
    _validate_prepared_arrays,
)


def _candidate_row(split: str = "train", observation_index: int = 7) -> dict[str, object]:
    return {
        "split": split,
        "observation_index": observation_index,
        "table": "icu/chartevents",
        "raw_row_index": 3,
        "column": "valuenum",
        "current_value": "500",
        "dirty_score": 0.99,
        "field_rule": "icu/chartevents.valuenum",
        "reason": "accepted_field_rule",
        "candidates": [
            {"value": "5", "rule_id": "scale", "evidence": "divide by 100"}
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_value_embedding_text_matches_typed_value_graph_contract() -> None:
    assert json.loads(value_embedding_text("icu/chartevents", "valuenum", " 5.0 ")) == {
        "domain": "icu/chartevents.valuenum",
        "kind": "value",
        "shared": False,
        "value": "5.0",
        "value_type": "numeric",
    }
    assert json.loads(value_embedding_text("icu/chartevents", "stay_id", "30001")) == {
        "domain": "entity.stay_id",
        "kind": "value",
        "shared": True,
        "value": "30001",
        "value_type": "identifier",
    }
    assert json.loads(value_embedding_text(
        "hosp/diagnoses_icd", "icd_code", "I10", icd_version="10"
    ))["value"] == "10:I10"
    assert json.loads(value_embedding_text("hosp/patients", "gender", "NULL"))["value"] == "<MISSING>"


def test_exact_gold_does_not_collapse_normalized_missing_aliases() -> None:
    assert value_embedding_text("hosp/patients", "gender", "") == value_embedding_text(
        "hosp/patients", "gender", "NULL"
    )
    assert exact_candidate_position(["NULL"], "") == -1
    assert exact_candidate_position([""], "") == 0


def test_candidate_loader_enforces_split_schema_and_private_field_boundary(tmp_path: Path) -> None:
    valid = tmp_path / "candidate_values_train.jsonl"
    _write_jsonl(valid, [_candidate_row()])
    loaded = load_candidate_artifact(valid, "train")
    assert loaded[7]["candidate_values"] == ["5"]

    private = _candidate_row()
    private["clean_value"] = "5"
    private_path = tmp_path / "private.jsonl"
    _write_jsonl(private_path, [private])
    with pytest.raises(CandidateRankingError, match="private top-level schema"):
        load_candidate_artifact(private_path, "train")

    wrong_split_path = tmp_path / "wrong.jsonl"
    _write_jsonl(wrong_split_path, [_candidate_row("validation")])
    with pytest.raises(CandidateRankingError, match="expected only train"):
        load_candidate_artifact(wrong_split_path, "train")


def test_candidate_loader_rejects_duplicate_values_and_internal_test(tmp_path: Path) -> None:
    row = _candidate_row()
    row["candidates"] = [
        {"value": "5", "rule_id": "a", "evidence": "x"},
        {"value": "5", "rule_id": "b", "evidence": "y"},
    ]
    duplicate = tmp_path / "duplicates.jsonl"
    _write_jsonl(duplicate, [row])
    with pytest.raises(CandidateRankingError, match="unique"):
        load_candidate_artifact(duplicate, "train")

    internal = tmp_path / "internal_test" / "candidate_values.jsonl"
    _write_jsonl(internal, [_candidate_row()])
    with pytest.raises(CandidateRankingError, match="Internal Test"):
        load_candidate_artifact(internal, "train")


def test_scope_gold_exports_only_folds_zero_through_three(tmp_path: Path) -> None:
    graph = tmp_path / "graph"
    supervision = tmp_path / "supervision"
    graph.mkdir()
    supervision.mkdir()
    (graph / "graph_manifest.json").write_text('{"status":"SUCCESS"}\n')
    (supervision / "supervision_manifest.json").write_text('{"status":"SUCCESS"}\n')
    with (graph / "cell_observations.jsonl").open("w", encoding="utf-8") as handle:
        for fold in range(5):
            handle.write(json.dumps({
                "table": "icu/chartevents",
                "row_number": fold + 1,
                "column": "valuenum",
                "row_id": fold,
                "raw_value": f"dirty-{fold}",
            }) + "\n")
    np.savez_compressed(
        supervision / "supervision_masks.npz",
        cell_indices=np.arange(5, dtype=np.int64),
        cell_labels=np.ones(5, dtype=np.int8),
        cell_folds=np.arange(5, dtype=np.int8),
        cell_source=np.ones(5, dtype=np.int8),
    )
    source = tmp_path / "merged_injection_log.csv"
    fields = ["raw_file", "raw_row_index", "column", "dirty_value", "clean_value"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for fold in range(5):
            writer.writerow({
                "raw_file": "icu/chartevents.csv",
                "raw_row_index": fold,
                "column": "valuenum",
                "dirty_value": f"dirty-{fold}",
                "clean_value": str(fold),
            })

    predictions = tmp_path / "predictions.csv"
    prediction_fields = [
        "split", "observation_index", "fold", "label", "prediction",
        "model_type", "seed",
    ]
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields, lineterminator="\n")
        writer.writeheader()
        for fold in range(5):
            writer.writerow({
                "split": "train" if fold < 3 else ("validation" if fold == 3 else "internal_test"),
                "observation_index": fold,
                "fold": fold,
                "label": 1,
                "prediction": 1,
                "model_type": "strict_rgcn",
                "seed": 666,
            })

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    detector_report = tmp_path / "report.json"
    detector_report.write_text(json.dumps({
        "status": "SUCCESS",
        "model_type": "strict_rgcn",
        "seed": 666,
        "training_identity": {"identity": "fixture"},
        "outputs": {
            "best_checkpoint": {"sha256": "a" * 64},
            "predictions": {"sha256": sha256(predictions)},
        },
    }) + "\n", encoding="utf-8")

    output = tmp_path / "scoped"
    report = export_train_validation_gold(
        graph_dir=graph,
        supervision_dir=supervision,
        paired_log=source,
        predictions=predictions,
        detector_report=detector_report,
        output_dir=output,
    )
    assert report["record_count"] == 4
    assert report["folds"] == [0, 1, 2, 3]
    assert report["internal_test_gold_exported"] is False
    assert report["prediction_count"] == 4
    with (output / "paired_gold_train_validation.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["clean_value"] for row in rows] == ["0", "1", "2", "3"]
    assert "4" not in {row["clean_value"] for row in rows}
    with (output / "predictions_train_validation.csv").open(encoding="utf-8") as handle:
        scoped_predictions = list(csv.DictReader(handle))
    assert {row["split"] for row in scoped_predictions} == {"train", "validation"}


def test_training_rejects_manifest_that_accessed_internal_test_before_torch(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "prepared_manifest.json").write_text(json.dumps({
        "workflow": "strict_rgcn_complex_candidate_ranking_prepare",
        "folds": {"train": [0, 1, 2], "validation": [3]},
        "internal_test_gold_accessed": True,
        "internal_test_candidates_accessed": False,
    }) + "\n")
    with pytest.raises(CandidateRankingError, match="Train/Validation-only"):
        train_complex_ranker(RankTrainingConfig(
            prepared_dir=str(prepared),
            output_dir=str(tmp_path / "output"),
            seed=666,
        ))


def test_prepared_array_contract_rejects_fold_four_context() -> None:
    manifest = {"counts": {
        "context_count": 2,
        "train_pair_count": 1,
        "validation_detector_positive_count": 1,
        "validation_candidate_count": 1,
        "unique_value_count": 2,
    }}
    examples = {
        "context_observation": np.asarray([10, 20], dtype=np.int64),
        "context_relation": np.asarray([0, 0], dtype=np.int64),
        "context_fold": np.asarray([0, 3], dtype=np.int8),
        "pair_context": np.asarray([0], dtype=np.int64),
        "pair_positive": np.asarray([0], dtype=np.int64),
        "pair_negative": np.asarray([1], dtype=np.int64),
        "pair_weight": np.asarray([1.0], dtype=np.float32),
        "validation_group_observation": np.asarray([20], dtype=np.int64),
        "validation_group_context": np.asarray([1], dtype=np.int64),
        "validation_group_gold": np.asarray([0], dtype=np.int64),
        "validation_group_gold_position": np.asarray([0], dtype=np.int8),
        "validation_group_label": np.asarray([1], dtype=np.int8),
        "validation_offsets": np.asarray([0, 1], dtype=np.int64),
        "validation_candidate_ids": np.asarray([0], dtype=np.int64),
    }
    hidden = np.zeros((2, 8), dtype=np.float32)
    relation = np.zeros((1, 8), dtype=np.float32)
    _validate_prepared_arrays(manifest, hidden, relation, hidden, examples)

    examples["context_fold"] = np.asarray([0, 4], dtype=np.int8)
    with pytest.raises(CandidateRankingError, match="fold/relation"):
        _validate_prepared_arrays(manifest, hidden, relation, hidden, examples)


def test_complex_score_head_has_expected_shape() -> None:
    torch = pytest.importorskip("torch")
    from graph.candidate_ranking import _complex_class

    model = _complex_class(torch)(input_dim=8, rank_dim=4, dropout=0.0)
    scores = model(
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.randn(3, 8),
    )
    assert tuple(scores.shape) == (3,)
    scores.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
