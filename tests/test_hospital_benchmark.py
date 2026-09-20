from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark_experiments.hospital_curated10_v1.build_dataset import (
    EXPECTED_CLEAN_FIELDS,
    EXPECTED_DIRTY_FIELDS,
    build_hospital_dataset,
)
from benchmark_experiments.hospital_curated10_v1.adapt_strict_test_snapshot import (
    adapt_snapshot,
)
from benchmark_experiments.hospital_curated10_v1.score_cells import score_cells


FIELDS = ["index", "provider_number", "city", "state"]


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)


def _read_indices(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row["index"] for row in csv.DictReader(handle)]


def test_builder_selects_deterministic_coverage_examples(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty.csv"
    clean = tmp_path / "clean.csv"
    dirty_rows = [
        ["1", "10x", "alpha", "aa"],
        ["2", "101", "xlpha", "aa"],
        ["3", "102", "gamma", "xx"],
        ["4", "1x3", "delta", "dd"],
        ["5", "104", "epsilon", "ee"],
    ]
    clean_rows = [
        ["1", "100", "alpha", "aa"],
        ["2", "101", "alpha", "aa"],
        ["3", "102", "gamma", "cc"],
        ["4", "103", "delta", "dd"],
        ["5", "104", "epsilon", "ee"],
    ]
    _write_rows(dirty, dirty_rows)
    _write_rows(clean, clean_rows)

    report = build_hospital_dataset(
        dirty_source=dirty,
        clean_source=clean,
        output_dir=tmp_path / "dataset",
        train_count=3,
        expected_rows=5,
    )

    dataset = tmp_path / "dataset"
    assert report["counts"] == {"all": 5, "train": 3, "correction": 2}
    assert report["difference_counts"]["cells"] == 4
    assert report["selection"] == "curated_greedy_changed_column_coverage_v1"
    assert _read_indices(dataset / "train/raw/hospital.csv") == ["1", "2", "3"]
    assert _read_indices(dataset / "correction/raw/hospital.csv") == ["4", "5"]
    assert (
        dataset / "train/raw/hospital.csv"
    ).read_bytes() == (dataset / "train_only_replay/raw/hospital.csv").read_bytes()
    assert (
        dataset / "train/reference/hospital.csv"
    ).read_bytes() == (dataset / "train_only_replay/gold/hospital.csv").read_bytes()
    private_diffs = (dataset / "host_private/cell_diff.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(private_diffs) == 4
    assert "dirty_value" in json.loads(private_diffs[0])


def test_cell_scorer_separates_detection_repair_and_preservation(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty/hospital.csv"
    clean = tmp_path / "clean/hospital.csv"
    result = tmp_path / "result/hospital.csv"
    _write_rows(
        dirty,
        [
            ["1", "10x", "alpha", "aa"],
            ["2", "101", "xlpha", "bb"],
        ],
    )
    _write_rows(
        clean,
        [
            ["1", "100", "alpha", "aa"],
            ["2", "101", "alpha", "bb"],
        ],
    )
    _write_rows(
        result,
        [
            ["1", "100", "alpha", "zz"],
            ["2", "101", "wrong", "bb"],
        ],
    )

    report = score_cells(
        dirty_root=dirty.parent,
        clean_root=clean.parent,
        result_root=result.parent,
        output_path=tmp_path / "score.json",
    )

    metrics = report["metrics"]
    assert metrics["ground_truth_error_cells"] == 2
    assert metrics["predicted_error_cells"] == 3
    assert metrics["detected_error_cells"] == 2
    assert metrics["exact_repair_cells"] == 1
    assert metrics["detection_precision"] == 2 / 3
    assert metrics["detection_recall"] == 1.0
    assert metrics["exact_repair_precision"] == 1 / 3
    assert metrics["exact_repair_recall"] == 0.5
    assert metrics["candidate_recall_at_1"] == 0.5
    assert metrics["clean_cells_modified"] == 1
    assert report["by_column"]["provider_number"]["exact_repair_recall"] == 1.0
    assert report["by_column"]["city"]["exact_repair_recall"] == 0.0


def test_builder_accepts_only_the_official_hospital_schema_mapping(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty.csv"
    clean = tmp_path / "clean.csv"
    dirty_values = ["1", *[f"value-{index}" for index in range(1, 20)]]
    clean_values = list(dirty_values)
    clean_values[1] = "corrected"
    second_values = ["2", *[f"second-{index}" for index in range(1, 20)]]
    _write_custom_csv(dirty, EXPECTED_DIRTY_FIELDS, [dirty_values, second_values])
    _write_custom_csv(clean, EXPECTED_CLEAN_FIELDS, [clean_values, second_values])

    report = build_hospital_dataset(
        dirty_source=dirty,
        clean_source=clean,
        output_dir=tmp_path / "mapped",
        train_count=1,
        expected_rows=2,
    )

    assert report["clean_schema_mode"] == "raha_official_positional_mapping_v1"


def test_builder_can_create_disjoint_seeded_validation_split(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty.csv"
    clean = tmp_path / "clean.csv"
    dirty_rows = [
        [str(index), f"10{index}", f"city-{index}", "aa"]
        for index in range(1, 9)
    ]
    clean_rows = [list(row) for row in dirty_rows]
    for offset in range(5):
        dirty_rows[offset][1] = f"x0{offset + 1}"
    _write_rows(dirty, dirty_rows)
    _write_rows(clean, clean_rows)

    report = build_hospital_dataset(
        dirty_source=dirty,
        clean_source=clean,
        output_dir=tmp_path / "split",
        train_count=2,
        validation_count=2,
        validation_seed=666,
        expected_rows=8,
    )

    root = tmp_path / "split"
    train = set(_read_indices(root / "train/raw/hospital.csv"))
    validation = set(_read_indices(root / "validation/raw/hospital.csv"))
    correction = set(_read_indices(root / "correction/raw/hospital.csv"))
    assert report["counts"] == {
        "all": 8,
        "train": 2,
        "validation": 2,
        "correction": 4,
    }
    assert not train & validation
    assert not train & correction
    assert not validation & correction
    assert train | validation | correction == {str(index) for index in range(1, 9)}
    assert report["validation_selection"] == "seeded_stable_hash_over_remaining_rows_v1"


@pytest.mark.parametrize("train_raw_flag", ["--train-raw", "--train_raw"])
def test_strict_test_adapter_changes_only_submission_contract(
    tmp_path: Path, train_raw_flag: str
) -> None:
    source = tmp_path / "source"
    snapshot = source / "host/reproducible_snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline.py").write_text("print('unchanged')\n", encoding="utf-8")
    (snapshot / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result_root": "result_package",
                "replay": {
                    "argv": [
                        "python3",
                        "pipeline.py",
                        train_raw_flag,
                        "{train_raw}",
                        "--train-reference",
                        "{train_reference}",
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
    (source / "host/run_report.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS_REPRODUCIBLE",
                "reproducible_snapshot": str(snapshot),
            }
        ),
        encoding="utf-8",
    )
    (source / "host/run_manifest.json").write_text(
        json.dumps({"train_reference": str(tmp_path / "train_reference")}),
        encoding="utf-8",
    )

    output = tmp_path / "adapted"
    report = adapt_snapshot(source, output)

    adapted_submission = json.loads(
        (output / "host/reproducible_snapshot/submission.json").read_text(
            encoding="utf-8"
        )
    )
    argv = adapted_submission["replay"]["argv"]
    assert "{train_raw}" not in argv
    assert argv.count("{train_reference}") == 2
    assert (
        output / "host/reproducible_snapshot/pipeline.py"
    ).read_bytes() == (snapshot / "pipeline.py").read_bytes()
    assert report["cleaning_implementation_modified"] is False
    assert report["test_data_accessed"] is False


def _write_custom_csv(path: Path, fields: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)
