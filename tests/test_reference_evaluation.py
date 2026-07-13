from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import workflow.reference_evaluation as reference_evaluation

from workflow.reference_evaluation import SCORER_VERSION, score_reference_directory


FILE_COLUMNS = {
    "cohort/cohort_icu_mortality_0__.csv": ["stay_id", "value"],
    "csv/labels.csv": ["stay_id", "label"],
    "features/preproc_chart_icu.csv": ["stay_id", "itemid", "event_time_from_admit", "value"],
    "features/preproc_diag_icu.csv": ["stay_id", "new_icd_code", "value"],
    "features/preproc_med_icu.csv": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "endtime",
        "orderid",
        "value",
    ],
    "features/preproc_out_icu.csv": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "value",
    ],
    "features/preproc_proc_icu.csv": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "value",
    ],
    "summary/chart_features.csv": ["itemid", "value"],
    "summary/chart_summary.csv": ["itemid", "value"],
    "summary/diag_features.csv": ["new_icd_code", "value"],
    "summary/diag_summary.csv": ["new_icd_code", "value"],
    "summary/med_features.csv": ["itemid", "value"],
    "summary/med_summary.csv": ["itemid", "value"],
    "summary/out_features.csv": ["itemid", "value"],
    "summary/out_summary.csv": ["itemid", "value"],
    "summary/proc_features.csv": ["itemid", "value"],
    "summary/proc_summary.csv": ["itemid", "value"],
}


def _default_row(columns: list[str], suffix: str = "1") -> list[str]:
    values = {
        "subject_id": f"10{suffix}",
        "hadm_id": f"20{suffix}",
        "stay_id": f"30{suffix}",
        "itemid": f"40{suffix}",
        "event_time_from_admit": f"{suffix}:00:00",
        "starttime": f"2025-01-0{suffix} 00:00:00",
        "endtime": f"2025-01-0{suffix} 01:00:00",
        "orderid": f"50{suffix}",
        "charttime": f"2025-01-0{suffix} 00:30:00",
        "new_icd_code": f"A0{suffix}",
        "label": suffix,
        "value": f"value-{suffix}",
    }
    return [values[column] for column in columns]


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_complete_package(root: Path) -> None:
    for relative, columns in FILE_COLUMNS.items():
        _write_csv(root / relative, columns, [_default_row(columns)])


def _file_report(report: dict, relative_path: str) -> dict:
    return next(item for item in report["file_reports"] if item["relative_path"] == relative_path)


def test_exact_package_scores_one_with_schema_v2(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    result = tmp_path / "result"
    _write_complete_package(reference)
    _write_complete_package(result)

    report = score_reference_directory(result, reference)

    assert report["schema_version"] == 2
    assert report["scorer_version"] == SCORER_VERSION
    assert report["metrics"]["core_score"] == 1.0
    assert report["metrics"]["feature_score"] == 1.0
    assert report["metrics"]["summary_score"] == 1.0
    assert report["metrics"]["composite_score"] == 1.0


def test_row_order_does_not_change_score(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    result = tmp_path / "result"
    _write_complete_package(reference)
    _write_complete_package(result)
    columns = FILE_COLUMNS["cohort/cohort_icu_mortality_0__.csv"]
    rows = [_default_row(columns, "1"), _default_row(columns, "2")]
    _write_csv(reference / "cohort/cohort_icu_mortality_0__.csv", columns, rows)
    _write_csv(result / "cohort/cohort_icu_mortality_0__.csv", columns, list(reversed(rows)))

    report = score_reference_directory(result, reference)

    assert report["metrics"]["composite_score"] == 1.0


def test_partial_row_gets_cell_credit_but_not_exact_row_credit(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    result = tmp_path / "result"
    _write_complete_package(reference)
    _write_complete_package(result)
    relative = "cohort/cohort_icu_mortality_0__.csv"
    columns = FILE_COLUMNS[relative]
    reference_rows = [_default_row(columns, "1"), _default_row(columns, "2")]
    result_rows = [list(reference_rows[0]), list(reference_rows[1])]
    result_rows[1][columns.index("value")] = "wrong"
    _write_csv(reference / relative, columns, reference_rows)
    _write_csv(result / relative, columns, result_rows)

    report = score_reference_directory(result, reference)
    cohort = _file_report(report, relative)

    assert cohort["metrics"]["schema_f1"] == 1.0
    assert cohort["metrics"]["key_structure_f1"] == 1.0
    assert cohort["metrics"]["row_aligned_cell_f1"] == 0.75
    assert cohort["metrics"]["exact_row_f1"] == 0.5
    assert cohort["metrics"]["file_score"] == 0.75
    assert report["metrics"]["composite_score"] == 0.9625


def test_duplicate_key_rows_use_maximum_cell_match_alignment(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    result = tmp_path / "result"
    _write_complete_package(reference)
    _write_complete_package(result)
    relative = "features/preproc_diag_icu.csv"
    columns = FILE_COLUMNS[relative]
    reference_rows = [
        ["301", "A01", "alpha"],
        ["301", "B02", "beta"],
    ]
    result_rows = [
        ["301", "B02", "beta"],
        ["301", "A01", "wrong"],
    ]
    _write_csv(reference / relative, columns, reference_rows)
    _write_csv(result / relative, columns, result_rows)

    report = score_reference_directory(result, reference)
    diag = _file_report(report, relative)

    assert diag["key_columns"] == ["stay_id"]
    assert diag["matched_key_rows"] == 2
    assert diag["exact_row_matches"] == 1
    assert diag["per_column"]["new_icd_code"]["f1"] == 1.0
    assert diag["per_column"]["value"]["f1"] == 0.5


def test_missing_file_loses_its_fixed_weight(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    result = tmp_path / "result"
    _write_complete_package(reference)
    _write_complete_package(result)
    (result / "features/preproc_chart_icu.csv").unlink()

    report = score_reference_directory(result, reference)

    assert report["metrics"]["feature_score"] == 0.8
    assert report["metrics"]["composite_score"] == 0.88
    assert report["missing_files"] == ["features/preproc_chart_icu.csv"]


def test_missing_required_key_zeroes_key_cell_and_exact_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    result = tmp_path / "result"
    _write_complete_package(reference)
    _write_complete_package(result)
    relative = "features/preproc_chart_icu.csv"
    reference_columns = FILE_COLUMNS[relative]
    result_columns = [column for column in reference_columns if column != "event_time_from_admit"]
    _write_csv(result / relative, result_columns, [_default_row(result_columns)])

    report = score_reference_directory(result, reference)
    chart = _file_report(report, relative)

    assert chart["missing_key_columns"] == ["event_time_from_admit"]
    assert chart["metrics"]["key_structure_f1"] == 0.0
    assert chart["metrics"]["row_aligned_cell_f1"] == 0.0
    assert chart["metrics"]["exact_row_f1"] == 0.0
    assert chart["metrics"]["file_score"] == pytest.approx(0.1 * (6 / 7), abs=1e-6)


def test_evaluation_json_replace_failure_preserves_previous_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "evaluation_report.json"
    path.write_text('{"version": 1}\n', encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(reference_evaluation.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        reference_evaluation._write_json(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert not list(tmp_path.glob(".evaluation_report.json.*.tmp"))
