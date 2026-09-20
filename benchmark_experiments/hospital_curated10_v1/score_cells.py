from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TABLE_NAME = "hospital.csv"
KEY_COLUMN = "index"


def score_cells(
    *,
    dirty_root: str | Path,
    clean_root: str | Path,
    result_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    dirty_path = Path(dirty_root).expanduser().resolve() / TABLE_NAME
    clean_path = Path(clean_root).expanduser().resolve() / TABLE_NAME
    result_path = Path(result_root).expanduser().resolve() / TABLE_NAME
    dirty_fields, dirty_rows = _read_csv(dirty_path)
    clean_fields, clean_rows = _read_csv(clean_path)
    result_fields, result_rows = _read_csv(result_path)
    if dirty_fields != clean_fields or dirty_fields != result_fields:
        raise ValueError("dirty, clean, and result schemas differ")
    if KEY_COLUMN not in dirty_fields:
        raise ValueError(f"missing key column: {KEY_COLUMN}")
    if not (len(dirty_rows) == len(clean_rows) == len(result_rows)):
        raise ValueError("dirty, clean, and result row counts differ")
    keys = [row[KEY_COLUMN] for row in dirty_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("dirty keys are not unique")
    if keys != [row[KEY_COLUMN] for row in clean_rows]:
        raise ValueError("clean row order or keys differ")
    if keys != [row[KEY_COLUMN] for row in result_rows]:
        raise ValueError("result row order or keys differ")

    columns = [field for field in dirty_fields if field != KEY_COLUMN]
    totals = _empty_counts()
    by_column = {column: _empty_counts() for column in columns}
    exact_rows = 0
    for dirty, clean, result in zip(dirty_rows, clean_rows, result_rows, strict=True):
        if result == clean:
            exact_rows += 1
        for column in columns:
            _record_cell(totals, dirty[column], clean[column], result[column])
            _record_cell(by_column[column], dirty[column], clean[column], result[column])

    metrics = _metrics(totals)
    report = {
        "schema_version": 1,
        "status": "SUCCESS",
        "workflow": "raha_hospital_cell_evaluation",
        "definitions": {
            "predicted_error": "result != dirty",
            "ground_truth_error": "dirty != clean",
            "exact_repair": "ground_truth_error and result == clean",
        },
        "metrics": {
            **metrics,
            "rows": len(dirty_rows),
            "columns_excluding_key": len(columns),
            "exact_clean_rows": exact_rows,
            "exact_clean_row_rate": _divide(exact_rows, len(dirty_rows)),
        },
        "by_column": {column: _metrics(counts) for column, counts in by_column.items()},
        "integrity": {
            "schema_match": True,
            "row_count_match": True,
            "key_order_match": True,
            "key_unique": True,
        },
        "sha256": {
            "dirty": _file_sha256(dirty_path),
            "clean": _file_sha256(clean_path),
            "result": _file_sha256(result_path),
        },
    }
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _empty_counts() -> dict[str, int]:
    return {
        "cells": 0,
        "ground_truth_error_cells": 0,
        "ground_truth_clean_cells": 0,
        "predicted_error_cells": 0,
        "detected_error_cells": 0,
        "exact_repair_cells": 0,
        "missed_error_cells": 0,
        "incorrect_repair_cells": 0,
        "clean_cells_modified": 0,
        "clean_cells_preserved": 0,
    }


def _record_cell(counts: dict[str, int], dirty: str, clean: str, result: str) -> None:
    truth_error = dirty != clean
    predicted_error = result != dirty
    exact_repair = truth_error and result == clean
    counts["cells"] += 1
    counts["ground_truth_error_cells" if truth_error else "ground_truth_clean_cells"] += 1
    if predicted_error:
        counts["predicted_error_cells"] += 1
    if truth_error and predicted_error:
        counts["detected_error_cells"] += 1
    if exact_repair:
        counts["exact_repair_cells"] += 1
    if truth_error and not predicted_error:
        counts["missed_error_cells"] += 1
    if truth_error and predicted_error and not exact_repair:
        counts["incorrect_repair_cells"] += 1
    if not truth_error and predicted_error:
        counts["clean_cells_modified"] += 1
    if not truth_error and not predicted_error:
        counts["clean_cells_preserved"] += 1


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    detection_precision = _divide(
        counts["detected_error_cells"], counts["predicted_error_cells"]
    )
    detection_recall = _divide(
        counts["detected_error_cells"], counts["ground_truth_error_cells"]
    )
    exact_precision = _divide(
        counts["exact_repair_cells"], counts["predicted_error_cells"]
    )
    exact_recall = _divide(
        counts["exact_repair_cells"], counts["ground_truth_error_cells"]
    )
    return {
        **counts,
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": _f1(detection_precision, detection_recall),
        "exact_repair_precision": exact_precision,
        "exact_repair_recall": exact_recall,
        "exact_repair_f1": _f1(exact_precision, exact_recall),
        "candidate_recall_at_1": exact_recall,
        "clean_preservation": _divide(
            counts["clean_cells_preserved"], counts["ground_truth_clean_cells"]
        ),
    }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [{field: str(row.get(field) or "") for field in fields} for row in reader]
    return fields, rows


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score frozen Hospital cell repairs.")
    parser.add_argument("--dirty-root", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = score_cells(
        dirty_root=args.dirty_root,
        clean_root=args.clean_root,
        result_root=args.result_root,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
