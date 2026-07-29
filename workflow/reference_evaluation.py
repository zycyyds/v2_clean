from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd


SCORER_VERSION = "balanced_row_aligned_v2"
NUMERIC_NORMALIZATION_PLACES = Decimal("0.000001")
STRUCTURED_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tsv.gz",
    ".parquet",
    ".jsonl",
    ".json",
    ".xlsx",
    ".xls",
)
IGNORED_PACKAGE_FILES = {
    "package_manifest.json",
    "result_manifest.json",
    "target_field_mapping.json",
    "reference.csv",
}
FEATURE_FILES = {
    "features/preproc_chart_icu.csv",
    "features/preproc_diag_icu.csv",
    "features/preproc_med_icu.csv",
    "features/preproc_out_icu.csv",
    "features/preproc_proc_icu.csv",
}
SUMMARY_FILES = {
    f"summary/{source}_{kind}.csv"
    for source in ("chart", "diag", "med", "out", "proc")
    for kind in ("features", "summary")
}


def score_reference_directory(
    result_package: str | Path,
    reference_root: str | Path,
    *,
    key_column: str = "",
) -> dict[str, Any]:
    package_root = Path(result_package).expanduser().resolve()
    gold_root = Path(reference_root).expanduser().resolve()
    reference_files = [
        path
        for path in _structured_package_files(gold_root)
        if path.name not in IGNORED_PACKAGE_FILES
    ]
    if not reference_files:
        return _empty_report("reference directory contains no comparable structured files")

    file_reports: list[dict[str, Any]] = []
    missing_files: list[str] = []
    missing_columns: dict[str, list[str]] = {}
    issues: list[str] = []
    category_weighted = {"core": 0.0, "feature": 0.0, "summary": 0.0}
    category_budgets = {"core": 0.20, "feature": 0.60, "summary": 0.20}
    metric_totals = {
        "schema_f1": 0.0,
        "key_structure_f1": 0.0,
        "row_aligned_cell_f1": 0.0,
        "exact_row_f1": 0.0,
    }
    matched_files = 0

    for reference_file in reference_files:
        relative = reference_file.relative_to(gold_root).as_posix()
        category, weight = _file_weight(relative)
        if weight == 0.0:
            issues.append(f"unweighted reference file: {relative}")
        result_file = package_root / relative
        if not result_file.is_file():
            missing_files.append(relative)
            report = _missing_file_report(relative, reference_file, result_file, category, weight)
        else:
            try:
                report = _compare_file(
                    reference_file=reference_file,
                    result_file=result_file,
                    relative_path=relative,
                    category=category,
                    weight=weight,
                    fallback_key_column=key_column,
                )
                matched_files += 1
            except Exception as exc:
                report = _missing_file_report(relative, reference_file, result_file, category, weight)
                report.update({"status": "read_error", "issue": f"{type(exc).__name__}: {exc}"})
                issues.append(f"cannot compare {relative}: {type(exc).__name__}: {exc}")
        if report.get("missing_columns"):
            missing_columns[relative] = list(report["missing_columns"])
        raw_metrics = report.pop("_raw_metrics", report["metrics"])
        file_reports.append(report)
        if category in category_weighted:
            category_weighted[category] += weight * float(raw_metrics["file_score"])
        for name in metric_totals:
            metric_totals[name] += weight * float(raw_metrics[name])

    core_score = category_weighted["core"] / category_budgets["core"]
    feature_score = category_weighted["feature"] / category_budgets["feature"]
    summary_score = category_weighted["summary"] / category_budgets["summary"]
    composite = sum(category_weighted.values())
    expected_files = len(reference_files)
    metrics = {
        "core_score": _rounded(core_score),
        "feature_score": _rounded(feature_score),
        "summary_score": _rounded(summary_score),
        "schema_f1": _rounded(metric_totals["schema_f1"]),
        "key_structure_f1": _rounded(metric_totals["key_structure_f1"]),
        "row_aligned_cell_f1": _rounded(metric_totals["row_aligned_cell_f1"]),
        "exact_row_f1": _rounded(metric_totals["exact_row_f1"]),
        "file_coverage": _rounded(matched_files / max(expected_files, 1)),
        "composite_score": _rounded(composite),
    }
    return {
        "schema_version": 2,
        "scorer_version": SCORER_VERSION,
        "status": "SUCCESS",
        "mode": "balanced_reference_directory",
        "metrics": metrics,
        "weighting": {
            "core": 0.20,
            "feature": 0.60,
            "summary": 0.20,
            "per_file": {
                "cohort": 0.15,
                "labels": 0.05,
                "feature": 0.12,
                "summary": 0.02,
            },
            "within_file": {
                "schema_f1": 0.10,
                "key_structure_f1": 0.20,
                "row_aligned_cell_f1": 0.40,
                "exact_row_f1": 0.30,
            },
        },
        "detail_metrics": {
            "reference_file_count": expected_files,
            "matched_file_count": matched_files,
            "weighted_score_sum": _rounded(composite),
        },
        "missing_files": sorted(missing_files),
        "missing_columns": missing_columns,
        "issues": issues,
        "file_reports": file_reports,
    }


def evaluate_reference_directory_package(
    *,
    result_package: str | Path,
    validation_reference_root: str | Path | None,
    output_dir: str | Path,
    key_column: str = "",
) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not validation_reference_root:
        report = _empty_report("reference directory is unavailable")
    else:
        reference = Path(validation_reference_root).expanduser().resolve()
        if not reference.is_dir():
            report = _empty_report("reference directory is unavailable")
        else:
            report = score_reference_directory(result_package, reference, key_column=key_column)
    public_feedback = public_feedback_from_report(report)
    report_path = output / "evaluation_report.json"
    public_path = output / "public_feedback.json"
    private_path = output / "private_report.json"
    _write_json(report_path, report)
    _write_json(public_path, public_feedback)
    _write_json(private_path, {**report, "privacy": "private_reference_directory_metrics"})
    return {
        "evaluation_report": str(report_path),
        "public_feedback": str(public_path),
        "private_report": str(private_path),
    }


def public_feedback_from_report(report: dict[str, Any]) -> dict[str, Any]:
    repair_targets: list[dict[str, Any]] = []
    for relative in report.get("missing_files") or []:
        repair_targets.append({"type": "missing_file", "relative_path": relative})
    for relative, columns in (report.get("missing_columns") or {}).items():
        repair_targets.append(
            {"type": "missing_columns", "relative_path": relative, "columns": list(columns)}
        )
    for item in report.get("file_reports") or []:
        if item.get("status") != "compared":
            continue
        metrics = item.get("metrics") or {}
        if float(metrics.get("key_structure_f1", 0.0)) < 0.98:
            repair_targets.append(
                {
                    "type": "key_structure_mismatch",
                    "relative_path": item.get("relative_path", ""),
                    "key_columns": item.get("key_columns", []),
                    "reference_rows": item.get("reference_rows", 0),
                    "result_rows": item.get("result_rows", 0),
                    "f1": metrics.get("key_structure_f1", 0.0),
                    "suggestion": "检查业务 key、过滤范围、重复粒度和 join 逻辑。",
                }
            )
        if float(metrics.get("exact_row_f1", 0.0)) < 0.98:
            repair_targets.append(
                {
                    "type": "low_exact_row_f1",
                    "relative_path": item.get("relative_path", ""),
                    "f1": metrics.get("exact_row_f1", 0.0),
                    "suggestion": "在 key 对齐后检查整行序列化、派生规则和多余或缺失记录。",
                }
            )
        for column, column_report in (item.get("per_column") or {}).items():
            if float(column_report.get("f1", 0.0)) >= 0.98:
                continue
            repair_targets.append(
                {
                    "type": "low_column_f1",
                    "relative_path": item.get("relative_path", ""),
                    "column": column,
                    "precision": column_report.get("precision", 0.0),
                    "recall": column_report.get("recall", 0.0),
                    "f1": column_report.get("f1", 0.0),
                    "suggestion": "检查该列来源、格式、单位、聚合和缺失值规则。",
                }
            )
    return {
        "schema_version": 2,
        "scorer_version": report.get("scorer_version", SCORER_VERSION),
        "status": report.get("status", "SUCCESS"),
        "mode": report.get("mode", "balanced_reference_directory"),
        "metrics": report.get("metrics") or {},
        "repair_targets": repair_targets[:200],
    }


def _compare_file(
    *,
    reference_file: Path,
    result_file: Path,
    relative_path: str,
    category: str,
    weight: float,
    fallback_key_column: str,
    key_columns_override: Sequence[str] | None = None,
) -> dict[str, Any]:
    reference_columns = _table_columns(reference_file)
    result_columns = _table_columns(result_file)
    matched_columns = [column for column in reference_columns if column in result_columns]
    missing_columns = [column for column in reference_columns if column not in result_columns]
    extra_columns = [column for column in result_columns if column not in reference_columns]
    schema_matches = len(matched_columns)
    schema_precision = schema_matches / max(len(result_columns), 1)
    schema_recall = schema_matches / max(len(reference_columns), 1)
    schema_f1 = _f1(schema_matches, len(result_columns), len(reference_columns))
    key_columns = (
        [str(column) for column in key_columns_override]
        if key_columns_override is not None
        else _key_columns(relative_path, reference_columns, fallback_key_column)
    )
    missing_key_columns = [column for column in key_columns if column not in result_columns]

    if not reference_columns:
        comparison = _zero_row_comparison(reference_columns)
    elif not key_columns or missing_key_columns:
        comparison = _zero_row_comparison(reference_columns)
        comparison["reference_rows"] = _count_rows(reference_file)
        comparison["result_rows"] = _count_rows(result_file)
    elif _use_disk_assisted(reference_file, result_file):
        comparison = _compare_rows_disk_assisted(
            reference_file,
            result_file,
            reference_columns,
            result_columns,
            key_columns,
        )
    else:
        comparison = _compare_rows_in_memory(
            reference_file,
            result_file,
            reference_columns,
            result_columns,
            key_columns,
        )

    key_f1 = _f1(
        comparison["matched_key_rows"],
        comparison["result_rows"],
        comparison["reference_rows"],
    )
    per_column: dict[str, dict[str, Any]] = {}
    column_f1_values: list[float] = []
    for index, column in enumerate(reference_columns):
        matched = comparison["column_matches"][index]
        precision = matched / max(comparison["result_rows"], 1)
        recall = matched / max(comparison["reference_rows"], 1)
        f1 = _f1(matched, comparison["result_rows"], comparison["reference_rows"])
        column_f1_values.append(f1)
        per_column[column] = {
            "status": "matched" if column in result_columns else "missing_column",
            "matched_cells": matched,
            "reference_cells": comparison["reference_rows"],
            "result_cells": comparison["result_rows"],
            "precision": _rounded(precision),
            "recall": _rounded(recall),
            "f1": _rounded(f1),
        }
    cell_f1 = sum(column_f1_values) / max(len(reference_columns), 1)
    exact_f1 = _f1(
        comparison["exact_row_matches"],
        comparison["result_rows"],
        comparison["reference_rows"],
    )
    if missing_key_columns or not key_columns:
        key_f1 = 0.0
        cell_f1 = 0.0
        exact_f1 = 0.0
        for column_report in per_column.values():
            column_report.update({"matched_cells": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
    file_score = 0.10 * schema_f1 + 0.20 * key_f1 + 0.40 * cell_f1 + 0.30 * exact_f1
    return {
        "relative_path": relative_path,
        "reference_file": str(reference_file),
        "result_file": str(result_file),
        "status": "compared",
        "category": category,
        "weight": weight,
        "comparison_mode": comparison["comparison_mode"],
        "key_columns": key_columns,
        "missing_key_columns": missing_key_columns,
        "reference_columns": reference_columns,
        "result_columns": result_columns,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "reference_rows": comparison["reference_rows"],
        "result_rows": comparison["result_rows"],
        "matched_key_rows": comparison["matched_key_rows"],
        "exact_row_matches": comparison["exact_row_matches"],
        "missing_reference_rows": max(comparison["reference_rows"] - comparison["matched_key_rows"], 0),
        "extra_result_rows": max(comparison["result_rows"] - comparison["matched_key_rows"], 0),
        "schema_precision": _rounded(schema_precision),
        "schema_recall": _rounded(schema_recall),
        "per_column": per_column,
        "metrics": {
            "schema_f1": _rounded(schema_f1),
            "key_structure_f1": _rounded(key_f1),
            "row_aligned_cell_f1": _rounded(cell_f1),
            "exact_row_f1": _rounded(exact_f1),
            "file_score": _rounded(file_score),
        },
        "_raw_metrics": {
            "schema_f1": schema_f1,
            "key_structure_f1": key_f1,
            "row_aligned_cell_f1": cell_f1,
            "exact_row_f1": exact_f1,
            "file_score": file_score,
        },
    }


def _compare_rows_in_memory(
    reference_file: Path,
    result_file: Path,
    reference_columns: list[str],
    result_columns: list[str],
    key_columns: list[str],
) -> dict[str, Any]:
    reference_groups: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    result_groups: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for row in _normalized_rows(reference_file, reference_columns, reference_columns):
        reference_groups[_project_key(row, reference_columns, key_columns)].append(row)
    for row in _normalized_rows(result_file, reference_columns, result_columns):
        result_groups[_project_key(row, reference_columns, key_columns)].append(row)
    return _compare_group_maps(reference_groups, result_groups, len(reference_columns), "row_aligned")


def _compare_rows_disk_assisted(
    reference_file: Path,
    result_file: Path,
    reference_columns: list[str],
    result_columns: list[str],
    key_columns: list[str],
) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="reference-evaluation-"))
    database = temp_root / "rows.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("CREATE TABLE rows (side INTEGER NOT NULL, key TEXT NOT NULL, row TEXT NOT NULL)")
        reference_count = _insert_rows(
            connection,
            side=0,
            path=reference_file,
            reference_columns=reference_columns,
            source_columns=reference_columns,
            key_columns=key_columns,
        )
        result_count = _insert_rows(
            connection,
            side=1,
            path=result_file,
            reference_columns=reference_columns,
            source_columns=result_columns,
            key_columns=key_columns,
        )
        connection.execute("CREATE INDEX rows_key_side_row ON rows(key, side, row)")
        column_matches = [0] * len(reference_columns)
        matched_key_rows = 0
        exact_row_matches = 0
        current_key: str | None = None
        reference_rows: list[tuple[str, ...]] = []
        result_rows: list[tuple[str, ...]] = []
        cursor = connection.execute("SELECT key, side, row FROM rows ORDER BY key, side, row")
        for key_text, side, row_text in cursor:
            if current_key is not None and key_text != current_key:
                matched, exact, columns = _compare_duplicate_group(reference_rows, result_rows)
                matched_key_rows += matched
                exact_row_matches += exact
                column_matches = [left + right for left, right in zip(column_matches, columns)]
                reference_rows = []
                result_rows = []
            current_key = key_text
            values = tuple(json.loads(row_text))
            (reference_rows if side == 0 else result_rows).append(values)
        if current_key is not None:
            matched, exact, columns = _compare_duplicate_group(reference_rows, result_rows)
            matched_key_rows += matched
            exact_row_matches += exact
            column_matches = [left + right for left, right in zip(column_matches, columns)]
        return {
            "comparison_mode": "disk_assisted_row_aligned",
            "reference_rows": reference_count,
            "result_rows": result_count,
            "matched_key_rows": matched_key_rows,
            "exact_row_matches": exact_row_matches,
            "column_matches": column_matches,
        }
    finally:
        connection.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def _insert_rows(
    connection: sqlite3.Connection,
    *,
    side: int,
    path: Path,
    reference_columns: list[str],
    source_columns: list[str],
    key_columns: list[str],
) -> int:
    count = 0
    batch: list[tuple[int, str, str]] = []
    for row in _normalized_rows(path, reference_columns, source_columns):
        key = _project_key(row, reference_columns, key_columns)
        batch.append((side, json.dumps(key, ensure_ascii=True), json.dumps(row, ensure_ascii=True)))
        count += 1
        if len(batch) >= 10_000:
            connection.executemany("INSERT INTO rows(side, key, row) VALUES (?, ?, ?)", batch)
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO rows(side, key, row) VALUES (?, ?, ?)", batch)
    connection.commit()
    return count


def _compare_group_maps(
    reference_groups: dict[tuple[str, ...], list[tuple[str, ...]]],
    result_groups: dict[tuple[str, ...], list[tuple[str, ...]]],
    column_count: int,
    comparison_mode: str,
) -> dict[str, Any]:
    reference_count = sum(len(rows) for rows in reference_groups.values())
    result_count = sum(len(rows) for rows in result_groups.values())
    matched_key_rows = 0
    exact_row_matches = 0
    column_matches = [0] * column_count
    for key in sorted(set(reference_groups) | set(result_groups)):
        matched, exact, columns = _compare_duplicate_group(
            reference_groups.get(key, []),
            result_groups.get(key, []),
        )
        matched_key_rows += matched
        exact_row_matches += exact
        column_matches = [left + right for left, right in zip(column_matches, columns)]
    return {
        "comparison_mode": comparison_mode,
        "reference_rows": reference_count,
        "result_rows": result_count,
        "matched_key_rows": matched_key_rows,
        "exact_row_matches": exact_row_matches,
        "column_matches": column_matches,
    }


def _compare_duplicate_group(
    reference_rows: Sequence[tuple[str, ...]],
    result_rows: Sequence[tuple[str, ...]],
) -> tuple[int, int, list[int]]:
    column_count = len(reference_rows[0]) if reference_rows else len(result_rows[0]) if result_rows else 0
    matches = min(len(reference_rows), len(result_rows))
    column_matches = [0] * column_count
    if not matches:
        return 0, 0, column_matches
    left = sorted(reference_rows)
    right = sorted(result_rows)
    weights = [
        [sum(ref_value == result_value for ref_value, result_value in zip(ref_row, result_row)) for result_row in right]
        for ref_row in left
    ]
    assignment = _maximum_weight_assignment(weights)
    exact = 0
    for ref_index, result_index in assignment:
        ref_row = left[ref_index]
        result_row = right[result_index]
        row_matches = [left_value == right_value for left_value, right_value in zip(ref_row, result_row)]
        exact += int(all(row_matches))
        for index, matched in enumerate(row_matches):
            column_matches[index] += int(matched)
    return matches, exact, column_matches


def _maximum_weight_assignment(weights: list[list[int]]) -> list[tuple[int, int]]:
    if not weights or not weights[0]:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    transposed = row_count > column_count
    matrix = [list(row) for row in weights]
    if transposed:
        matrix = [list(row) for row in zip(*matrix)]
    rows = len(matrix)
    columns = len(matrix[0])
    max_weight = max(max(row) for row in matrix)
    costs = [[max_weight - value for value in row] for row in matrix]
    u = [0] * (rows + 1)
    v = [0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for i in range(1, rows + 1):
        p[0] = i
        minimum = [10**9] * (columns + 1)
        used = [False] * (columns + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = 10**9
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment: list[tuple[int, int]] = []
    for column in range(1, columns + 1):
        if p[column] == 0:
            continue
        row_index = p[column] - 1
        column_index = column - 1
        assignment.append(
            (column_index, row_index) if transposed else (row_index, column_index)
        )
    return sorted(assignment)


def _normalized_rows(
    path: Path,
    reference_columns: list[str],
    source_columns: list[str],
) -> Iterator[tuple[str, ...]]:
    source_set = set(source_columns)
    for row in _iter_table_dicts(path):
        yield tuple(
            _normalize_value(row.get(column, "")) if column in source_set else "__missing_column__"
            for column in reference_columns
        )


def _project_key(
    row: tuple[str, ...],
    columns: list[str],
    key_columns: list[str],
) -> tuple[str, ...]:
    positions = [columns.index(column) for column in key_columns]
    return tuple(row[position] for position in positions)


def _key_columns(relative_path: str, reference_columns: list[str], fallback: str) -> list[str]:
    lower = relative_path.casefold()
    if lower.startswith("cohort/") or lower.endswith("/labels.csv") or lower == "csv/labels.csv":
        return ["stay_id"]
    if lower == "features/preproc_chart_icu.csv":
        return ["stay_id", "itemid", "event_time_from_admit"]
    if lower == "features/preproc_diag_icu.csv":
        return ["stay_id"]
    if lower == "features/preproc_med_icu.csv":
        return ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime", "orderid"]
    if lower == "features/preproc_out_icu.csv":
        return ["subject_id", "hadm_id", "stay_id", "itemid", "charttime"]
    if lower == "features/preproc_proc_icu.csv":
        return ["subject_id", "hadm_id", "stay_id", "itemid", "starttime"]
    if lower.startswith("summary/diag_"):
        return ["new_icd_code"]
    if lower.startswith("summary/"):
        return ["itemid"]
    if fallback and fallback in reference_columns:
        return [fallback]
    for candidate in ("stay_id", "hadm_id", "subject_id", "itemid", "new_icd_code"):
        if candidate in reference_columns:
            return [candidate]
    return []


def _file_weight(relative_path: str) -> tuple[str, float]:
    lower = relative_path.casefold()
    if lower.startswith("cohort/") and lower.endswith((".csv", ".csv.gz")):
        return "core", 0.15
    if lower == "csv/labels.csv" or lower.endswith("/labels.csv"):
        return "core", 0.05
    if lower in FEATURE_FILES:
        return "feature", 0.12
    if lower in SUMMARY_FILES:
        return "summary", 0.02
    return "unweighted", 0.0


def _missing_file_report(
    relative_path: str,
    reference_file: Path,
    result_file: Path,
    category: str,
    weight: float,
    key_columns_override: Sequence[str] | None = None,
) -> dict[str, Any]:
    columns = _table_columns(reference_file)
    key_columns = (
        [str(column) for column in key_columns_override]
        if key_columns_override is not None
        else _key_columns(relative_path, columns, "")
    )
    return {
        "relative_path": relative_path,
        "reference_file": str(reference_file),
        "result_file": str(result_file),
        "status": "missing_file",
        "category": category,
        "weight": weight,
        "key_columns": key_columns,
        "missing_key_columns": key_columns,
        "missing_columns": columns,
        "extra_columns": [],
        "reference_rows": _count_rows(reference_file),
        "result_rows": 0,
        "matched_key_rows": 0,
        "exact_row_matches": 0,
        "per_column": {},
        "metrics": {
            "schema_f1": 0.0,
            "key_structure_f1": 0.0,
            "row_aligned_cell_f1": 0.0,
            "exact_row_f1": 0.0,
            "file_score": 0.0,
        },
    }


def _zero_row_comparison(columns: list[str]) -> dict[str, Any]:
    return {
        "comparison_mode": "missing_required_key",
        "reference_rows": 0,
        "result_rows": 0,
        "matched_key_rows": 0,
        "exact_row_matches": 0,
        "column_matches": [0] * len(columns),
    }


def _use_disk_assisted(reference_file: Path, result_file: Path) -> bool:
    try:
        return reference_file.stat().st_size + result_file.stat().st_size > 128 * 1024 * 1024
    except OSError:
        return False


def _table_columns(path: Path) -> list[str]:
    name = path.name.casefold()
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        with _open_text_table(path) as handle:
            delimiter = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
            reader = csv.reader(handle, delimiter=delimiter)
            return [str(column) for column in next(reader, [])]
    frame = _read_table(path, nrows=0)
    return [str(column) for column in frame.columns]


def _count_rows(path: Path) -> int:
    return sum(1 for _ in _iter_table_dicts(path))


def _iter_table_dicts(path: Path) -> Iterator[dict[str, Any]]:
    name = path.name.casefold()
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        with _open_text_table(path) as handle:
            delimiter = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                if None in row:
                    raise ValueError(
                        f"row {reader.line_num} contains extra CSV cells beyond the declared header"
                    )
                yield {str(key): value for key, value in row.items() if key is not None}
        return
    frame = _read_table(path, nrows=None)
    frame.columns = [str(column) for column in frame.columns]
    yield from frame.to_dict(orient="records")


def _open_text_table(path: Path):
    if path.name.casefold().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_table(path: Path, *, nrows: int | None) -> pd.DataFrame:
    lower = path.name.casefold()
    if lower.endswith(".parquet"):
        frame = pd.read_parquet(path)
        return frame.head(nrows) if nrows is not None else frame
    if lower.endswith(".jsonl"):
        frame = pd.read_json(path, lines=True)
        return frame.head(nrows) if nrows is not None else frame
    if lower.endswith(".json"):
        frame = pd.read_json(path)
        return frame.head(nrows) if nrows is not None else frame
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def _structured_package_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in IGNORED_PACKAGE_FILES and path.name.casefold().endswith(STRUCTURED_SUFFIXES)
    )


def _normalize_value(value: Any) -> str:
    if _is_empty(value):
        return ""
    text = str(value).strip()
    duration = _duration_text_to_hours(text)
    if duration is not None:
        return _normalize_decimal(duration)
    try:
        numeric = Decimal(text)
        if numeric.is_nan():
            return ""
        return _normalize_decimal(numeric)
    except (InvalidOperation, ValueError):
        return " ".join(text.casefold().split())


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.casefold() in {"nan", "nat", "none", "null"}


def _normalize_decimal(value: Decimal) -> str:
    normalized_decimal = value.quantize(NUMERIC_NORMALIZATION_PLACES, rounding=ROUND_HALF_UP)
    normalized = format(normalized_decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _duration_text_to_hours(text: str) -> Decimal | None:
    import re

    match = re.fullmatch(
        r"(?:(?P<days>-?\d+)\s+days?\s+)?(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2}(?:\.\d+)?)",
        text.strip().casefold(),
    )
    if not match:
        return None
    days = Decimal(match.group("days") or "0")
    hours = Decimal(match.group("hours"))
    minutes = Decimal(match.group("minutes"))
    seconds = Decimal(match.group("seconds"))
    sign = Decimal("-1") if days < 0 else Decimal("1")
    return days * Decimal(24) + sign * (hours + minutes / Decimal(60) + seconds / Decimal(3600))


def _f1(matches: int | float, predicted: int | float, expected: int | float) -> float:
    if predicted == 0 and expected == 0:
        return 1.0
    precision = float(matches) / max(float(predicted), 1.0)
    recall = float(matches) / max(float(expected), 1.0)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _empty_report(issue: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scorer_version": SCORER_VERSION,
        "status": "NEEDS_REPAIR",
        "mode": "balanced_reference_directory",
        "metrics": {
            "core_score": 0.0,
            "feature_score": 0.0,
            "summary_score": 0.0,
            "schema_f1": 0.0,
            "key_structure_f1": 0.0,
            "row_aligned_cell_f1": 0.0,
            "exact_row_f1": 0.0,
            "file_coverage": 0.0,
            "composite_score": 0.0,
        },
        "missing_files": [],
        "missing_columns": {},
        "issues": [issue],
        "file_reports": [],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
