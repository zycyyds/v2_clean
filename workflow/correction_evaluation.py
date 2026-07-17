from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


def evaluate_correction_package(
    *,
    dirty_root: str | Path,
    result_root: str | Path,
    clean_root: str | Path,
    modification_log: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    dirty = Path(dirty_root).expanduser().resolve()
    result = Path(result_root).expanduser().resolve()
    clean = Path(clean_root).expanduser().resolve()
    log_path = Path(modification_log).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for name, path in (("dirty", dirty), ("result", result), ("clean", clean)):
        if not path.is_dir():
            raise ValueError(f"{name} package does not exist: {path}")
    if not log_path.is_file():
        raise ValueError(f"correction modification log does not exist: {log_path}")
    output.mkdir(parents=True, exist_ok=True)

    clean_files = _table_files(clean)
    dirty_files = _table_files(dirty)
    result_files = _table_files(result)
    if clean_files != dirty_files or clean_files != result_files:
        raise ValueError("dirty, result, and clean business file lists must match")
    modifications = _read_modifications(log_path)
    database = output / ".row_multiset.sqlite"
    connection = sqlite3.connect(database)
    try:
        _initialize_database(connection)
        schemas: dict[str, dict[str, list[str]]] = defaultdict(dict)
        for source_name, root in (("clean", clean), ("dirty", dirty), ("result", result)):
            for relative in clean_files:
                fields, hashes = _iter_row_hashes(root / relative)
                schemas[relative][source_name] = fields
                _add_hashes(connection, relative, source_name, hashes)
        connection.commit()
        schema_issues = [
            relative
            for relative, values in schemas.items()
            if values["clean"] != values["dirty"] or values["clean"] != values["result"]
        ]
        if schema_issues:
            raise ValueError("result schemas differ for: " + ", ".join(schema_issues))

        error_reports = _score_modifications(connection, modifications)
        metrics = _aggregate_error_metrics(error_reports)
        preservation = _preservation_metrics(connection, error_reports)
        metrics.update(preservation)
        attempted = metrics["exact_repairs"] + metrics["incorrect_repairs"] + metrics["collateral_row_changes"]
        precision = metrics["exact_repairs"] / attempted if attempted else 0.0
        recall = metrics["exact_repair_recall"]
        metrics["correction_precision"] = precision
        metrics["correction_f1"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        report = {
            "schema_version": 1,
            "status": "SUCCESS",
            "metrics": metrics,
            "by_error_class": _group_metrics(error_reports, "error_class"),
            "by_error_subtype": _group_metrics(error_reports, "error_subtype"),
            "error_reports": error_reports,
            "files": clean_files,
        }
        report_path = output / "correction_evaluation_report.json"
        _write_json(report_path, report)
        report["evaluation_report"] = str(report_path)
        return report
    finally:
        connection.close()
        database.unlink(missing_ok=True)


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE row_counts (
            relative_path TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            clean_count INTEGER NOT NULL DEFAULT 0,
            dirty_count INTEGER NOT NULL DEFAULT 0,
            result_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (relative_path, row_hash)
        ) WITHOUT ROWID
        """
    )


def _add_hashes(
    connection: sqlite3.Connection,
    relative: str,
    source_name: str,
    hashes: Iterator[str],
) -> None:
    column = f"{source_name}_count"
    statement = (
        f"INSERT INTO row_counts(relative_path, row_hash, {column}) VALUES (?, ?, 1) "
        f"ON CONFLICT(relative_path, row_hash) DO UPDATE SET {column}={column}+1"
    )
    batch: list[tuple[str, str]] = []
    for row_hash in hashes:
        batch.append((relative, row_hash))
        if len(batch) >= 10_000:
            connection.executemany(statement, batch)
            batch.clear()
    if batch:
        connection.executemany(statement, batch)


def _score_modifications(
    connection: sqlite3.Connection,
    modifications: list[dict[str, str]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for index, row in enumerate(modifications, start=1):
        relative = row["file"]
        dirty_hash = _json_row_hash(row["dirty_row_json"])
        clean_hash = _json_row_hash(row.get("clean_row_json", "")) if row.get("clean_row_json") else ""
        dirty_counts = _counts(connection, relative, dirty_hash)
        clean_counts = _counts(connection, relative, clean_hash) if clean_hash else (0, 0, 0)
        operation = row["operation"]
        if operation == "row_insert":
            exact = dirty_counts[2] == dirty_counts[0]
            missed = dirty_counts[2] == dirty_counts[1]
        else:
            exact = dirty_counts[2] == dirty_counts[0] and clean_counts[2] == clean_counts[0]
            missed = dirty_counts[2] == dirty_counts[1] and clean_counts[2] == clean_counts[1]
        status = "exact" if exact else "missed" if missed else "incorrect"
        reports.append(
            {
                "error_id": index,
                "file": relative,
                "operation": operation,
                "error_class": row.get("error_class", ""),
                "error_subtype": row.get("error_subtype", ""),
                "status": status,
                "clean_row_hash": clean_hash,
                "dirty_row_hash": dirty_hash,
            }
        )
    return reports


def _aggregate_error_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(reports)
    exact = sum(item["status"] == "exact" for item in reports)
    missed = sum(item["status"] == "missed" for item in reports)
    incorrect = total - exact - missed
    inserted = [item for item in reports if item["operation"] == "row_insert"]
    inserted_exact = sum(item["status"] == "exact" for item in inserted)
    return {
        "ground_truth_errors": total,
        "exact_repairs": exact,
        "missed_repairs": missed,
        "incorrect_repairs": incorrect,
        "exact_repair_recall": exact / total if total else 1.0,
        "inserted_row_errors": len(inserted),
        "inserted_rows_deleted": inserted_exact,
        "inserted_row_deletion_rate": inserted_exact / len(inserted) if inserted else 1.0,
    }


def _preservation_metrics(
    connection: sqlite3.Connection,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    affected_clean = {
        (item["file"], item["clean_row_hash"])
        for item in reports
        if item["clean_row_hash"]
    }
    clean_total = 0
    clean_matched = 0
    missing_unaffected = 0
    invented = 0
    for relative, row_hash, clean_count, dirty_count, result_count in connection.execute(
        "SELECT relative_path, row_hash, clean_count, dirty_count, result_count FROM row_counts"
    ):
        if (relative, row_hash) not in affected_clean:
            clean_total += clean_count
            clean_matched += min(clean_count, result_count)
            missing_unaffected += max(clean_count - result_count, 0)
        invented += max(result_count - max(clean_count, dirty_count), 0)
    return {
        "clean_rows_outside_changed_cells": clean_total,
        "clean_rows_preserved": clean_matched,
        "clean_preservation": clean_matched / clean_total if clean_total else 1.0,
        "missing_unaffected_rows": missing_unaffected,
        "invented_rows": invented,
        "collateral_row_changes": missing_unaffected + invented,
    }


def _group_metrics(reports: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reports:
        groups[str(item.get(field) or "unknown")].append(item)
    return {name: _aggregate_error_metrics(items) for name, items in sorted(groups.items())}


def _counts(connection: sqlite3.Connection, relative: str, row_hash: str) -> tuple[int, int, int]:
    row = connection.execute(
        "SELECT clean_count, dirty_count, result_count FROM row_counts WHERE relative_path=? AND row_hash=?",
        (relative, row_hash),
    ).fetchone()
    return tuple(row) if row else (0, 0, 0)


def _table_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and (path.suffix.casefold() == ".csv" or path.name.endswith(".csv.gz"))
    )


def _iter_row_hashes(path: Path) -> tuple[list[str], Iterator[str]]:
    @contextmanager
    def reader_context() -> Iterator[TextIO]:
        if path.name.endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
                yield handle
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                yield handle

    context = reader_context()
    handle = context.__enter__()
    reader = csv.DictReader(handle)
    fields = list(reader.fieldnames or [])

    def hashes() -> Iterator[str]:
        try:
            for row in reader:
                yield _row_hash(dict(row))
        finally:
            context.__exit__(None, None, None)

    return fields, hashes()


def _row_hash(row: dict[str, str]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_row_hash(payload: str) -> str:
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("row JSON must contain an object")
    return _row_hash({str(key): str(value) for key, value in loaded.items()})


def _read_modifications(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"operation", "file", "error_class", "error_subtype", "dirty_row_json"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("correction modification log is missing required fields")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
