from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


class CleanRawBuildError(ValueError):
    pass


def _csv_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv") if not any(part.startswith(".") for part in path.relative_to(root).parts))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_log(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_error_id", "operation", "raw_file", "raw_row_index", "column", "clean_value", "dirty_value"}
    if not rows or not required.issubset(rows[0]):
        raise CleanRawBuildError(f"injection log is missing required columns: {sorted(required)}")
    return rows


def _source_row_index(record: dict[str, str]) -> int:
    try:
        index = int(str(record.get("raw_row_index") or ""))
    except ValueError as exc:
        raise CleanRawBuildError(f"invalid raw_row_index: {record}") from exc
    if index < 0:
        raise CleanRawBuildError(f"raw_row_index must be non-negative: {record}")
    return index


def _rewrite_table(source: Path, destination: Path, records: list[dict[str, str]]) -> dict[str, int]:
    updates: dict[int, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    inserted_rows: set[int] = set()
    for log_order, record in enumerate(records):
        row_index = _source_row_index(record)
        operation = str(record.get("operation") or "")
        if operation == "cell_update":
            updates[row_index].append((log_order, record))
        elif operation == "row_insert":
            inserted_rows.add(row_index)
        else:
            raise CleanRawBuildError(f"unsupported operation in {source}: {operation}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.reconstructing")
    restored_cells = 0
    removed_rows = 0
    seen_rows: set[int] = set()
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as source_handle, temporary.open("w", encoding="utf-8", newline="") as destination_handle:
            reader = csv.DictReader(source_handle)
            fields = list(reader.fieldnames or [])
            if not fields:
                raise CleanRawBuildError(f"CSV has no header: {source}")
            writer = csv.DictWriter(destination_handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row_index, row in enumerate(reader):
                seen_rows.add(row_index)
                if row_index in inserted_rows:
                    removed_rows += 1
                    continue
                # Later injections overwrite earlier injections. Undo them in
                # reverse log order so the original clean value is restored.
                for _, record in sorted(updates.get(row_index, []), key=lambda item: item[0], reverse=True):
                    column = str(record.get("column") or "")
                    if column not in fields:
                        raise CleanRawBuildError(f"logged column is absent from {source}: {column}")
                    expected_dirty = str(record.get("dirty_value") or "")
                    observed = str(row.get(column) or "")
                    if observed != expected_dirty:
                        raise CleanRawBuildError(
                            f"dirty value mismatch at {source}:{row_index}:{column} "
                            f"for source_error_id={record.get('source_error_id')}"
                        )
                    row[column] = str(record.get("clean_value") or "")
                    restored_cells += 1
                writer.writerow(row)
        missing_insertions = inserted_rows - seen_rows
        missing_updates = set(updates) - seen_rows
        if missing_insertions or missing_updates:
            raise CleanRawBuildError(f"log points outside {source}: inserts={sorted(missing_insertions)[:5]} updates={sorted(missing_updates)[:5]}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"restored_cells": restored_cells, "removed_rows": removed_rows}


def reconstruct_clean_raw(dirty_raw: str | Path, injection_log: str | Path, output_raw: str | Path) -> dict[str, Any]:
    """Reconstruct a clean raw split without modifying dirty raw or private Gold."""
    dirty_raw = Path(dirty_raw).expanduser().resolve()
    injection_log = Path(injection_log).expanduser().resolve()
    output_raw = Path(output_raw).expanduser().resolve()
    if not dirty_raw.is_dir():
        raise CleanRawBuildError(f"dirty raw does not exist: {dirty_raw}")
    if not injection_log.is_file():
        raise CleanRawBuildError(f"injection log does not exist: {injection_log}")
    if output_raw == dirty_raw or dirty_raw in output_raw.parents:
        raise CleanRawBuildError("output_raw must be outside dirty_raw")
    if output_raw.exists() and any(output_raw.iterdir()):
        raise CleanRawBuildError(f"output_raw must be new or empty: {output_raw}")

    log_rows = _read_log(injection_log)
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in log_rows:
        by_file[str(record.get("raw_file") or "").strip()].append(record)

    output_raw.mkdir(parents=True, exist_ok=True)
    stats = {"restored_cells": 0, "removed_rows": 0, "copied_files": 0, "rewritten_files": 0}
    for source in _csv_files(dirty_raw):
        relative = source.relative_to(dirty_raw).as_posix()
        destination = output_raw / relative
        records = by_file.get(relative, [])
        if records:
            result = _rewrite_table(source, destination, records)
            stats["restored_cells"] += result["restored_cells"]
            stats["removed_rows"] += result["removed_rows"]
            stats["rewritten_files"] += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            stats["copied_files"] += 1

    unknown_files = sorted(set(by_file) - {path.relative_to(dirty_raw).as_posix() for path in _csv_files(dirty_raw)})
    if unknown_files:
        raise CleanRawBuildError(f"injection log references missing raw files: {unknown_files}")

    manifest = {
        "schema_version": 1,
        "status": "SUCCESS",
        "workflow": "reconstruct_clean_raw",
        "input": {"dirty_raw_name": dirty_raw.name, "injection_record_count": len(log_rows)},
        "output": {"clean_raw_name": output_raw.name},
        **stats,
        "source_files": len(_csv_files(dirty_raw)),
    }
    (output_raw / "clean_raw_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
