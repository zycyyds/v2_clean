from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from workflow.reference_evaluation import (
    IGNORED_PACKAGE_FILES,
    _iter_table_dicts,
    _key_columns,
    _normalize_value,
    _structured_package_files,
    _table_columns,
)


MAX_IN_MEMORY_DUPLICATE_SCAN_BYTES = 256 * 1024 * 1024
MIMIC_EXPECTED_BUSINESS_FILE_COUNT = 17


def validate_business_result_package(
    *,
    result_package: str | Path,
    train_reference_root: str | Path,
    split_keys: str | Path,
    key_column: str,
    split_mode: str,
    required_file_count: int | None = None,
) -> dict[str, Any]:
    """Apply the shared validation/test business package gate."""
    package = Path(result_package).expanduser().resolve()
    reference = Path(train_reference_root).expanduser().resolve()
    keys_path = Path(split_keys).expanduser().resolve()
    issues: list[str] = []
    diagnostics: list[str] = []
    files: list[dict[str, Any]] = []
    expected_files = [
        path
        for path in _structured_package_files(reference)
        if path.name not in IGNORED_PACKAGE_FILES
    ]
    expected_relatives = {
        path.relative_to(reference).as_posix() for path in expected_files
    }
    result_relatives = {
        path.relative_to(package).as_posix()
        for path in _structured_package_files(package)
        if path.name not in IGNORED_PACKAGE_FILES
    } if package.is_dir() else set()
    if required_file_count is not None and len(expected_relatives) != required_file_count:
        issues.append(
            "train reference business file count mismatch: "
            f"expected={required_file_count}, actual={len(expected_relatives)}"
        )
    extra_files = sorted(result_relatives - expected_relatives)
    if extra_files:
        issues.append("unexpected business files: " + ", ".join(extra_files[:20]))
    split_key_values = _read_key_values(keys_path, key_column)

    for reference_file in expected_files:
        relative = reference_file.relative_to(reference).as_posix()
        result_file = package / relative
        file_report: dict[str, Any] = {
            "relative_path": relative,
            "passed": True,
            "issues": [],
            "diagnostics": [],
        }
        if not result_file.is_file():
            file_report["passed"] = False
            file_report["issues"].append("missing file")
        else:
            try:
                expected_columns = _table_columns(reference_file)
                result_columns = _table_columns(result_file)
                is_cohort = relative.startswith("cohort/")
                is_labels = relative.endswith("/labels.csv")
                require_unique_primary_key = is_cohort or is_labels
                if result_columns != expected_columns:
                    file_report["passed"] = False
                    file_report["issues"].append("schema differs from train reference")

                business_key_columns = _key_columns(relative, expected_columns, key_column)
                result_business_duplicates = _duplicate_business_key_count(
                    result_file,
                    business_key_columns,
                )
                reference_business_duplicates = _duplicate_business_key_count(
                    reference_file,
                    business_key_columns,
                )
                row_count, observed_keys, duplicate_keys, placeholder_only = _scan_gate_file(
                    result_file,
                    result_columns,
                    key_column=key_column,
                    require_unique_key=require_unique_primary_key,
                )
                file_report.update(
                    {
                        "row_count": row_count,
                        "observed_key_count": len(observed_keys),
                        "duplicate_key_count": duplicate_keys,
                        "business_key_columns": business_key_columns,
                        "duplicate_business_key_count": result_business_duplicates,
                        "train_duplicate_business_key_count": reference_business_duplicates,
                    }
                )
                if result_business_duplicates is None:
                    file_report["diagnostics"].append(
                        "duplicate business key scan deferred for large file; unified scorer will account for duplicates"
                    )
                elif result_business_duplicates:
                    file_report["diagnostics"].append(
                        f"contains {result_business_duplicates} duplicate business keys"
                    )
                if row_count == 0:
                    file_report["passed"] = False
                    file_report["issues"].append("empty output file")
                if placeholder_only:
                    file_report["passed"] = False
                    file_report["issues"].append("output contains only placeholder values")
                if key_column in result_columns:
                    unknown_keys = sorted(observed_keys - split_key_values)
                    if unknown_keys:
                        file_report["passed"] = False
                        file_report["issues"].append(
                            f"contains {len(unknown_keys)} unknown {split_mode} keys"
                        )
                if is_cohort and key_column in result_columns:
                    missing_keys = sorted(split_key_values - observed_keys)
                    if missing_keys:
                        file_report["passed"] = False
                        file_report["issues"].append(
                            f"missing {len(missing_keys)} {split_mode} keys"
                        )
                if require_unique_primary_key and key_column in result_columns and duplicate_keys:
                    file_report["passed"] = False
                    file_report["issues"].append(
                        f"contains {duplicate_keys} duplicate primary keys"
                    )
            except Exception as exc:
                file_report["passed"] = False
                file_report["issues"].append(f"read error: {type(exc).__name__}: {exc}")

        if not file_report["passed"]:
            issues.extend(f"{relative}: {item}" for item in file_report["issues"])
        diagnostics.extend(
            f"{relative}: {item}" for item in file_report["diagnostics"]
        )
        files.append(file_report)

    fingerprint_payload = {
        "valid": not issues,
        "expected_file_count": len(expected_files),
        "actual_file_count": len(result_relatives),
        "key_count": len(split_key_values),
        "issues": issues,
        "diagnostics": diagnostics,
        "files": files,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "valid": not issues,
        "split_mode": split_mode,
        "expected_file_count": len(expected_files),
        "actual_file_count": len(result_relatives),
        "split_key_count": len(split_key_values),
        "business_gate_fingerprint": fingerprint,
        "issues": issues,
        "diagnostics": diagnostics,
        "files": files,
    }


def _scan_gate_file(
    path: Path,
    columns: list[str],
    *,
    key_column: str,
    require_unique_key: bool,
) -> tuple[int, set[str], int, bool]:
    observed_keys: set[str] = set()
    duplicate_keys = 0
    row_count = 0
    non_placeholder_value = False
    placeholders = {"placeholder", "todo", "unknown", "n/a", "not available"}
    for row in _iter_table_dicts(path):
        row_count += 1
        if key_column in columns:
            value = str(row.get(key_column, "")).strip()
            if require_unique_key and value in observed_keys:
                duplicate_keys += 1
            observed_keys.add(value)
        for column in columns:
            if column == key_column:
                continue
            value = str(row.get(column, "")).strip().casefold()
            if value and value not in placeholders:
                non_placeholder_value = True
    placeholder_only = (
        row_count > 0
        and len(columns) > int(key_column in columns)
        and not non_placeholder_value
    )
    return row_count, observed_keys, duplicate_keys, placeholder_only


def _duplicate_business_key_count(path: Path, key_columns: list[str]) -> int | None:
    if not key_columns:
        return 0
    if path.stat().st_size > MAX_IN_MEMORY_DUPLICATE_SCAN_BYTES:
        return None
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in _iter_table_dicts(path):
        key = tuple(_normalize_value(row.get(column, "")) for column in key_columns)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def _read_key_values(path: Path, key_column: str) -> set[str]:
    if not path.is_file():
        raise ValueError(f"split keys do not exist: {path}")
    values = {
        str(row.get(key_column, "")).strip()
        for row in _iter_table_dicts(path)
        if str(row.get(key_column, "")).strip()
    }
    if not values:
        raise ValueError(f"split keys contain no {key_column} values: {path}")
    return values


def _count_file_rows(path: Path) -> int:
    return sum(1 for _ in _iter_table_dicts(path))
