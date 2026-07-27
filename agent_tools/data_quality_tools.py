from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from agentscope.tool import ToolResponse

from .context import EngineerToolContext
from .tools import _load_json_string_list, _load_structured_frame, _response


QUALITY_TOOL_NAMES = {
    "ProfileDataQuality",
    "ProfileByGroup",
    "TestDataConstraint",
    "AuditRepairDelta",
}
SUPPORTED_CONSTRAINT_TYPES = {
    "not_null",
    "unique",
    "allowed_values",
    "range",
    "regex",
    "functional_dependency",
    "foreign_key",
    "temporal_order",
}


def _python_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    return [
        {str(column): _python_value(value) for column, value in row.items()}
        for row in frame.head(limit).to_dict(orient="records")
    ]


def _missing_mask(series: pd.Series) -> pd.Series:
    missing = series.isna()
    if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
        missing = missing | series.fillna("").astype(str).str.strip().eq("")
    return missing


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str = "columns") -> list[str]:
    values = [str(column) for column in columns]
    if not values:
        raise ValueError(f"{label} must not be empty")
    missing = [column for column in values if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} are missing: {', '.join(missing)}")
    return values


def _logical_type(series: pd.Series) -> tuple[str, dict[str, float]]:
    values = series[~_missing_mask(series)]
    if values.empty:
        return "empty", {"numeric": 0.0, "datetime": 0.0, "boolean": 0.0}
    text = values.astype(str).str.strip()
    numeric_ratio = float(pd.to_numeric(text, errors="coerce").notna().mean())
    datetime_ratio = 0.0
    if text.str.contains(r"[-/:T]", regex=True).any():
        datetime_ratio = float(
            pd.to_datetime(text, errors="coerce", format="mixed").notna().mean()
        )
    boolean_ratio = float(text.str.casefold().isin({"true", "false", "0", "1", "yes", "no"}).mean())
    if numeric_ratio >= 0.95:
        logical = "numeric"
    elif datetime_ratio >= 0.95:
        logical = "datetime"
    elif boolean_ratio == 1.0:
        logical = "boolean"
    else:
        logical = "string"
    return logical, {
        "numeric": round(numeric_ratio, 6),
        "datetime": round(datetime_ratio, 6),
        "boolean": round(boolean_ratio, 6),
    }


def _format_pattern(value: str) -> str:
    output: list[str] = []
    for character in value[:200]:
        if character.isupper():
            token = "A"
        elif character.islower():
            token = "a"
        elif character.isdigit():
            token = "9"
        elif character.isspace():
            token = "_"
        else:
            token = character
        if not output or output[-1] != token or token not in {"A", "a", "9", "_"}:
            output.append(token)
    return "".join(output)


def _column_profile(series: pd.Series, *, top_k: int) -> dict[str, Any]:
    missing = _missing_mask(series)
    present = series[~missing]
    logical_type, parse_rates = _logical_type(series)
    counts = present.astype(str).value_counts(dropna=False).head(top_k)
    profile: dict[str, Any] = {
        "physical_dtype": str(series.dtype),
        "logical_type": logical_type,
        "parse_rates": parse_rates,
        "missing_count": int(missing.sum()),
        "missing_rate": round(float(missing.mean()) if len(series) else 0.0, 6),
        "distinct_count": int(present.astype(str).nunique(dropna=True)),
        "top_values": [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ],
    }
    if logical_type == "numeric" and not present.empty:
        numeric = pd.to_numeric(present, errors="coerce").dropna()
        quantiles = numeric.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        profile["numeric_summary"] = {
            "min": _python_value(quantiles.loc[0.0]),
            "q25": _python_value(quantiles.loc[0.25]),
            "median": _python_value(quantiles.loc[0.5]),
            "q75": _python_value(quantiles.loc[0.75]),
            "max": _python_value(quantiles.loc[1.0]),
        }
    elif logical_type == "string" and not present.empty:
        text = present.astype(str)
        lengths = text.str.len()
        patterns = text.map(_format_pattern).value_counts().head(top_k)
        profile["text_length"] = {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "max": int(lengths.max()),
        }
        profile["format_patterns"] = [
            {"pattern": str(pattern), "count": int(count)}
            for pattern, count in patterns.items()
        ]
    return profile


def _parse_json_object(value: str, label: str) -> dict[str, Any]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _load_frame(path: Path) -> pd.DataFrame:
    frame, _ = _load_structured_frame(path)
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _constraint_samples(frame: pd.DataFrame, mask: pd.Series, limit: int) -> list[dict[str, Any]]:
    return _records(frame.loc[mask], limit)


def _key_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].map(
            lambda value: "__NULL__" if _python_value(value) is None else str(_python_value(value))
        )
    return normalized.apply(lambda row: tuple(row.tolist()), axis=1)


def _row_signature(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple("__NULL__" if _python_value(row[column]) is None else str(_python_value(row[column])) for column in columns)


class DataQualityEvidenceTools:
    """Evidence-only data quality tools for the correction Agent."""

    def __init__(self, context: EngineerToolContext) -> None:
        self.context = context

    def ProfileDataQuality(
        self,
        file_path: str,
        columns_json: str = "[]",
        top_k: int = 10,
    ) -> ToolResponse:
        try:
            if top_k < 1 or top_k > 50:
                raise ValueError("top_k must be between 1 and 50")
            path = self.context.resolve_read_path(file_path)
            if not path.is_file():
                raise ValueError(f"file_path is not a file: {path}")
            frame = _load_frame(path)
            requested = _load_json_string_list(columns_json)
            columns = requested or list(frame.columns)
            _require_columns(frame, columns)
            self.context.mark_read(path)
            return _response(
                "SUCCESS",
                f"Profiled {path.name} without modifying it.",
                {
                    "file_path": str(path),
                    "row_count": int(len(frame)),
                    "column_count": int(len(frame.columns)),
                    "profiled_columns": columns,
                    "duplicate_row_count": int(frame.duplicated().sum()),
                    "columns": {
                        column: _column_profile(frame[column], top_k=top_k)
                        for column in columns
                    },
                    "interpretation": "Statistical evidence only; anomalies are not proven errors.",
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Data quality profiling failed.", issues=[str(exc)])

    def ProfileByGroup(
        self,
        file_path: str,
        group_columns_json: str,
        value_columns_json: str = "[]",
        max_groups: int = 100,
    ) -> ToolResponse:
        try:
            if max_groups < 1 or max_groups > 1_000:
                raise ValueError("max_groups must be between 1 and 1000")
            path = self.context.resolve_read_path(file_path)
            if not path.is_file():
                raise ValueError(f"file_path is not a file: {path}")
            frame = _load_frame(path)
            group_columns = _require_columns(frame, _load_json_string_list(group_columns_json), "group_columns")
            requested_values = _load_json_string_list(value_columns_json)
            value_columns = requested_values or [column for column in frame.columns if column not in group_columns]
            _require_columns(frame, value_columns, "value_columns")
            grouped = list(frame.groupby(group_columns, dropna=False, sort=True))
            if len(grouped) > max_groups:
                raise ValueError(
                    f"observed {len(grouped)} groups, exceeding max_groups={max_groups}; choose a narrower scope"
                )
            reports: list[dict[str, Any]] = []
            for raw_key, group in grouped:
                values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
                key = {column: _python_value(value) for column, value in zip(group_columns, values)}
                column_reports: dict[str, Any] = {}
                for column in value_columns:
                    profile = _column_profile(group[column], top_k=5)
                    profile["inconsistent"] = profile["distinct_count"] > 1
                    column_reports[column] = profile
                reports.append(
                    {
                        "key": key,
                        "row_count": int(len(group)),
                        "columns": column_reports,
                    }
                )
            reports.sort(key=lambda item: (-item["row_count"], json.dumps(item["key"], sort_keys=True)))
            self.context.mark_read(path)
            return _response(
                "SUCCESS",
                f"Profiled {len(reports)} groups without modifying the source.",
                {
                    "file_path": str(path),
                    "group_columns": group_columns,
                    "value_columns": value_columns,
                    "group_count": len(reports),
                    "groups": reports,
                    "interpretation": "Group differences are evidence, not automatic repair decisions.",
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Group profiling failed.", issues=[str(exc)])

    def TestDataConstraint(
        self,
        constraints_json: str,
        sample_limit: int = 10,
    ) -> ToolResponse:
        try:
            if sample_limit < 0 or sample_limit > 50:
                raise ValueError("sample_limit must be between 0 and 50")
            constraints = json.loads(constraints_json or "[]")
            if not isinstance(constraints, list) or not constraints:
                raise ValueError("constraints_json must be a non-empty JSON list")
            reports: list[dict[str, Any]] = []
            cache: dict[Path, pd.DataFrame] = {}

            def load(raw_path: Any) -> tuple[Path, pd.DataFrame]:
                path = self.context.resolve_read_path(str(raw_path or ""))
                if not path.is_file():
                    raise ValueError(f"constraint file is not a file: {path}")
                if path not in cache:
                    cache[path] = _load_frame(path)
                    self.context.mark_read(path)
                return path, cache[path]

            for index, constraint in enumerate(constraints, start=1):
                if not isinstance(constraint, dict):
                    raise ValueError(f"constraint #{index} must be a JSON object")
                constraint_id = str(constraint.get("id") or f"constraint_{index:04d}")
                kind = str(constraint.get("type") or "").strip()
                if kind not in SUPPORTED_CONSTRAINT_TYPES:
                    raise ValueError(f"unsupported constraint type: {kind or '(empty)'}")
                path, frame = load(constraint.get("file_path"))
                mask = pd.Series(False, index=frame.index)

                if kind == "not_null":
                    columns = _require_columns(frame, constraint.get("columns") or [])
                    mask = pd.concat([_missing_mask(frame[column]) for column in columns], axis=1).any(axis=1)
                elif kind == "unique":
                    columns = _require_columns(frame, constraint.get("columns") or [])
                    mask = frame.duplicated(subset=columns, keep=False)
                elif kind == "allowed_values":
                    column = _require_columns(frame, [constraint.get("column")])[0]
                    allowed = {str(value) for value in constraint.get("values") or []}
                    if not allowed:
                        raise ValueError("allowed_values requires a non-empty values list")
                    present = ~_missing_mask(frame[column])
                    mask = present & ~frame[column].astype(str).isin(allowed)
                elif kind == "range":
                    column = _require_columns(frame, [constraint.get("column")])[0]
                    numeric = pd.to_numeric(frame[column], errors="coerce")
                    present = ~_missing_mask(frame[column])
                    mask = present & numeric.isna()
                    if constraint.get("min") is not None:
                        mask = mask | (numeric < float(constraint["min"]))
                    if constraint.get("max") is not None:
                        mask = mask | (numeric > float(constraint["max"]))
                elif kind == "regex":
                    column = _require_columns(frame, [constraint.get("column")])[0]
                    pattern = str(constraint.get("pattern") or "")
                    if not pattern:
                        raise ValueError("regex requires pattern")
                    compiled = re.compile(pattern)
                    present = ~_missing_mask(frame[column])
                    mask = present & ~frame[column].astype(str).map(lambda value: compiled.fullmatch(value) is not None)
                elif kind == "functional_dependency":
                    determinant = _require_columns(frame, constraint.get("determinant") or [], "determinant")
                    dependent = _require_columns(frame, constraint.get("dependent") or [], "dependent")
                    combined = determinant + dependent
                    distinct = frame[combined].drop_duplicates().groupby(determinant, dropna=False).size()
                    violating_keys = set(distinct[distinct > 1].index.tolist())
                    determinant_keys = _key_series(frame, determinant)
                    normalized_violations = {
                        key if isinstance(key, tuple) else (key,)
                        for key in violating_keys
                    }
                    mask = determinant_keys.isin(normalized_violations)
                elif kind == "foreign_key":
                    columns = _require_columns(frame, constraint.get("columns") or [])
                    reference_path, reference = load(constraint.get("reference_file_path"))
                    reference_columns = _require_columns(
                        reference,
                        constraint.get("reference_columns") or [],
                        "reference_columns",
                    )
                    if len(columns) != len(reference_columns):
                        raise ValueError("foreign_key columns and reference_columns must have equal length")
                    known = set(_key_series(reference, reference_columns).tolist())
                    local = _key_series(frame, columns)
                    null_key = pd.concat([_missing_mask(frame[column]) for column in columns], axis=1).any(axis=1)
                    mask = ~null_key & ~local.isin(known)
                    constraint = {**constraint, "reference_file_path": str(reference_path)}
                elif kind == "temporal_order":
                    start = _require_columns(frame, [constraint.get("start_column")], "start_column")[0]
                    end = _require_columns(frame, [constraint.get("end_column")], "end_column")[0]
                    start_numeric = pd.to_numeric(frame[start], errors="coerce")
                    end_numeric = pd.to_numeric(frame[end], errors="coerce")
                    if start_numeric.notna().mean() >= 0.95 and end_numeric.notna().mean() >= 0.95:
                        mask = start_numeric > end_numeric
                    else:
                        start_time = pd.to_datetime(frame[start], errors="coerce")
                        end_time = pd.to_datetime(frame[end], errors="coerce")
                        mask = start_time > end_time

                violation_count = int(mask.sum())
                reports.append(
                    {
                        "id": constraint_id,
                        "type": kind,
                        "file_path": str(path),
                        "row_count": int(len(frame)),
                        "violation_count": violation_count,
                        "violation_rate": round(violation_count / len(frame), 6) if len(frame) else 0.0,
                        "examples": _constraint_samples(frame, mask, sample_limit),
                        "passed": violation_count == 0,
                    }
                )
            return _response(
                "SUCCESS",
                f"Tested {len(reports)} candidate constraints.",
                {
                    "constraints": reports,
                    "passed_count": sum(item["passed"] for item in reports),
                    "failed_count": sum(not item["passed"] for item in reports),
                    "interpretation": "Constraint violations are evidence; candidate constraints are not proven business rules.",
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Constraint testing failed.", issues=[str(exc)])

    def AuditRepairDelta(
        self,
        dirty_path: str,
        repaired_path: str,
        key_columns_json: str = "[]",
        key_map_json: str = "{}",
        sample_limit: int = 10,
    ) -> ToolResponse:
        try:
            if sample_limit < 0 or sample_limit > 50:
                raise ValueError("sample_limit must be between 0 and 50")
            dirty = self.context.resolve_read_path(dirty_path)
            repaired = self.context.resolve_read_path(repaired_path)
            if dirty.is_file() != repaired.is_file() or dirty.is_dir() != repaired.is_dir():
                raise ValueError("dirty_path and repaired_path must both be files or both be directories")
            default_keys = _load_json_string_list(key_columns_json)
            key_map_raw = _parse_json_object(key_map_json, "key_map_json")
            key_map = {
                str(relative): [str(column) for column in columns]
                for relative, columns in key_map_raw.items()
                if isinstance(columns, list)
            }
            if dirty.is_file():
                pairs = [(dirty.name, dirty, repaired)]
            elif dirty.is_dir():
                dirty_files = _business_files(dirty)
                repaired_files = _business_files(repaired)
                relatives = sorted(set(dirty_files) | set(repaired_files))
                pairs = [
                    (relative, dirty_files.get(relative), repaired_files.get(relative))
                    for relative in relatives
                ]
            else:
                raise ValueError("dirty_path does not exist")

            reports: list[dict[str, Any]] = []
            for relative, dirty_file, repaired_file in pairs:
                keys = key_map.get(relative, default_keys)
                reports.append(
                    _audit_file_delta(
                        relative,
                        dirty_file,
                        repaired_file,
                        keys,
                        sample_limit,
                    )
                )
                if dirty_file is not None:
                    self.context.mark_read(dirty_file)
                if repaired_file is not None:
                    self.context.mark_read(repaired_file)
            totals = {
                field: sum(int(report.get(field) or 0) for report in reports)
                for field in (
                    "inserted_rows",
                    "deleted_rows",
                    "changed_rows",
                    "changed_cells",
                    "duplicate_key_groups",
                    "ambiguous_key_groups",
                )
            }
            return _response(
                "SUCCESS",
                f"Audited {len(reports)} files without judging repair correctness.",
                {
                    "dirty_path": str(dirty),
                    "repaired_path": str(repaired),
                    "files": reports,
                    "totals": totals,
                    "interpretation": "This is a before/after delta only; correctness requires host-side hidden evaluation.",
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Repair delta audit failed.", issues=[str(exc)])


def _business_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".csv") or path.name.endswith(".csv.gz"))
    }


def _audit_file_delta(
    relative: str,
    dirty_path: Path | None,
    repaired_path: Path | None,
    keys: list[str],
    sample_limit: int,
) -> dict[str, Any]:
    if dirty_path is None:
        repaired = _load_frame(repaired_path)  # type: ignore[arg-type]
        return {
            "relative_path": relative,
            "status": "inserted_file",
            "inserted_rows": int(len(repaired)),
            "deleted_rows": 0,
            "changed_rows": 0,
            "changed_cells": 0,
            "changed_by_column": {},
            "duplicate_key_groups": 0,
            "ambiguous_key_groups": 0,
            "schema_added_columns": list(repaired.columns),
            "schema_removed_columns": [],
            "examples": [],
        }
    if repaired_path is None:
        dirty = _load_frame(dirty_path)
        return {
            "relative_path": relative,
            "status": "deleted_file",
            "inserted_rows": 0,
            "deleted_rows": int(len(dirty)),
            "changed_rows": 0,
            "changed_cells": 0,
            "changed_by_column": {},
            "duplicate_key_groups": 0,
            "ambiguous_key_groups": 0,
            "schema_added_columns": [],
            "schema_removed_columns": list(dirty.columns),
            "examples": [],
        }

    dirty = _load_frame(dirty_path)
    repaired = _load_frame(repaired_path)
    dirty_columns = list(dirty.columns)
    repaired_columns = list(repaired.columns)
    common_columns = [column for column in dirty_columns if column in repaired_columns]
    added_columns = [column for column in repaired_columns if column not in dirty_columns]
    removed_columns = [column for column in dirty_columns if column not in repaired_columns]
    changed_by_column: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    inserted = deleted = changed_rows = changed_cells = duplicate_groups = ambiguous_groups = 0

    if keys:
        _require_columns(dirty, keys, "key_columns")
        _require_columns(repaired, keys, "key_columns")
        dirty_keys = _key_series(dirty, keys)
        repaired_keys = _key_series(repaired, keys)
        dirty_groups = {key: group for key, group in dirty.assign(__key=dirty_keys).groupby("__key", sort=False)}
        repaired_groups = {key: group for key, group in repaired.assign(__key=repaired_keys).groupby("__key", sort=False)}
        for key in sorted(set(dirty_groups) | set(repaired_groups), key=str):
            before = dirty_groups.get(key)
            after = repaired_groups.get(key)
            if before is None:
                inserted += len(after)
                continue
            if after is None:
                deleted += len(before)
                continue
            if len(before) == 1 and len(after) == 1:
                before_row = before.iloc[0]
                after_row = after.iloc[0]
                row_changes: dict[str, dict[str, Any]] = {}
                for column in common_columns:
                    if column in keys:
                        continue
                    old = _python_value(before_row[column])
                    new = _python_value(after_row[column])
                    if old != new:
                        row_changes[column] = {"before": old, "after": new}
                        changed_by_column[column] += 1
                if row_changes:
                    changed_rows += 1
                    changed_cells += len(row_changes)
                    if len(examples) < sample_limit:
                        examples.append(
                            {
                                "key": {column: _python_value(before_row[column]) for column in keys},
                                "changes": row_changes,
                            }
                        )
                continue

            duplicate_groups += 1
            before_counts = Counter(_row_signature(row, common_columns) for _, row in before.iterrows())
            after_counts = Counter(_row_signature(row, common_columns) for _, row in after.iterrows())
            removed = sum((before_counts - after_counts).values())
            added = sum((after_counts - before_counts).values())
            deleted += removed
            inserted += added
            if removed or added or added_columns or removed_columns:
                ambiguous_groups += 1
    else:
        before_counts = Counter(_row_signature(row, common_columns) for _, row in dirty.iterrows())
        after_counts = Counter(_row_signature(row, common_columns) for _, row in repaired.iterrows())
        deleted = sum((before_counts - after_counts).values())
        inserted = sum((after_counts - before_counts).values())

    return {
        "relative_path": relative,
        "status": "compared",
        "key_columns": keys,
        "dirty_rows": int(len(dirty)),
        "repaired_rows": int(len(repaired)),
        "inserted_rows": int(inserted),
        "deleted_rows": int(deleted),
        "changed_rows": int(changed_rows),
        "changed_cells": int(changed_cells),
        "changed_by_column": dict(sorted(changed_by_column.items())),
        "duplicate_key_groups": int(duplicate_groups),
        "ambiguous_key_groups": int(ambiguous_groups),
        "schema_added_columns": added_columns,
        "schema_removed_columns": removed_columns,
        "examples": examples,
    }
