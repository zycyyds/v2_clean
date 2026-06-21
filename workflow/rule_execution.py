from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}
EXECUTABLE_STATUSES = {"supported", "active", "frozen"}
NULL_MARKERS = {"", "na", "n/a", "null", "none", "nan"}


def extract_structured_rule_values(
    rules_path: str | Path,
    input_root: str | Path,
    output_dir: str | Path,
    *,
    rule_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Execute dataset-agnostic direct structured rules into normalized field values."""
    rules_payload = _read_json_object(Path(rules_path))
    rules = rules_payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError("field_extraction_rules.json must contain a rules list")
    selected = set(rule_ids or [])
    root = Path(input_root).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Input root does not exist: {root}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_grain = str(rules_payload.get("record_grain") or "case_id")
    values: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        field_id = str(rule.get("target_field_id") or "")
        if selected and field_id not in selected:
            continue
        reason = _rule_support_reason(rule)
        if reason:
            unsupported.append(_unsupported(rule, reason))
            continue
        source_file = str(rule["source_files"][0])
        source_column = str(rule["source_columns"][0])
        source_paths = _find_source_paths(root, source_file)
        if not source_paths:
            unsupported.append(_unsupported(rule, f"source file not found: {source_file}"))
            continue
        occurrence = 0
        extracted = 0
        for source_path in source_paths:
            frame = _read_frame(source_path)
            if source_column not in frame.columns:
                unsupported.append(_unsupported(rule, f"source column not found: {source_column}"))
                continue
            case_column = _case_column(rule, frame, record_grain)
            fallback_case = _case_id_from_path(root, source_path)
            for _, row in frame.iterrows():
                raw_value = row.get(source_column)
                if _is_missing(raw_value):
                    continue
                case_id = row.get(case_column) if case_column else fallback_case
                if _is_missing(case_id):
                    continue
                occurrence += 1
                extracted += 1
                values.append(
                    {
                        "case_id": str(case_id).strip(),
                        "target_field_id": field_id,
                        "target_field_path": str(rule.get("target_field_path") or field_id),
                        "occurrence_id": f"occurrence_{occurrence:06d}",
                        "value": _json_value(raw_value),
                        "source_artifact": str(source_path),
                        "source_column": source_column,
                    }
                )
        if extracted == 0 and not any(item["target_field_id"] == field_id for item in unsupported):
            unsupported.append(_unsupported(rule, "no non-empty values extracted"))

    values_path = output / "field_values.jsonl"
    unsupported_path = output / "unsupported_rules.json"
    _write_jsonl(values_path, values)
    _write_json(unsupported_path, {"unsupported": unsupported})
    return {
        "field_values": str(values_path),
        "unsupported": str(unsupported_path),
        "value_count": len(values),
        "unsupported_rule_count": len(unsupported),
    }


def join_source_records(
    sources: list[dict[str, Any]],
    join_keys: list[str],
    output_path: str | Path,
    *,
    how: str = "left",
) -> str:
    if len(sources) < 2:
        raise ValueError("At least two source artifacts are required")
    if not join_keys:
        raise ValueError("join_keys cannot be empty")
    if how not in {"left", "inner", "outer"}:
        raise ValueError("how must be left, inner, or outer")
    frames: list[pd.DataFrame] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not source.get("path"):
            raise ValueError(f"sources[{index}] requires path")
        frame = _read_frame(Path(str(source["path"])), sheet_name=str(source.get("sheet_name") or ""))
        missing = [key for key in join_keys if key not in frame.columns]
        if missing:
            raise ValueError(f"Source {index} is missing join keys: {missing}")
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=join_keys, how=how, validate=None, suffixes=("", "_right"))
    target = Path(output_path).expanduser().resolve()
    _write_frame(merged, target)
    return str(target)


def aggregate_rule_values(
    input_path: str | Path,
    output_path: str | Path,
    *,
    group_keys: list[str],
    aggregations: dict[str, list[str]],
) -> str:
    """Aggregate repeated source records with an explicit, dataset-agnostic contract."""
    if not group_keys:
        raise ValueError("group_keys cannot be empty")
    if not aggregations:
        raise ValueError("aggregations cannot be empty")
    frame = _read_frame(Path(input_path))
    missing = [column for column in [*group_keys, *aggregations] if column not in frame.columns]
    if missing:
        raise ValueError(f"Aggregation references missing columns: {missing}")
    allowed = {"count", "mean", "min", "max", "sum", "first", "last", "list"}
    output_rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_keys[0] if len(group_keys) == 1 else group_keys
    for group_value, group in frame.groupby(grouper, dropna=False, sort=True):
        values = (group_value,) if len(group_keys) == 1 else tuple(group_value)
        row = {key: _json_value(value) for key, value in zip(group_keys, values)}
        for column, operations in aggregations.items():
            series = group[column].dropna()
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            for operation in operations:
                if operation not in allowed:
                    raise ValueError(f"Unsupported aggregation operation: {operation}")
                target = f"{column}_{operation}"
                if operation == "count":
                    row[target] = int(series.count())
                elif operation == "list":
                    list_values = numeric if len(numeric) == len(series) else series
                    row[target] = json.dumps([_json_value(value) for value in list_values], ensure_ascii=False)
                elif operation in {"mean", "min", "max", "sum"}:
                    row[target] = None if numeric.empty else _json_value(getattr(numeric, operation)())
                elif operation == "first":
                    row[target] = None if series.empty else _json_value(series.iloc[0])
                elif operation == "last":
                    row[target] = None if series.empty else _json_value(series.iloc[-1])
        output_rows.append(row)
    target = Path(output_path).expanduser().resolve()
    _write_frame(pd.DataFrame(output_rows), target)
    return str(target)


def derive_target_fields(
    input_path: str | Path,
    derivations: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    frame = _read_frame(Path(input_path))
    for index, spec in enumerate(derivations):
        target = str(spec.get("target") or "").strip()
        operation = str(spec.get("operation") or "").strip()
        columns = [str(value) for value in spec.get("columns", [])]
        if not target or not operation or not columns:
            raise ValueError(f"derivations[{index}] requires target, operation, and columns")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"derivations[{index}] references missing columns: {missing}")
        if operation == "coalesce":
            frame[target] = frame[columns].bfill(axis=1).iloc[:, 0]
        elif operation == "concat":
            separator = str(spec.get("separator", " "))
            frame[target] = frame[columns].fillna("").astype(str).agg(separator.join, axis=1)
        elif operation in {"sum", "mean", "min", "max"}:
            numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
            frame[target] = getattr(numeric, operation)(axis=1)
        elif operation == "count_non_null":
            frame[target] = frame[columns].notna().sum(axis=1)
        elif operation == "copy":
            frame[target] = frame[columns[0]]
        elif operation == "datetime_difference_days":
            start = pd.to_datetime(frame[columns[0]], errors="coerce")
            end = pd.to_datetime(frame[columns[1]], errors="coerce")
            frame[target] = (end - start).dt.total_seconds() / 86400.0
        elif operation == "adjusted_age":
            anchor_age = pd.to_numeric(frame[columns[0]], errors="coerce")
            anchor_year = pd.to_numeric(frame[columns[1]], errors="coerce")
            event_year = pd.to_datetime(frame[columns[2]], errors="coerce").dt.year
            frame[target] = anchor_age + event_year - anchor_year
        else:
            raise ValueError(f"Unsupported reversible derivation operation: {operation}")
    target_path = Path(output_path).expanduser().resolve()
    _write_frame(frame, target_path)
    return str(target_path)


def normalize_reversible_table(
    input_path: str | Path,
    output_path: str | Path,
    *,
    schema: dict[str, str] | None = None,
) -> str:
    """Normalize strings/null markers/types without imputing or changing clinical values."""
    frame = _read_frame(Path(input_path))
    schema = schema or {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            stripped = series.map(lambda value: value.strip() if isinstance(value, str) else value)
            frame[column] = stripped.map(
                lambda value: pd.NA
                if isinstance(value, str) and value.casefold() in NULL_MARKERS
                else value
            )
    for column, target_type in schema.items():
        if column not in frame.columns:
            raise ValueError(f"Schema references missing column: {column}")
        kind = str(target_type).casefold()
        if kind in {"number", "numeric", "float", "integer", "int"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        elif kind in {"date", "datetime", "timestamp"}:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            frame[column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S").where(parsed.notna(), pd.NA)
        elif kind in {"string", "text"}:
            frame[column] = frame[column].astype("string")
        elif kind in {"boolean", "bool"}:
            frame[column] = frame[column].astype("boolean")
        else:
            raise ValueError(f"Unsupported normalization type for {column}: {target_type}")
    target = Path(output_path).expanduser().resolve()
    _write_frame(frame, target)
    return str(target)


def _rule_support_reason(rule: dict[str, Any]) -> str:
    if rule.get("status") not in EXECUTABLE_STATUSES:
        return f"rule status is {rule.get('status') or 'missing'}"
    if rule.get("capability_type") != "structured_extract":
        return f"capability requires another Skill: {rule.get('capability_type')}"
    if (rule.get("derivation_logic") or {}).get("operation", "direct") != "direct":
        return "derivation is not direct"
    if len(rule.get("source_files") or []) != 1 or len(rule.get("source_columns") or []) != 1:
        return "direct extraction requires exactly one source file and one source column"
    return ""


def _find_source_paths(root: Path, source_file: str) -> list[Path]:
    if root.is_file():
        return [root] if root.name == Path(source_file).name else []
    direct = root / source_file
    if direct.is_file():
        return [direct.resolve()]
    normalized = Path(source_file).as_posix().lstrip("./")
    matches = [
        path.resolve()
        for path in root.rglob(Path(source_file).name)
        if path.is_file() and path.as_posix().endswith(normalized)
    ]
    return sorted(set(matches))


def _case_column(rule: dict[str, Any], frame: pd.DataFrame, record_grain: str) -> str:
    candidates = [record_grain, "case_id"]
    return next((str(column) for column in candidates if str(column) in frame.columns), "")


def _case_id_from_path(root: Path, path: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return path.parent.name
    return parts[0] if len(parts) > 1 else ""


def _read_frame(path: Path, *, sheet_name: str = "") -> pd.DataFrame:
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix not in TABULAR_SUFFIXES:
        raise ValueError(f"Unsupported tabular format: {path}")
    if suffix == ".csv":
        return pd.read_csv(path, dtype=object)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=object)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0, dtype=object)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    return pd.read_json(path, dtype=False)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        frame.to_excel(path, index=False)
    elif suffix == ".jsonl":
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _unsupported(rule: dict[str, Any], reason: str) -> dict[str, str]:
    return {
        "target_field_id": str(rule.get("target_field_id") or ""),
        "target_field_path": str(rule.get("target_field_path") or ""),
        "reason": reason,
    }


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
