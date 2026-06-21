from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


CONTROL_SHEETS = ("_cases", "_provenance", "_unsupported")
_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")


def export_gold_workbook(
    output_dir: str | Path,
    *,
    cases: Sequence[str | Mapping[str, Any]],
    category_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    field_values: Sequence[Mapping[str, Any]] | None = None,
    provenance: Sequence[Mapping[str, Any]] | None = None,
    unsupported: Sequence[Mapping[str, Any]] | None = None,
    field_mappings: Sequence[Mapping[str, Any]] | None = None,
    target_categories: Sequence[str] | None = None,
    filename: str = "final_dataset.xlsx",
) -> dict[str, str]:
    """Export a dataset-agnostic, category-oriented workbook result package."""
    if category_rows is not None and field_values is not None:
        raise ValueError("Provide category_rows or field_values, not both")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    workbook_path = output / filename
    csv_path = output / "final_dataset.csv"

    case_frame = _case_frame(cases)
    if field_values is not None:
        categories, mapping_specs = _categories_from_field_values(field_values)
        case_csv_frame = _case_level_frame(case_frame, field_values)
    else:
        categories = OrderedDict(
            (str(category), [dict(row) for row in rows])
            for category, rows in (category_rows or {}).items()
        )
        mapping_specs = [dict(item) for item in (field_mappings or [])]
        case_csv_frame = case_frame.copy()

    if target_categories:
        case_rows = [{"case_id": value} for value in case_frame["case_id"].astype(str).tolist()]
        ordered: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for category in target_categories:
            name = str(category).strip()
            if name and name not in ordered:
                ordered[name] = categories.get(name, [dict(row) for row in case_rows])
        for category, rows in categories.items():
            ordered.setdefault(category, rows)
        categories = ordered

    used_names = {name.casefold() for name in CONTROL_SHEETS}
    category_sheets: OrderedDict[str, str] = OrderedDict()
    for category in categories:
        category_sheets[category] = _unique_sheet_name(category, used_names)

    frames: OrderedDict[str, pd.DataFrame] = OrderedDict(
        [
            ("_cases", case_frame),
            ("_provenance", _metadata_frame(provenance, ("target_field_id", "skill"))),
            ("_unsupported", _metadata_frame(unsupported, ("target_field_id", "reason"))),
        ]
    )
    for category, rows in categories.items():
        frames[category_sheets[category]] = _category_frame(rows)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    case_csv_frame.to_csv(csv_path, index=False)

    manifest_items: list[dict[str, Any]] = [
        {
            "alias": "case_level_csv",
            "path": str(csv_path),
            "sheet_name": "",
            "case_id_column": "case_id",
            "occurrence_id_column": "",
            "grain": "case",
            "row_count": int(len(case_csv_frame)),
        }
    ]
    alias_by_sheet: dict[str, str] = {}
    for index, (sheet_name, frame) in enumerate(frames.items()):
        alias = f"sheet_{index:03d}"
        alias_by_sheet[sheet_name] = alias
        has_case = "case_id" in frame.columns
        has_occurrence = "occurrence_id" in frame.columns
        manifest_items.append(
            {
                "alias": alias,
                "path": str(workbook_path),
                "sheet_name": sheet_name,
                "case_id_column": "case_id" if has_case else "",
                "occurrence_id_column": "occurrence_id" if has_occurrence else "",
                "grain": "occurrence" if has_occurrence else ("case" if has_case else "metadata"),
                "row_count": int(len(frame)),
            }
        )

    mappings = _resolve_mappings(mapping_specs, category_sheets, alias_by_sheet, frames)
    manifest_path = output / "result_manifest.json"
    mapping_path = output / "target_field_mapping.json"
    _write_json(manifest_path, {"artifacts": manifest_items})
    _write_json(mapping_path, {"mappings": mappings})
    return {
        "csv": str(csv_path),
        "workbook": str(workbook_path),
        "result_manifest": str(manifest_path),
        "target_field_mapping": str(mapping_path),
    }


def _case_level_frame(
    case_frame: pd.DataFrame,
    field_values: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    values_by_field: OrderedDict[str, dict[str, list[Any]]] = OrderedDict()
    for item in field_values:
        case_id = str(item.get("case_id") or "").strip()
        field_id = str(item.get("target_field_id") or "").strip()
        if not case_id or not field_id:
            continue
        values_by_field.setdefault(field_id, {}).setdefault(case_id, []).append(item.get("value"))
    result = case_frame.copy()
    for field_id, by_case in values_by_field.items():
        column = []
        for case_id in result["case_id"].astype(str):
            values = by_case.get(case_id, [])
            if not values:
                column.append(pd.NA)
            elif len(values) == 1:
                column.append(values[0])
            else:
                column.append(json.dumps(values, ensure_ascii=False, default=str))
        result[field_id] = column
    return result


def _case_frame(cases: Sequence[str | Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in cases:
        row = dict(item) if isinstance(item, Mapping) else {"case_id": item}
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("Every case must define a non-empty case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        row["case_id"] = case_id
        rows.append(row)
    columns = ["case_id", *sorted({key for row in rows for key in row if key != "case_id"})]
    return pd.DataFrame(rows, columns=columns)


def _categories_from_field_values(
    values: Sequence[Mapping[str, Any]],
) -> tuple[OrderedDict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    category_records: OrderedDict[str, OrderedDict[tuple[str, str], dict[str, Any]]] = OrderedDict()
    mappings: OrderedDict[str, dict[str, Any]] = OrderedDict()
    repeated_categories: set[str] = set()
    for index, item in enumerate(values):
        case_id = str(item.get("case_id") or "").strip()
        field_id = str(item.get("target_field_id") or "").strip()
        field_path = str(item.get("target_field_path") or "").strip()
        if not case_id or not field_id or not field_path:
            raise ValueError(f"field_values[{index}] requires case_id, target_field_id and target_field_path")
        category, column, path_repeated = _split_field_path(field_path)
        occurrence = str(item.get("occurrence_id") or "").strip()
        repeated = bool(occurrence or path_repeated)
        if repeated:
            repeated_categories.add(category)
            occurrence = occurrence or f"occurrence_{index + 1:06d}"
        key = (case_id, occurrence if repeated else "")
        records = category_records.setdefault(category, OrderedDict())
        row = records.setdefault(key, {"case_id": case_id})
        if repeated:
            row["occurrence_id"] = occurrence
        value = item.get("value")
        if column in row and not _values_equal(row[column], value):
            raise ValueError(f"Conflicting values for {field_id} in case {case_id}, occurrence {occurrence}")
        row[column] = value
        existing = mappings.get(field_id)
        spec = {"target_field_id": field_id, "category": category, "source_column": column}
        if existing is not None and existing != spec:
            raise ValueError(f"target_field_id maps to multiple output columns: {field_id}")
        mappings[field_id] = spec

    result: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for category, records in category_records.items():
        rows = list(records.values())
        if category in repeated_categories:
            for row_index, row in enumerate(rows, start=1):
                row.setdefault("occurrence_id", f"occurrence_{row_index:06d}")
        result[category] = rows
    return result, list(mappings.values())


def _split_field_path(field_path: str) -> tuple[str, str, bool]:
    first, separator, remainder = field_path.partition(".")
    repeated = "[]" in field_path
    category = first.replace("[]", "").strip() or "data"
    column = (remainder if separator else "value").replace("[]", "").strip(".") or "value"
    return category, column, repeated


def _category_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    copied = [dict(row) for row in rows]
    if any(not str(row.get("case_id") or "").strip() for row in copied):
        raise ValueError("Every category row must define a non-empty case_id")
    prefix = ["case_id"]
    if any("occurrence_id" in row for row in copied):
        prefix.append("occurrence_id")
        for index, row in enumerate(copied, start=1):
            row.setdefault("occurrence_id", f"occurrence_{index:06d}")
    remaining = sorted({str(key) for row in copied for key in row if key not in prefix})
    return pd.DataFrame(copied, columns=[*prefix, *remaining])


def _metadata_frame(
    rows: Sequence[Mapping[str, Any]] | None,
    default_columns: Iterable[str],
) -> pd.DataFrame:
    copied = [dict(row) for row in (rows or [])]
    columns = list(default_columns)
    columns.extend(sorted({str(key) for row in copied for key in row if key not in columns}))
    return pd.DataFrame(copied, columns=columns)


def _unique_sheet_name(raw_name: str, used: set[str]) -> str:
    base = _INVALID_SHEET_CHARS.sub("_", str(raw_name)).strip().strip("'") or "data"
    base = base[:31]
    candidate = base
    suffix = 2
    while candidate.casefold() in used:
        marker = f"_{suffix}"
        candidate = f"{base[: 31 - len(marker)]}{marker}"
        suffix += 1
    used.add(candidate.casefold())
    return candidate


def _resolve_mappings(
    specs: Sequence[Mapping[str, Any]],
    category_sheets: Mapping[str, str],
    alias_by_sheet: Mapping[str, str],
    frames: Mapping[str, pd.DataFrame],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(specs):
        field_id = str(item.get("target_field_id") or "").strip()
        category = str(item.get("category") or "").strip()
        column = str(item.get("source_column") or "").strip()
        if not field_id or not category or not column:
            raise ValueError(f"field_mappings[{index}] requires target_field_id, category and source_column")
        if field_id in seen:
            raise ValueError(f"Duplicate target_field_id mapping: {field_id}")
        if category not in category_sheets:
            raise ValueError(f"Unknown mapping category for {field_id}: {category}")
        sheet_name = category_sheets[category]
        frame = frames[sheet_name]
        if column not in frame.columns:
            raise ValueError(f"Sheet {sheet_name} is missing mapped column {column} for {field_id}")
        result.append(
            {
                "target_field_id": field_id,
                "artifact": alias_by_sheet[sheet_name],
                "sheet_name": sheet_name,
                "source_column": column,
                "case_id_column": "case_id",
                "occurrence_id_column": "occurrence_id" if "occurrence_id" in frame.columns else "",
            }
        )
        seen.add(field_id)
    return result


def _values_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:
        return False


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
