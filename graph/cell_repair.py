from __future__ import annotations

import ast
import asyncio
import csv
import gzip
import hashlib
import inspect
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


PAIR_SCHEMA_VERSION = 2
RULE_SCHEMA_VERSION = 2
REPAIR_SCHEMA_VERSION = 2
FCORR_MINIMUM_ACCURACY = 0.85
TRAIN_FOLDS = (0, 1, 2)
FOLD_SPLITS = {0: "train", 1: "train", 2: "train", 3: "validation", 4: "internal_test"}
EXPECTED_PROTOCOL_COUNTS = {
    "train": {"dirty": 9560, "clean": 4301},
    "validation": {"dirty": 5210, "clean": 2418},
    "internal_test": {"dirty": 5230, "clean": 3281},
}
ALLOWED_ACTIONS = {"replace", "unresolved"}
LOCATION_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "canonical_stay_id",
    "patient_id",
}
PRIVATE_MARKERS = (
    "reference_private",
    "host_private",
    "merged_injection_log",
    "validation/reference",
    "test/reference",
)
FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "hasattr",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
SAFE_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}
SAFE_STRING_METHODS = {
    "casefold",
    "count",
    "endswith",
    "find",
    "format",
    "get",
    "group",
    "groups",
    "index",
    "isdigit",
    "isalnum",
    "isalpha",
    "islower",
    "isnumeric",
    "isspace",
    "isupper",
    "items",
    "join",
    "keys",
    "lstrip",
    "lower",
    "replace",
    "rfind",
    "rstrip",
    "split",
    "startswith",
    "strip",
    "upper",
    "values",
    "zfill",
}
SAFE_RE_METHODS = {"fullmatch", "match", "search", "sub"}
GIDCL_SYSTEM_MESSAGE = (
    "You are an AI assistant that follows instruction extremely well. "
    "The user will give you a question. Answer as faithfully as you can."
)


class CellRepairError(ValueError):
    pass


Completion = Callable[[list[dict[str, str]], dict[str, Any], int], tuple[str, str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CellRepairError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CellRepairError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise CellRepairError(f"cannot read JSONL: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CellRepairError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise CellRepairError(f"JSONL record is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _new_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise CellRepairError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise CellRepairError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _coordinate(table: str, row_number: int, column: str) -> tuple[str, int, str]:
    return str(table), int(row_number), str(column)


def _field_key(table: str, column: str) -> str:
    return f"{table}.{column}"


def _field_id(table: str, column: str) -> str:
    return hashlib.sha256(f"{table}\0{column}".encode("utf-8")).hexdigest()[:16]


def _normalize_table_name(raw_file: str) -> str:
    value = str(raw_file).replace("\\", "/")
    if value.endswith(".csv.gz"):
        return value[:-7]
    if value.endswith(".csv"):
        return value[:-4]
    return value


def _read_injection_log(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise CellRepairError(f"cannot read paired evidence: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"raw_file", "raw_row_index", "column", "clean_value", "dirty_value"}
        if not required.issubset(reader.fieldnames or ()):
            raise CellRepairError("paired evidence has an invalid schema")
        result: dict[tuple[str, int, str], dict[str, str]] = {}
        for row in reader:
            table = _normalize_table_name(row["raw_file"])
            key = _coordinate(table, int(row["raw_row_index"]) + 1, row["column"])
            if key in result:
                raise CellRepairError(f"duplicate paired Cell coordinate: {key}")
            result[key] = {str(name): str(value or "") for name, value in row.items()}
    return result


def _read_raw_rows(
    raw_dir: Path,
    requests: Mapping[str, set[int]],
) -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for table, row_numbers in sorted(requests.items()):
        path = raw_dir / f"{table}.csv"
        if not path.is_file():
            path = raw_dir / f"{table}.csv.gz"
        if not path.is_file():
            raise CellRepairError(f"raw table is missing: {table}.csv or {table}.csv.gz")
        remaining = set(row_numbers)
        context = (
            gzip.open(path, "rt", encoding="utf-8-sig", newline="")
            if path.suffix == ".gz"
            else path.open("r", encoding="utf-8-sig", newline="")
        )
        with context as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=1):
                if row_number not in remaining:
                    continue
                rows[(table, row_number)] = {
                    str(name): str(value or "") for name, value in row.items()
                }
                remaining.remove(row_number)
                if not remaining:
                    break
        if remaining:
            raise CellRepairError(
                f"raw table {table} is missing requested rows: {sorted(remaining)[:5]}"
            )
    return rows


def _load_supervision_masks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path) as arrays:
            required = {"cell_indices", "cell_labels", "cell_folds", "cell_source"}
            if not required.issubset(arrays.files):
                raise CellRepairError("supervision masks have an invalid schema")
            indices = np.asarray(arrays["cell_indices"], dtype=np.int64)
            labels = np.asarray(arrays["cell_labels"], dtype=np.int8)
            folds = np.asarray(arrays["cell_folds"], dtype=np.int8)
            sources = np.asarray(arrays["cell_source"], dtype=np.int8)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, CellRepairError):
            raise
        raise CellRepairError(f"invalid supervision masks: {path}") from exc
    if len({len(indices), len(labels), len(folds), len(sources)}) != 1:
        raise CellRepairError("supervision mask arrays have different lengths")
    if len(np.unique(indices)) != len(indices):
        raise CellRepairError("supervised observation indices must be unique")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise CellRepairError("supervision labels must be binary")
    if set(np.unique(folds).tolist()) != {0, 1, 2, 3, 4}:
        raise CellRepairError("supervision folds must be exactly 0..4")
    if np.any((sources == 0) != (labels == 0)):
        raise CellRepairError("sampled clean source and labels disagree")
    return indices, labels, folds, sources


def _protocol_count_map(
    expected_counts: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, dict[str, int]] | None:
    if expected_counts is None:
        return None
    return {
        str(split): {str(label): int(count) for label, count in values.items()}
        for split, values in expected_counts.items()
    }


def _safe_table_path(table: str) -> Path:
    normalized = str(table).replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or normalized.startswith("/") or any(
        not part or part in {".", ".."} or ":" in part for part in parts
    ):
        raise CellRepairError(f"unsafe graph table path: {table}")
    return Path(*parts)


def recover_raw_from_graph(
    *,
    graph_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    graph = Path(graph_dir).expanduser().resolve()
    if not graph.is_dir():
        raise CellRepairError(f"graph directory does not exist: {graph}")
    graph_manifest_path = graph / "graph_manifest.json"
    observations_path = graph / "cell_observations.jsonl"
    graph_manifest = _read_json(graph_manifest_path, "graph manifest")
    expected_table_counts = {
        str(table): int(count)
        for table, count in (graph_manifest.get("table_counts") or {}).items()
    }
    if not expected_table_counts:
        raise CellRepairError("graph manifest has no table counts")
    if not observations_path.is_file():
        raise CellRepairError(f"Cell observations do not exist: {observations_path}")
    output = _new_output(output_dir)

    current_table: str | None = None
    current_row_number = 0
    current_row: dict[str, str] = {}
    columns: list[str] | None = None
    handle: Any = None
    writer: csv.DictWriter | None = None
    temporary_path: Path | None = None
    destination_path: Path | None = None
    table_row_count = 0
    observation_count = 0
    observation_digest = hashlib.sha256()
    completed_tables: dict[str, dict[str, Any]] = {}

    def flush_row() -> None:
        nonlocal columns, handle, writer, temporary_path, destination_path, table_row_count
        if current_table is None or not current_row:
            return
        row_columns = list(current_row)
        if columns is None:
            columns = row_columns
            table_path = _safe_table_path(current_table)
            destination_path = output / table_path.parent / f"{table_path.name}.csv"
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination_path.with_name(f".{destination_path.name}.recovering")
            handle = temporary_path.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
        elif row_columns != columns:
            raise CellRepairError(
                f"Cell observation columns changed in {current_table} row {current_row_number}"
            )
        assert writer is not None
        writer.writerow(current_row)
        table_row_count += 1

    def finish_table() -> None:
        nonlocal handle, writer, temporary_path, destination_path
        if current_table is None:
            return
        flush_row()
        if handle is None or temporary_path is None or destination_path is None or columns is None:
            raise CellRepairError(f"graph table has no recoverable rows: {current_table}")
        handle.close()
        handle = None
        writer = None
        expected_rows = expected_table_counts.get(current_table)
        if expected_rows is None:
            temporary_path.unlink(missing_ok=True)
            raise CellRepairError(f"Cell observations contain an unknown table: {current_table}")
        if table_row_count != expected_rows:
            temporary_path.unlink(missing_ok=True)
            raise CellRepairError(
                f"recovered row count mismatch for {current_table}: "
                f"expected={expected_rows} actual={table_row_count}"
            )
        os.replace(temporary_path, destination_path)
        completed_tables[current_table] = {
            "path": destination_path.relative_to(output).as_posix(),
            "row_count": table_row_count,
            "column_count": len(columns),
            "columns": columns,
            "sha256": _sha256(destination_path),
            "serialized_bytes": destination_path.stat().st_size,
        }
        temporary_path = None
        destination_path = None

    try:
        with observations_path.open("rb") as observations_handle:
            for line_number, raw_line in enumerate(observations_handle, start=1):
                observation_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    observation = json.loads(raw_line)
                    table = str(observation["table"])
                    row_number = int(observation["row_number"])
                    column = str(observation["column"])
                    raw_value = observation.get("raw_value")
                    if raw_value is not None and not isinstance(raw_value, str):
                        raise TypeError("raw_value must be a string or null")
                    value = "" if raw_value is None else str(raw_value)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise CellRepairError(
                        f"invalid Cell observation at line {line_number}"
                    ) from exc
                _safe_table_path(table)
                if table != current_table:
                    finish_table()
                    if table not in expected_table_counts:
                        raise CellRepairError(
                            f"Cell observations contain an unknown table: {table}"
                        )
                    if table in completed_tables:
                        raise CellRepairError(f"Cell observation table is not contiguous: {table}")
                    current_table = table
                    current_row_number = 0
                    current_row = {}
                    columns = None
                    table_row_count = 0
                if row_number != current_row_number:
                    if current_row_number:
                        if row_number != current_row_number + 1:
                            raise CellRepairError(
                                f"non-contiguous row numbers in {table}: "
                                f"previous={current_row_number} current={row_number}"
                            )
                        flush_row()
                    elif row_number != 1:
                        raise CellRepairError(f"first row number is not 1 in {table}")
                    current_row_number = row_number
                    current_row = {}
                if column in current_row:
                    raise CellRepairError(
                        f"duplicate Cell observation column in {table} row {row_number}: {column}"
                    )
                current_row[column] = value
                observation_count += 1
        finish_table()
    except Exception:
        if handle is not None:
            handle.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        shutil.rmtree(output, ignore_errors=True)
        raise

    try:
        if set(completed_tables) != set(expected_table_counts):
            missing = sorted(set(expected_table_counts) - set(completed_tables))
            extra = sorted(set(completed_tables) - set(expected_table_counts))
            raise CellRepairError(
                f"recovered tables differ from graph manifest: missing={missing} extra={extra}"
            )
        expected_observations = graph_manifest.get("observation_count")
        if expected_observations is not None and observation_count != int(expected_observations):
            raise CellRepairError(
                "recovered observation count differs from graph manifest: "
                f"expected={expected_observations} actual={observation_count}"
            )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    report = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "recover_dirty_raw_from_cell_observations",
        "source_of_truth": "graph_cell_observations",
        "raw_root_name": graph_manifest.get("raw_root_name"),
        "table_count": len(completed_tables),
        "row_count": sum(item["row_count"] for item in completed_tables.values()),
        "observation_count": observation_count,
        "tables": {table: completed_tables[table] for table in sorted(completed_tables)},
        "inputs": {
            "graph_manifest_sha256": _sha256(graph_manifest_path),
            "cell_observations_sha256": observation_digest.hexdigest(),
        },
    }
    _write_json(output / "recovered_raw_manifest.json", report)
    return report


def build_field_pairs(
    *,
    graph_dir: str | Path,
    supervision_dir: str | Path,
    raw_dir: str | Path,
    paired_log: str | Path,
    output_dir: str | Path,
    expected_counts: Mapping[str, Mapping[str, int]] | None = EXPECTED_PROTOCOL_COUNTS,
) -> dict[str, Any]:
    graph = Path(graph_dir).expanduser().resolve()
    supervision = Path(supervision_dir).expanduser().resolve()
    raw = Path(raw_dir).expanduser().resolve()
    log_path = Path(paired_log).expanduser().resolve()
    for label, path in (("graph", graph), ("supervision", supervision), ("raw", raw)):
        if not path.is_dir():
            raise CellRepairError(f"{label} directory does not exist: {path}")
    if not log_path.is_file():
        raise CellRepairError(f"paired evidence does not exist: {log_path}")
    output = _new_output(output_dir)

    indices, labels, folds, _ = _load_supervision_masks(supervision / "supervision_masks.npz")
    selected = {
        int(index): (int(label), int(fold))
        for index, label, fold in zip(indices, labels, folds)
    }
    observed: dict[int, dict[str, Any]] = {}
    requests: dict[str, set[int]] = defaultdict(set)
    observations_path = graph / "cell_observations.jsonl"
    with observations_path.open(encoding="utf-8") as handle:
        for observation_index, line in enumerate(handle):
            if observation_index not in selected:
                continue
            try:
                observation = json.loads(line)
                table = str(observation["table"])
                row_number = int(observation["row_number"])
                column = str(observation["column"])
                current = str(observation.get("raw_value") or "")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise CellRepairError(
                    f"invalid supervised observation at index {observation_index}"
                ) from exc
            observed[observation_index] = {
                "table": table,
                "row_number": row_number,
                "column": column,
                "current": current,
            }
            requests[table].add(row_number)
    missing = sorted(set(selected) - set(observed))
    if missing:
        raise CellRepairError(f"supervision references missing observations: {missing[:5]}")

    raw_rows = _read_raw_rows(raw, requests)
    raw_table_hashes: dict[str, str] = {}
    for table in sorted(requests):
        raw_path = raw / f"{table}.csv"
        if not raw_path.is_file():
            raw_path = raw / f"{table}.csv.gz"
        raw_table_hashes[table] = _sha256(raw_path)
    injection = _read_injection_log(log_path)
    dirty_coordinates: set[tuple[str, int, str]] = set()
    counts = {
        split: {"dirty": 0, "clean": 0}
        for split in ("train", "validation", "internal_test")
    }
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"dirty_clean_pairs": [], "clean_examples": []}
    )
    excluded_locator_counts: Counter[str] = Counter()

    for observation_index in sorted(selected):
        label, fold = selected[observation_index]
        split = FOLD_SPLITS[fold]
        observation = observed[observation_index]
        table = observation["table"]
        row_number = observation["row_number"]
        column = observation["column"]
        current = observation["current"]
        raw_current = str(raw_rows[(table, row_number)].get(column) or "")
        if raw_current != current:
            raise CellRepairError(
                f"raw/graph value mismatch at {(table, row_number, column)}"
            )
        coordinate = _coordinate(table, row_number, column)
        if label == 1:
            counts[split]["dirty"] += 1
            truth = injection.get(coordinate)
            if truth is None:
                raise CellRepairError(f"dirty Cell is missing paired evidence: {coordinate}")
            if current != str(truth["dirty_value"]):
                raise CellRepairError(f"dirty value mismatch at {coordinate}")
            if coordinate in dirty_coordinates:
                raise CellRepairError(f"duplicate supervised dirty Cell coordinate: {coordinate}")
            dirty_coordinates.add(coordinate)
            if split == "train":
                if column.lower() in LOCATION_COLUMNS:
                    excluded_locator_counts[_field_key(table, column)] += 1
                else:
                    groups[(table, column)]["dirty_clean_pairs"].append({
                        "dirty": current,
                        "clean": str(truth["clean_value"]),
                    })
        else:
            counts[split]["clean"] += 1
            if coordinate in injection:
                raise CellRepairError(f"clean Cell unexpectedly appears in paired evidence: {coordinate}")
            if split == "train" and column.lower() not in LOCATION_COLUMNS:
                groups[(table, column)]["clean_examples"].append(current)

    if dirty_coordinates != set(injection):
        missing_dirty = sorted(set(injection) - dirty_coordinates)
        extra_dirty = sorted(dirty_coordinates - set(injection))
        raise CellRepairError(
            "paired evidence and supervised dirty Cells differ: "
            f"missing={missing_dirty[:3]} extra={extra_dirty[:3]}"
        )
    expected = _protocol_count_map(expected_counts)
    if expected is not None and counts != expected:
        raise CellRepairError(f"supervision split counts differ from frozen protocol: {counts}")

    fields_dir = output / "fields"
    fields_dir.mkdir()
    field_entries: list[dict[str, Any]] = []
    exported_dirty = 0
    exported_clean = 0
    for (table, column), values in sorted(groups.items()):
        pairs = values["dirty_clean_pairs"]
        clean_examples = values["clean_examples"]
        if not pairs:
            continue
        field_id = _field_id(table, column)
        payload = {
            "table": table,
            "column": column,
            "dirty_clean_pairs": pairs,
            "clean_examples": clean_examples,
        }
        serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        lowered = serialized.lower().replace("\\", "/")
        if any(marker in lowered for marker in PRIVATE_MARKERS):
            raise CellRepairError(f"private marker leaked into field evidence: {table}.{column}")
        path = fields_dir / f"{field_id}.json"
        _atomic_text(path, serialized)
        mapping: dict[str, set[str]] = defaultdict(set)
        for pair in pairs:
            mapping[str(pair["dirty"])].add(str(pair["clean"]))
        conflict_count = sum(len(clean_values) > 1 for clean_values in mapping.values())
        entry = {
            "field_id": field_id,
            "table": table,
            "column": column,
            "path": f"fields/{field_id}.json",
            "sha256": _sha256(path),
            "dirty_pair_count": len(pairs),
            "clean_example_count": len(clean_examples),
            "conflicting_dirty_value_count": conflict_count,
            "serialized_bytes": path.stat().st_size,
        }
        field_entries.append(entry)
        exported_dirty += len(pairs)
        exported_clean += len(clean_examples)

    manifest = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "gidcl_train_field_pairs",
        "train_folds": list(TRAIN_FOLDS),
        "split_counts": counts,
        "field_count": len(field_entries),
        "exported_train_dirty_pair_count": exported_dirty,
        "exported_train_clean_example_count": exported_clean,
        "excluded_locator_pair_count": sum(excluded_locator_counts.values()),
        "excluded_locator_fields": dict(sorted(excluded_locator_counts.items())),
        "fields": field_entries,
        "inputs": {
            "supervision_masks_sha256": _sha256(supervision / "supervision_masks.npz"),
            "cell_observations_sha256": _sha256(observations_path),
            "paired_log_sha256": _sha256(log_path),
            "raw_table_sha256": raw_table_hashes,
        },
        "privacy": {
            "llm_visible_splits": ["train"],
            "locator_fields_excluded": sorted(LOCATION_COLUMNS),
            "private_taxonomy_exported": False,
        },
    }
    _write_json(output / "field_pairs_manifest.json", manifest)
    return manifest


class _CorrectionAstValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[str] = []
        self.nested_helpers: set[str] = set()
        self.depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.depth += 1
        if self.depth > 1:
            if node.name != "is_dirty" or self.nested_helpers:
                self.issues.append(
                    f"only one nested is_dirty helper is allowed at line {node.lineno}"
                )
            self.nested_helpers.add(node.name)
        self.generic_visit(node)
        self.depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        self.issues.append(f"imports are forbidden at line {node.lineno}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.issues.append(f"imports are forbidden at line {node.lineno}")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.issues.append(f"class definitions are forbidden at line {node.lineno}")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.issues.append(f"async functions are forbidden at line {node.lineno}")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.issues.append(f"lambda expressions are forbidden at line {node.lineno}")

    def visit_While(self, node: ast.While) -> None:
        self.issues.append(f"while loops are forbidden at line {node.lineno}")

    def visit_For(self, node: ast.For) -> None:
        self.issues.append(f"for loops are forbidden at line {node.lineno}")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.issues.append(f"comprehensions are forbidden at line {node.lineno}")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.issues.append(f"comprehensions are forbidden at line {node.lineno}")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.issues.append(f"comprehensions are forbidden at line {node.lineno}")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.issues.append(f"comprehensions are forbidden at line {node.lineno}")

    def visit_Try(self, node: ast.Try) -> None:
        self.issues.append(f"try statements are forbidden at line {node.lineno}")

    def visit_With(self, node: ast.With) -> None:
        self.issues.append(f"with statements are forbidden at line {node.lineno}")

    def visit_Raise(self, node: ast.Raise) -> None:
        self.issues.append(f"raise statements are forbidden at line {node.lineno}")

    def visit_Global(self, node: ast.Global) -> None:
        self.issues.append(f"global statements are forbidden at line {node.lineno}")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.issues.append(f"nonlocal statements are forbidden at line {node.lineno}")

    def visit_Yield(self, node: ast.Yield) -> None:
        self.issues.append(f"yield expressions are forbidden at line {node.lineno}")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.issues.append(f"yield expressions are forbidden at line {node.lineno}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") or node.id in LOCATION_COLUMNS:
            self.issues.append(f"forbidden identifier '{node.id}' at line {node.lineno}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        allowed = SAFE_STRING_METHODS | SAFE_RE_METHODS
        if node.attr.startswith("_") or node.attr not in allowed:
            self.issues.append(f"unsafe attribute '{node.attr}' at line {node.lineno}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            allowed = set(SAFE_BUILTINS) | self.nested_helpers | {"is_dirty"}
            if node.func.id in FORBIDDEN_CALLS or node.func.id not in allowed:
                self.issues.append(f"forbidden call '{node.func.id}' at line {node.lineno}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in SAFE_STRING_METHODS | SAFE_RE_METHODS:
                self.issues.append(
                    f"forbidden method call '{node.func.attr}' at line {node.lineno}"
                )
        else:
            self.issues.append(f"indirect calls are forbidden at line {node.lineno}")
        if any(isinstance(argument, ast.Starred) for argument in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            self.issues.append(f"expanded call arguments are forbidden at line {node.lineno}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            value = node.value.strip()
            lowered = value.lower().replace("\\", "/")
            if value.lower() in LOCATION_COLUMNS:
                self.issues.append(f"forbidden locator key at line {node.lineno}")
            if re.fullmatch(r"\d{7,}", value):
                self.issues.append(f"suspicious hard-coded identifier at line {node.lineno}")
            if any(marker in lowered for marker in PRIVATE_MARKERS):
                self.issues.append(f"private source marker at line {node.lineno}")
            if re.search(r"(?:^|/)(?:users|home|mnt|private|var)/", lowered) or re.match(
                r"^[a-z]:/", lowered
            ):
                self.issues.append(f"absolute path literal at line {node.lineno}")
        elif isinstance(node.value, (int, float)) and abs(node.value) >= 1_000_000:
            self.issues.append(f"oversized numeric literal at line {node.lineno}")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Pow, ast.LShift, ast.RShift, ast.MatMult)):
            self.issues.append(f"unsafe arithmetic operator at line {node.lineno}")
        self.generic_visit(node)


def _static_rule_issues(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} at line {exc.lineno}"]
    top_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    issues: list[str] = []
    if len(top_functions) != 1 or top_functions[0].name != "Correction":
        issues.append("source must define exactly one top-level Correction function")
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            issues.append(
                f"top-level {type(node).__name__} is forbidden at line {getattr(node, 'lineno', 0)}"
            )
    if top_functions:
        function = top_functions[0]
        if function.decorator_list or function.returns is not None:
            issues.append("decorators and return annotations are forbidden")
        if [argument.arg for argument in function.args.args] != ["input_string"]:
            issues.append("Correction must accept exactly input_string")
        if function.args.defaults or function.args.kw_defaults:
            issues.append("default parameters are forbidden")
        if any(argument.annotation is not None for argument in function.args.args):
            issues.append("parameter annotations are forbidden")
    validator = _CorrectionAstValidator()
    validator.visit(tree)
    return sorted(set(issues + validator.issues))


def _compile_correction(source: str) -> Callable[[str], str]:
    issues = _static_rule_issues(source)
    if issues:
        raise CellRepairError("invalid correction source: " + "; ".join(issues))
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "re": re}
    exec(compile(source, "correction.py", "exec"), namespace, namespace)
    function = namespace.get("Correction")
    if not callable(function) or list(inspect.signature(function).parameters) != ["input_string"]:
        raise CellRepairError("Correction is not callable with input_string")
    return function


def _source_candidates(text: str) -> list[str]:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return [*blocks, text]


def _extract_correction_source(text: str) -> str:
    valid_sources: list[str] = []
    for candidate in _source_candidates(text):
        lines = candidate.strip().splitlines()
        start = next(
            (index for index, line in enumerate(lines) if line.lstrip().startswith("def Correction(")),
            None,
        )
        if start is None:
            continue
        relevant = lines[start:]
        for end in range(len(relevant), 0, -1):
            source = "\n".join(relevant[:end]).rstrip() + "\n"
            if not _static_rule_issues(source):
                valid_sources.append(source)
                break
    unique_sources = list(dict.fromkeys(valid_sources))
    if len(unique_sources) != 1:
        raise CellRepairError(
            "MiniMax response must contain exactly one valid Correction function"
        )
    return unique_sources[0]


def _validate_evidence(evidence: Mapping[str, Any]) -> None:
    if set(evidence) != {"table", "column", "dirty_clean_pairs", "clean_examples"}:
        raise CellRepairError("field evidence has an invalid schema")
    if not str(evidence["table"]) or not str(evidence["column"]):
        raise CellRepairError("field evidence requires table and column")
    if str(evidence["column"]).lower() in LOCATION_COLUMNS:
        raise CellRepairError("locator fields are forbidden in MiniMax evidence")
    pairs = evidence["dirty_clean_pairs"]
    clean = evidence["clean_examples"]
    if not isinstance(pairs, list) or not pairs:
        raise CellRepairError("field evidence requires dirty-clean pairs")
    if not isinstance(clean, list):
        raise CellRepairError("field clean examples must be a list")
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"dirty", "clean"}:
            raise CellRepairError("dirty-clean pair has an invalid schema")
        if not all(isinstance(pair[key], str) for key in ("dirty", "clean")):
            raise CellRepairError("dirty-clean values must be strings")
    if not all(isinstance(value, str) for value in clean):
        raise CellRepairError("clean examples must be strings")
    serialized = json.dumps(evidence, ensure_ascii=True, separators=(",", ":")).lower()
    if any(marker in serialized.replace("\\", "/") for marker in PRIVATE_MARKERS):
        raise CellRepairError("private source marker leaked into MiniMax evidence")


def validate_fcorr(
    source: str,
    evidence: Mapping[str, Any],
    *,
    minimum_accuracy: float = 0.85,
) -> dict[str, Any]:
    if not 0.0 <= minimum_accuracy < 1.0:
        raise CellRepairError("minimum accuracy must be in [0, 1)")
    _validate_evidence(evidence)
    static_issues = _static_rule_issues(source)
    if static_issues:
        return {"status": "FAILED", "issues": static_issues, "metrics": {}}
    try:
        function = _compile_correction(source)
    except CellRepairError as exc:
        return {"status": "FAILED", "issues": [str(exc)], "metrics": {}}

    wrong: list[dict[str, str]] = []
    runtime_issues: list[str] = []
    deterministic = True
    correct = 0
    pairs = list(evidence["dirty_clean_pairs"])
    for index, pair in enumerate(pairs):
        dirty = pair["dirty"]
        expected = pair["clean"]
        try:
            first = function(dirty)
            second = function(dirty)
        except Exception as exc:
            runtime_issues.append(f"pair {index}: {type(exc).__name__}: {exc}")
            wrong.append({"dirty": dirty, "expected_clean": expected, "actual_output": "<ERROR>"})
            continue
        if not isinstance(first, str) or not isinstance(second, str):
            runtime_issues.append(f"pair {index}: Correction must return str")
            wrong.append({
                "dirty": dirty,
                "expected_clean": expected,
                "actual_output": f"<{type(first).__name__}>",
            })
            continue
        deterministic = deterministic and first == second
        if first == expected:
            correct += 1
        else:
            wrong.append({"dirty": dirty, "expected_clean": expected, "actual_output": first})

    preserved = 0
    clean_runtime_issues = 0
    clean_examples = list(evidence["clean_examples"])
    for index, value in enumerate(clean_examples):
        try:
            first = function(value)
            second = function(value)
        except Exception as exc:
            runtime_issues.append(f"clean {index}: {type(exc).__name__}: {exc}")
            clean_runtime_issues += 1
            continue
        if not isinstance(first, str) or not isinstance(second, str):
            runtime_issues.append(f"clean {index}: Correction must return str")
            clean_runtime_issues += 1
            continue
        deterministic = deterministic and first == second
        preserved += first == value

    accuracy = correct / len(pairs)
    metrics = {
        "pair_count": len(pairs),
        "exact_count": correct,
        "wrong_count": len(pairs) - correct,
        "exact_match_accuracy": accuracy,
        "clean_example_count": len(clean_examples),
        "clean_preserved_count": preserved,
        "clean_preservation_rate": (
            preserved / len(clean_examples) if clean_examples else None
        ),
        "clean_runtime_issue_count": clean_runtime_issues,
        "deterministic": deterministic,
    }
    issues = list(runtime_issues)
    if not deterministic:
        issues.append("Correction output is not deterministic")
    if accuracy <= minimum_accuracy:
        issues.append(
            f"exact-match accuracy {accuracy:.6f} does not exceed {minimum_accuracy:.6f}"
        )
    return {
        "status": "SUCCESS" if not issues else "FAILED",
        "issues": issues,
        "metrics": metrics,
        "wrongly_corrected_pairs": wrong,
    }


def _initial_prompt(evidence: Mapping[str, Any]) -> str:
    pairs = evidence["dirty_clean_pairs"]
    dirty_values = [pair["dirty"] for pair in pairs]
    payload = {
        "table": evidence["table"],
        "column": evidence["column"],
        "dirty_clean_pairs": pairs,
        "dirty_values": dirty_values,
        "clean_examples": evidence["clean_examples"],
    }
    return (
        "Please conclude a general pattern for dirty and clean cells, and write a "
        "correction function with Python module re, with simple and precise regular "
        "expressions, to correct a dirty value to its corresponding clean value.\n\n"
        "The function contract is exactly:\n"
        "def Correction(input_string):\n"
        "    ...\n"
        "    return corrected_string\n\n"
        "Return one deterministic Correction function. You may include a short explanation "
        "or a Python code fence. Do not import modules, access files or networks, use random "
        "behavior, use row context, or hard-code patient, admission, stay, row, or observation "
        "identifiers. Return the input unchanged when no demonstrated general correction "
        "pattern applies.\n\n"
        "Train-only field evidence:\n"
        + json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )


def _revision_prompt(source: str, validation: Mapping[str, Any]) -> str:
    feedback = {
        "previous_function": source,
        "issues": validation.get("issues", []),
        "metrics": validation.get("metrics", {}),
        "wrongly_corrected_pairs": validation.get("wrongly_corrected_pairs", []),
    }
    return (
        "The previous correction function failed Train-only validation. Revise the complete "
        "Correction(input_string) function using the wrongly corrected dirty-clean pairs. "
        "Preserve general rules and do not memorize private identifiers. Return one revised "
        "deterministic function under the same contract.\n\n"
        + json.dumps(feedback, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )


def _normalized_model_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _context_too_large(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "context length",
        "context_length",
        "maximum context",
        "max context",
        "too many tokens",
        "request entity too large",
        "payload too large",
        "status code: 413",
        "error 413",
    )
    return any(marker in text for marker in markers)


async def _minimax_completion(
    messages: Sequence[Mapping[str, str]],
    *,
    agent_key: str,
) -> tuple[str, str]:
    from lib.agent_runtime import (
        create_openai_model_and_formatter,
        format_message_content,
        make_assistant_msg,
        make_system_msg,
        make_user_msg,
        resolve_model_name,
    )

    model_name = resolve_model_name(agent_key, "MiniMax-M3")
    if "minimaxm3" not in _normalized_model_name(model_name):
        raise CellRepairError(f"resolved synthesis model is not MiniMax M3: {model_name}")
    model, formatter = create_openai_model_and_formatter(
        agent_key,
        "MiniMax-M3",
        generate_overrides={"temperature": 0.0, "seed": 666},
    )
    formatted_messages = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        if role == "system":
            formatted_messages.append(make_system_msg(content))
        elif role == "user":
            formatted_messages.append(make_user_msg("fcorr_rule_synthesizer", content))
        elif role == "assistant":
            formatted_messages.append(make_assistant_msg(content))
        else:
            raise CellRepairError(f"unsupported conversation role: {role}")
    try:
        request = await formatter.format(formatted_messages)
        response = await model(request)
        return format_message_content(response.content), model_name
    finally:
        closer = getattr(model, "aclose", None)
        if callable(closer):
            await closer()


def _read_pair_manifest(evidence_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = evidence_dir / "field_pairs_manifest.json"
    manifest = _read_json(manifest_path, "field pair manifest")
    if manifest.get("status") != "SUCCESS" or manifest.get("workflow") != "gidcl_train_field_pairs":
        raise CellRepairError("field pair manifest is not a successful Train-only artifact")
    fields = manifest.get("fields")
    if not isinstance(fields, list):
        raise CellRepairError("field pair manifest has no fields")
    for entry in fields:
        path = (evidence_dir / str(entry.get("path") or "")).resolve()
        if not path.is_relative_to(evidence_dir):
            raise CellRepairError(f"field evidence escapes evidence directory: {path}")
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise CellRepairError(f"field evidence hash mismatch: {entry.get('field_id')}")
    return manifest, fields


def _invoke_completion(
    messages: list[dict[str, str]],
    evidence: dict[str, Any],
    attempt: int,
    *,
    agent_key: str,
    completion: Completion | None,
) -> tuple[str, str]:
    if completion is not None:
        return completion(messages, evidence, attempt)
    return asyncio.run(_minimax_completion(messages, agent_key=agent_key))


def synthesize_fcorr(
    *,
    evidence_dir: str | Path,
    output_dir: str | Path,
    agent_key: str = "react_planner",
    max_attempts: int = 12,
    minimum_accuracy: float = FCORR_MINIMUM_ACCURACY,
    fields: Sequence[str] = (),
    completion: Completion | None = None,
) -> dict[str, Any]:
    if not 1 <= max_attempts <= 12:
        raise CellRepairError("max_attempts must be between 1 and 12")
    if minimum_accuracy != FCORR_MINIMUM_ACCURACY:
        raise CellRepairError(
            f"the frozen protocol requires minimum_accuracy={FCORR_MINIMUM_ACCURACY}"
        )
    evidence_path = Path(evidence_dir).expanduser().resolve()
    pair_manifest, entries = _read_pair_manifest(evidence_path)
    selected_fields = {str(value) for value in fields}
    known_fields = {_field_key(str(entry["table"]), str(entry["column"])) for entry in entries}
    unknown = selected_fields - known_fields
    if unknown:
        raise CellRepairError(f"unknown requested fields: {sorted(unknown)}")
    output = _new_output(output_dir)
    (output / "fields").mkdir()
    frozen_pair_manifest = output / "pair_manifest.json"
    shutil.copyfile(evidence_path / "field_pairs_manifest.json", frozen_pair_manifest)
    results: list[dict[str, Any]] = []
    models: set[str] = set()

    for entry in entries:
        field_name = _field_key(str(entry["table"]), str(entry["column"]))
        if selected_fields and field_name not in selected_fields:
            continue
        field_id = str(entry["field_id"])
        evidence_file = evidence_path / str(entry["path"])
        evidence = _read_json(evidence_file, f"field evidence {field_name}")
        _validate_evidence(evidence)
        field_output = output / "fields" / field_id
        field_output.mkdir()
        frozen_evidence_path = field_output / "evidence.json"
        shutil.copyfile(evidence_file, frozen_evidence_path)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": GIDCL_SYSTEM_MESSAGE},
            {"role": "user", "content": _initial_prompt(evidence)},
        ]
        attempts: list[dict[str, Any]] = []
        accepted_source: str | None = None
        final_validation: dict[str, Any] = {}
        status = "FCORR_REJECTED"
        model_name = "unknown"

        for attempt in range(1, max_attempts + 1):
            request_text = json.dumps(messages, ensure_ascii=True, separators=(",", ":"))
            source = ""
            try:
                response_text, model_name = _invoke_completion(
                    messages,
                    evidence,
                    attempt,
                    agent_key=agent_key,
                    completion=completion,
                )
            except Exception as exc:
                if _context_too_large(exc):
                    status = "CONTEXT_TOO_LARGE"
                    context_report = {
                        "attempt": attempt,
                        "request_sha256": _text_sha256(request_text),
                        "status": status,
                        "error": f"{type(exc).__name__}: {exc}",
                        "consumes_rule_attempt": True,
                    }
                    attempts.append(context_report)
                    _write_json(
                        field_output / f"attempt_{attempt:02d}_validation.json",
                        context_report,
                    )
                    final_validation = {
                        "status": "FAILED",
                        "issues": [context_report["error"]],
                        "metrics": {},
                        "wrongly_corrected_pairs": [],
                    }
                    break
                response_text = ""
                validation = {
                    "status": "FAILED",
                    "issues": [f"{type(exc).__name__}: {exc}"],
                    "metrics": {},
                    "wrongly_corrected_pairs": [],
                }
            else:
                if "minimaxm3" not in _normalized_model_name(model_name):
                    validation = {
                        "status": "FAILED",
                        "issues": [f"completion model is not MiniMax M3: {model_name}"],
                        "metrics": {},
                        "wrongly_corrected_pairs": [],
                    }
                else:
                    models.add(model_name)
                    try:
                        source = _extract_correction_source(response_text)
                        validation = validate_fcorr(
                            source,
                            evidence,
                            minimum_accuracy=minimum_accuracy,
                        )
                    except Exception as exc:
                        source = ""
                        validation = {
                            "status": "FAILED",
                            "issues": [f"{type(exc).__name__}: {exc}"],
                            "metrics": {},
                            "wrongly_corrected_pairs": [],
                        }
            _atomic_text(field_output / f"attempt_{attempt:02d}_response.txt", response_text)
            attempt_report = {
                "attempt": attempt,
                "request_sha256": _text_sha256(request_text),
                "response_sha256": _text_sha256(response_text),
                "model": model_name,
                "validation": validation,
                "consumes_rule_attempt": True,
            }
            attempts.append(attempt_report)
            _write_json(field_output / f"attempt_{attempt:02d}_validation.json", attempt_report)
            final_validation = validation
            messages.append({"role": "assistant", "content": response_text})
            if validation.get("status") == "SUCCESS" and source:
                accepted_source = source
                status = "FROZEN"
                break
            if attempt < max_attempts:
                messages.append({"role": "user", "content": _revision_prompt(source, validation)})

        _write_json(field_output / "conversation.json", {"messages": messages})
        field_manifest: dict[str, Any] = {
            "schema_version": RULE_SCHEMA_VERSION,
            "status": status,
            "workflow": "gidcl_direct_fcorr",
            "field_id": field_id,
            "table": evidence["table"],
            "column": evidence["column"],
            "model": model_name,
            "agent_key": agent_key,
            "attempt_count": len(attempts),
            "maximum_attempts": max_attempts,
            "minimum_accuracy_exclusive": minimum_accuracy,
            "generation_config": {"temperature": 0.0, "seed": 666},
            "evidence_path": f"fields/{field_id}/evidence.json",
            "evidence_sha256": _sha256(frozen_evidence_path),
            "conversation_path": f"fields/{field_id}/conversation.json",
            "conversation_sha256": _sha256(field_output / "conversation.json"),
            "validation": final_validation,
            "test_time_llm_access": False,
        }
        if accepted_source is not None:
            source_path = field_output / "correction.py"
            _atomic_text(source_path, accepted_source.rstrip() + "\n")
            field_manifest["source_sha256"] = _sha256(source_path)
        _write_json(field_output / "field_manifest.json", field_manifest)
        results.append({
            "field_id": field_id,
            "table": evidence["table"],
            "column": evidence["column"],
            "status": status,
            "field_manifest": f"fields/{field_id}/field_manifest.json",
        })

    synthesis_manifest = {
        "schema_version": RULE_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "gidcl_direct_fcorr_synthesis",
        "field_result_count": len(results),
        "status_counts": dict(Counter(item["status"] for item in results)),
        "models": sorted(models),
        "agent_key": agent_key,
        "maximum_attempts": max_attempts,
        "minimum_accuracy_exclusive": minimum_accuracy,
        "pair_manifest_path": "pair_manifest.json",
        "pair_manifest_sha256": _sha256(frozen_pair_manifest),
        "fields": results,
    }
    _write_json(output / "synthesis_manifest.json", synthesis_manifest)
    registry = freeze_rule_registry(synthesis_dir=output)
    return {**synthesis_manifest, "registry": registry}


def freeze_rule_registry(*, synthesis_dir: str | Path) -> dict[str, Any]:
    synthesis = Path(synthesis_dir).expanduser().resolve()
    manifest = _read_json(synthesis / "synthesis_manifest.json", "synthesis manifest")
    if manifest.get("status") != "SUCCESS" or manifest.get("workflow") != (
        "gidcl_direct_fcorr_synthesis"
    ):
        raise CellRepairError("synthesis manifest is not a successful F_corr artifact")
    pair_manifest_path = synthesis / str(manifest.get("pair_manifest_path") or "")
    if not pair_manifest_path.is_file() or _sha256(pair_manifest_path) != manifest.get(
        "pair_manifest_sha256"
    ):
        raise CellRepairError("synthesis pair manifest hash mismatch")
    accepted: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("fields", []):
        field_manifest_path = synthesis / str(entry["field_manifest"])
        field_manifest = _read_json(field_manifest_path, "field synthesis manifest")
        if field_manifest.get("status") != "FROZEN":
            continue
        field_id = str(field_manifest["field_id"])
        source_path = synthesis / "fields" / field_id / "correction.py"
        evidence_path = synthesis / str(field_manifest.get("evidence_path") or "")
        conversation_path = synthesis / str(field_manifest.get("conversation_path") or "")
        validation = field_manifest.get("validation") or {}
        accuracy = (validation.get("metrics") or {}).get("exact_match_accuracy")
        if field_manifest.get("workflow") != "gidcl_direct_fcorr":
            raise CellRepairError(f"accepted field has an invalid workflow: {field_id}")
        if "minimaxm3" not in _normalized_model_name(str(field_manifest.get("model") or "")):
            raise CellRepairError(f"accepted field model is not MiniMax M3: {field_id}")
        if field_manifest.get("generation_config") != {"temperature": 0.0, "seed": 666}:
            raise CellRepairError(f"accepted field generation config changed: {field_id}")
        if validation.get("status") != "SUCCESS" or not isinstance(accuracy, (int, float)) or (
            float(accuracy) <= FCORR_MINIMUM_ACCURACY
        ):
            raise CellRepairError(f"accepted field failed frozen Train threshold: {field_id}")
        if not evidence_path.is_file() or _sha256(evidence_path) != field_manifest.get(
            "evidence_sha256"
        ):
            raise CellRepairError(f"accepted field evidence hash mismatch: {field_id}")
        if not conversation_path.is_file() or _sha256(conversation_path) != field_manifest.get(
            "conversation_sha256"
        ):
            raise CellRepairError(f"accepted field conversation hash mismatch: {field_id}")
        if not source_path.is_file() or _sha256(source_path) != field_manifest.get("source_sha256"):
            raise CellRepairError(f"accepted field source hash mismatch: {field_id}")
        _compile_correction(source_path.read_text(encoding="utf-8"))
        key = _field_key(str(field_manifest["table"]), str(field_manifest["column"]))
        if key in accepted:
            raise CellRepairError(f"duplicate accepted field rule: {key}")
        accepted[key] = {
            "field_id": field_id,
            "table": field_manifest["table"],
            "column": field_manifest["column"],
            "source_path": f"fields/{field_id}/correction.py",
            "source_sha256": field_manifest["source_sha256"],
            "evidence_path": field_manifest["evidence_path"],
            "evidence_sha256": field_manifest["evidence_sha256"],
            "conversation_path": field_manifest["conversation_path"],
            "conversation_sha256": field_manifest["conversation_sha256"],
            "field_manifest_path": str(entry["field_manifest"]),
            "field_manifest_sha256": _sha256(field_manifest_path),
            "train_exact_match_accuracy": field_manifest["validation"]["metrics"][
                "exact_match_accuracy"
            ],
        }
    registry = {
        "schema_version": RULE_SCHEMA_VERSION,
        "status": "FROZEN",
        "workflow": "gidcl_fcorr_rule_registry",
        "rule_count": len(accepted),
        "rules": {key: accepted[key] for key in sorted(accepted)},
        "pair_manifest_path": manifest["pair_manifest_path"],
        "pair_manifest_sha256": manifest["pair_manifest_sha256"],
        "synthesis_manifest_sha256": _sha256(synthesis / "synthesis_manifest.json"),
        "test_time_llm_access": False,
    }
    _write_json(synthesis / "rule_registry.json", registry)
    return registry


def _load_rule_registry(rule_dir: Path) -> tuple[dict[str, Callable[[str], str]], dict[str, Any]]:
    registry = _read_json(rule_dir / "rule_registry.json", "rule registry")
    if registry.get("status") != "FROZEN" or registry.get("test_time_llm_access") is not False:
        raise CellRepairError("rule registry is not frozen for offline execution")
    synthesis_manifest_path = rule_dir / "synthesis_manifest.json"
    if not synthesis_manifest_path.is_file() or _sha256(synthesis_manifest_path) != registry.get(
        "synthesis_manifest_sha256"
    ):
        raise CellRepairError("frozen synthesis manifest hash mismatch")
    pair_manifest_path = rule_dir / str(registry.get("pair_manifest_path") or "")
    if not pair_manifest_path.is_file() or _sha256(pair_manifest_path) != registry.get(
        "pair_manifest_sha256"
    ):
        raise CellRepairError("frozen pair manifest hash mismatch")
    rules: dict[str, Callable[[str], str]] = {}
    for key, entry in registry.get("rules", {}).items():
        source_path = rule_dir / str(entry["source_path"])
        field_manifest_path = rule_dir / str(entry["field_manifest_path"])
        evidence_path = rule_dir / str(entry["evidence_path"])
        conversation_path = rule_dir / str(entry["conversation_path"])
        if not source_path.is_file() or _sha256(source_path) != entry.get("source_sha256"):
            raise CellRepairError(f"frozen rule source hash mismatch: {key}")
        if not field_manifest_path.is_file() or _sha256(field_manifest_path) != entry.get(
            "field_manifest_sha256"
        ):
            raise CellRepairError(f"frozen field manifest hash mismatch: {key}")
        if not evidence_path.is_file() or _sha256(evidence_path) != entry.get("evidence_sha256"):
            raise CellRepairError(f"frozen field evidence hash mismatch: {key}")
        if not conversation_path.is_file() or _sha256(conversation_path) != entry.get(
            "conversation_sha256"
        ):
            raise CellRepairError(f"frozen field conversation hash mismatch: {key}")
        rules[str(key)] = _compile_correction(source_path.read_text(encoding="utf-8"))
    if len(rules) != int(registry.get("rule_count", -1)):
        raise CellRepairError("frozen registry rule count mismatch")
    return rules, registry


def _read_predictions(path: Path, split: str) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "observation_index", "dirty_score", "threshold", "prediction"}
        if not required.issubset(reader.fieldnames or ()):
            raise CellRepairError("predictions CSV has an invalid schema")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            if row["split"] != split or row["prediction"] != "1":
                continue
            observation_index = int(row["observation_index"])
            if observation_index in rows:
                raise CellRepairError(
                    f"duplicate predicted Cell observation index: {observation_index}"
                )
            rows[observation_index] = row
    if not rows:
        raise CellRepairError(f"no predicted dirty Cells for split {split}")
    return rows


def build_repair_targets(
    *,
    predictions: str | Path,
    graph_dir: str | Path,
    raw_dir: str | Path,
    split: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    predictions_path = Path(predictions).expanduser().resolve()
    graph = Path(graph_dir).expanduser().resolve()
    raw = Path(raw_dir).expanduser().resolve()
    output = _new_output(output_dir)
    prediction_rows = _read_predictions(predictions_path, split)
    observations: dict[int, dict[str, Any]] = {}
    with (graph / "cell_observations.jsonl").open(encoding="utf-8") as handle:
        for observation_index, line in enumerate(handle):
            if observation_index in prediction_rows:
                observations[observation_index] = json.loads(line)
    missing = sorted(set(prediction_rows) - set(observations))
    if missing:
        raise CellRepairError(f"predictions reference missing observations: {missing[:5]}")
    requests: dict[str, set[int]] = defaultdict(set)
    for observation in observations.values():
        requests[str(observation["table"])].add(int(observation["row_number"]))
    raw_rows = _read_raw_rows(raw, requests)
    fields = [
        "split",
        "observation_index",
        "table",
        "row_number",
        "raw_row_index",
        "column",
        "current_value",
        "dirty_score",
        "threshold",
    ]
    target_path = output / "repair_targets.csv"
    temporary = target_path.with_name(f".{target_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for observation_index in sorted(prediction_rows):
            prediction = prediction_rows[observation_index]
            observation = observations[observation_index]
            table = str(observation["table"])
            row_number = int(observation["row_number"])
            column = str(observation["column"])
            current = str(raw_rows[(table, row_number)].get(column) or "")
            if current != str(observation.get("raw_value") or ""):
                raise CellRepairError(
                    f"raw/graph value mismatch at {(table, row_number, column)}"
                )
            writer.writerow({
                "split": split,
                "observation_index": observation_index,
                "table": table,
                "row_number": row_number,
                "raw_row_index": row_number - 1,
                "column": column,
                "current_value": current,
                "dirty_score": prediction["dirty_score"],
                "threshold": prediction["threshold"],
            })
    os.replace(temporary, target_path)
    manifest = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "build_repair_targets",
        "split": split,
        "target_count": len(prediction_rows),
        "predictions_sha256": _sha256(predictions_path),
        "targets_sha256": _sha256(target_path),
        "private_labels_exported": False,
    }
    _write_json(output / "target_manifest.json", manifest)
    return manifest


def _read_targets(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split",
            "observation_index",
            "table",
            "raw_row_index",
            "column",
            "current_value",
            "dirty_score",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise CellRepairError("repair targets CSV has an invalid schema")
        return [dict(row) for row in reader]


def run_frozen_rules(
    *,
    targets: str | Path,
    rule_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    target_path = Path(targets).expanduser().resolve()
    rules_path = Path(rule_dir).expanduser().resolve()
    output = _new_output(output_dir)
    functions, registry = _load_rule_registry(rules_path)
    target_rows = _read_targets(target_path)
    value_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for row in target_rows:
        field = _field_key(row["table"], row["column"])
        current = str(row["current_value"])
        function = functions.get(field)
        candidate: str | None = None
        reason = "no_accepted_field_rule"
        if function is not None:
            try:
                produced = function(current)
                if not isinstance(produced, str):
                    reason = "rule_returned_non_string"
                elif produced == current:
                    reason = "rule_returned_input"
                else:
                    candidate = produced
                    reason = "accepted_field_rule"
            except Exception as exc:
                reason = f"rule_runtime_error:{type(exc).__name__}"
        action = "replace" if candidate is not None else "unresolved"
        action_counts[action] += 1
        common = {
            "split": row["split"],
            "observation_index": int(row["observation_index"]),
            "table": row["table"],
            "raw_row_index": int(row["raw_row_index"]),
            "column": row["column"],
            "current_value": current,
            "dirty_score": float(row["dirty_score"]),
            "field_rule": field if function is not None else "",
            "reason": reason,
        }
        value_rows.append({**common, "candidate_value": candidate})
        plan_rows.append({
            **common,
            "action": action,
            "replacement_value": candidate or "",
        })
    values_path = output / "repair_values.jsonl"
    plan_path = output / "repair_plan.jsonl"
    _write_jsonl(values_path, value_rows)
    _write_jsonl(plan_path, plan_rows)
    manifest = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "offline_frozen_fcorr_execution",
        "split": target_rows[0]["split"] if target_rows else "",
        "target_count": len(target_rows),
        "action_counts": dict(action_counts),
        "targets_sha256": _sha256(target_path),
        "rule_registry_sha256": _sha256(rules_path / "rule_registry.json"),
        "repair_values_sha256": _sha256(values_path),
        "repair_plan_sha256": _sha256(plan_path),
        "llm_called": False,
        "frozen_rule_count": registry["rule_count"],
    }
    _write_json(output / "repair_execution_manifest.json", manifest)
    return manifest


def _csv_open(path: Path, mode: str, *, compressed: bool | None = None):
    if compressed is None:
        compressed = path.suffix == ".gz"
    if compressed:
        return gzip.open(
            path,
            mode,
            encoding="utf-8-sig" if "r" in mode else "utf-8",
            newline="",
        )
    return path.open(mode, encoding="utf-8-sig" if "r" in mode else "utf-8", newline="")


def apply_repair_plan(
    *,
    raw_dir: str | Path,
    repair_plan: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    raw = Path(raw_dir).expanduser().resolve()
    plan_path = Path(repair_plan).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not raw.is_dir():
        raise CellRepairError(f"raw directory does not exist: {raw}")
    if output == raw or raw in output.parents:
        raise CellRepairError("repair output must be outside the input raw directory")
    output = _new_output(output)
    plans = _read_jsonl(plan_path)
    replacements: dict[tuple[str, int, str], dict[str, Any]] = {}
    for plan in plans:
        action = str(plan.get("action") or "")
        if action not in ALLOWED_ACTIONS:
            raise CellRepairError(f"invalid repair action: {action}")
        if action != "replace":
            continue
        key = (str(plan["table"]), int(plan["raw_row_index"]), str(plan["column"]))
        if key in replacements:
            raise CellRepairError(f"duplicate replacement coordinate: {key}")
        replacements[key] = plan
    shutil.copytree(raw, output, dirs_exist_ok=True)
    by_table: dict[str, dict[tuple[int, str], dict[str, Any]]] = defaultdict(dict)
    for (table, row_index, column), plan in replacements.items():
        by_table[table][(row_index, column)] = plan
    applied = 0
    for table, table_plans in sorted(by_table.items()):
        source = raw / f"{table}.csv"
        destination = output / f"{table}.csv"
        if not source.is_file():
            source = raw / f"{table}.csv.gz"
            destination = output / f"{table}.csv.gz"
        if not source.is_file():
            raise CellRepairError(f"repair plan references missing table: {table}")
        temporary = destination.with_name(f".{destination.name}.repairing")
        seen: set[tuple[int, str]] = set()
        plans_by_row: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for (row_index, column), plan in table_plans.items():
            plans_by_row[row_index].append((column, plan))
        compressed = source.suffix == ".gz"
        with _csv_open(source, "rt", compressed=compressed) as source_handle, _csv_open(
            temporary, "wt", compressed=compressed
        ) as output_handle:
            reader = csv.DictReader(source_handle)
            fields = list(reader.fieldnames or [])
            writer = csv.DictWriter(output_handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row_index, row in enumerate(reader):
                for column, plan in plans_by_row.get(row_index, []):
                    if column not in fields:
                        raise CellRepairError(f"repair column is missing from {table}: {column}")
                    observed = str(row.get(column) or "")
                    if observed != str(plan["current_value"]):
                        raise CellRepairError(
                            f"repair precondition mismatch at {(table, row_index, column)}"
                        )
                    row[column] = str(plan["replacement_value"])
                    seen.add((row_index, column))
                    applied += 1
                writer.writerow(row)
        missing = set(table_plans) - seen
        if missing:
            temporary.unlink(missing_ok=True)
            raise CellRepairError(f"repair rows are missing from {table}: {sorted(missing)[:5]}")
        os.replace(temporary, destination)
    manifest = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "apply_cell_repair_plan",
        "plan_record_count": len(plans),
        "replacement_count": len(replacements),
        "applied_replacement_count": applied,
        "repair_plan_sha256": _sha256(plan_path),
        "input_file_count": sum(path.is_file() for path in raw.rglob("*")),
        "output_business_file_count": sum(
            path.is_file() and path.name != "repair_apply_manifest.json"
            for path in output.rglob("*")
        ),
    }
    _write_json(output / "repair_apply_manifest.json", manifest)
    return manifest


def _metric_block(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dirty = [row for row in reports if row["is_dirty"]]
    clean = [row for row in reports if not row["is_dirty"]]
    predicted_dirty = [row for row in reports if row["detector_prediction"] == 1]
    detection_tp = sum(row["is_dirty"] for row in predicted_dirty)
    detection_fp = len(predicted_dirty) - detection_tp
    detection_fn = sum(row["detector_prediction"] == 0 for row in dirty)
    detection_tn = sum(row["detector_prediction"] == 0 for row in clean)
    dirty_candidates = [row for row in dirty if row["has_rule_candidate"]]
    exact = sum(row["repair_status"] == "exact" for row in dirty)
    incorrect = sum(row["repair_status"] == "incorrect" for row in dirty)
    clean_modified = sum(row["repair_status"] == "collateral" for row in clean)
    precision_denominator = exact + incorrect + clean_modified
    precision = exact / precision_denominator if precision_denominator else 0.0
    recall = exact / len(dirty) if dirty else 0.0
    return {
        "target_count": len(reports),
        "ground_truth_dirty_count": len(dirty),
        "ground_truth_clean_count": len(clean),
        "detector_confusion_matrix": {
            "tp": detection_tp,
            "fp": detection_fp,
            "fn": detection_fn,
            "tn": detection_tn,
        },
        "dirty_rule_candidate_count": len(dirty_candidates),
        "dirty_rule_candidate_coverage": len(dirty_candidates) / len(dirty) if dirty else 0.0,
        "rule_only_exact_accuracy": (
            exact / detection_tp
            if detection_tp
            else 0.0
        ),
        "candidate_exact_accuracy": (
            sum(row["candidate_exact"] for row in dirty_candidates) / len(dirty_candidates)
            if dirty_candidates
            else 0.0
        ),
        "exact_repairs": exact,
        "incorrect_repairs": incorrect,
        "unresolved_dirty_targets": sum(row["repair_status"] == "unresolved" for row in dirty),
        "clean_targets_preserved": len(clean) - clean_modified,
        "clean_targets_modified": clean_modified,
        "clean_preservation_rate": (
            (len(clean) - clean_modified) / len(clean) if clean else 0.0
        ),
        "replacement_count": sum(row["action"] == "replace" for row in reports),
        "correction_precision": precision,
        "correction_recall": recall,
        "correction_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate_repairs(
    *,
    repair_values: str | Path,
    repair_plan: str | Path,
    injection_log: str | Path,
    predictions: str | Path,
    graph_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    values_path = Path(repair_values).expanduser().resolve()
    plan_path = Path(repair_plan).expanduser().resolve()
    log_path = Path(injection_log).expanduser().resolve()
    predictions_path = Path(predictions).expanduser().resolve()
    graph = Path(graph_dir).expanduser().resolve()
    output = _new_output(output_dir)
    value_rows = _read_jsonl(values_path)
    plan_rows = _read_jsonl(plan_path)
    if not value_rows:
        raise CellRepairError("repair values are empty")
    splits = sorted({str(row.get("split") or "") for row in value_rows})
    if len(splits) != 1 or not splits[0]:
        raise CellRepairError("repair values must contain exactly one non-empty split")
    split = splits[0]
    values = {int(row["observation_index"]): row for row in value_rows}
    if len(values) != len(value_rows):
        raise CellRepairError("repair values have duplicate observation indices")
    plans = {int(row["observation_index"]): row for row in plan_rows}
    if len(plans) != len(plan_rows):
        raise CellRepairError("repair plan has duplicate observation indices")
    if set(values) != set(plans):
        raise CellRepairError("repair value and plan observation indices differ")

    with predictions_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "observation_index", "prediction", "label"}
        if not required.issubset(reader.fieldnames or ()):
            raise CellRepairError("private evaluation predictions have an invalid schema")
        prediction_rows: dict[int, dict[str, str]] = {}
        for row in reader:
            if row["split"] != split:
                continue
            observation_index = int(row["observation_index"])
            if observation_index in prediction_rows:
                raise CellRepairError(
                    f"duplicate evaluation prediction: {observation_index}"
                )
            if row["prediction"] not in {"0", "1"} or row["label"] not in {"0", "1"}:
                raise CellRepairError("evaluation predictions must contain binary labels")
            prediction_rows[observation_index] = dict(row)
    if not prediction_rows:
        raise CellRepairError(f"no prediction rows for evaluation split {split}")
    predicted_positive = {
        index for index, row in prediction_rows.items() if row["prediction"] == "1"
    }
    if set(values) != predicted_positive:
        raise CellRepairError(
            "repair artifacts must cover every and only detector-positive Cell"
        )

    observations: dict[int, dict[str, Any]] = {}
    observations_path = graph / "cell_observations.jsonl"
    with observations_path.open(encoding="utf-8") as handle:
        for observation_index, line in enumerate(handle):
            if observation_index in prediction_rows:
                observations[observation_index] = json.loads(line)
    missing = sorted(set(prediction_rows) - set(observations))
    if missing:
        raise CellRepairError(f"evaluation predictions reference missing observations: {missing[:5]}")

    gold = _read_injection_log(log_path)
    reports: list[dict[str, Any]] = []
    for observation_index in sorted(prediction_rows):
        prediction = prediction_rows[observation_index]
        observation = observations[observation_index]
        table = str(observation["table"])
        row_number = int(observation["row_number"])
        column = str(observation["column"])
        current = str(observation.get("raw_value") or "")
        coordinate = _coordinate(table, row_number, column)
        truth = gold.get(coordinate)
        is_dirty = prediction["label"] == "1"
        if is_dirty != (truth is not None):
            raise CellRepairError(
                f"prediction label and paired evidence disagree at observation {observation_index}"
            )
        clean_value = str(truth["clean_value"]) if truth else current
        value_row = values.get(observation_index)
        plan = plans.get(observation_index)
        if value_row is not None:
            expected_coordinate = (table, row_number - 1, column, current)
            actual_coordinate = (
                str(value_row["table"]),
                int(value_row["raw_row_index"]),
                str(value_row["column"]),
                str(value_row["current_value"]),
            )
            if actual_coordinate != expected_coordinate:
                raise CellRepairError(
                    f"repair artifact coordinate mismatch at observation {observation_index}"
                )
        candidate = value_row.get("candidate_value") if value_row is not None else None
        has_candidate = isinstance(candidate, str) and candidate != current
        candidate_exact = has_candidate and str(candidate) == clean_value
        action = str(plan["action"]) if plan is not None else "not_targeted"
        if truth:
            if action == "replace" and str(plan["replacement_value"]) == clean_value:
                repair_status = "exact"
            elif action == "replace":
                repair_status = "incorrect"
            else:
                repair_status = "unresolved"
        else:
            repair_status = "collateral" if action == "replace" else "preserved"
        reports.append({
            "split": split,
            "observation_index": observation_index,
            "table": table,
            "raw_row_index": row_number - 1,
            "column": column,
            "field": _field_key(table, column),
            "is_dirty": is_dirty,
            "detector_prediction": int(prediction["prediction"]),
            "error_class": str(truth.get("error_class") or "") if truth else "sampled_clean",
            "error_subtype": str(truth.get("error_subtype") or "") if truth else "sampled_clean",
            "has_rule_candidate": has_candidate,
            "candidate_exact": candidate_exact,
            "action": action,
            "repair_status": repair_status,
            "reason": plan.get("reason", "") if plan is not None else "detector_negative",
        })

    by_field = {
        field: _metric_block([row for row in reports if row["field"] == field])
        for field in sorted({row["field"] for row in reports})
    }
    by_subtype = {
        subtype: _metric_block([row for row in reports if row["error_subtype"] == subtype])
        for subtype in sorted({row["error_subtype"] for row in reports})
    }
    report = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "private_fcorr_repair_evaluation",
        "evaluation_role": (
            "development_audit" if split == "internal_test" else "validation_or_external"
        ),
        "splits": [split],
        "metrics": _metric_block(reports),
        "by_field": by_field,
        "by_error_subtype": by_subtype,
        "target_reports": reports,
        "inputs": {
            "repair_values_sha256": _sha256(values_path),
            "repair_plan_sha256": _sha256(plan_path),
            "injection_log_sha256": _sha256(log_path),
            "predictions_sha256": _sha256(predictions_path),
            "cell_observations_sha256": _sha256(observations_path),
        },
    }
    _write_json(output / "repair_evaluation_report.json", report)
    return report
