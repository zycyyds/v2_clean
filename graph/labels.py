from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


class LabelBuildError(ValueError):
    pass


def _iter_jsonl(path: Path):
    if not path.is_file():
        raise LabelBuildError(f"missing graph observation file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _read_log(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise LabelBuildError(f"missing host-side injection log: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"operation", "raw_file", "raw_row_index", "column", "error_class", "error_subtype"}
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        raise LabelBuildError(f"injection log is missing columns: {missing}")
    return rows


def build_labels(
    graph_dir: str | Path,
    injection_log: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create host-side cell/row labels from an auditable injection log.

    The log is the only source of dirty supervision. ``reference`` and all
    private Gold directories are intentionally not accepted as inputs. Row
    coordinates use the injector's zero-based output-row index, while graph
    observations use one-based data-row numbers, so the conversion is explicit.
    """
    graph_dir = Path(graph_dir).expanduser().resolve()
    output_dir = Path(output_dir or graph_dir).expanduser().resolve()
    log_path = Path(injection_log).expanduser().resolve()
    graph_manifest_path = graph_dir / "graph_manifest.json"
    if not graph_manifest_path.is_file():
        raise LabelBuildError(f"missing graph manifest: {graph_manifest_path}")
    if output_dir == log_path or log_path in output_dir.parents:
        raise LabelBuildError("label output cannot be inside the injection-log directory")

    log_rows = _read_log(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_cells: set[tuple[str, int, str]] = set()
    target_rows: set[tuple[str, int]] = set()

    for log_index, record in enumerate(log_rows):
        raw_file = str(record.get("raw_file") or "").strip()
        try:
            raw_row_index = int(str(record.get("raw_row_index") or ""))
        except ValueError as exc:
            raise LabelBuildError(f"invalid raw_row_index at log row {log_index}") from exc
        table = raw_file.removesuffix(".csv")
        row_number = raw_row_index + 1
        operation = str(record.get("operation") or "").strip()
        if operation == "cell_update":
            column = str(record.get("column") or "").strip()
            target_cells.add((table, row_number, column))
        elif operation == "row_insert":
            target_rows.add((table, row_number))
        else:
            raise LabelBuildError(f"unsupported injection operation at log row {log_index}: {operation}")

    observation_count = 0
    cell_dirty_count = 0
    matched_coordinates: set[tuple[str, int, str]] = set()
    matched_rows: set[tuple[str, int]] = set()
    with (graph_dir / "cell_observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observation = json.loads(line)
            observation_count += 1
            coordinate = (str(observation["table"]), int(observation["row_number"]), str(observation["column"]))
            row_key = coordinate[:2]
            if coordinate in target_cells or row_key in target_rows:
                cell_dirty_count += 1
                matched_coordinates.add(coordinate)
                matched_rows.add(row_key)

    unmatched = target_cells - matched_coordinates
    unmatched_rows = target_rows - matched_rows
    if unmatched or unmatched_rows:
        preview = ", ".join(f"{table}:{row}" for table, row, *_ in sorted(unmatched)[:5])
        preview += ", ".join(f"{table}:{row}" for table, row in sorted(unmatched_rows)[:5])
        raise LabelBuildError(f"injection log rows did not map to graph observations ({len(unmatched) + len(unmatched_rows)}): {preview}")

    nodes_path = graph_dir / "nodes.jsonl"
    node_index_by_row: dict[tuple[str, int], int] = {}
    node_count = 0
    row_node_count = 0
    for node in _iter_jsonl(nodes_path):
        node_id = int(node["node_id"])
        node_count = max(node_count, node_id + 1)
        if node.get("node_type") == "Row":
            row_node_count += 1
            node_index_by_row[(str(node.get("table") or ""), int(node["row_number"]))] = node_id
    dirty_row_keys = matched_rows
    row_dirty_nodes = np.zeros(node_count, dtype=np.int8)
    for row_key in dirty_row_keys:
        node_id = node_index_by_row.get(row_key)
        if node_id is None:
            raise LabelBuildError(f"dirty row is missing from graph nodes: {row_key}")
        row_dirty_nodes[node_id] = 1

    cell_labels = np.zeros(observation_count, dtype=np.int8)
    observation_index = 0
    with (graph_dir / "cell_observations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            observation = json.loads(line)
            coordinate = (str(observation["table"]), int(observation["row_number"]), str(observation["column"]))
            if coordinate in target_cells or coordinate[:2] in target_rows:
                cell_labels[observation_index] = 1
            observation_index += 1

    labels_path = output_dir / "edge_labels.npz"
    np.savez_compressed(
        labels_path,
        cell_dirty=cell_labels,
        row_dirty_node=row_dirty_nodes,
    )
    with (output_dir / "label_records.jsonl").open("w", encoding="utf-8") as handle:
        for log_index, record in enumerate(log_rows):
            raw_file = str(record.get("raw_file") or "").strip()
            row_number = int(str(record.get("raw_row_index") or "")) + 1
            handle.write(json.dumps({
                "log_index": log_index,
                "source_error_id": str(record.get("source_error_id") or log_index),
                "table": raw_file.removesuffix(".csv"),
                "row_number": row_number,
                "column": str(record.get("column") or ""),
                "operation": str(record.get("operation") or ""),
                "dirty_cell_count": 1 if str(record.get("operation") or "") == "cell_update" else 0,
                "error_class": str(record.get("error_class") or ""),
                "error_subtype": str(record.get("error_subtype") or ""),
            }, ensure_ascii=True, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "label_source": "host_private_injection_log",
        "graph_manifest": graph_manifest_path.name,
        "observation_count": observation_count,
        "log_record_count": len(log_rows),
        "matched_record_count": len(log_rows),
        "dirty_cell_count": cell_dirty_count,
        "dirty_row_count": int(row_dirty_nodes.sum()),
        "row_node_count": row_node_count,
        "clean_cell_count": int(observation_count - cell_dirty_count),
        "by_operation": dict(Counter(str(row.get("operation") or "") for row in log_rows)),
        "by_error_class": dict(Counter(str(record.get("error_class") or "") for record in log_rows)),
        "by_error_subtype": dict(Counter(str(record.get("error_subtype") or "") for record in log_rows)),
        "outputs": {"edge_labels": labels_path.name, "label_records": "label_records.jsonl"},
    }
    (output_dir / "label_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
