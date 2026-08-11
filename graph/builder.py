from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.lib.format import open_memmap

from .features import canonical_scalar
from .schema import DEFAULT_SCHEMA, GraphSchema


class GraphBuildError(ValueError):
    pass


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_MISSING = {"", "nan", "none", "null", "nat"}


@dataclass(frozen=True)
class TableScan:
    path: Path
    table: str
    columns: tuple[str, ...]
    row_count: int

    @property
    def observation_count(self) -> int:
        return self.row_count * len(self.columns)


def _table_id(path: Path, raw_root: Path) -> str:
    return path.relative_to(raw_root).with_suffix("").as_posix()


def _safe_csv_files(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        raise GraphBuildError(f"raw root does not exist or is not a directory: {raw_root}")
    return sorted(
        path
        for path in raw_root.rglob("*.csv")
        if not any(part.startswith(".") for part in path.relative_to(raw_root).parts)
    )


def _read_rows(path: Path, *, max_rows: int | None = None):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise GraphBuildError(f"CSV has no header: {path}")
        columns = [str(column).strip() for column in reader.fieldnames]
        for row_number, row in enumerate(reader, start=1):
            if max_rows is not None and row_number > max_rows:
                break
            yield row_number, columns, {
                str(key).strip(): (value or "")
                for key, value in row.items()
                if key is not None
            }


def _scan_tables(raw_root: Path, max_rows_per_table: int | None) -> list[TableScan]:
    scans: list[TableScan] = []
    for path in _safe_csv_files(raw_root):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                columns = tuple(str(column).strip() for column in next(reader))
            except StopIteration as exc:
                raise GraphBuildError(f"CSV has no header: {path}") from exc
            if not columns or any(not column for column in columns):
                raise GraphBuildError(f"CSV has an empty column name: {path}")
            row_count = 0
            for _ in reader:
                row_count += 1
                if max_rows_per_table is not None and row_count >= max_rows_per_table:
                    break
        scans.append(TableScan(path, _table_id(path, raw_root), columns, row_count))
    return scans


def _node_key(node_type: str, key: str) -> str:
    return f"{node_type}:{key}"


def _is_private_or_unauthorized(raw_root: Path) -> None:
    lowered = {part.lower() for part in raw_root.parts}
    forbidden = {"reference_private", "test_gold", "gold", "host_private", "workflow"}
    if lowered & forbidden:
        raise GraphBuildError("raw_root points to a forbidden/private directory")


def _serialized(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _ordinary_value(value: str) -> str:
    stripped = value.strip()
    return "<MISSING>" if stripped.lower() in _MISSING else stripped


def _value_type(value: str, *, shared: bool) -> str:
    if value == "<MISSING>":
        return "missing"
    if shared:
        return "identifier"
    if _NUMBER.fullmatch(value):
        return "numeric"
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "text"
    return "datetime"


def _relation_vocabulary(
    scans: list[TableScan],
    relation_ids: Mapping[str, int] | None,
) -> dict[str, int]:
    discovered: list[str] = []
    for scan in scans:
        for column in scan.columns:
            relation = f"{scan.table}.{column}"
            discovered.extend((relation, relation + "__rev"))
    if relation_ids is None:
        return {name: index for index, name in enumerate(discovered)}
    fixed = dict(relation_ids)
    if sorted(fixed.values()) != list(range(len(fixed))):
        raise GraphBuildError("relation_ids must use contiguous zero-based integer IDs")
    missing = [name for name in discovered if name not in fixed]
    if missing:
        raise GraphBuildError(f"unknown relation outside fixed vocabulary: {missing[0]}")
    return fixed


def build_graph(
    raw_root: str | Path,
    output_dir: str | Path,
    schema: GraphSchema = DEFAULT_SCHEMA,
    *,
    split: str = "train",
    max_rows_per_table: int | None = None,
    relation_ids: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    raw_root = Path(raw_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if split not in {"train", "validation", "test"}:
        raise GraphBuildError(f"unsupported split: {split}")
    if max_rows_per_table is not None and max_rows_per_table <= 0:
        raise GraphBuildError("max_rows_per_table must be positive")
    _is_private_or_unauthorized(raw_root)
    if output_dir == raw_root or raw_root in output_dir.parents:
        raise GraphBuildError("output_dir must not be inside raw_root")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise GraphBuildError(f"output_dir must be new or empty: {output_dir}")

    scans = _scan_tables(raw_root, max_rows_per_table)
    relation_id_map = _relation_vocabulary(scans, relation_ids)
    observation_count = sum(scan.observation_count for scan in scans)
    directed_edge_count = observation_count * 2
    node_ids: dict[str, int] = {}
    table_counts = {scan.table: scan.row_count for scan in scans}

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        edge_index = open_memmap(
            output_dir / "edge_index.npy",
            mode="w+",
            dtype=np.int64,
            shape=(2, directed_edge_count),
        )
        edge_type = open_memmap(
            output_dir / "edge_type.npy",
            mode="w+",
            dtype=np.int16,
            shape=(directed_edge_count,),
        )
        edge_cursor = 0
        observation_index = 0

        with (
            (output_dir / "nodes.jsonl").open("w", encoding="utf-8") as nodes_handle,
            (output_dir / "cell_observations.jsonl").open("w", encoding="utf-8") as cells_handle,
        ):
            def intern(node_type: str, key: str, **metadata: Any) -> int:
                node_key = _node_key(node_type, key)
                existing = node_ids.get(node_key)
                if existing is not None:
                    return existing
                node_id = len(node_ids)
                node_ids[node_key] = node_id
                record = {"node_id": node_id, "node_type": node_type, "key": key, **metadata}
                nodes_handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                return node_id

            for scan in scans:
                for row_number, columns, row in _read_rows(scan.path, max_rows=max_rows_per_table):
                    row_key = f"{scan.table}#row_{row_number}"
                    row_text = _serialized({
                        "cells": [[column, row.get(column, "")] for column in columns],
                        "kind": "row",
                        "table": scan.table,
                    })
                    row_id = intern(
                        "Row",
                        row_key,
                        table=scan.table,
                        row_number=row_number,
                        embedding_text=row_text,
                    )
                    for column in columns:
                        raw_value = row.get(column, "")
                        field_spec = schema.field_for(scan.table, column)
                        icd_version = row.get("icd_version", "")
                        canonical = (
                            canonical_scalar(raw_value, field_spec.canonicalizer, icd_version=icd_version)
                            if field_spec
                            else None
                        )
                        if field_spec and canonical is not None:
                            domain = field_spec.domain
                            normalized = canonical
                            shared = True
                        else:
                            domain = f"{scan.table}.{column}"
                            normalized = _ordinary_value(raw_value)
                            shared = False
                        value_key = f"{domain}:{normalized}"
                        value_kind = _value_type(normalized, shared=shared)
                        value_text = _serialized({
                            "domain": domain,
                            "kind": "value",
                            "shared": shared,
                            "value": normalized,
                            "value_type": value_kind,
                        })
                        value_id = intern(
                            "Value",
                            value_key,
                            domain=domain,
                            canonical_value=normalized,
                            shared=shared,
                            value_type=value_kind,
                            embedding_text=value_text,
                        )
                        relation = f"{scan.table}.{column}"
                        forward_edge_id = edge_cursor
                        reverse_edge_id = edge_cursor + 1
                        edge_index[:, forward_edge_id] = (row_id, value_id)
                        edge_type[forward_edge_id] = relation_id_map[relation]
                        edge_index[:, reverse_edge_id] = (value_id, row_id)
                        edge_type[reverse_edge_id] = relation_id_map[relation + "__rev"]
                        edge_cursor += 2
                        cells_handle.write(json.dumps({
                            "column": column,
                            "forward_edge_id": forward_edge_id,
                            "normalized_value": normalized,
                            "observation_index": observation_index,
                            "raw_value": raw_value,
                            "relation": relation,
                            "reverse_edge_id": reverse_edge_id,
                            "row_id": row_id,
                            "row_number": row_number,
                            "shared_domain": field_spec.domain if shared and field_spec else None,
                            "table": scan.table,
                            "value_node_id": value_id,
                        }, ensure_ascii=True, sort_keys=True) + "\n")
                        observation_index += 1

        if edge_cursor != directed_edge_count or observation_index != observation_count:
            raise GraphBuildError(
                f"graph count mismatch: edges={edge_cursor}/{directed_edge_count} "
                f"observations={observation_index}/{observation_count}"
            )
        edge_index.flush()
        edge_type.flush()
        del edge_index, edge_type

        relations = [name for name, _ in sorted(relation_id_map.items(), key=lambda item: item[1])]
        with (output_dir / "relation_texts.jsonl").open("w", encoding="utf-8") as handle:
            for relation_id, name in enumerate(relations):
                reverse = name.endswith("__rev")
                forward_name = name.removesuffix("__rev")
                table, column = forward_name.rsplit(".", 1)
                handle.write(json.dumps({
                    "embedding_text": _serialized({
                        "column": column,
                        "direction": "value_to_row" if reverse else "row_to_value",
                        "kind": "relation",
                        "table": table,
                    }),
                    "relation": name,
                    "relation_id": relation_id,
                }, ensure_ascii=True, sort_keys=True) + "\n")

        manifest = {
            "schema_version": 2,
            "split": split,
            "raw_root_name": raw_root.name,
            "node_types": list(schema.node_types),
            "value_domains": sorted({field.domain for field in schema.shared_fields}),
            "node_count": len(node_ids),
            "observation_count": observation_count,
            "edge_count": directed_edge_count,
            "triple_count": observation_count,
            "relation_count": len(relation_id_map),
            "table_counts": dict(sorted(table_counts.items())),
            "relations": relations,
            "schema": schema.to_dict(),
            "storage": {
                "cell_observations": "cell_observations.jsonl",
                "edge_index": "edge_index.npy",
                "edge_type": "edge_type.npy",
                "nodes": "nodes.jsonl",
                "relation_texts": "relation_texts.jsonl",
            },
        }
        (output_dir / "graph_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "relation_ids.json").write_text(
            json.dumps(relation_id_map, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
