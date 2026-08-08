from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .features import canonical_scalar, node_feature
from .io import write_graph_arrays, write_jsonl
from .schema import DEFAULT_SCHEMA, GraphSchema


class GraphBuildError(ValueError):
    pass


def _table_id(path: Path, raw_root: Path) -> str:
    return path.relative_to(raw_root).with_suffix("").as_posix()


def _safe_csv_files(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        raise GraphBuildError(f"raw root does not exist or is not a directory: {raw_root}")
    return sorted(path for path in raw_root.rglob("*.csv") if not any(part.startswith(".") for part in path.relative_to(raw_root).parts))


def _read_rows(path: Path, *, max_rows: int | None = None):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise GraphBuildError(f"CSV has no header: {path}")
        columns = [str(column).strip() for column in reader.fieldnames]
        for row_number, row in enumerate(reader, start=1):
            if max_rows is not None and row_number > max_rows:
                break
            yield row_number, columns, {str(key).strip(): (value or "") for key, value in row.items() if key is not None}


def _node_key(node_type: str, key: str) -> str:
    return f"{node_type}:{key}"


def _is_private_or_unauthorized(raw_root: Path) -> None:
    lowered = {part.lower() for part in raw_root.parts}
    forbidden = {"reference_private", "test_gold", "gold", "host_private", "workflow"}
    if lowered & forbidden:
        raise GraphBuildError("raw_root points to a forbidden/private directory")


def build_graph(
    raw_root: str | Path,
    output_dir: str | Path,
    schema: GraphSchema = DEFAULT_SCHEMA,
    *,
    split: str = "train",
    max_rows_per_table: int | None = None,
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
    output_dir.mkdir(parents=True, exist_ok=True)

    node_ids: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []
    edge_pairs: list[tuple[int, int]] = []
    edge_types: list[int] = []
    relation_ids: dict[str, int] = {}
    degrees: Counter[int] = Counter()
    frequencies: Counter[int] = Counter()
    table_counts: Counter[str] = Counter()
    triple_count = 0

    def intern(node_type: str, key: str, **metadata: Any) -> int:
        node_key = _node_key(node_type, key)
        existing = node_ids.get(node_key)
        if existing is not None:
            return existing
        index = len(nodes)
        node_ids[node_key] = index
        nodes.append({"node_id": index, "node_type": node_type, "key": key, **metadata})
        return index

    def relation_id(relation: str) -> int:
        if relation not in relation_ids:
            relation_ids[relation] = len(relation_ids)
        return relation_ids[relation]

    def add_edge(source: int, relation: str, target: int, *, reverse: bool = False) -> None:
        relation_name = relation + "__rev" if reverse else relation
        r_id = relation_id(relation_name)
        edge_pairs.append((source, target))
        edge_types.append(r_id)
        degrees[source] += 1
        degrees[target] += 1

    triples_path = output_dir / "triples.jsonl"
    cells_path = output_dir / "cell_observations.jsonl"
    with triples_path.open("w", encoding="utf-8") as triples_handle, cells_path.open("w", encoding="utf-8") as cells_handle:
        for csv_path in _safe_csv_files(raw_root):
            table = _table_id(csv_path, raw_root)
            for row_number, columns, row in _read_rows(csv_path, max_rows=max_rows_per_table):
                table_counts[table] += 1
                row_key = f"{table}#row_{row_number}"
                row_id = intern("Row", row_key, table=table, row_number=row_number)
                for column in columns:
                    raw_value = row.get(column, "")
                    field_spec = schema.field_for(table, column)
                    icd_version = row.get("icd_version", "")
                    if field_spec:
                        canonical = canonical_scalar(raw_value, field_spec.canonicalizer, icd_version=icd_version)
                        value_key = f"{field_spec.domain}:{canonical}" if canonical is not None else None
                        value_meta = {"domain": field_spec.domain, "canonical_value": canonical, "shared": True}
                    else:
                        canonical = raw_value.strip() or "<MISSING>"
                        # Ordinary cells stay as observations. Materializing
                        # one Value node per cell makes full MIMIC graphs too large.
                        value_key = None
                        value_meta = {}

                    relation = f"{table}.{column}"
                    value_id = None
                    if field_spec and value_key is not None:
                        value_id = intern("Value", value_key, **value_meta)
                        add_edge(row_id, relation, value_id)
                        add_edge(value_id, relation, row_id, reverse=True)
                        frequencies[value_id] += 1
                        triples_handle.write(json.dumps({"head": row_id, "relation": relation, "tail": value_id, "head_key": row_key, "tail_key": value_key}, ensure_ascii=True, sort_keys=True) + "\n")
                        triple_count += 1
                    cells_handle.write(json.dumps({"row_id": row_id, "table": table, "row_number": row_number, "column": column, "raw_value": raw_value, "normalized_value": canonical, "value_node_id": value_id, "relation": relation, "shared_domain": field_spec.domain if field_spec else None}, ensure_ascii=True, sort_keys=True) + "\n")

    features = np.vstack([node_feature(node, degree=degrees[index], frequency=frequencies[index]) for index, node in enumerate(nodes)]) if nodes else np.empty((0, 64), dtype=np.float32)
    write_jsonl(output_dir / "nodes.jsonl", nodes)
    write_graph_arrays(output_dir, edge_pairs, edge_types, features)
    manifest = {
        "schema_version": 1,
        "split": split,
        "raw_root_name": raw_root.name,
        "node_types": list(schema.node_types),
        "value_domains": sorted({field.domain for field in schema.shared_fields}),
        "node_count": len(nodes),
        "edge_count": len(edge_pairs),
        "triple_count": triple_count,
        "relation_count": len(relation_ids),
        "table_counts": dict(sorted(table_counts.items())),
        "relations": [name for name, _ in sorted(relation_ids.items(), key=lambda item: item[1])],
        "schema": schema.to_dict(),
    }
    (output_dir / "graph_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "relation_ids.json").write_text(json.dumps(relation_ids, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
