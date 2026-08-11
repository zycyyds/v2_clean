from __future__ import annotations

import argparse
import json
from pathlib import Path

from .builder import build_graph
from .schema import DEFAULT_SCHEMA, GraphSchema


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a typed-value graph from authorized MIMIC raw tables.")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--relation-ids", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--max-rows-per-table", type=int)
    args = parser.parse_args()
    schema = GraphSchema.load(args.schema) if args.schema else DEFAULT_SCHEMA
    relation_ids = json.loads(args.relation_ids.read_text(encoding="utf-8")) if args.relation_ids else None
    manifest = build_graph(
        args.raw_root,
        args.output_dir,
        schema,
        split=args.split,
        max_rows_per_table=args.max_rows_per_table,
        relation_ids=relation_ids,
    )
    print(f"SUCCESS nodes={manifest['node_count']} edges={manifest['edge_count']} triples={manifest['triple_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
