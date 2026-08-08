from __future__ import annotations

import argparse
from pathlib import Path

from .labels import build_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Build host-side clean/dirty labels from an injection log.")
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--injection-log", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = build_labels(args.graph_dir, args.injection_log, args.output_dir)
    print(f"SUCCESS observations={manifest['observation_count']} dirty_cells={manifest['dirty_cell_count']} dirty_rows={manifest['dirty_row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
