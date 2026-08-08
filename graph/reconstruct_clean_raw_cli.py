from __future__ import annotations

import argparse
from pathlib import Path

from .reconstruct_clean_raw import reconstruct_clean_raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct clean raw tables from a dirty raw split and host injection log.")
    parser.add_argument("--dirty-raw", required=True, type=Path)
    parser.add_argument("--injection-log", required=True, type=Path)
    parser.add_argument("--output-raw", required=True, type=Path)
    args = parser.parse_args()
    manifest = reconstruct_clean_raw(args.dirty_raw, args.injection_log, args.output_raw)
    print(f"SUCCESS restored_cells={manifest['restored_cells']} removed_rows={manifest['removed_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
