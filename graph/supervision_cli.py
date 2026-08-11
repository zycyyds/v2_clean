from __future__ import annotations

import argparse
from pathlib import Path

from .supervision import build_supervised_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic dirty supervision graph from authorized Train raw.")
    parser.add_argument("--base-dirty-raw", required=True, type=Path)
    parser.add_argument("--clean-raw", required=True, type=Path)
    parser.add_argument("--clean-graph", required=True, type=Path)
    parser.add_argument("--base-injection-log", required=True, type=Path)
    parser.add_argument("--output-raw", required=True, type=Path)
    parser.add_argument("--output-graph", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--dirty-target", type=int, default=20_000)
    parser.add_argument("--clean-target", type=int, default=10_000)
    parser.add_argument("--seeds", default="666,667")
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    result = build_supervised_graph(
        base_dirty_raw=args.base_dirty_raw,
        clean_raw=args.clean_raw,
        clean_graph=args.clean_graph,
        base_injection_log=args.base_injection_log,
        output_raw=args.output_raw,
        output_graph=args.output_graph,
        private_output=args.private_output,
        dirty_target=args.dirty_target,
        clean_target=args.clean_target,
        seeds=seeds,
    )
    print(
        "SUCCESS "
        f"dirty={result['dirty_observation_count']} "
        f"clean={result['clean_training_count']} "
        f"synthetic={result['synthetic_log_record_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
