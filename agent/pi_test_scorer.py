"""Gold-aware scorer entrypoint used only by the strict Test host process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflow.pi_harness_evaluation import (
    load_evaluation_manifest,
    score_equal_weight_reference_directory,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--gold-root", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result_root = Path(args.result_root).resolve()
    gold_root = Path(args.gold_root).resolve()
    key_columns = load_evaluation_manifest(args.evaluation_manifest, gold_root)
    report = score_equal_weight_reference_directory(result_root, gold_root, key_columns)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
