from __future__ import annotations

import argparse
from pathlib import Path

from .cell_training_aggregate import aggregate_training_reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate formal 666/667/668 Cell classifier reports."
    )
    parser.add_argument(
        "--reports",
        required=True,
        nargs="+",
        type=Path,
        help="Three run directories or report.json files.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-nonformal-seeds",
        action="store_true",
        help="Allow development aggregates that are not exactly seeds 666/667/668.",
    )
    args = parser.parse_args()
    report = aggregate_training_reports(
        args.reports,
        args.output,
        require_formal_seeds=not args.allow_nonformal_seeds,
    )
    test = report["splits"]["internal_test"]
    print(
        f"SUCCESS model={report['model_type']} seeds={report['seeds']} "
        f"test_average_precision={test['average_precision']['mean']:.6f}"
        f"+/-{test['average_precision']['standard_deviation']:.6f} "
        f"test_macro_f1={test['macro_f1']['mean']:.6f}"
        f"+/-{test['macro_f1']['standard_deviation']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
