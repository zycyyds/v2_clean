"""CLI for the persistent Pi-style validation Harness."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.pi_harness import PiValidationHarness, PiValidationHarnessConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden validation feedback against one persistent Pi-style Agent.",
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--train-raw", required=True)
    parser.add_argument("--train-reference", required=True)
    parser.add_argument("--validation-raw", required=True)
    parser.add_argument("--validation-gold", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--target-score", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--skills-dir", action="append", default=[])
    parser.add_argument("--replay-timeout", type=float, default=1_800.0)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    harness = PiValidationHarness(
        PiValidationHarnessConfig(
            project_root=Path(__file__).parents[1],
            experiment_dir=Path(args.experiment_dir),
            train_raw=Path(args.train_raw),
            train_reference=Path(args.train_reference),
            validation_raw=Path(args.validation_raw),
            validation_gold=Path(args.validation_gold),
            evaluation_manifest=Path(args.evaluation_manifest),
            max_rounds=args.max_rounds,
            patience=args.patience,
            target_score=args.target_score,
            max_iters=args.max_iters,
            skill_dirs=tuple(Path(item) for item in args.skills_dir),
            replay_timeout_seconds=args.replay_timeout,
        ),
    )
    result = await harness.run(prompt)
    print(
        json.dumps(
            {
                "status": result.status,
                "stop_reason": result.stop_reason,
                "rounds": result.rounds,
                "repair_rounds": result.repair_rounds,
                "best_score": result.best_score,
                "reproducible_score": result.reproducible_score,
                "best_snapshot": str(result.best_snapshot),
                "reproducible_snapshot": str(result.reproducible_snapshot),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    if result.status == "SUCCESS_REPRODUCIBLE":
        return 0
    if result.status.startswith("INTERRUPTED"):
        return 130
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
