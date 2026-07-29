"""CLI for independent frozen-pipeline Test evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.pi_test_harness import PiTestHarness, PiTestHarnessConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen Validation pipeline on hidden Test data.")
    parser.add_argument("--validation-experiment", required=True)
    parser.add_argument("--test-experiment")
    parser.add_argument("--test-raw", required=True)
    parser.add_argument("--test-gold", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--max-declaration-repairs", type=int, default=3)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--replay-timeout", type=float, default=1_800.0)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    validation = Path(args.validation_experiment).expanduser().resolve()
    test_experiment = (
        Path(args.test_experiment).expanduser().resolve()
        if args.test_experiment
        else validation.with_name(validation.name + "_test")
    )
    result = await PiTestHarness(
        PiTestHarnessConfig(
            project_root=Path(__file__).parents[1],
            validation_experiment=validation,
            test_experiment=test_experiment,
            test_raw=Path(args.test_raw),
            test_gold=Path(args.test_gold),
            evaluation_manifest=Path(args.evaluation_manifest),
            max_declaration_repairs=args.max_declaration_repairs,
            replay_timeout_seconds=args.replay_timeout,
            max_iters=args.max_iters,
        )
    ).run()
    print(json.dumps({
        "status": result.status,
        "score": result.score,
        "declaration_agent_used": result.declaration_agent_used,
        "declaration_rounds": result.declaration_rounds,
        "replay_status": result.replay_status,
        "result_package": str(result.result_package) if result.result_package else "",
    }, ensure_ascii=False, indent=2))
    return 0 if result.status == "SUCCESS" else 1


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(parse_args(argv)))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
