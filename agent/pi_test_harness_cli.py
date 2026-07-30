"""CLI for strict one-shot frozen-pipeline Test evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.pi_harness_cli import _run_with_terminal_signals
from agent.pi_test_harness import PiTestHarness, PiTestHarnessConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a frozen Validation pipeline on Test exactly once.",
    )
    parser.add_argument("--validation-experiment", required=True)
    parser.add_argument("--preflight-attestation")
    parser.add_argument("--test-experiment", required=True)
    parser.add_argument("--test-raw", required=True)
    parser.add_argument("--test-gold", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--replay-timeout", type=float, default=1_800.0)
    parser.add_argument("--scoring-timeout", type=float, default=3_600.0)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    harness = PiTestHarness(
        PiTestHarnessConfig(
            project_root=Path(__file__).parents[1],
            validation_experiment=Path(args.validation_experiment),
            preflight_attestation=(
                Path(args.preflight_attestation)
                if args.preflight_attestation is not None
                else None
            ),
            test_experiment=Path(args.test_experiment),
            test_raw=Path(args.test_raw),
            test_gold=Path(args.test_gold),
            evaluation_manifest=Path(args.evaluation_manifest),
            replay_timeout_seconds=args.replay_timeout,
            scoring_timeout_seconds=args.scoring_timeout,
        ),
    )
    result, received_signal = await _run_with_terminal_signals(harness.run())
    print(json.dumps(_cli_payload(result), ensure_ascii=False, indent=2))
    if received_signal is not None:
        return 128 + int(received_signal)
    if result.status == "INTERRUPTED":
        return 130
    return 0 if result.status == "SUCCESS" else 1


def _cli_payload(result) -> dict:
    return {
        "status": result.status,
        "phase": result.phase,
        "scored": result.scored,
        "score": result.score,
        "test_execution_count": result.test_execution_count,
        "replay_status": result.replay_status,
        "frozen_snapshot_sha256": result.frozen_snapshot_sha256,
        "result_package": str(result.result_package) if result.result_package else "",
    }


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(parse_args(argv)))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
