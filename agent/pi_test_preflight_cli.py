"""CLI for public large-scale preflight of a frozen Validation pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.pi_harness_cli import _run_with_terminal_signals
from agent.pi_test_harness import PiTestPreflight, PiTestPreflightConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight a frozen Validation pipeline without opening Gold or Test raw.",
    )
    parser.add_argument("--validation-experiment", required=True)
    parser.add_argument("--preflight-experiment", required=True)
    parser.add_argument("--preflight-raw", required=True)
    parser.add_argument("--replay-timeout", type=float, default=1_800.0)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    harness = PiTestPreflight(
        PiTestPreflightConfig(
            project_root=Path(__file__).parents[1],
            validation_experiment=Path(args.validation_experiment),
            preflight_experiment=Path(args.preflight_experiment),
            preflight_raw=Path(args.preflight_raw),
            replay_timeout_seconds=args.replay_timeout,
        ),
    )
    result, received_signal = await _run_with_terminal_signals(harness.run())
    print(
        json.dumps(
            {
                "status": result.status,
                "phase": result.phase,
                "scored": result.scored,
                "score": result.score,
                "replay_status": result.replay_status,
                "frozen_snapshot_sha256": result.frozen_snapshot_sha256,
                "attestation": str(result.attestation) if result.attestation else "",
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    if received_signal is not None:
        return 128 + int(received_signal)
    if result.status == "INTERRUPTED":
        return 130
    return 0 if result.status == "SUCCESS" else 1


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(parse_args(argv)))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
