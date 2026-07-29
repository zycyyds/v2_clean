"""Command-line entry point for the generic Pi-style AgentScope runtime."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.pi_runtime import PiAgentConfig, PiAgentRuntime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a persistent Pi-style coding agent on AgentScope 2.x.",
    )
    parser.add_argument("prompt", help="Task prompt for the coding agent.")
    parser.add_argument(
        "--workdir",
        required=True,
        help="Existing task working directory.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=10_000,
        help="Maximum ReAct iterations for this turn.",
    )
    parser.add_argument(
        "--skills-dir",
        action="append",
        default=[],
        help="Skill root to scan recursively; may be supplied more than once.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    runtime = PiAgentRuntime(
        PiAgentConfig(
            workdir=Path(args.workdir),
            max_iters=args.max_iters,
            skill_dirs=tuple(Path(item) for item in args.skills_dir),
        ),
    )
    try:
        await runtime.initialize()
        result = await runtime.run_turn(args.prompt)
        print(json.dumps(asdict_result(result), ensure_ascii=False, indent=2))
        return 0 if result.status == "SUCCESS" else 1
    finally:
        await runtime.close()


def asdict_result(result) -> dict:
    return {
        "status": result.status,
        "text": result.text,
        "finished_reason": result.finished_reason,
        "model_calls": result.model_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "react_iterations": result.react_iterations,
        "duration_seconds": result.duration_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
