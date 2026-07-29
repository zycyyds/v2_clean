from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO

from agent.pi_runtime import PiAgentConfig, PiAgentRuntime


def require_sandbox_launcher() -> None:
    if os.environ.get("PI_HARNESS_SANDBOX") != "1":
        raise RuntimeError("pi_worker must be started by the trusted Harness launcher")


def _write_message(writer: TextIO, payload: dict[str, Any]) -> None:
    writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
    writer.flush()


def _result_payload(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


async def serve_worker(
    runtime: Any,
    *,
    reader: TextIO = sys.stdin,
    writer: TextIO = sys.stdout,
) -> None:
    await runtime.initialize()
    _write_message(writer, {"event": "ready"})
    closed = False
    try:
        for raw_line in reader:
            if not raw_line.strip():
                continue
            request_id: Any = None
            try:
                request = json.loads(raw_line)
                request_id = request.get("id")
                command = request.get("command")
                if command == "run_turn":
                    prompt = str(request.get("prompt") or "")
                    result = await runtime.run_turn(prompt)
                    _write_message(
                        writer,
                        {"id": request_id, "result": _result_payload(result)},
                    )
                elif command == "clear_file_cache":
                    if runtime.agent is None:
                        raise RuntimeError("runtime is not initialized")
                    await runtime.agent.state.tool_context.clean_file_cache()
                    _write_message(
                        writer,
                        {"id": request_id, "result": {"status": "SUCCESS"}},
                    )
                elif command == "close":
                    await runtime.close()
                    closed = True
                    _write_message(
                        writer,
                        {"id": request_id, "result": {"status": "CLOSED"}},
                    )
                    break
                else:
                    raise ValueError(f"unsupported worker command: {command}")
            except Exception as exc:
                _write_message(
                    writer,
                    {
                        "id": request_id,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    },
                )
    finally:
        if not closed:
            await runtime.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent JSONL worker for PiAgentRuntime.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--skills-dir", action="append", default=[])
    parser.add_argument("--tool-profile", choices=("full", "test_declaration"), default="full")
    parser.add_argument("--read-root", action="append", default=[])
    parser.add_argument("--runner-spec")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    runtime = PiAgentRuntime(
        PiAgentConfig(
            workdir=Path(args.workdir),
            max_iters=args.max_iters,
            skill_dirs=tuple(Path(item) for item in args.skills_dir),
            tool_profile=args.tool_profile,
            read_roots=tuple(Path(item) for item in args.read_root),
            runner_spec_path=Path(args.runner_spec) if args.runner_spec else None,
        ),
        output=sys.stderr,
    )
    await serve_worker(runtime)


def main(argv: list[str] | None = None) -> int:
    require_sandbox_launcher()
    args = parse_args(argv)
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
