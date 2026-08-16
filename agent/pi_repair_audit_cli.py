"""CLI for Train-only Pi Agent repair synthesis and one-shot Test audit."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.pi_harness_cli import _run_with_terminal_signals
from agent.pi_repair_test import PiRepairTestConfig, PiRepairTestHarness
from agent.pi_train_repair import (
    PiTrainRepairConfig,
    PiTrainRepairHarness,
    default_repair_skill_dirs,
)
from workflow.pi_repair_audit import build_public_train_view


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a de-identified paired Train view, synthesize a frozen Pi repair pipeline, "
            "and audit raw repairs once on Internal Test without model access."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-view", help="Build the de-identified 20k/10k Train view.")
    build.add_argument("--dirty-raw", required=True)
    build.add_argument("--clean-raw", required=True)
    build.add_argument("--graph-dir", required=True)
    build.add_argument("--supervision-dir", required=True)
    build.add_argument("--paired-log", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--expected-dirty", type=int, default=20_000)
    build.add_argument("--expected-clean", type=int, default=10_000)
    build.add_argument("--expected-table-count", type=int, default=29)

    train = subparsers.add_parser("train", help="Run one persistent Train-only Pi Agent session.")
    _add_train_arguments(train)

    test = subparsers.add_parser("test", help="Run one frozen pipeline Internal Test audit.")
    _add_test_arguments(test)

    full = subparsers.add_parser(
        "full", help="Run Train-only synthesis and then exactly one Internal Test audit."
    )
    _add_train_arguments(full)
    _add_test_arguments(full, include_train_experiment=False)
    return parser.parse_args(argv)


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=str(Path(__file__).parents[1]))
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--public-train-root", required=True)
    parser.add_argument("--train-gold-log", required=True)
    parser.add_argument("--train-row-gold-log", required=True)
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--max-repair-turns", type=int, default=2)
    parser.add_argument("--skill-dir", action="append", default=[])
    parser.add_argument("--replay-timeout", type=float, default=1_800.0)


def _add_test_arguments(
    parser: argparse.ArgumentParser, *, include_train_experiment: bool = True
) -> None:
    if include_train_experiment:
        parser.add_argument("--project-root", default=str(Path(__file__).parents[1]))
        parser.add_argument("--train-experiment", required=True)
    parser.add_argument("--test-experiment", required=True)
    parser.add_argument("--test-raw", required=True)
    parser.add_argument("--test-gold-log", required=True)
    parser.add_argument("--test-replay-timeout", type=float, default=1_800.0)


def _train_config(args: argparse.Namespace) -> PiTrainRepairConfig:
    project = Path(args.project_root)
    skills = (
        tuple(Path(item) for item in args.skill_dir)
        if args.skill_dir
        else default_repair_skill_dirs(project)
    )
    return PiTrainRepairConfig(
        project_root=project,
        experiment_dir=Path(args.experiment_dir),
        public_train_root=Path(args.public_train_root),
        train_gold_log=Path(args.train_gold_log),
        train_row_gold_log=Path(args.train_row_gold_log),
        max_iters=args.max_iters,
        max_repair_turns=args.max_repair_turns,
        skill_dirs=skills,
        replay_timeout_seconds=args.replay_timeout,
    )


def _test_config(
    args: argparse.Namespace, *, train_experiment: Path | None = None
) -> PiRepairTestConfig:
    return PiRepairTestConfig(
        project_root=Path(args.project_root),
        train_experiment=train_experiment or Path(args.train_experiment),
        test_experiment=Path(args.test_experiment),
        test_raw=Path(args.test_raw),
        test_gold_log=Path(args.test_gold_log),
        replay_timeout_seconds=args.test_replay_timeout,
    )


def _prompt(args: argparse.Namespace) -> str:
    if not args.prompt_file:
        return (
            "Learn a deterministic, evidence-backed raw error detector and repair pipeline from "
            "the paired Train files. Do not build downstream formatted outputs."
        )
    return Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")


async def _run_async(args: argparse.Namespace) -> tuple[int, dict]:
    if args.command == "train":
        result, signal = await _run_with_terminal_signals(
            PiTrainRepairHarness(_train_config(args)).run(_prompt(args))
        )
        payload = _train_payload(result)
        code = (
            128 + int(signal)
            if signal
            else (0 if result.status == "SUCCESS_REPRODUCIBLE" else 1)
        )
        return code, payload
    if args.command == "test":
        result, signal = await _run_with_terminal_signals(
            PiRepairTestHarness(_test_config(args)).run()
        )
        payload = _test_payload(result)
        code = 128 + int(signal) if signal else (0 if result.status == "SUCCESS" else 1)
        return code, payload
    if args.command == "full":
        train_result, signal = await _run_with_terminal_signals(
            PiTrainRepairHarness(_train_config(args)).run(_prompt(args))
        )
        payload: dict = {"train": _train_payload(train_result), "test": {}}
        if signal:
            return 128 + int(signal), payload
        if train_result.status != "SUCCESS_REPRODUCIBLE":
            return 1, payload
        test_result, signal = await _run_with_terminal_signals(
            PiRepairTestHarness(
                _test_config(args, train_experiment=Path(args.experiment_dir))
            ).run()
        )
        payload["test"] = _test_payload(test_result)
        if signal:
            return 128 + int(signal), payload
        return (0 if test_result.status == "SUCCESS" else 1), payload
    raise AssertionError(args.command)


def _train_payload(result) -> dict:
    return {
        "status": result.status,
        "repair_turns": result.repair_turns,
        "frozen_snapshot": str(result.frozen_snapshot),
        "frozen_snapshot_sha256": result.frozen_snapshot_sha256,
        "output_root": str(result.output_root) if result.output_root else "",
        "train_report": str(result.train_report) if result.train_report else "",
    }


def _test_payload(result) -> dict:
    return {
        "status": result.status,
        "phase": result.phase,
        "test_execution_count": result.test_execution_count,
        "frozen_snapshot_sha256": result.frozen_snapshot_sha256,
        "output_root": str(result.output_root) if result.output_root else "",
        "score_report": str(result.score_report) if result.score_report else "",
        "replay_status": result.replay_status,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-view":
        result = build_public_train_view(
            dirty_raw=args.dirty_raw,
            clean_raw=args.clean_raw,
            graph_dir=args.graph_dir,
            supervision_dir=args.supervision_dir,
            paired_log=args.paired_log,
            output_dir=args.output_dir,
            expected_dirty=args.expected_dirty,
            expected_clean=args.expected_clean,
            expected_table_count=args.expected_table_count,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    try:
        code, payload = asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        return 130
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
