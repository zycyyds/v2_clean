"""CLI for automatic Validation followed by one-shot Test evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agent.pi_full_experiment import PiFullExperiment, PiFullExperimentConfig
from agent.pi_harness import PiValidationHarnessConfig
from agent.pi_harness_cli import _run_with_terminal_signals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Validation and automatically evaluate its frozen pipeline on Test.",
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--train-raw", required=True)
    parser.add_argument("--train-reference", required=True)
    parser.add_argument("--validation-raw", required=True)
    parser.add_argument("--validation-gold", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--target-score", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--skills-dir", action="append", default=[])
    parser.add_argument("--replay-timeout", type=float, default=1_800.0)
    parser.add_argument("--test-experiment", required=True)
    parser.add_argument("--test-raw", required=True)
    parser.add_argument("--test-gold", required=True)
    parser.add_argument("--test-replay-timeout", type=float, default=1_800.0)
    parser.add_argument("--test-scoring-timeout", type=float, default=3_600.0)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    prompt = Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    evaluation_manifest = Path(args.evaluation_manifest)
    validation = PiValidationHarnessConfig(
        project_root=Path(__file__).parents[1],
        experiment_dir=Path(args.experiment_dir),
        train_raw=Path(args.train_raw),
        train_reference=Path(args.train_reference),
        validation_raw=Path(args.validation_raw),
        validation_gold=Path(args.validation_gold),
        evaluation_manifest=evaluation_manifest,
        dataset_manifest=Path(args.dataset_manifest),
        max_rounds=args.max_rounds,
        patience=args.patience,
        target_score=args.target_score,
        max_iters=args.max_iters,
        skill_dirs=tuple(Path(item) for item in args.skills_dir),
        replay_timeout_seconds=args.replay_timeout,
    )
    experiment = PiFullExperiment(
        PiFullExperimentConfig(
            validation=validation,
            test_experiment=Path(args.test_experiment),
            test_raw=Path(args.test_raw),
            test_gold=Path(args.test_gold),
            evaluation_manifest=evaluation_manifest,
            test_replay_timeout_seconds=args.test_replay_timeout,
            test_scoring_timeout_seconds=args.test_scoring_timeout,
        ),
    )
    result, received_signal = await _run_with_terminal_signals(experiment.run(prompt))
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            _cli_payload(report, result.report_path),
            ensure_ascii=False,
            indent=2,
        ),
    )
    report_status = report.get("status")
    if received_signal is not None or report_status == "INTERRUPTED":
        return 130
    return 0 if report_status == "SUCCESS" else 1


def _cli_payload(report: dict, report_path: Path) -> dict:
    validation = report["validation"]
    test = report["test"]
    return {
        "status": report["status"],
        "phase": report["phase"],
        "validation": {
            "rounds": validation["rounds"],
            "best_score": validation["best_score"],
            "reproducible_score": validation["reproducible_score"],
        },
        "test": (
            {
                "status": test["status"],
                "score": test["score"],
                "test_execution_count": report["test_execution_count"],
                "frozen_snapshot_sha256": test["frozen_snapshot_sha256"],
            }
            if test is not None
            else None
        ),
        "report_path": str(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_run(parse_args(argv)))
    except KeyboardInterrupt:
        return 130
    except Exception:
        print(
            json.dumps(
                {
                    "status": "CLI_FAILED",
                    "error_code": "FULL_EXPERIMENT_EXCEPTION",
                },
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
