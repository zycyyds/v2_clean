"""Command-line entry point for the MIMIC reference-package workflow."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "output" / "logs"


def _setup_logging() -> Path:
    """Write CLI output to a run log while preserving terminal output."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio", "agentscope.model"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log_file = log_path.open("w", encoding="utf-8", buffering=1)

    class Tee:
        def __init__(self, original) -> None:
            self.original = original

        def write(self, value: str) -> None:
            self.original.write(value)
            log_file.write(value)

        def flush(self) -> None:
            self.original.flush()
            log_file.flush()

        def fileno(self):
            return self.original.fileno()

        def isatty(self) -> bool:
            return False

    sys.stdout = Tee(sys.__stdout__)
    sys.stderr = Tee(sys.__stderr__)
    return log_path


def _parse_split_ratios(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if len(parts) != 3:
        raise SystemExit("--split-ratios 必须是三个逗号分隔的数字，例如 0.6,0.2,0.2")
    try:
        ratios = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise SystemExit("--split-ratios 必须是数字") from exc
    if any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise SystemExit("--split-ratios 必须非负且总和大于零")
    return ratios


def _parse_split_counts(value: str) -> tuple[int, int, int | None]:
    parts = [part.strip().casefold() for part in str(value or "").split(",") if part.strip()]
    if len(parts) != 3:
        raise SystemExit("--split-counts 必须是三个值，例如 10,4000,5000")
    counts: list[int | None] = []
    for index, part in enumerate(parts):
        if part in {"rest", "remaining", "剩余", "剩下"}:
            if index != 2:
                raise SystemExit("rest 只能作为 --split-counts 的第三个值")
            counts.append(None)
            continue
        try:
            count = int(part)
        except ValueError as exc:
            raise SystemExit("--split-counts 必须是整数或第三位 rest") from exc
        if count < 0:
            raise SystemExit("--split-counts 不能为负数")
        counts.append(count)
    return int(counts[0] or 0), int(counts[1] or 0), counts[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIMIC ICU mortality reference-package workflow.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="",
        help="reference-guided-train-validate 的任务提示词。",
    )
    parser.add_argument(
        "--workflow",
        required=True,
        choices=(
            "prepare-reference-splits",
            "prepare-correction-split",
            "reference-guided-train-validate",
            "reference-guided-correct",
            "reference-test-evaluate",
        ),
        help="要运行的主链路阶段。",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=60,
        help="整个 validation Agent 生命周期的 ReAct 迭代上限。",
    )
    parser.add_argument("--round-limit", type=int, default=None, help="本次最多完成的成功晋升 loop 数。")
    parser.add_argument("--patience", type=int, default=2, help="连续有效但未提升的候选上限。")
    parser.add_argument("--max-attempts", type=int, default=100, help="本次候选尝试的安全上限。")
    parser.add_argument("--target-score", type=float, default=None, help="达到该综合分后提前冻结最佳包。")
    parser.add_argument("--dataset-split", default="", help="含 train/validation/test 的物理 split 目录。")
    parser.add_argument("--experiment-dir", default="", help="实验输出目录。")
    parser.add_argument("--adapter-script", default="", help="测试阶段要执行的已冻结 adapter 脚本。")
    parser.add_argument("--raw-root", default="", help="生成 split 时使用的原始 MIMIC 根目录。")
    parser.add_argument("--reference", default="", help="生成 split 时使用的单表 reference。")
    parser.add_argument("--reference-root", default="", help="生成 split 时使用的目录型 reference package。")
    parser.add_argument("--split-output", default="", help="生成的物理 split 输出目录。")
    parser.add_argument("--raw-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--decompress-gzip", action="store_true")
    parser.add_argument("--split-ratios", default="0.6,0.2,0.2")
    parser.add_argument("--split-counts", default="")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-key", default="")
    parser.add_argument("--source-archive", default="", help="纠错数据集来源压缩包。")
    parser.add_argument("--train-count", type=int, default=10, help="纠错标准示例 stay 数。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="显式恢复同一失败实验；仅 reference-guided-correct 使用。",
    )
    return parser.parse_args(argv)


def _required(args: argparse.Namespace, *names: str) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if not str(getattr(args, name) or "").strip()]
    if missing:
        raise SystemExit("缺少必填参数：" + ", ".join(missing))


def _print_result(result: dict) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    logging.info("=== final output ===\n%s", text)


def run_prepare_reference_splits(args: argparse.Namespace) -> None:
    _required(args, "raw_root", "split_output")
    if not str(args.reference or "").strip() and not str(args.reference_root or "").strip():
        raise SystemExit("prepare-reference-splits 需要 --reference 或 --reference-root")
    from workflow.reference_splits import ReferenceSplitConfig, prepare_reference_splits

    result = prepare_reference_splits(
        ReferenceSplitConfig(
            raw_root=args.raw_root,
            reference=args.reference,
            reference_root=args.reference_root or None,
            split_output=args.split_output,
            task_text=args.input,
            counts=_parse_split_counts(args.split_counts) if args.split_counts.strip() else (10, 4000, None),
            ratios=_parse_split_ratios(args.split_ratios),
            seed=args.split_seed,
            key_column=args.split_key,
            raw_mode=args.raw_mode,
            decompress_gzip=args.decompress_gzip,
        )
    )
    _print_result(result)


def run_reference_guided_train_validate(args: argparse.Namespace) -> None:
    _required(args, "dataset_split", "experiment_dir")
    if not args.input.strip():
        raise SystemExit("reference-guided-train-validate 需要任务提示词")
    from workflow.reference_guided import ReferenceGuidedConfig, ReferenceGuidedWorkflow

    result = ReferenceGuidedWorkflow(
        ReferenceGuidedConfig(
            dataset_split=args.dataset_split,
            experiment_dir=args.experiment_dir,
            task_text=args.input,
            round_limit=args.round_limit,
            patience=args.patience,
            max_attempts=args.max_attempts,
            target_score=args.target_score,
            max_iters=args.max_iters,
        )
    ).run_sync()
    _print_result(result)


def run_prepare_correction_split(args: argparse.Namespace) -> None:
    _required(args, "source_archive", "split_output")
    from workflow.correction_dataset import build_correction_dataset

    _print_result(
        build_correction_dataset(
            args.source_archive,
            args.split_output,
            train_count=args.train_count,
        )
    )


def run_reference_guided_correct(args: argparse.Namespace) -> None:
    _required(args, "dataset_split", "experiment_dir")
    if not args.input.strip():
        raise SystemExit("reference-guided-correct 需要任务提示词")
    from workflow.reference_correction import (
        ReferenceCorrectionConfig,
        ReferenceCorrectionWorkflow,
    )

    result = ReferenceCorrectionWorkflow(
        ReferenceCorrectionConfig(
            dataset_split=args.dataset_split,
            experiment_dir=args.experiment_dir,
            task_text=args.input,
            max_iters=args.max_iters,
            resume=args.resume,
        )
    ).run_sync()
    _print_result(result)


def run_reference_test_evaluate(args: argparse.Namespace) -> None:
    _required(args, "dataset_split", "experiment_dir", "adapter_script")
    from workflow.reference_test_stage import ReferenceTestStageConfig, run_reference_test_stage

    result = run_reference_test_stage(
        ReferenceTestStageConfig(
            dataset_split=args.dataset_split,
            experiment_dir=args.experiment_dir,
            adapter_script=args.adapter_script,
        )
    )
    _print_result(result)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    log_path = _setup_logging()
    print(f"日志保存至: {log_path}")
    runners = {
        "prepare-reference-splits": run_prepare_reference_splits,
        "prepare-correction-split": run_prepare_correction_split,
        "reference-guided-train-validate": run_reference_guided_train_validate,
        "reference-guided-correct": run_reference_guided_correct,
        "reference-test-evaluate": run_reference_test_evaluate,
    }
    runners[args.workflow](args)


if __name__ == "__main__":
    main()
