"""Agent 4 standalone entrypoint for Step4 task-oriented column clipping.

The executable Step4 logic lives in ``step-4`` and is shared with the main
orchestrator handoff path. This module only keeps Agent4-facing defaults and CLI.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
EXECUTION_DIR = ORCHESTRATOR_DIR / "execution"
STEP4_DIR = PROJECT_ROOT / "step-4"
for path in (PROJECT_ROOT, ORCHESTRATOR_DIR, EXECUTION_DIR, STEP4_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step4_runtime import get_default_output_root, resolve_step4_task_text  # noqa: E402
from supervisors import Step4Supervisor  # noqa: E402


def get_agent_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_project_root() -> str:
    return str(PROJECT_ROOT)


def get_default_input_dir() -> str:
    return os.path.join(get_agent_root(), "data_input")


def get_default_output_dir() -> str:
    return get_default_output_root()


def run_task_oriented_clipping(
    input_path: str | None = None,
    output_dir: str | None = None,
    task_text: str | None = None,
) -> dict:
    supervisor = Step4Supervisor(output_root=output_dir or get_default_output_dir())
    result = asyncio.run(
        supervisor.run_from_input(
            input_path=input_path or get_default_input_dir(),
            task_text=task_text,
        )
    )
    if result.status.value != "SUCCESS":
        raise RuntimeError("; ".join(result.issues) or result.summary)
    return {
        "filtered_csv_path": result.artifacts["filtered_csv_path"],
        "selection_report_path": result.artifacts["selection_report_path"],
        "next_input_csv": result.artifacts.get("next_input_csv"),
        "next_selection_report": result.artifacts.get("next_selection_report"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent_4 task-oriented column clipping.")
    parser.add_argument(
        "--input",
        "-i",
        default=get_default_input_dir(),
        help=f"输入 CSV/XLSX 文件或目录（默认: {get_default_input_dir()}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=get_default_output_dir(),
        help=f"输出目录（默认: {get_default_output_dir()}）",
    )
    parser.add_argument(
        "--task",
        "-t",
        default=resolve_step4_task_text()["task_text"],
        help="任务文本；也可通过 AGENT4_TASK_TEXT 环境变量设置。",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    result = run_task_oriented_clipping(
        input_path=args.input,
        output_dir=args.output,
        task_text=args.task,
    )
    print("Agent4 task clipping completed.")
    print(f"Filtered CSV: {result['filtered_csv_path']}")
    print(f"Selection Report: {result['selection_report_path']}")
    return result


if __name__ == "__main__":
    main()
