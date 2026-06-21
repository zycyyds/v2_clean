"""Entry point for single-agent, two-phase, and train-validation workflows.

Replaces ``main_orchestrator/main_orchestrator.py``. No router, no state
machine, no handoffs. Single ReAct agent + 37 skills.

Usage:
    python v2/main.py                          # interactive
    python v2/main.py "<your free text>"       # one-shot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

V2_DIR = Path(__file__).resolve().parent
LIB_DIR = V2_DIR / "lib"
LOG_DIR = V2_DIR / "output" / "logs"
for path in (V2_DIR, LIB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_runtime import format_message_content, make_user_msg
from agent_artifacts import (
    clear_phase_context,
    create_run_session,
    init_phase_session,
    resolve_canonical_phase_handoff,
    set_phase_context,
)

from agent.planner_agent import create_planner_agent


def _setup_logging() -> Path:
    """Tee both stdout and stderr into a log file, silence HTTP noise."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{ts}.log"

    # silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio",
                  "agentscope.model", "agentscope.utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        """Write to both original stream and log file."""
        def __init__(self, orig):
            self._orig = orig
        def write(self, s):
            self._orig.write(s)
            log_file.write(s)
            log_file.flush()
        def flush(self):
            self._orig.flush()
            log_file.flush()
        def fileno(self):
            return self._orig.fileno()
        def isatty(self):
            return False

    sys.stdout = _Tee(sys.__stdout__)
    sys.stderr = _Tee(sys.__stderr__)

    return log_path


def _patch_react_logging(log_file) -> None:
    pass  # no longer needed — stdout tee handles everything

EXIT_COMMANDS = {"exit", "quit", "q", "退出", "结束"}
HELP_COMMANDS = {"help", "?", "帮助"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset processing workflows via ReAct agent(s).",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="",
        help="Free-form user task (path + goal). Omit for interactive mode.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=30,
        help="ReAct loop budget per agent.",
    )
    parser.add_argument(
        "--two-phase",
        action="store_true",
        help="Use two-phase mode: DataExplorer → FeatureEngineer.",
    )
    parser.add_argument(
        "--workflow",
        choices=("single", "two-phase", "train-validate"),
        default="",
        help="Select an explicit execution workflow.",
    )
    parser.add_argument("--train-examples", default="", help="Training raw/gold example directory.")
    parser.add_argument("--validation-raw", default="", help="Validation raw-data path visible to Engineer.")
    parser.add_argument("--validation-gold", default="", help="Hidden validation gold path visible only to Evaluator.")
    parser.add_argument("--experiment-dir", default="", help="Experiment-scoped bundle and evaluation directory.")
    parser.add_argument(
        "--round-limit",
        type=int,
        default=None,
        help="Stop after this many validation rounds in the current invocation; resume with the same experiment directory.",
    )
    parser.add_argument(
        "--enable-feedback-agent",
        action="store_true",
        help="Use the optional Evaluator Agent to enrich deterministic public feedback. Disabled by default for stable loops.",
    )
    parser.add_argument(
        "--enable-explorer-agent",
        action="store_true",
        help="Use the optional DataExplorer Agent to enrich deterministic field rules. Disabled by default for stable loops.",
    )
    parser.add_argument(
        "--enable-engineer-agent",
        action="store_true",
        help="Use the optional FeatureEngineer code Agent before deterministic completion. Disabled by default for stable loops.",
    )
    parser.add_argument(
        "--show-manifest",
        action="store_true",
        help="Print loaded skill manifest and exit.",
    )
    return parser.parse_args(argv)


def _should_use_two_phase(user_input: str) -> bool:
    text = str(user_input or "").lower()
    dataset_keywords = ("数据集", "训练集", "dataset", "ml dataset")
    build_keywords = ("构造", "生成", "build", "create")
    diagnosis_keywords = ("诊断", "预测", "分类", "肝病", "疾病")
    return any(k in text for k in dataset_keywords) and (
        any(k in text for k in build_keywords) or any(k in text for k in diagnosis_keywords)
    )


async def run_one_shot(user_input: str, max_iters: int) -> str:
    session = create_run_session(mode="planner")
    phase = init_phase_session(session["run_root"], "planner")
    set_phase_context(session["run_id"], session["run_root"], "planner", phase["phase_root"])
    try:
        agent, _manifest = create_planner_agent(max_iters=max_iters)
        msg = make_user_msg(name="user", content=user_input)
        response = await agent(msg)
        return format_message_content(getattr(response, "content", "")).strip()
    finally:
        clear_phase_context()


async def run_two_phase(user_input: str, max_iters: int) -> str:
    from agent.two_phase_agents import (
        create_explorer_agent,
        create_engineer_agent,
        infer_split_mode,
    )

    session = create_run_session(mode="two_phase")
    explorer_phase = init_phase_session(session["run_root"], "explorer")
    engineer_phase = init_phase_session(session["run_root"], "engineer")

    # Phase 1: DataExplorer
    print("\n" + "="*60)
    print("【阶段1】DataExplorer — 深度数据探索")
    print("="*60)
    set_phase_context(session["run_id"], session["run_root"], "explorer", explorer_phase["phase_root"])
    try:
        explorer = create_explorer_agent(max_iters=max_iters, task_text=user_input)
        explore_msg = make_user_msg(
            name="user",
            content=(
                f"{user_input}\n\n"
                f"请先完整探索所有数据表，输出数据分析报告。当前 phase 输出目录: {explorer_phase['phase_root']}"
            ),
        )
        explore_resp = await explorer(explore_msg)
        explore_result = format_message_content(getattr(explore_resp, "content", "")).strip()
    finally:
        clear_phase_context()
    print("\n--- DataExplorer 完成 ---")
    print(explore_result)

    aliases = (
        "analysis_report",
        "field_extraction_rules",
        "extraction_task_plan",
        "explorer_run_record",
        "gold_field_provenance_report",
        "planner_extraction_brief",
        "learned_rules",
    )
    report_paths = {
        alias: path
        for alias in aliases
        if (path := resolve_canonical_phase_handoff(session["run_root"], "explorer", alias))
    }
    report_path = report_paths.get("analysis_report")
    if not report_path:
        raise RuntimeError(
            "DataExplorer 未调用 publish_analysis_report，缺少 canonical analysis_report handoff；"
            "FeatureEngineer 未启动。",
        )
    split_mode = infer_split_mode(user_input)
    handoff_lines = "\n".join(f"- {alias}: {path}" for alias, path in report_paths.items())
    engineer_context = (
        "以下是 DataExplorer 已发布的 canonical handoff，必须逐一用 Read 读取并在 "
        "plan_skill_usage 中记录指纹：\n"
        f"{handoff_lines}\n"
        f"当前数据划分模式: {split_mode}\n"
        "只能依据这些已发布文件制定抽取计划。"
    )

    # Phase 2: FeatureEngineer
    print("\n" + "="*60)
    print("【阶段2】FeatureEngineer — 特征工程")
    print("="*60)
    set_phase_context(session["run_id"], session["run_root"], "engineer", engineer_phase["phase_root"])
    try:
        engineer = create_engineer_agent(
            max_iters=max_iters,
            task_text=user_input,
            explorer_phase_root=explorer_phase["phase_root"],
            report_paths=report_paths,
            split_mode=split_mode,
        )
        engineer_msg = make_user_msg(
            name="user",
            content=(
                f"{user_input}\n\n{engineer_context}\n\n"
                f"当前 phase 输出目录: {engineer_phase['phase_root']}"
            ),
        )
        engineer_resp = await engineer(engineer_msg)
        engineer_result = format_message_content(getattr(engineer_resp, "content", "")).strip()
    finally:
        clear_phase_context()
    print("\n--- FeatureEngineer 完成 ---")
    return engineer_result



def _print_help() -> None:
    print(
        "\n输入格式示例：\n"
        "  请处理 /abs/path/to/mimic-mini，然后任务裁剪目标设为肝癌诊断\n"
        "  从 /abs/path/to/program/output/step1_results 续跑，目标是死亡预测\n"
        "  /abs/path/to/raw 只跑 step1\n"
        "\n命令: help / ?（帮助）  exit / quit / 退出（退出）\n",
    )


async def interactive_loop(max_iters: int) -> None:
    print("[v2] PlannerReActAgent 交互模式。输入 help 看示例，exit 退出。")
    while True:
        try:
            user_input = input("\nPlanner> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[v2] bye.")
            return
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS or user_input in EXIT_COMMANDS:
            print("[v2] bye.")
            return
        if user_input.lower() in HELP_COMMANDS or user_input in HELP_COMMANDS:
            _print_help()
            continue
        use_two_phase = _should_use_two_phase(user_input)
        if use_two_phase:
            print("[v2] 检测到数据集构造任务，自动切换为 two-phase 模式。")
        try:
            if use_two_phase:
                text = await run_two_phase(user_input, max_iters=max_iters)
            else:
                text = await run_one_shot(user_input, max_iters=max_iters)
        except Exception as exc:
            print(f"[v2][error] {exc}")
            continue
        print("\n--- Planner final ---")
        print(text)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.show_manifest:
        from skills._registry import load_all_skills, render_skill_manifest
        from agentscope.tool import Toolkit

        manifest = load_all_skills(Toolkit())
        print(render_skill_manifest(manifest))
        return

    log_path = _setup_logging()
    print(f"[v2] 日志保存至: {log_path}")

    if args.workflow == "train-validate":
        required = {
            "--train-examples": args.train_examples,
            "--validation-raw": args.validation_raw,
            "--validation-gold": args.validation_gold,
            "--experiment-dir": args.experiment_dir,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise SystemExit("train-validate requires: " + ", ".join(missing))
        if not str(args.input or "").strip():
            raise SystemExit("train-validate requires a task prompt")
        from workflow.agent_runners import engineer_runner, evaluator_feedback_enricher, explorer_runner
        from workflow.evaluator import create_embedding_semantic_scorer
        from workflow.orchestrator import TrainValidateConfig, TrainValidateWorkflow

        workflow = TrainValidateWorkflow(
            TrainValidateConfig(
                train_examples=args.train_examples,
                validation_raw=args.validation_raw,
                validation_gold=args.validation_gold,
                experiment_dir=args.experiment_dir,
                task_text=args.input,
                round_limit=args.round_limit,
            ),
            engineer_runner=engineer_runner(max_iters=args.max_iters if args.enable_engineer_agent else 0),
            feedback_enricher=(
                evaluator_feedback_enricher(max_iters=min(args.max_iters, 20))
                if args.enable_feedback_agent
                else None
            ),
            explorer_runner=explorer_runner(
                max_iters=min(args.max_iters, 40) if args.enable_explorer_agent else 0
            ),
            semantic_scorer=create_embedding_semantic_scorer(),
        )
        result = workflow.run_sync()
        text = json.dumps(result, ensure_ascii=False, indent=2)
        print(text)
        logging.info("=== final output ===\n%s", text)
        return

    if not str(args.input or "").strip():
        asyncio.run(interactive_loop(max_iters=args.max_iters))
        return

    use_two_phase = args.workflow == "two-phase" or args.two_phase or (
        args.workflow != "single" and _should_use_two_phase(args.input)
    )
    if use_two_phase and not args.two_phase:
        print("[v2] 检测到数据集构造任务，自动切换为 two-phase 模式。")

    if use_two_phase:
        text = asyncio.run(run_two_phase(args.input, max_iters=args.max_iters))
    else:
        text = asyncio.run(run_one_shot(args.input, max_iters=args.max_iters))

    print(text)
    logging.info("=== final output ===\n%s", text)


if __name__ == "__main__":
    main()
