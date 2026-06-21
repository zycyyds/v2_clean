from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from agent.two_phase_agents import (
    ENGINEER_CODE_READ_ROOTS,
    V2_DIR,
    _make_model,
    create_engineer_agent,
    create_engineer_toolkit,
    create_explorer_agent,
)
from agent_tools import EngineerToolContext, ExplorerToolContext, register_explorer_atomic_tools
from agent_tools.report_tools import ExplorerReportTools
from lib.agent_artifacts import (
    clear_phase_context,
    init_phase_session,
    resolve_canonical_phase_handoff,
    set_phase_context,
)
from lib.agent_runtime import format_message_content, make_user_msg
from skills.analyze_gold_examples.skill import analyze_gold_examples_tool
from workflow.orchestrator import EngineerRoundContext, EngineerRoundResult, ExplorerRunContext
from workflow.dataset_profiles import build_dataset_profile_suggestions
from workflow.engineer_fallback import complete_engineer_result_package
from workflow.gold_analysis import validate_field_rule_completion, write_data_analysis_report
from workflow.skill_adapter import restore_bundle_variants


async def run_engineer_round(context: EngineerRoundContext, *, max_iters: int = 60) -> EngineerRoundResult:
    attempt_root = _engineer_attempt_root(context.candidate_bundle, context.round_index)
    report_paths = {
        "analysis_report": str(context.analysis_report_path),
        "field_extraction_rules": str(context.rules_path),
        "extraction_task_plan": str(context.task_plan_path),
        "explorer_run_record": str(context.explorer_run_record_path),
    }
    if context.public_feedback_path is not None:
        report_paths["public_feedback"] = str(context.public_feedback_path)
    task_text = _engineer_task_text(context)
    model_rule_updates: Path | None = None
    if max_iters > 0:
        model_root = attempt_root / "model_agent"
        phase = init_phase_session(model_root, "engineer")
        set_phase_context(attempt_root.name, model_root, "engineer", phase["phase_root"])
        try:
            created = create_engineer_agent(
                max_iters=min(max_iters, 24),
                task_text=task_text,
                explorer_phase_root=str(context.analysis_report_path.parent),
                report_paths=report_paths,
                split_mode="validation",
                return_context=True,
                bundle_managed_rules=True,
            )
            agent, tool_context = created
            restore_bundle_variants(context.active_bundle, tool_context.variant_root)
            try:
                response = await agent(make_user_msg(name="user", content=task_text))
                response_text = format_message_content(getattr(response, "content", "")).strip()
                (Path(phase["phase_root"]) / "agent_response.txt").write_text(response_text, encoding="utf-8")
            except Exception as exc:
                (Path(phase["phase_root"]) / "agent_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}",
                    encoding="utf-8",
                )
            model_rule_updates = _latest_named_file(Path(phase["phase_root"]), "field_rule_updates.json")
            package = _complete_result_package_paths(Path(phase["phase_root"]))
            if package is not None:
                return EngineerRoundResult(
                    package["manifest"],
                    package["mapping"],
                    tool_context.variant_root,
                    model_rule_updates,
                    package["skill_usage_report"],
                    package["task_execution"],
                )
        finally:
            clear_phase_context()

    fallback_root = attempt_root / "deterministic_completion"
    fallback_phase = init_phase_session(fallback_root, "engineer")
    set_phase_context(attempt_root.name, fallback_root, "engineer", fallback_phase["phase_root"])
    try:
        tool_context = EngineerToolContext.from_task(
            task_text=task_text,
            engineer_phase_root=fallback_phase["phase_root"],
            explorer_phase_root=str(context.analysis_report_path.parent),
            additional_read_roots=ENGINEER_CODE_READ_ROOTS,
            require_skill_plan=True,
            split_mode="validation",
            required_report_paths=report_paths,
        )
        _, manifest = create_engineer_toolkit(
            tool_context,
            bundle_managed_rules=True,
            task_text=task_text,
        )
        skill_names = {str(item.get("name")) for item in manifest if item.get("name")}
        restore_bundle_variants(context.active_bundle, tool_context.variant_root)
        package = complete_engineer_result_package(
            validation_raw=context.validation_raw,
            rules_path=context.rules_path,
            task_plan_path=context.task_plan_path,
            tool_context=tool_context,
            skill_names=skill_names,
            skills_root=V2_DIR / "skills",
        )
        return EngineerRoundResult(
            package["result_manifest"],
            package["target_field_mapping"],
            tool_context.variant_root,
            model_rule_updates,
            package["skill_usage_report"],
            package["task_execution"],
        )
    finally:
        clear_phase_context()


def _complete_result_package_paths(root: Path) -> dict[str, Path] | None:
    paths = {
        "manifest": _latest_named_file(root, "result_manifest.json"),
        "mapping": _latest_named_file(root, "target_field_mapping.json"),
        "skill_usage_report": _latest_named_file(root, "skill_usage_report.json"),
        "task_execution": _latest_named_file(root, "extraction_task_execution.json"),
    }
    if any(path is None for path in paths.values()):
        return None
    audit = json.loads(paths["skill_usage_report"].read_text(encoding="utf-8"))
    if audit.get("status") != "SUCCESS" or audit.get("unexplained_deviations") not in (None, []):
        return None
    return {name: path for name, path in paths.items() if path is not None}


def _engineer_attempt_root(candidate_bundle: Path, round_index: int) -> Path:
    """Allocate an isolated Agent run directory for each retry of one logical round."""
    experiment_dir = candidate_bundle.parent.parent
    state_path = experiment_dir / "experiment_state.json"
    attempt_count = 0
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        attempt_count = sum(
            1
            for item in state.get("round_attempts", [])
            if isinstance(item, dict) and int(item.get("round", -1)) == round_index
        )
    return (
        experiment_dir
        / "agent_runs"
        / f"round_{round_index:04d}"
        / f"attempt_{attempt_count + 1:04d}"
    )


async def run_explorer_analysis(context: ExplorerRunContext, *, max_iters: int = 40) -> dict[str, str]:
    run_root = context.experiment_dir / "agent_runs" / "explorer"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    try:
        existing = _canonical_explorer_artifacts(run_root)
        if existing and _rules_match_example_root(existing["field_extraction_rules"], context.train_examples):
            return _finalize_explorer_artifacts(
                context,
                phase,
                existing,
                dataset_profile="resumed",
            )
        locked = analyze_gold_examples_tool(
            task_text=context.task_text,
            gold_examples_path=str(context.train_examples),
        )
        locked_payload = json.loads(locked.content[0]["text"])
        if locked_payload.get("status") != "SUCCESS":
            raise RuntimeError(f"Deterministic gold analysis failed: {locked_payload.get('issues')}")
        locked_artifacts = locked_payload.get("artifacts") or {}
        rules_path = Path(str(locked_artifacts["field_extraction_rules"]))
        profile = build_dataset_profile_suggestions(rules_path, context.train_examples)
        if profile.get("updates"):
            updates = [
                {key: value for key, value in item.items() if key != "target_field_path"}
                for item in profile["updates"]
            ]
            profile_result = json.loads(
                ExplorerReportTools().publish_field_rule_updates(
                    str(rules_path),
                    json.dumps({"updates": updates}, ensure_ascii=False),
                ).content[0]["text"]
            )
            if profile_result.get("status") != "SUCCESS":
                raise RuntimeError(f"Dataset profile publication failed: {profile_result.get('issues')}")
            rules_path = Path(profile_result["artifacts"]["file_path"])
            task_plan_path = Path(profile_result["artifacts"]["task_plan_path"])
        else:
            task_plan_path = Path(str(locked_artifacts["extraction_task_plan"]))
        if max_iters <= 0:
            artifacts = _canonical_explorer_artifacts(run_root)
            if not artifacts:
                raise RuntimeError("Deterministic DataExplorer did not publish all required canonical artifacts")
            return _finalize_explorer_artifacts(
                context,
                phase,
                artifacts,
                dataset_profile=profile.get("dataset_profile", "generic"),
            )
        task_text = f"""{context.task_text}

【当前阶段硬边界】
你现在只执行 DataExplorer 阶段。用户所列 final_dataset、manifest、mapping、执行记录和 Skill 审计属于后续 FeatureEngineer，不得在本阶段创建，也不得把训练 Gold 值写成数据集。
本阶段唯一交付是完整字段规则、抽取任务计划、Explorer 执行记录和 data_analysis_report。publish_analysis_report 成功后立即结束。

训练示例目录：{context.train_examples}
完整叶子字段规则：{rules_path}
当前抽取任务计划：{task_plan_path}
数据集候选知识：{profile.get('dataset_profile', 'generic')}。候选已经过schema验证，但你仍需使用原子工具检查其语义、join和派生逻辑。
不得重新调用 analyze_gold_examples，不得缩减字段清单。需要修正规则时，调用 publish_field_rule_updates；最后发布补充后的 data_analysis_report.json。
"""
        explorer = create_explorer_agent(max_iters=min(max_iters, 24), task_text=task_text)
        response = await explorer(make_user_msg(name="user", content=task_text))
        response_text = format_message_content(getattr(response, "content", "")).strip()
        (Path(phase["phase_root"]) / "agent_response.txt").write_text(response_text, encoding="utf-8")
        artifacts = _canonical_explorer_artifacts(run_root)
        if not artifacts:
            raise RuntimeError("DataExplorer did not publish all required canonical artifacts")
        return _finalize_explorer_artifacts(
            context,
            phase,
            artifacts,
            dataset_profile=profile.get("dataset_profile", "generic"),
        )
    finally:
        clear_phase_context()


def _canonical_explorer_artifacts(run_root: Path) -> dict[str, str]:
    aliases = {
        "field_extraction_rules": "field_extraction_rules",
        "extraction_task_plan": "extraction_task_plan",
        "data_analysis_report": "analysis_report",
        "explorer_run_record": "explorer_run_record",
    }
    artifacts = {
        output_name: path
        for output_name, alias in aliases.items()
        if (path := resolve_canonical_phase_handoff(run_root, "explorer", alias))
    }
    return artifacts if set(artifacts) == set(aliases) else {}


def _rules_match_example_root(rules_path: str, examples_root: Path) -> bool:
    try:
        payload = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        recorded = Path(str(payload.get("example_root") or "")).expanduser().resolve()
        return recorded == examples_root.expanduser().resolve()
    except Exception:
        return False


def _finalize_explorer_artifacts(
    context: ExplorerRunContext,
    phase: dict[str, str],
    artifacts: dict[str, str],
    *,
    dataset_profile: str,
) -> dict[str, str]:
    completion = validate_field_rule_completion(
        artifacts["field_extraction_rules"],
        context.train_examples,
    )
    final_report_dir = Path(phase["artifacts_dir"]) / "final_analysis"
    final_report_dir.mkdir(parents=True, exist_ok=True)
    final_report_path = write_data_analysis_report(
        artifacts["field_extraction_rules"],
        final_report_dir / "data_analysis_report.json",
        base_report_path=artifacts["data_analysis_report"],
    )
    publish_report = json.loads(
        ExplorerReportTools().publish_analysis_report(file_path=final_report_path).content[0]["text"]
    )
    if publish_report.get("status") != "SUCCESS":
        raise RuntimeError(f"Final analysis report publication failed: {publish_report.get('issues')}")
    artifacts = dict(artifacts)
    artifacts["data_analysis_report"] = str(final_report_path)
    record_path = Path(artifacts["explorer_run_record"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["analysis_schema_version"] = 2
    record["dataset_profile"] = dataset_profile
    record["completion"] = completion
    record["agent_manifest_path"] = phase["manifest_path"]
    record["agent_response_path"] = str(Path(phase["phase_root"]) / "agent_response.txt")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifacts


async def enrich_public_feedback(
    public_feedback_path: Path,
    context: EngineerRoundContext,
    *,
    max_iters: int = 20,
) -> None:
    round_root = context.candidate_bundle.parent.parent / "evaluator_agent_runs" / f"round_{context.round_index:04d}"
    phase = init_phase_session(round_root, "evaluator")
    set_phase_context(round_root.name, round_root, "evaluator", phase["phase_root"])
    try:
        tool_context = ExplorerToolContext(
            engineer_phase_root=phase["phase_root"],
            read_roots=[context.validation_raw, public_feedback_path, context.rules_path],
        )
        toolkit = Toolkit()
        register_explorer_atomic_tools(toolkit, tool_context)
        model, formatter = _make_model(agent_key="evaluator")
        agent = ReActAgent(
            name="ValidationFeedbackAnalyst",
            sys_prompt=_EVALUATOR_PROMPT,
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=InMemoryMemory(),
            parallel_tool_calls=False,
            max_iters=max_iters,
            print_hint_msg=False,
        )
        prompt = (
            f"验证原始数据目录: {context.validation_raw}\n"
            f"字段规则: {context.rules_path}\n"
            f"确定性 public feedback: {public_feedback_path}\n"
            "读取这些文件并返回字段级修复建议 JSON。"
        )
        response = await agent(make_user_msg(name="user", content=prompt))
        text = format_message_content(getattr(response, "content", "")).strip()
        suggestions = _extract_json_object(text)
        _merge_public_suggestions(public_feedback_path, suggestions)
    finally:
        clear_phase_context()


def engineer_runner(max_iters: int = 60):
    return partial(run_engineer_round, max_iters=max_iters)


def evaluator_feedback_enricher(max_iters: int = 20):
    return partial(enrich_public_feedback, max_iters=max_iters)


def explorer_runner(max_iters: int = 40):
    return partial(run_explorer_analysis, max_iters=max_iters)


def _engineer_task_text(context: EngineerRoundContext) -> str:
    feedback = str(context.public_feedback_path) if context.public_feedback_path else "无（第一轮）"
    return f"""{context.task_text}

当前为验证集第 {context.round_index} 轮。你只能读取验证原始数据，不得访问验证 gold。

验证原始数据：{context.validation_raw}
验证来源路径规则：每个 case 的任务输入位于 `<validation_raw>/<case_id>/raw_mimic/<input_artifacts.path>`；例如 `structured/admissions.csv` 必须解析到 `raw_mimic/structured/admissions.csv`。
DataExplorer 总报告：{context.analysis_report_path}
完整叶子字段规则：{context.rules_path}
确定性细粒度任务计划：{context.task_plan_path}
Explorer 执行记录：{context.explorer_run_record_path}
当前 active extraction bundle：{context.active_bundle}
上一轮公开反馈：{feedback}

必须逐个执行 extraction_task_plan.json 中 required=true 的任务，并覆盖 field_extraction_rules.json 中全部可执行规则。优先复用现有 Skill 的步骤和工具；固定接口不足时，使用 inspect_skill 和 create_skill_variant 创建 adapter，不能修改原始 skills/ 或 lib/。

最终必须在当前 Engineer phase 内生成：
1. result_manifest.json：列出所有真实数据产物，每项包含 alias、path、case_id_column。
2. target_field_mapping.json：每个目标字段包含 target_field_id、artifact、source_column。
3. `final_dataset.csv` 与 `final_dataset.xlsx`：CSV每个case一行；Excel把规则中的 `target_categories` 原样传给 `export_gold_workbook`，按 Gold 顶层类别动态生成 Sheet，并包含 `_cases`、`_provenance`、`_unsupported` 控制 Sheet；不得写死类别名称。
4. 如果上一轮 public feedback 要求修改或新增规则，生成 field_rule_updates.json；新增字段必须来自 missing_rule 反馈，不得自行发明。
5. 调用 audit_skill_usage 并确保 skill_usage_report.json 的状态为 SUCCESS；任何 NEEDS_REPAIR 都不能结束本轮。
6. 每个 required extraction task 完成后调用 record_extraction_task；生成 extraction_task_execution.json，不能静默跳过字段。

必须读取并验证这些文件后才能结束。不得返回或猜测验证 gold 值。
"""


def _latest_named_file(root: Path, name: str) -> Path | None:
    matches = [path for path in root.rglob(name) if path.is_file()]
    return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Evaluator Agent response must be a JSON object")
    return value


def _merge_public_suggestions(path: Path, suggestions: dict[str, Any]) -> None:
    allowed_keys = {
        "target_field_id",
        "source_files",
        "source_columns",
        "join_keys",
        "derivation_logic",
        "recommended_action",
        "capability_type",
    }
    safe_items = []
    for item in suggestions.get("field_feedback", []):
        if isinstance(item, dict) and item.get("target_field_id"):
            safe_items.append({key: item[key] for key in allowed_keys if key in item})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluator_agent_suggestions"] = safe_items
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_EVALUATOR_PROMPT = """你是独立的验证反馈分析 Agent。你只能看到验证原始数据、字段规则和去值后的错误统计。
使用 Glob、Read、Grep、InspectDataFile 和 ExecuteAnalysisPython 检查失败字段可能对应的来源文件、列、join 和派生逻辑。
不得请求、推断或输出任何病例级 gold 值或预测值。只返回 JSON：
{"field_feedback":[{"target_field_id":"...","source_files":[],"source_columns":[],"join_keys":[],"derivation_logic":{},"recommended_action":"...","capability_type":"..."}]}
不要返回 Markdown 或额外说明。"""
