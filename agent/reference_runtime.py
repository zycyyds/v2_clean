"""Runtime construction for the retained reference-guided code agent."""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO

import yaml
from agentscope.agent import Agent, ContextConfig, ReActConfig
from agentscope.event import (
    ExceedMaxItersEvent,
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import AssistantMsg, Msg
from agentscope.model import OpenAIChatModel
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.skill import LocalSkillLoader
from agentscope.state import AgentState
from agentscope.tool import ToolBase, Toolkit
from agentscope.workspace import LocalWorkspace
from pydantic import BaseModel, Field

from agent_tools.context import EngineerToolContext
from agent_tools.agentscope2_tools import build_reference_tools
from lib.agent_runtime import create_openai_model_and_formatter


V2_DIR = Path(__file__).resolve().parent.parent
DATA_CLEANING_AGENT_PIPELINE_SKILLS = {
    "pipeline_build_cohort",
    "pipeline_filter_disease_cohort",
    "pipeline_extract_diag_features",
    "pipeline_extract_proc_features",
    "pipeline_extract_lab_features",
    "pipeline_extract_med_features",
    "pipeline_extract_icu_event_features",
    "pipeline_clean_feature_table",
    "pipeline_assemble_reference_package",
}
DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS = {
    "correction_intra_table_errors",
    "correction_entity_alignment_errors",
    "correction_cross_table_errors",
    "correction_task_oriented_errors",
}
AGENTSCOPE_VERSION = "2.0.4.post1"
LOGGER = logging.getLogger(__name__)


class ReferenceCompressionSummary(BaseModel):
    """Evidence-grounded continuation state retained across compression."""

    task_overview: str = Field(description="用户目标、成功条件和当前任务边界。")
    current_state: str = Field(description="已经完成的工作、当前阶段和最近一次有效结果。")
    verified_rules_and_evidence: str = Field(
        description="已经由工具输出、训练回放或文件证据验证的规则；包含证据路径或步骤。",
    )
    rejected_hypotheses_and_counterexamples: str = Field(
        description="已经否定的假设、反例及不得重复尝试的原因。",
    )
    unresolved_work: str = Field(description="尚未解决的问题、歧义和当前阻塞点。")
    artifacts_and_paths: str = Field(
        description="需要继续使用的脚本、报告、结果包和其他真实文件路径。",
    )
    next_steps: str = Field(description="按优先级排列的具体下一步，不重新开始已完成分析。")
    constraints_to_preserve: str = Field(
        description="数据隔离、不可读取路径、输出格式和其他必须持续遵守的约束。",
    )


REFERENCE_COMPRESSION_PROMPT = """\
<system-hint>
上下文即将压缩。请生成一份可以让 Data Cleaning Agent 无损继续工作的结构化续接摘要。

要求：
1. 只保留对后续执行有用的事实，不复述冗长工具输出。
2. 明确区分已验证规则、被否定假设、反例、未解决问题和下一步。
3. 已验证规则必须来自当前对话中的工具输出、训练回放或真实文件；尽量保留证据路径、步骤编号和关键计数。
4. 不得在压缩时发明新规则、补全缺失证据或把待验证假设写成结论。
5. 保留当前正式脚本、结果包、报告和工作目录的准确路径。
6. 保留用户原始目标、数据访问限制和禁止读取历史实验或隐藏答案等约束。
7. 如果前一份压缩摘要已经包含仍然有效的结论，继续保留；只有出现新证据时才能修正或删除。
8. 摘要必须足以避免重新扫描全部数据或重复已经失败的诊断。
</system-hint>
"""


REFERENCE_COMPRESSION_TEMPLATE = """\
<system-info>
# 任务目标与边界
{task_overview}

# 当前状态
{current_state}

# 已验证规则与证据
{verified_rules_and_evidence}

# 已否定假设与反例
{rejected_hypotheses_and_counterexamples}

# 未解决工作
{unresolved_work}

# 关键产物与路径
{artifacts_and_paths}

# 下一步
{next_steps}

# 必须保留的约束
{constraints_to_preserve}
</system-info>
"""


def make_reference_model() -> tuple[OpenAIChatModel, OpenAIChatFormatter]:
    require_agentscope_version()
    return create_openai_model_and_formatter("react_planner", "gpt-4.1-mini")


def require_agentscope_version() -> None:
    installed = version("agentscope")
    if installed != AGENTSCOPE_VERSION:
        raise RuntimeError(
            f"Data Cleaning Agent requires agentscope=={AGENTSCOPE_VERSION}; installed={installed}",
        )


def create_reference_context_config(model_name: str = "") -> ContextConfig:
    trigger_ratio = 0.8999 if "minimax" in model_name.lower() or "m3" in model_name.lower() else 0.8
    return ContextConfig(
        trigger_ratio=trigger_ratio,
        reserve_ratio=0.1,
        compression_prompt=REFERENCE_COMPRESSION_PROMPT,
        summary_template=REFERENCE_COMPRESSION_TEMPLATE,
        summary_schema=ReferenceCompressionSummary.model_json_schema(),
        tool_result_limit=4_000,
    )


def create_reference_react_config(max_iters: int) -> ReActConfig:
    return ReActConfig(
        max_iters=max_iters,
        stop_on_reject=False,
        interruption_raise_cancelled_error=True,
    )


def _skill_manifest(names: set[str]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for name in sorted(names):
        skill_dir = V2_DIR / "skills" / name
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        if not text.startswith("---\n") or "\n---\n" not in text[4:]:
            raise RuntimeError(f"Skill frontmatter is invalid: {skill_dir / 'SKILL.md'}")
        metadata = yaml.safe_load(text.split("\n---\n", 1)[0][4:]) or {}
        if metadata.get("name") != name:
            raise RuntimeError(f"Skill name is invalid: {skill_dir / 'SKILL.md'}")
        manifest.append(
            {
                "name": name,
                "dir": name,
                "description": str(metadata.get("description") or "").strip(),
                "tool_names": [],
                "agentscope_agent_skill": True,
            },
        )
    return manifest


def reference_code_read_roots(
    *,
    include_pipeline_skills: bool = True,
    include_error_view_skills: bool = False,
) -> tuple[Path, ...]:
    names: set[str] = set()
    if include_pipeline_skills:
        names.update(DATA_CLEANING_AGENT_PIPELINE_SKILLS)
    if include_error_view_skills:
        names.update(DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS)
    return (
        V2_DIR / "workflow",
        V2_DIR / "lib",
        *(V2_DIR / "skills" / name for name in sorted(names)),
    )


def reference_code_denied_read_roots(
    *,
    include_pipeline_skills: bool = True,
    include_error_view_skills: bool = False,
) -> tuple[Path, ...]:
    included: set[str] = set()
    if include_pipeline_skills:
        included.update(DATA_CLEANING_AGENT_PIPELINE_SKILLS)
    if include_error_view_skills:
        included.update(DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS)
    excluded = (
        DATA_CLEANING_AGENT_PIPELINE_SKILLS
        | DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS
    ) - included
    return tuple(V2_DIR / "skills" / name for name in sorted(excluded))


def error_view_skill_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for name in sorted(DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS):
        path = V2_DIR / "skills" / name / "SKILL.md"
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def create_reference_toolkit(
    context: EngineerToolContext,
    *,
    include_pipeline_skills: bool = True,
    include_error_view_skills: bool = False,
    include_quality_tools: bool = False,
    codegraph_tool: ToolBase | None = None,
) -> tuple[Toolkit, list[dict]]:
    """Build the per-attempt native AgentScope 2.x Toolkit."""
    require_agentscope_version()
    pipeline_manifest = (
        _skill_manifest(DATA_CLEANING_AGENT_PIPELINE_SKILLS)
        if include_pipeline_skills
        else []
    )
    error_view_manifest = (
        _skill_manifest(DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS)
        if include_error_view_skills
        else []
    )
    manifest = [*pipeline_manifest, *error_view_manifest]
    context.register_executable_skills(
        str(item.get("name")) for item in pipeline_manifest
    )
    tools = build_reference_tools(
        context,
        include_quality_tools=include_quality_tools,
    )
    if codegraph_tool is not None:
        tools.append(codegraph_tool)
    options: dict[str, Any] = {"tools": tools}
    if manifest:
        options["skills_or_loaders"] = [
            LocalSkillLoader(
                directory=str(V2_DIR / "skills" / str(item["name"])),
                scan_subdir=False,
            )
            for item in manifest
        ]
    toolkit = Toolkit(**options)
    return toolkit, manifest


async def create_reference_agent(
    *,
    name: str,
    system_prompt: str,
    toolkit: Toolkit,
    workspace_dir: str | Path,
    max_iters: int,
) -> tuple[Agent, LocalWorkspace]:
    model, _ = make_reference_model()
    workspace = LocalWorkspace(
        workdir=str(Path(workspace_dir).expanduser().resolve()),
        instructions=(
            "<workspace>Persistent context and oversized tool results are stored under {workdir}. "
            "Filesystem permissions are enforced by the provided tools.</workspace>"
        ),
    )
    await workspace.initialize()
    state = AgentState(
        permission_context=PermissionContext(mode=PermissionMode.BYPASS),
    )
    agent = Agent(
        name=name,
        system_prompt=system_prompt,
        model=model,
        toolkit=toolkit,
        state=state,
        offloader=workspace,
        context_config=create_reference_context_config(str(model.model)),
        react_config=create_reference_react_config(max_iters),
    )
    return agent, workspace


async def close_reference_agent(
    agent: Agent | None,
    workspace: LocalWorkspace | None,
) -> None:
    """Release attempt resources before the owning event loop is closed."""
    try:
        if workspace is not None:
            await workspace.close()
    finally:
        model = getattr(agent, "model", None) if agent is not None else None
        closer = getattr(model, "aclose", None) or getattr(model, "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            LOGGER.warning("Failed to close Data Cleaning Agent model client: %s", exc)


async def stream_reference_agent_reply(
    agent: Agent,
    inputs: Any,
    *,
    output: TextIO | None = None,
    skill_usage_report_path: str | Path | None = None,
    exposed_pipeline_skills: set[str] | frozenset[str] = frozenset(),
    exposed_error_view_skills: set[str] | frozenset[str] = frozenset(),
) -> Msg:
    """Run one native reply while rendering observable events to the terminal."""
    stream = output or sys.stdout
    text_blocks: dict[str, str] = {}
    text_block_order: list[str] = []
    skill_calls: list[dict[str, str]] = []
    pending_skill_calls: dict[str, dict[str, str]] = {}
    stream_status = "RUNNING"
    started_monotonic = time.monotonic()
    model_calls = 0
    input_tokens = 0
    output_tokens = 0

    def line(text: str = "") -> None:
        print(text, file=stream, flush=True)

    def value(item: Any) -> str:
        return str(getattr(item, "value", item))

    try:
        async for event in agent.reply_stream(inputs):
            if isinstance(event, ReplyStartEvent):
                line(f"[Reply Start] {event.name}")
            elif isinstance(event, ModelCallStartEvent):
                line(f"[Model] {event.model_name}")
            elif isinstance(event, ModelCallEndEvent):
                model_calls += 1
                input_tokens += int(event.input_tokens or 0)
                output_tokens += int(event.output_tokens or 0)
                line(
                    f"[Model Usage] input={event.input_tokens} "
                    f"output={event.output_tokens} reason={value(event.finished_reason)}",
                )
            elif isinstance(event, ToolCallStartEvent):
                line(f"[Tool Call] {event.tool_call_name}")
                if event.tool_call_name == "Skill":
                    pending_skill_calls[event.tool_call_id] = {
                        "arguments": "",
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                    }
            elif isinstance(event, ToolCallDeltaEvent):
                stream.write(event.delta)
                stream.flush()
                if event.tool_call_id in pending_skill_calls:
                    pending_skill_calls[event.tool_call_id]["arguments"] += event.delta
            elif isinstance(event, ToolCallEndEvent):
                line()
            elif isinstance(event, ToolResultStartEvent):
                line(f"[Tool Result] {event.tool_call_name}")
            elif isinstance(event, ToolResultTextDeltaEvent):
                stream.write(event.delta)
                stream.flush()
            elif isinstance(event, ToolResultEndEvent):
                line()
                line(f"[Tool Result End] {value(event.state)}")
                pending = pending_skill_calls.pop(event.tool_call_id, None)
                if pending is not None:
                    try:
                        arguments = json.loads(pending["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    raw_status = value(event.state)
                    skill_calls.append(
                        {
                            "tool_call_id": event.tool_call_id,
                            "skill_name": str(arguments.get("skill") or ""),
                            "status": raw_status.rsplit(".", 1)[-1].upper(),
                            "started_at": pending["started_at"],
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
            elif isinstance(event, TextBlockDeltaEvent):
                if event.block_id not in text_blocks:
                    text_blocks[event.block_id] = ""
                    text_block_order.append(event.block_id)
                text_blocks[event.block_id] += event.delta
                stream.write(event.delta)
                stream.flush()
            elif isinstance(event, ExceedMaxItersEvent):
                line()
                line(f"[Max Iters] {event.name}")
            elif isinstance(event, ReplyEndEvent):
                line()
                line(f"[Reply End] {value(event.finished_reason)}")
        stream_status = "SUCCESS"
    except BaseException:
        stream_status = "ERROR"
        raise
    finally:
        if skill_usage_report_path is not None:
            for tool_call_id, pending in pending_skill_calls.items():
                skill_calls.append(
                    {
                        "tool_call_id": tool_call_id,
                        "skill_name": "",
                        "status": "INTERRUPTED",
                        "started_at": pending["started_at"],
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
            _write_skill_usage_report(
                Path(skill_usage_report_path),
                status=stream_status,
                calls=skill_calls,
                pipeline_skills=set(exposed_pipeline_skills),
                error_view_skills=set(exposed_error_view_skills),
                agent_usage={
                    "model_calls": model_calls,
                    "react_iterations": int(getattr(agent.state, "cur_iter", 0) or 0),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "duration_seconds": round(time.monotonic() - started_monotonic, 6),
                },
            )

    if text_block_order:
        return AssistantMsg(
            name=agent.name,
            content=text_blocks[text_block_order[-1]],
        )
    for message in reversed(agent.state.context):
        if message.role == "assistant" and message.name == agent.name:
            return message
    raise RuntimeError("Data Cleaning Agent produced no final assistant message.")


def _write_skill_usage_report(
    path: Path,
    *,
    status: str,
    calls: list[dict[str, str]],
    pipeline_skills: set[str],
    error_view_skills: set[str],
    agent_usage: dict[str, int | float],
) -> None:
    successful = {
        call["skill_name"]
        for call in calls
        if call.get("status") == "SUCCESS" and call.get("skill_name")
    }
    loaded_pipeline = sorted(successful & pipeline_skills)
    loaded_error_views = sorted(successful & error_view_skills)

    def call_counts(names: set[str]) -> dict[str, int]:
        return {
            name: sum(1 for call in calls if call.get("skill_name") == name)
            for name in sorted(names)
        }

    payload = {
        "schema_version": 1,
        "status": status,
        "exposed": {
            "pipeline": sorted(pipeline_skills),
            "error_view": sorted(error_view_skills),
        },
        "loaded": {
            "pipeline": loaded_pipeline,
            "error_view": loaded_error_views,
        },
        "not_loaded": {
            "pipeline": sorted(pipeline_skills - set(loaded_pipeline)),
            "error_view": sorted(error_view_skills - set(loaded_error_views)),
        },
        "call_counts": {
            "pipeline": call_counts(pipeline_skills),
            "error_view": call_counts(error_view_skills),
        },
        "calls": calls,
        "agent_usage": agent_usage,
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
