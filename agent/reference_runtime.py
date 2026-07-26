"""Runtime construction for the retained reference-guided code agent."""
from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml
from agentscope.agent import Agent, ContextConfig, ReActConfig
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.skill import LocalSkillLoader
from agentscope.state import AgentState
from agentscope.tool import TaskCreate, TaskGet, TaskList, TaskUpdate, Toolkit
from agentscope.workspace import LocalWorkspace
from pydantic import BaseModel, Field

from agent_tools.agentscope2_tools import (
    RunPipelineTool,
    SubmitCandidateTool,
    ToolAuditMiddleware,
    TestRunnerSpecServices,
    ValidateDraftTool,
    ValidationToolServices,
    build_scoped_atomic_tools,
    build_test_tools,
)
from agent_tools.context import EngineerToolContext
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
ENGINEER_CODE_READ_ROOTS = (V2_DIR / "skills", V2_DIR / "workflow", V2_DIR / "lib")
AGENTSCOPE_VERSION = "2.0.4.post1"


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
            f"Data Cleaning Agent requires agentscope=={AGENTSCOPE_VERSION}; installed={installed}"
        )


def create_reference_context_config() -> ContextConfig:
    """Use AgentScope 2.x semantic compression and workspace offload."""
    return ContextConfig(
        trigger_ratio=0.8,
        reserve_ratio=0.1,
        compression_prompt=REFERENCE_COMPRESSION_PROMPT,
        summary_template=REFERENCE_COMPRESSION_TEMPLATE,
        summary_schema=ReferenceCompressionSummary.model_json_schema(),
        tool_result_limit=50_000,
    )


def create_reference_react_config(max_iters: int) -> ReActConfig:
    return ReActConfig(
        max_iters=max_iters,
        stop_on_reject=False,
        interruption_raise_cancelled_error=True,
    )


def _skill_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for name in sorted(DATA_CLEANING_AGENT_PIPELINE_SKILLS):
        skill_dir = V2_DIR / "skills" / name
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise RuntimeError(f"Missing retained Skill instructions: {skill_md}")
        text = skill_md.read_text(encoding="utf-8")
        if not text.startswith("---\n") or "\n---\n" not in text[4:]:
            raise RuntimeError(f"Skill frontmatter is invalid: {skill_md}")
        frontmatter = text.split("\n---\n", 1)[0][4:]
        metadata = yaml.safe_load(frontmatter) or {}
        if metadata.get("name") != name or not str(metadata.get("description") or "").strip():
            raise RuntimeError(f"Skill name/description is invalid: {skill_md}")
        manifest.append(
            {
                "name": name,
                "description": str(metadata["description"]).strip(),
                "dir": name,
                "skill_path": str(skill_dir),
                "agentscope_agent_skill": True,
                "tool_names": [],
            }
        )
    return manifest


def create_reference_toolkit(
    context: EngineerToolContext,
    services: ValidationToolServices | None = None,
) -> tuple[Toolkit, list[dict]]:
    """Build one always-visible validation toolkit using AgentScope 2.x APIs."""
    require_agentscope_version()
    audit_path = context.workspace_dir / "tool_audit.jsonl"
    tools = [
        TaskCreate(),
        TaskGet(),
        TaskList(),
        TaskUpdate(),
        *build_scoped_atomic_tools(context, audit_path=audit_path),
    ]
    if services is not None:
        boundary_audit = [ToolAuditMiddleware(audit_path)]
        tools.extend(
            [
                RunPipelineTool(services, middlewares=boundary_audit),
                ValidateDraftTool(services, middlewares=boundary_audit),
                SubmitCandidateTool(services, middlewares=boundary_audit),
            ]
        )
    toolkit = Toolkit(
        tools=tools,
        skills_or_loaders=[
            LocalSkillLoader(directory=str(V2_DIR / "skills"), scan_subdir=True)
        ],
    )
    manifest = _skill_manifest()
    return toolkit, manifest


def create_test_toolkit(
    context: EngineerToolContext,
    services: TestRunnerSpecServices,
) -> Toolkit:
    """Build the isolated Test Agent toolkit without edit, Python or Skill tools."""
    require_agentscope_version()
    return Toolkit(
        tools=build_test_tools(
            context,
            services,
            audit_path=context.workspace_dir / "tool_audit.jsonl",
        )
    )


async def create_reference_agent(
    *,
    name: str,
    system_prompt: str,
    toolkit: Toolkit,
    workspace_dir: str | Path,
    max_iters: int,
    state: AgentState | None = None,
) -> tuple[Agent, LocalWorkspace]:
    require_agentscope_version()
    model, _ = make_reference_model()
    workspace = LocalWorkspace(
        workdir=str(Path(workspace_dir).expanduser().resolve()),
        instructions=(
            "<workspace>All persistent files for this experiment are under {workdir}. "
            "Use only the provided scoped tools; do not assume shell or network access.</workspace>"
        ),
    )
    await workspace.initialize()
    agent = Agent(
        name=name,
        system_prompt=system_prompt,
        model=model,
        toolkit=toolkit,
        state=state,
        offloader=workspace,
        context_config=create_reference_context_config(),
        react_config=create_reference_react_config(max_iters),
    )
    return agent, workspace


def load_agent_state(path: str | Path) -> AgentState | None:
    state_path = Path(path).expanduser().resolve()
    if not state_path.is_file():
        return None
    return AgentState.model_validate_json(state_path.read_text(encoding="utf-8"))


def save_agent_state(path: str | Path, state: AgentState) -> None:
    state_path = Path(path).expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(state.model_dump_json(indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, state_path)
