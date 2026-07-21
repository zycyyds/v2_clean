"""Runtime construction for the retained reference-guided code agent."""
from __future__ import annotations

import os
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.token import OpenAITokenCounter
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

from agent.bounded_memory import (
    BoundedInMemoryMemory,
    _compact_threshold,
    _context_window_tokens,
)
from agent_tools.context import EngineerToolContext
from agent_tools.reference_variants import register_reference_variant_tools
from agent_tools.tools import register_engineer_atomic_tools
from config_loader import get_agent_config
from skills._registry import load_skills


V2_DIR = Path(__file__).resolve().parent.parent
REFERENCE_CODE_AGENT_PIPELINE_SKILLS = {
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
上下文即将压缩。请生成一份可以让 ReferenceCodeAgent 无损继续工作的结构化续接摘要。

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
    cfg = get_agent_config("react_planner")
    api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key") or None
    base_url = (
        os.environ.get("OPENAI_API_BASE")
        or cfg.get("base_url")
        or cfg.get("api_base")
        or "https://api.openai.com/v1"
    )
    model_name = (
        os.environ.get("MODEL_NAME")
        or cfg.get("model_name")
        or cfg.get("model")
        or "gpt-4.1-mini"
    )
    generation = {key: cfg[key] for key in ("temperature", "seed") if key in cfg}
    return (
        OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=False,
            client_kwargs={"base_url": base_url},
            generate_kwargs=generation or None,
        ),
        OpenAIChatFormatter(),
    )


def create_reference_memory(
    phase_context: dict[str, str],
    model: OpenAIChatModel,
) -> BoundedInMemoryMemory:
    workflow_hint = (
        phase_context.get("workflow")
        or phase_context.get("run_id")
        or phase_context.get("run_root")
        or phase_context.get("phase_root")
        or ""
    )
    model_name = str(
        getattr(model, "model_name", "")
        or getattr(model, "model", "")
        or os.environ.get("MODEL_NAME")
        or ""
    )
    return BoundedInMemoryMemory(
        model_name=model_name,
        report_dir=Path(phase_context["phase_root"]) / "context",
        workflow_hint=workflow_hint,
    )


def create_reference_compression_config(
    model: OpenAIChatModel,
) -> ReActAgent.CompressionConfig:
    """Build token-triggered semantic compression for ReferenceCodeAgent."""
    model_name = str(
        getattr(model, "model_name", "")
        or getattr(model, "model", "")
        or os.environ.get("MODEL_NAME")
        or ""
    )
    context_window = _context_window_tokens(model_name)
    trigger_threshold = int(context_window * _compact_threshold(None))
    return ReActAgent.CompressionConfig(
        enable=True,
        agent_token_counter=OpenAITokenCounter(model_name),
        trigger_threshold=trigger_threshold,
        keep_recent=14,
        compression_prompt=REFERENCE_COMPRESSION_PROMPT,
        summary_template=REFERENCE_COMPRESSION_TEMPLATE,
        summary_schema=ReferenceCompressionSummary,
    )


def create_reference_toolkit(context: EngineerToolContext) -> tuple[Toolkit, list[dict]]:
    """Expose only the pipeline Skills and local-code tools used by this workflow."""
    toolkit = Toolkit()
    manifest = load_skills(toolkit, allowed_names=REFERENCE_CODE_AGENT_PIPELINE_SKILLS)
    for item in manifest:
        for tool_name in item.get("tool_names") or []:
            toolkit.tools.pop(tool_name, None)
        item["tool_names"] = []
        skill_dir = V2_DIR / "skills" / str(item.get("dir") or item.get("name") or "")
        if (skill_dir / "SKILL.md").is_file():
            toolkit.register_agent_skill(str(skill_dir))
            item["agentscope_agent_skill"] = True
    context.register_executable_skills(str(item.get("name")) for item in manifest)
    register_engineer_atomic_tools(toolkit, context)
    register_reference_variant_tools(
        toolkit,
        context,
        skill_names={str(item.get("name")) for item in manifest},
        skills_root=V2_DIR / "skills",
    )
    return toolkit, manifest
