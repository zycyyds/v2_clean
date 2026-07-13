"""Runtime construction for the retained reference-guided code agent."""
from __future__ import annotations

import os
from pathlib import Path

from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from agent.bounded_memory import BoundedInMemoryMemory
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
