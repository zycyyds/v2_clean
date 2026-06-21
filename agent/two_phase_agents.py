"""Two-phase agent factories: DataExplorer + FeatureEngineer."""
from __future__ import annotations

import os
import json
from pathlib import Path
import sys

V2_DIR = Path(__file__).resolve().parents[1]
LIB_DIR = V2_DIR / "lib"
for path in (V2_DIR, LIB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from config_loader import get_agent_config
from agent.bounded_memory import BoundedInMemoryMemory
from skills._registry import load_skills, render_skill_manifest
from agent.system_prompt import build_explorer_prompt, build_engineer_prompt
from agent_tools import (
    EngineerToolContext,
    ExplorerToolContext,
    register_engineer_atomic_tools,
    register_engineer_skill_lifecycle_tools,
    register_explorer_atomic_tools,
    register_explorer_report_tools,
)
from lib.agent_artifacts import get_phase_context

# Explorer uses atomic tools for discovery/reading and Skills for domain reports.
EXPLORER_SKILLS = {
    "analyze_gold_examples",
}

# engineer only needs process/label/output skills
ENGINEER_SKILLS = {
    "profile_table", "assess_data_quality",
    "clean_data", "select_features", "build_label_from_icd",
    "build_label_from_event", "build_label_from_column",
    "discover_label_candidates", "export_ml_dataset",
    "validate_output", "save_report", "build_mimic_liver_dataset",
    "learned_extraction_rules",
    "extract_text_features", "run_ocr",
    "extract_structured_fields", "join_source_records",
    "derive_target_fields", "aggregate_records", "normalize_reversible_values",
    "export_gold_workbook",
}

LABEL_SKILLS = {
    "build_label_from_icd",
    "build_label_from_event",
    "build_label_from_column",
    "discover_label_candidates",
    "export_ml_dataset",
}

ENGINEER_CODE_READ_ROOTS = (
    V2_DIR / "skills",
    V2_DIR / "lib",
    V2_DIR / "workflow",
)


def _make_model(agent_key: str = "react_planner",
                default_model: str = "gpt-4.1-mini") -> tuple:
    cfg = get_agent_config(agent_key)
    api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key") or None
    base_url = (
        os.environ.get("OPENAI_API_BASE")
        or cfg.get("base_url") or cfg.get("api_base")
        or "https://api.openai.com/v1"
    )
    model_name = (
        os.environ.get("MODEL_NAME")
        or cfg.get("model_name") or cfg.get("model")
        or default_model
    )
    gen_kwargs = {k: cfg[k] for k in ("temperature", "seed") if k in cfg}
    model = OpenAIChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=False,
        client_kwargs={"base_url": base_url},
        generate_kwargs=gen_kwargs or None,
    )
    return model, OpenAIChatFormatter()


def _filtered_toolkit(allowed: set[str]) -> tuple[Toolkit, list[dict]]:
    """Only register skills that are allowed in the current phase."""
    toolkit = Toolkit()
    manifest = load_skills(toolkit, allowed_names=allowed)
    return toolkit, manifest


def create_engineer_toolkit(
    context: EngineerToolContext,
    *,
    bundle_managed_rules: bool = False,
    task_text: str = "",
) -> tuple[Toolkit, list[dict]]:
    """Build the Engineer toolkit from domain skills plus atomic tools."""
    allowed = set(ENGINEER_SKILLS - LABEL_SKILLS)
    if _task_requests_label(task_text):
        allowed.update(LABEL_SKILLS)
    toolkit, manifest = _filtered_toolkit(allowed)
    if bundle_managed_rules:
        incompatible = {
            "learned_extraction_rules",
            "build_mimic_liver_dataset",
            "clean_data",
            "export_ml_dataset",
        }
        removed = [item for item in manifest if item.get("name") in incompatible]
        for item in removed:
            for tool_name in item.get("tool_names") or []:
                toolkit.tools.pop(tool_name, None)
        manifest = [item for item in manifest if item.get("name") not in incompatible]
    else:
        _configure_rule_policy(toolkit, context.split_mode)
    if context.split_mode == "test" and not bundle_managed_rules:
        toolkit.tools.pop("promote_validation_feedback_tool", None)
        toolkit.tools.pop("freeze_learned_rules_tool", None)
    _attach_skill_usage_audit(toolkit, manifest, context)
    register_engineer_atomic_tools(toolkit, context)
    register_engineer_skill_lifecycle_tools(
        toolkit,
        context,
        skill_names={str(item.get("name")) for item in manifest},
        skills_root=V2_DIR / "skills",
    )
    return toolkit, manifest


def _task_requests_label(task_text: str) -> bool:
    normalized = str(task_text or "").casefold()
    markers = (
        "标签", "构建label", "label column", "classification target",
        "分类目标", "预测目标", "监督学习目标",
    )
    return any(marker in normalized for marker in markers)


def _configure_rule_policy(toolkit: Toolkit, split_mode: str) -> None:
    registered = toolkit.tools.get("load_learned_rules_tool")
    if registered is None:
        return
    statuses = "frozen" if split_mode == "test" else "draft,active,frozen"
    registered.preset_kwargs["include_statuses"] = statuses
    parameters = registered.json_schema.get("function", {}).get("parameters", {})
    properties = parameters.get("properties") or {}
    properties.pop("include_statuses", None)
    required = parameters.get("required") or []
    if "include_statuses" in required:
        required.remove("include_statuses")


def _attach_skill_usage_audit(
    toolkit: Toolkit,
    manifest: list[dict],
    context: EngineerToolContext,
) -> None:
    """Record domain Skill calls without changing their implementations."""
    for item in manifest:
        skill_name = str(item.get("name") or "")
        for tool_name in item.get("tool_names") or []:
            registered = toolkit.tools.get(tool_name)
            if registered is None or registered.postprocess_func is not None:
                continue

            def postprocess(_tool_call, response, *, _skill=skill_name, _tool=tool_name):
                status = "UNKNOWN"
                artifacts: list[str] = []
                try:
                    block = response.content[0]
                    text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        status = str(payload.get("status") or status)
                        artifacts = _existing_artifact_paths(payload.get("artifacts"))
                except Exception:
                    pass
                context.record_skill_call(_skill, _tool, status, artifacts=artifacts)
                return None

            registered.postprocess_func = postprocess


def _existing_artifact_paths(value, *, parent_key: str = "") -> list[str]:
    """Collect real output paths from a Skill response without trusting declarations."""
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "manifest_path":
                continue
            paths.extend(_existing_artifact_paths(child, parent_key=str(key)))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_existing_artifact_paths(child, parent_key=parent_key))
    elif isinstance(value, str) and value.startswith("/"):
        path = Path(value).expanduser().resolve()
        if path.is_file():
            paths.append(str(path))
    return sorted(set(paths))


def create_explorer_toolkit(
    context: ExplorerToolContext | None = None,
) -> tuple[Toolkit, list[dict]]:
    """Build the Explorer toolkit with its canonical report publisher."""
    toolkit, manifest = _filtered_toolkit(EXPLORER_SKILLS)
    if context is not None:
        register_explorer_atomic_tools(toolkit, context)
    register_explorer_report_tools(toolkit)
    return toolkit, manifest


def infer_split_mode(task_text: str) -> str:
    """Infer whether learned rules must be frozen-only for a test split."""
    normalized = str(task_text or "").lower()
    test_markers = ("测试集", "test set", "test_set", "test split", "测试数据")
    return "test" if any(marker in normalized for marker in test_markers) else "training"


def create_explorer_agent(max_iters: int = 20, *, task_text: str = "") -> ReActAgent:
    phase_context = get_phase_context()
    if phase_context is None or phase_context.get("phase_name") != "explorer":
        raise RuntimeError("DataExplorer requires an active explorer phase context")
    context = ExplorerToolContext.from_task(
        task_text=task_text,
        explorer_phase_root=phase_context["phase_root"],
    )
    toolkit, manifest = create_explorer_toolkit(context)
    sys_prompt = build_explorer_prompt(render_skill_manifest(manifest))
    model, formatter = _make_model()
    return ReActAgent(
        name="DataExplorer",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=BoundedInMemoryMemory(),
        parallel_tool_calls=False,  # force serial tool calls
        max_iters=max_iters,
        print_hint_msg=False,
    )


def create_engineer_agent(
    max_iters: int = 40,
    *,
    task_text: str = "",
    explorer_phase_root: str | None = None,
    report_paths: dict[str, str] | None = None,
    split_mode: str | None = None,
    return_context: bool = False,
    bundle_managed_rules: bool = False,
) -> ReActAgent | tuple[ReActAgent, EngineerToolContext]:
    phase_context = get_phase_context()
    if phase_context is None:
        raise RuntimeError("FeatureEngineer requires an active phase context")
    learned_rules_dir = V2_DIR / "program" / "output" / "learned_extraction_rules" / "rules"
    additional_roots = (
        [learned_rules_dir]
        if learned_rules_dir.exists() and not bundle_managed_rules
        else []
    )
    additional_roots.extend(ENGINEER_CODE_READ_ROOTS)
    effective_split_mode = split_mode or infer_split_mode(task_text)
    context = EngineerToolContext.from_task(
        task_text=task_text,
        engineer_phase_root=phase_context["phase_root"],
        explorer_phase_root=explorer_phase_root,
        additional_read_roots=additional_roots,
        require_skill_plan=True,
        split_mode=effective_split_mode,
        required_report_paths=report_paths or {},
    )
    toolkit, manifest = create_engineer_toolkit(
        context,
        bundle_managed_rules=bundle_managed_rules,
        task_text=task_text,
    )
    sys_prompt = build_engineer_prompt(
        render_skill_manifest(manifest),
        split_mode=effective_split_mode,
    )
    model, formatter = _make_model()
    agent = ReActAgent(
        name="FeatureEngineer",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=BoundedInMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=max_iters,
        print_hint_msg=False,
    )
    return (agent, context) if return_context else agent
