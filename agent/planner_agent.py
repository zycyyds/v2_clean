"""PlannerReActAgent — single ReAct agent loaded with all skills.

Replaces the previous main_orchestrator + router + handoffs + supervisors
state machine. Lets the LLM plan and self-correct via the standard ReAct
loop (max_iters), with no external retry harness.
"""
from __future__ import annotations

from pathlib import Path
import sys

V2_DIR = Path(__file__).resolve().parents[1]
LIB_DIR = V2_DIR / "lib"
for path in (V2_DIR, LIB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit

from skills._registry import load_all_skills, render_skill_manifest
from agent.system_prompt import build_system_prompt

# Try v2-local config loader first; fall back to legacy agent_runtime if absent.
try:
    import os
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.model import OpenAIChatModel
    from config_loader import get_agent_config as _get_v2_cfg

    def _create_model_and_formatter(agent_key: str, default_model: str):
        cfg = _get_v2_cfg(agent_key)
        api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key") or None
        base_url = (
            os.environ.get("OPENAI_API_BASE")
            or cfg.get("base_url")
            or cfg.get("api_base")
            or "https://api.openai.com/v1"
        )
        model_name = os.environ.get("MODEL_NAME") or cfg.get("model_name") or cfg.get("model") or default_model
        gen_kwargs = {k: cfg[k] for k in ("temperature", "seed") if k in cfg}
        model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=False,
            client_kwargs={"base_url": base_url},
            generate_kwargs=gen_kwargs or None,
        )
        return model, OpenAIChatFormatter()

except Exception:
    from agent_runtime import create_openai_model_and_formatter as _create_model_and_formatter  # type: ignore


def create_planner_agent(
    *,
    agent_key: str = "react_planner",
    default_model: str = "gpt-4.1-mini",
    max_iters: int = 30,
) -> tuple[ReActAgent, list[dict]]:
    """Build the planner agent loaded with all skills.

    Returns the agent plus the manifest (handy for tests / debugging).
    """
    toolkit = Toolkit()
    manifest = load_all_skills(toolkit)
    sys_prompt = build_system_prompt(render_skill_manifest(manifest))

    model, formatter = _create_model_and_formatter(agent_key, default_model)
    agent = ReActAgent(
        name="MedicalPipelinePlanner",
        sys_prompt=sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=False,
        max_iters=max_iters,
        print_hint_msg=False,
    )
    return agent, manifest
