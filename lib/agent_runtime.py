from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock, UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.tool import ToolResponse

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import get_agent_config  # noqa: E402


def effective_agent_config(agent_key: str) -> dict[str, Any]:
    cfg = get_agent_config(agent_key)
    if not os.environ.get("OPENAI_API_KEY") and not cfg.get("api_key") and agent_key == "react_planner":
        fallback = get_agent_config("agent_1")
        if fallback.get("api_key"):
            return fallback
    return cfg


def format_message_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text") or ""))
                elif block_type == "tool_result":
                    parts.append(format_message_content(block.get("output")))
                elif block_type == "thinking":
                    continue
                else:
                    parts.append(str(block))
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text") or ""))
            elif hasattr(block, "output"):
                parts.append(format_message_content(getattr(block, "output")))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def json_tool_response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


def parse_json_text(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def parse_tool_response(response: ToolResponse) -> dict[str, Any]:
    return parse_json_text(format_message_content(response.content))


def parse_agent_text_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        return parse_json_text(stripped)
    except Exception:
        return None


def resolve_model_name(agent_key: str, default: str) -> str:
    env_model = os.environ.get("MODEL_NAME")
    if env_model:
        return env_model
    cfg = effective_agent_config(agent_key)
    return str(cfg.get("model_name") or cfg.get("model") or default)


def has_model_credentials(agent_key: str) -> bool:
    cfg = effective_agent_config(agent_key)
    return bool(os.environ.get("OPENAI_API_KEY") or cfg.get("api_key"))


def create_openai_model_and_formatter(agent_key: str, default_model: str):
    cfg = effective_agent_config(agent_key)
    api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key") or None
    base_url = (
        os.environ.get("OPENAI_API_BASE")
        or cfg.get("base_url")
        or cfg.get("api_base")
        or "https://api.openai.com/v1"
    )
    parameters: dict[str, Any] = {"parallel_tool_calls": False}
    if "temperature" in cfg:
        parameters["temperature"] = cfg.get("temperature")
    formatter = OpenAIChatFormatter()
    model = OpenAIChatModel(
        credential=OpenAICredential(
            api_key=api_key or "",
            base_url=base_url,
        ),
        model=resolve_model_name(agent_key, default_model),
        parameters=OpenAIChatModel.Parameters(**parameters),
        stream=False,
        formatter=formatter,
    )
    return model, formatter


async def collect_tool_results(agent: Any) -> dict[str, list[dict[str, Any]]]:
    collected: dict[str, list[dict[str, Any]]] = {}
    state = getattr(agent, "state", None)
    if state is None:
        return collected
    for msg in getattr(state, "context", []) or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_name = str(block.get("name") or "").strip()
            if not tool_name:
                continue
            output_text = format_message_content(block.get("output"))
            parsed = parse_agent_text_json(output_text)
            if parsed is not None:
                collected.setdefault(tool_name, []).append(parsed)
    return collected


def make_user_msg(name: str, content: str, metadata: dict[str, Any] | None = None) -> Msg:
    msg_metadata = dict(metadata or {})
    if name:
        msg_metadata.setdefault("sender_name", name)
    # Some OpenAI-compatible backends, including MiniMax, reject a chat history
    # when user-role messages carry different `name` values across turns.
    # Keep the protocol-level name stable and preserve the logical sender in
    # metadata for debugging.
    return UserMsg(name="user", content=content, metadata=msg_metadata or None)
