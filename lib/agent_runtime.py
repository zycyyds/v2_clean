from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock, ToolResultBlock, UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.tool import ToolResponse
from pydantic import PrivateAttr

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import get_agent_config  # noqa: E402


class ManagedOpenAIChatModel(OpenAIChatModel):
    """OpenAI model with an HTTP client whose lifecycle is owned by this process.

    AgentScope 2.0.4.post1 creates an ``openai.AsyncClient`` for every API
    call but does not close it. Supplying and later closing one shared HTTPX
    client prevents its transport cleanup from running after ``asyncio.run``
    has already closed the event loop.
    """

    _managed_http_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        client_kwargs = dict(kwargs.pop("client_kwargs", {}) or {})
        if "http_client" in client_kwargs:
            raise ValueError("ManagedOpenAIChatModel owns client_kwargs.http_client")
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0),
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
            follow_redirects=True,
        )
        client_kwargs["http_client"] = http_client
        super().__init__(client_kwargs=client_kwargs, **kwargs)
        self._managed_http_client = http_client

    async def aclose(self) -> None:
        if self._managed_http_client is not None:
            await self._managed_http_client.aclose()
            self._managed_http_client = None


class AllApiKeysRateLimitedError(RuntimeError):
    """Every configured OpenAI-compatible credential returned HTTP 429."""


def _is_rate_limit_error(exc: Exception) -> bool:
    return (
        getattr(exc, "status_code", None) == 429
        or type(exc).__name__ == "RateLimitError"
    )


class RotatingOpenAIChatModel(ManagedOpenAIChatModel):
    """Retry one request with the next local credential after an HTTP 429."""

    def __init__(self, *, api_keys: list[str], base_url: str, **kwargs: Any) -> None:
        if not api_keys:
            raise ValueError("RotatingOpenAIChatModel requires at least one API key")
        self._api_keys = tuple(api_keys)
        self._base_url = base_url
        self._active_key_index = 0
        self.on_failover = None
        super().__init__(
            credential=OpenAICredential(api_key=self._api_keys[0], base_url=base_url),
            max_retries=0,
            **kwargs,
        )

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for offset in range(len(self._api_keys)):
            key_index = (self._active_key_index + offset) % len(self._api_keys)
            self.credential = OpenAICredential(
                api_key=self._api_keys[key_index],
                base_url=self._base_url,
            )
            try:
                response = await super().__call__(*args, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                last_error = exc
                next_slot = ((key_index + 1) % len(self._api_keys)) + 1
                if self.on_failover is not None and offset + 1 < len(self._api_keys):
                    self.on_failover(
                        {
                            "event": "api_key_failover",
                            "from_slot": key_index + 1,
                            "to_slot": next_slot,
                            "reason": "http_429",
                        },
                    )
                continue
            self._active_key_index = key_index
            return response
        raise AllApiKeysRateLimitedError(
            f"all {len(self._api_keys)} configured API keys are rate limited",
        ) from last_error


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
            else:
                if hasattr(block, "text"):
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
    return bool(os.environ.get("OPENAI_API_KEY") or _configured_api_keys(cfg))


def _configured_api_keys(cfg: dict[str, Any]) -> list[str]:
    configured = cfg.get("api_keys")
    values = configured if isinstance(configured, list) else []
    if not values and cfg.get("api_key"):
        values = [cfg["api_key"]]
    keys: list[str] = []
    for value in values:
        key = str(value or "").strip()
        if key and key not in keys and key != "YOUR_API_KEY_HERE":
            keys.append(key)
    return keys


def _environment_api_keys() -> list[str]:
    raw = os.environ.get("OPENAI_API_KEYS_JSON", "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("OPENAI_API_KEYS_JSON must be valid JSON") from exc
        if not isinstance(payload, list):
            raise ValueError("OPENAI_API_KEYS_JSON must contain a JSON array")
        keys = [str(item).strip() for item in payload if str(item).strip()]
        if not keys:
            raise ValueError("OPENAI_API_KEYS_JSON must contain at least one key")
        return list(dict.fromkeys(keys))
    single = os.environ.get("OPENAI_API_KEY", "").strip()
    return [single] if single else []


def build_worker_model_environment(agent_key: str, default_model: str) -> dict[str, str]:
    """Serialize only model settings needed by a sandboxed Agent worker."""
    cfg = effective_agent_config(agent_key)
    keys = _environment_api_keys() or _configured_api_keys(cfg)
    if not keys:
        raise ValueError(f"no API key configured for {agent_key}")
    environment = {
        "OPENAI_API_KEYS_JSON": json.dumps(keys),
        "OPENAI_API_BASE": str(
            os.environ.get("OPENAI_API_BASE")
            or cfg.get("base_url")
            or cfg.get("api_base")
            or "https://api.openai.com/v1"
        ),
        "MODEL_NAME": str(
            os.environ.get("MODEL_NAME")
            or cfg.get("model_name")
            or cfg.get("model")
            or default_model
        ),
        "V2_SKIP_LOCAL_MODEL_CONFIG": "1",
    }
    if "temperature" in cfg:
        environment["AGENT_TEMPERATURE"] = str(cfg["temperature"])
    if "seed" in cfg:
        environment["AGENT_SEED"] = str(cfg["seed"])
    return environment


def create_openai_model_and_formatter(
    agent_key: str,
    default_model: str,
    *,
    stream: bool = False,
    context_size_override: int | None = None,
    parallel_tool_calls: bool = False,
    model_name_override: str | None = None,
):
    cfg = effective_agent_config(agent_key)
    api_keys = _environment_api_keys() or _configured_api_keys(cfg)
    base_url = (
        os.environ.get("OPENAI_API_BASE")
        or cfg.get("base_url")
        or cfg.get("api_base")
        or "https://api.openai.com/v1"
    )
    parameters: dict[str, Any] = {
        "parallel_tool_calls": parallel_tool_calls,
    }
    if os.environ.get("AGENT_TEMPERATURE") is not None:
        parameters["temperature"] = float(os.environ["AGENT_TEMPERATURE"])
    elif "temperature" in cfg:
        parameters["temperature"] = cfg.get("temperature")
    model_name = model_name_override or resolve_model_name(agent_key, default_model)
    context_size = context_size_override or _model_context_size(model_name)
    if os.environ.get("AGENT_SEED") is not None:
        extra_body = {"seed": int(os.environ["AGENT_SEED"])}
    else:
        extra_body = {"seed": cfg["seed"]} if "seed" in cfg else None
    model = RotatingOpenAIChatModel(
        api_keys=api_keys or [""],
        base_url=base_url,
        model=model_name,
        parameters=OpenAIChatModel.Parameters(**parameters),
        stream=stream,
        context_size=context_size,
        extra_body=extra_body,
        formatter=OpenAIChatFormatter(),
    )
    return model, model.formatter


def _model_context_size(model_name: str) -> int:
    for env_key in ("AGENT_CONTEXT_WINDOW_TOKENS", "MODEL_CONTEXT_TOKENS"):
        raw = os.environ.get(env_key)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    normalized = str(model_name or "").lower()
    if "minimax" in normalized or "m3" in normalized:
        return 204_800
    if "gpt-4.1" in normalized:
        return 1_000_000
    if "claude" in normalized:
        return 200_000
    return 128_000


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
            if isinstance(block, ToolResultBlock):
                tool_name = str(block.name or "").strip()
                output = block.output
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                tool_name = str(block.get("name") or "").strip()
                output = block.get("output")
            else:
                continue
            if not tool_name:
                continue
            output_text = format_message_content(output)
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
