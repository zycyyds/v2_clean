"""A small, persistent AgentScope coding-agent runtime."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO

import httpx
from agentscope.agent import Agent, ContextConfig, ModelConfig, ReActConfig
from agentscope.event import (
    ExceedMaxItersEvent,
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    ThinkingBlockDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.message import Msg, TextBlock, UserMsg
from agentscope.middleware import MiddlewareBase
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.skill import LocalSkillLoader
from agentscope.state import AgentState
from agentscope.tool import Bash, Edit, Glob, Grep, Read, Toolkit, Write
from agentscope.tool._builtin._backend import LocalBackend
from agentscope.workspace import LocalWorkspace
from pydantic import BaseModel, Field

from lib.agent_runtime import create_openai_model_and_formatter
from lib.restricted_local_backend import RestrictedLocalBackend


AGENTSCOPE_VERSION = "2.0.4.post1"
M3_CONTEXT_SIZE = 1_000_000
PI_RESERVE_TOKENS = 16_384
PI_KEEP_RECENT_TOKENS = 20_000
PI_TOOL_RESULT_LIMIT = 4_000
MODEL_SECRET_ENV_KEYS = ("OPENAI_API_KEY", "OPENAI_API_KEYS_JSON")


class PiCompressionSummary(BaseModel):
    task_and_constraints: str = Field(
        description="The original task, success criteria, and constraints.",
    )
    current_state: str = Field(
        description="Completed work, current phase, and latest valid result.",
    )
    verified_findings_and_evidence: str = Field(
        description="Verified findings with commands, files, counts, or other evidence.",
    )
    failed_attempts_and_counterexamples: str = Field(
        description="Rejected approaches, failures, and counterexamples not to repeat.",
    )
    files_read: str = Field(description="Important files read, with accurate paths.")
    files_created_or_modified: str = Field(
        description="Files created or modified and what changed.",
    )
    key_commands: str = Field(
        description="Important commands already run and their observed outcomes.",
    )
    artifacts_and_paths: str = Field(
        description="Scripts, outputs, reports, and other artifacts needed to continue.",
    )
    unresolved_work: str = Field(description="Open problems and uncertainties.")
    next_steps: str = Field(description="Concrete next actions in priority order.")


PI_COMPRESSION_PROMPT = """\
<system-hint>
The context must be compacted so the coding agent can continue without restarting.
Create an evidence-grounded continuation summary. Preserve the original task and
constraints, verified findings, failed approaches and counterexamples, exact file
paths, files read or changed, key commands and outcomes, current artifacts,
unresolved work, and concrete next steps. Carry forward still-valid facts from any
previous summary. Do not invent evidence or promote hypotheses to conclusions.
</system-hint>
"""


PI_COMPRESSION_TEMPLATE = """\
<system-info>
# Task and constraints
{task_and_constraints}

# Current state
{current_state}

# Verified findings and evidence
{verified_findings_and_evidence}

# Failed attempts and counterexamples
{failed_attempts_and_counterexamples}

# Files read
{files_read}

# Files created or modified
{files_created_or_modified}

# Key commands
{key_commands}

# Artifacts and paths
{artifacts_and_paths}

# Unresolved work
{unresolved_work}

# Next steps
{next_steps}
</system-info>
"""


PI_SYSTEM_PROMPT = """\
You are an expert coding agent. Work autonomously until the user's task is complete.
Inspect relevant files before changing them, use tools to act, observe real outputs,
fix failures, and verify the result before stopping. Prefer reproducible scripts over
one-off manual data edits. Keep responses concise and show paths clearly.

The user may declare allowed and forbidden paths in the task prompt. Obey those
boundaries. Do not inspect hidden references, private evaluation data, historical
experiments, credentials, or unrelated projects unless the user explicitly authorizes
them. These are behavioral instructions; the local tools are not a security sandbox.
"""


PI_TEST_DECLARATION_SYSTEM_PROMPT = """\
You are a Test Declaration Agent. Diagnose only how to invoke a frozen pipeline.
You may inspect the frozen snapshot, public Test raw data, Train reference schema,
and your work directory. You must write only runner_spec.json. Do not create or
modify Python or shell scripts, do not generate result data, and do not attempt to
change the frozen pipeline. You have no Bash tool. Hidden Test gold and scores are
never available to you.
"""


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _event_value(value: Any) -> str:
    """Normalize AgentScope enums and mock string values."""
    return str(getattr(value, "value", value))


def _exception_status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _exception_error_code(exc: BaseException) -> str:
    candidates = [
        getattr(exc, "code", None),
        getattr(getattr(exc, "error", None), "code", None),
    ]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        candidates.append(body.get("code"))
        nested = body.get("error")
        if isinstance(nested, dict):
            candidates.append(nested.get("code"))
    return " ".join(str(item).lower() for item in candidates if item)


def is_context_overflow(exc: BaseException) -> bool:
    """Recognize OpenAI-compatible context-window failures."""
    status = _exception_status_code(exc)
    code = _exception_error_code(exc)
    message = str(exc).lower()
    markers = (
        "context_length_exceeded",
        "context length",
        "context window",
        "input too long",
        "too many tokens",
        "maximum context",
    )
    return (
        "context_length_exceeded" in code
        or (status in {400, 413} and any(marker in message for marker in markers))
        or any(marker in message for marker in markers)
    )


def _is_transient_connection_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            asyncio.TimeoutError,
            ConnectionError,
            httpx.TimeoutException,
            httpx.TransportError,
        ),
    ):
        return True
    if _exception_status_code(exc) in {408, 500, 502, 503, 504}:
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }


class PiReasoningRetryMiddleware(MiddlewareBase):
    """Retry only recoverable reasoning failures before visible output."""

    def __init__(
        self,
        *,
        transient_retries: int = 2,
        base_delay: float = 1.0,
        sleep: Any = asyncio.sleep,
        on_retry: Any | None = None,
    ) -> None:
        self.transient_retries = transient_retries
        self.base_delay = base_delay
        self.sleep = sleep
        self.on_retry = on_retry

    async def on_reasoning(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Any,
    ):
        overflow_retried = False
        transient_attempt = 0
        incomplete_tool_retried = False
        suppress_next_start = False
        visible_types = (
            TextBlockDeltaEvent,
            ThinkingBlockDeltaEvent,
        )
        tool_event_types = (
            ToolCallStartEvent,
            ToolCallDeltaEvent,
            ToolCallEndEvent,
        )

        while True:
            visible_output = False
            buffered_tool_events: list[Any] = []
            retry_incomplete_tool = False
            try:
                async for event in next_handler(**input_kwargs):
                    if suppress_next_start and isinstance(event, ModelCallStartEvent):
                        suppress_next_start = False
                        continue
                    if isinstance(event, tool_event_types):
                        buffered_tool_events.append(event)
                        continue
                    if isinstance(event, ModelCallEndEvent):
                        if (
                            buffered_tool_events
                            and _event_value(event.finished_reason) == "interrupted"
                            and not incomplete_tool_retried
                        ):
                            retry_incomplete_tool = True
                            break
                        for tool_event in buffered_tool_events:
                            yield tool_event
                        buffered_tool_events.clear()
                    if isinstance(event, visible_types):
                        visible_output = True
                    yield event
                if retry_incomplete_tool:
                    incomplete_tool_retried = True
                    self._notify("incomplete_tool_call", 0.0, RuntimeError("interrupted"))
                    await agent.observe(
                        UserMsg(
                            name="system",
                            content=(
                                "The previous tool call was truncated before its arguments "
                                "were complete. It was not executed. Regenerate the complete "
                                "tool call with valid JSON arguments."
                            ),
                        ),
                    )
                    suppress_next_start = True
                    continue
                for tool_event in buffered_tool_events:
                    yield tool_event
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if visible_output:
                    raise
                if is_context_overflow(exc) and not overflow_retried:
                    overflow_retried = True
                    self._notify("context_overflow", 0.0, exc)
                    await agent.force_compress_context()
                    suppress_next_start = True
                    continue
                if (
                    _is_transient_connection_error(exc)
                    and transient_attempt < self.transient_retries
                ):
                    delay = self.base_delay * (2**transient_attempt)
                    transient_attempt += 1
                    self._notify("transient_connection", delay, exc)
                    await self.sleep(delay)
                    suppress_next_start = True
                    continue
                raise

    def _notify(self, reason: str, delay: float, exc: BaseException) -> None:
        if self.on_retry is not None:
            self.on_retry(
                {
                    "event": "model_retry",
                    "reason": reason,
                    "delay_seconds": delay,
                    "error_type": type(exc).__name__,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )


@dataclass(frozen=True)
class PiAgentConfig:
    workdir: str | Path
    max_iters: int = 10_000
    skill_dirs: tuple[str | Path, ...] = ()
    tool_profile: str = "full"
    read_roots: tuple[str | Path, ...] = ()
    runner_spec_path: str | Path | None = None

    def normalized(self) -> "PiAgentConfig":
        workdir = Path(self.workdir).expanduser().resolve()
        if not workdir.is_dir():
            raise ValueError(f"workdir must be an existing directory: {workdir}")
        if self.max_iters < 1:
            raise ValueError("max_iters must be positive")
        if self.tool_profile not in {"full", "test_declaration"}:
            raise ValueError("tool_profile must be full or test_declaration")
        skill_dirs = tuple(
            Path(item).expanduser().resolve() for item in self.skill_dirs
        )
        missing = [str(item) for item in skill_dirs if not item.is_dir()]
        if missing:
            raise ValueError("skill directory does not exist: " + ", ".join(missing))
        read_roots = tuple(Path(item).expanduser().resolve() for item in self.read_roots)
        runner_spec = (
            Path(self.runner_spec_path).expanduser().resolve()
            if self.runner_spec_path is not None
            else None
        )
        if self.tool_profile == "test_declaration" and runner_spec is None:
            raise ValueError("test_declaration requires runner_spec_path")
        return PiAgentConfig(
            workdir=workdir,
            max_iters=self.max_iters,
            skill_dirs=skill_dirs,
            tool_profile=self.tool_profile,
            read_roots=read_roots,
            runner_spec_path=runner_spec,
        )


@dataclass(frozen=True)
class PiCompressionPolicy:
    context_size: int = M3_CONTEXT_SIZE
    reserve_tokens: int = PI_RESERVE_TOKENS
    keep_recent_tokens: int = PI_KEEP_RECENT_TOKENS

    @property
    def trigger_tokens(self) -> int:
        return self.context_size - self.reserve_tokens

    @property
    def keep_recent_ratio(self) -> float:
        return self.keep_recent_tokens / self.context_size

    def should_compress(self, token_count: int) -> bool:
        return token_count >= self.trigger_tokens


@dataclass(frozen=True)
class AgentTurnResult:
    status: str
    text: str
    finished_reason: str
    model_calls: int
    input_tokens: int
    output_tokens: int
    react_iterations: int
    duration_seconds: float
    tool_calls: int
    tool_errors: int
    api_failovers: int
    compactions: int
    stream_event_counts: dict[str, int]
    text_block_ids: tuple[str, ...]
    thinking_block_ids: tuple[str, ...]


def create_pi_context_config() -> ContextConfig:
    return ContextConfig(
        # PiCompactionAgent performs the real 983,616-token check. This lower
        # internal threshold only lets AgentScope's compressor run once called.
        trigger_ratio=0.8999,
        reserve_ratio=PI_KEEP_RECENT_TOKENS / M3_CONTEXT_SIZE,
        compression_prompt=PI_COMPRESSION_PROMPT,
        summary_template=PI_COMPRESSION_TEMPLATE,
        summary_schema=PiCompressionSummary.model_json_schema(),
        tool_result_limit=PI_TOOL_RESULT_LIMIT,
    )


def make_pi_model():
    model, _ = create_openai_model_and_formatter(
        "react_planner",
        "MiniMax-M3",
        stream=True,
        context_size_override=M3_CONTEXT_SIZE,
        parallel_tool_calls=True,
        model_name_override="MiniMax-M3",
    )
    return model


def scrub_model_secrets_from_environment() -> None:
    """Prevent Bash descendants from inheriting model credentials."""
    for key in MODEL_SECRET_ENV_KEYS:
        os.environ.pop(key, None)


def build_pi_toolkit(
    workdir: str | Path,
    *,
    skill_dirs: tuple[str | Path, ...] = (),
    tool_profile: str = "full",
    read_roots: tuple[str | Path, ...] = (),
    runner_spec_path: str | Path | None = None,
) -> Toolkit:
    root = Path(workdir).expanduser().resolve()
    if tool_profile == "test_declaration":
        if runner_spec_path is None:
            raise ValueError("test_declaration requires runner_spec_path")
        backend = _RunnerSpecBackend(
            read_roots=(*read_roots, root),
            denied_read_roots=(),
            write_roots=(root,),
            cwd=root,
            runner_spec_path=runner_spec_path,
        )
        tools = [
            Read(backend=backend),
            Glob(backend=backend),
            Grep(backend=backend),
            Write(backend=backend),
            Edit(backend=backend),
        ]
    elif tool_profile == "full":
        backend = LocalBackend()
        tools = [
            Read(backend=backend),
            Write(backend=backend),
            Edit(backend=backend),
            Glob(backend=backend),
            Grep(backend=backend),
            Bash(cwd=str(root), backend=backend),
        ]
    else:
        raise ValueError("tool_profile must be full or test_declaration")
    loaders = [
        LocalSkillLoader(
            directory=str(Path(item).expanduser().resolve()),
            scan_subdir=True,
        )
        for item in skill_dirs
    ]
    return Toolkit(
        tools=tools,
        skills_or_loaders=loaders or None,
    )


class _RunnerSpecBackend(RestrictedLocalBackend):
    def __init__(self, *, runner_spec_path: str | Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.runner_spec_path = Path(runner_spec_path).expanduser().resolve()

    def _require_write(self, value: str | os.PathLike[str]) -> Path:
        path = super()._require_write(value)
        # AgentScope Write issues `mkdir -p <parent>` before writing the file.
        if path not in {self.runner_spec_path, self.runner_spec_path.parent}:
            raise PermissionError(f"only runner_spec.json may be written: {path}")
        return path


class PiCompactionAgent(Agent):
    """AgentScope Agent with Pi's fixed-token compaction trigger."""

    def __init__(self, *args: Any, compression_policy: PiCompressionPolicy, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compression_policy = compression_policy
        self.on_compaction = None

    async def compress_context(
        self,
        context_config: ContextConfig | None = None,
        instructions: Any | None = None,
    ) -> None:
        kwargs = await self._prepare_model_input()
        estimated_tokens = await self.model.count_tokens(**kwargs)
        if not self.compression_policy.should_compress(int(estimated_tokens)):
            return
        await self._run_compaction(
            context_config or self.context_config,
            instructions,
            estimated_tokens=int(estimated_tokens),
            reason="pi_threshold",
        )

    async def force_compress_context(self) -> None:
        """Compress once even when local estimation missed a server overflow."""
        kwargs = await self._prepare_model_input()
        estimated_tokens = int(await self.model.count_tokens(**kwargs))
        trigger_ratio = max(
            1e-9,
            min(0.8999, (max(estimated_tokens, 1) - 0.5) / self.model.context_size),
        )
        forced_config = self.context_config.model_copy(
            update={"trigger_ratio": trigger_ratio},
        )
        await self._run_compaction(
            forced_config,
            None,
            estimated_tokens=estimated_tokens,
            reason="context_overflow",
        )

    async def _run_compaction(
        self,
        context_config: ContextConfig,
        instructions: Any | None,
        *,
        estimated_tokens: int,
        reason: str,
    ) -> None:
        if self.on_compaction is not None:
            self.on_compaction(
                {
                    "event": "compression_start",
                    "reason": reason,
                    "tokens_before": estimated_tokens,
                    "trigger_tokens": self.compression_policy.trigger_tokens,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        await super().compress_context(
            context_config=context_config,
            instructions=instructions,
        )
        if self.on_compaction is not None:
            self.on_compaction(
                {
                    "event": "compression_end",
                    "reason": reason,
                    "tokens_before": estimated_tokens,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class PiAgentRuntime:
    def __init__(
        self,
        config: PiAgentConfig,
        *,
        output: TextIO | None = None,
    ) -> None:
        self.config = config.normalized()
        self.output = output or sys.stdout
        self.compression_policy = PiCompressionPolicy()
        self.agent: PiCompactionAgent | None = None
        self.workspace: LocalWorkspace | None = None
        self.run_dir: Path | None = None
        self._transcript_handle: TextIO | None = None
        self._stream: TextIO | None = None
        self._started_at = ""
        self._turns: list[dict[str, Any]] = []
        self._status = "INITIALIZED"
        self._closed = False
        self._active_turn_counters: dict[str, int] | None = None

    async def initialize(self) -> None:
        if self.agent is not None:
            return
        installed = version("agentscope")
        if installed != AGENTSCOPE_VERSION:
            raise RuntimeError(
                f"PiAgentRuntime requires agentscope=={AGENTSCOPE_VERSION}; installed={installed}",
            )
        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        self.run_dir = Path(self.config.workdir) / ".agent_runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._transcript_handle = (self.run_dir / "transcript.log").open(
            "w",
            encoding="utf-8",
            buffering=1,
        )
        self._stream = _Tee(self.output, self._transcript_handle)
        (self.run_dir / "tool_audit.jsonl").touch()
        workspace_dir = self.run_dir / "workspace"
        self.workspace = LocalWorkspace(
            workdir=str(workspace_dir),
            instructions=(
                "<workspace>Large tool results and compressed context are stored "
                "under {workdir}. The task working directory is managed by the "
                "native tools.</workspace>"
            ),
        )
        await self.workspace.initialize()
        toolkit = build_pi_toolkit(
            self.config.workdir,
            skill_dirs=self.config.skill_dirs,
            tool_profile=self.config.tool_profile,
            read_roots=self.config.read_roots,
            runner_spec_path=self.config.runner_spec_path,
        )
        tool_schemas = await toolkit.get_tool_schemas()
        model = make_pi_model()
        scrub_model_secrets_from_environment()
        self.agent = PiCompactionAgent(
            name="Pi-style Coding Agent",
            system_prompt=(
                PI_TEST_DECLARATION_SYSTEM_PROMPT
                if self.config.tool_profile == "test_declaration"
                else PI_SYSTEM_PROMPT
            ),
            model=model,
            toolkit=toolkit,
            state=AgentState(
                permission_context=PermissionContext(mode=PermissionMode.BYPASS),
            ),
            offloader=self.workspace,
            middlewares=[
                PiReasoningRetryMiddleware(on_retry=self._record_retry),
            ],
            model_config=ModelConfig(max_retries=0),
            context_config=create_pi_context_config(),
            react_config=ReActConfig(
                max_iters=self.config.max_iters,
                stop_on_reject=False,
                interruption_raise_cancelled_error=True,
            ),
            compression_policy=self.compression_policy,
        )
        self.agent.on_compaction = self._record_compaction
        if hasattr(model, "on_failover"):
            model.on_failover = self._record_model_failover
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._status = "READY"
        _atomic_write_json(
            self.run_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "started_at": self._started_at,
                "workdir": str(self.config.workdir),
                "agentscope_version": installed,
                "model": {
                    "name": str(model.model),
                    "context_size": int(model.context_size),
                    "stream": bool(model.stream),
                },
                "max_iters": self.config.max_iters,
                "skill_dirs": [str(item) for item in self.config.skill_dirs],
                "tools": [schema["function"]["name"] for schema in tool_schemas],
                "tool_profile": self.config.tool_profile,
                "permission_mode": "BYPASS_PROMPT_CONSTRAINT_ONLY",
                "compression": asdict(self.compression_policy),
            },
        )
        self._write_report()

    def _record_compaction(self, event: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        with (self.run_dir / "tool_audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if event.get("event") == "compression_start" and self._active_turn_counters is not None:
            self._active_turn_counters["compactions"] += 1
        if self._stream is not None:
            if event["event"] == "compression_start":
                self._line(
                    f"[Compression Start] tokens={event['tokens_before']} "
                    f"trigger={event['trigger_tokens']}",
                )
            else:
                self._line("[Compression End]")

    def _record_model_failover(self, event: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        with (self.run_dir / "tool_audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self._active_turn_counters is not None:
            self._active_turn_counters["api_failovers"] += 1
        self._line(
            f"[API Failover] slot={event['from_slot']}->{event['to_slot']} "
            f"reason={event['reason']}",
        )

    def _record_retry(self, event: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        with (self.run_dir / "tool_audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        reason = event["reason"]
        delay = event["delay_seconds"]
        self._line(f"[Model Retry] reason={reason} delay={delay:g}s")

    def _line(self, text: str = "") -> None:
        assert self._stream is not None
        print(text, file=self._stream, flush=True)

    async def run_turn(self, prompt: str) -> AgentTurnResult:
        if self.agent is None:
            raise RuntimeError("PiAgentRuntime.initialize() must be called first")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        started = time.monotonic()
        model_calls = 0
        input_tokens = 0
        output_tokens = 0
        counters = {"tool_calls": 0, "tool_errors": 0, "api_failovers": 0, "compactions": 0}
        self._active_turn_counters = counters
        stream_event_counts: dict[str, int] = {}
        text_block_ids: list[str] = []
        thinking_block_ids: list[str] = []
        text_blocks: dict[str, str] = {}
        text_order: list[str] = []
        pending_tools: dict[str, dict[str, Any]] = {}
        finished_reason = "unknown"
        status = "RUNNING"

        try:
            async for event in self.agent.reply_stream(
                UserMsg(name="user", content=prompt),
            ):
                event_name = type(event).__name__
                stream_event_counts[event_name] = stream_event_counts.get(event_name, 0) + 1
                if isinstance(event, ReplyStartEvent):
                    self._line(f"[Reply Start] {event.name}")
                elif isinstance(event, ModelCallStartEvent):
                    model_calls += 1
                    self._line(f"[Model] {event.model_name}")
                elif isinstance(event, ModelCallEndEvent):
                    input_tokens += int(event.input_tokens or 0)
                    output_tokens += int(event.output_tokens or 0)
                    self._line(
                        f"[Model Usage] input={event.input_tokens} "
                        f"output={event.output_tokens} "
                        f"reason={_event_value(event.finished_reason)}",
                    )
                elif isinstance(event, ToolCallStartEvent):
                    counters["tool_calls"] += 1
                    self._line(f"[Tool Call] {event.tool_call_name}")
                    pending_tools[event.tool_call_id] = {
                        "tool_call_id": event.tool_call_id,
                        "tool_name": event.tool_call_name,
                        "arguments": "",
                        "result_chars": 0,
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                        "started_monotonic": time.monotonic(),
                    }
                elif isinstance(event, ToolCallDeltaEvent):
                    self._stream.write(event.delta)
                    self._stream.flush()
                    if event.tool_call_id in pending_tools:
                        pending_tools[event.tool_call_id]["arguments"] += event.delta
                elif isinstance(event, ToolCallEndEvent):
                    self._line()
                elif isinstance(event, ToolResultStartEvent):
                    self._line(f"[Tool Result] {event.tool_call_name}")
                elif isinstance(event, ToolResultTextDeltaEvent):
                    self._stream.write(event.delta)
                    self._stream.flush()
                    if event.tool_call_id in pending_tools:
                        pending_tools[event.tool_call_id]["result_chars"] += len(event.delta)
                elif isinstance(event, ToolResultEndEvent):
                    self._line()
                    result_state = _event_value(event.state)
                    if result_state.lower() == "error":
                        counters["tool_errors"] += 1
                    self._line(f"[Tool Result End] {result_state}")
                    pending = pending_tools.pop(event.tool_call_id, None)
                    if pending is not None:
                        pending["status"] = result_state
                        pending["finished_at"] = datetime.now().isoformat(timespec="seconds")
                        pending["duration_seconds"] = round(
                            time.monotonic() - pending.pop("started_monotonic"),
                            6,
                        )
                        self._append_tool_audit(pending)
                elif isinstance(event, TextBlockDeltaEvent):
                    if event.block_id not in text_block_ids:
                        text_block_ids.append(event.block_id)
                    if event.block_id not in text_blocks:
                        text_blocks[event.block_id] = ""
                        text_order.append(event.block_id)
                    text_blocks[event.block_id] += event.delta
                    self._stream.write(event.delta)
                    self._stream.flush()
                elif isinstance(event, ThinkingBlockDeltaEvent):
                    if event.block_id not in thinking_block_ids:
                        thinking_block_ids.append(event.block_id)
                    self._stream.write(event.delta)
                    self._stream.flush()
                elif isinstance(event, ExceedMaxItersEvent):
                    status = "MAX_ITERS"
                    self._line(f"[Max Iters] {event.name}")
                elif isinstance(event, ReplyEndEvent):
                    finished_reason = _event_value(event.finished_reason)
                    if status == "RUNNING":
                        status = (
                            "SUCCESS"
                            if finished_reason == "completed"
                            else finished_reason.upper()
                        )
                    self._line()
                    self._line(f"[Reply End] {finished_reason}")
            if status == "RUNNING":
                status = "SUCCESS"
        except asyncio.CancelledError:
            status = "INTERRUPTED"
            finished_reason = "interrupted"
            raise
        except BaseException:
            status = "ERROR"
            finished_reason = "error"
            raise
        finally:
            for pending in pending_tools.values():
                pending["status"] = "INTERRUPTED"
                pending["finished_at"] = datetime.now().isoformat(timespec="seconds")
                pending["duration_seconds"] = round(
                    time.monotonic() - pending.pop("started_monotonic"),
                    6,
                )
                self._append_tool_audit(pending)
            text = text_blocks[text_order[-1]] if text_order else self._latest_text()
            result = AgentTurnResult(
                status=status,
                text=text,
                finished_reason=finished_reason,
                model_calls=model_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                react_iterations=int(self.agent.state.cur_iter or 0),
                duration_seconds=round(time.monotonic() - started, 6),
                tool_calls=counters["tool_calls"],
                tool_errors=counters["tool_errors"],
                api_failovers=counters["api_failovers"],
                compactions=counters["compactions"],
                stream_event_counts=stream_event_counts,
                text_block_ids=tuple(text_block_ids),
                thinking_block_ids=tuple(thinking_block_ids),
            )
            self._turns.append(asdict(result))
            self._status = status
            self._write_report()
            self._active_turn_counters = None
        return result

    def _latest_text(self) -> str:
        if self.agent is None:
            return ""
        for message in reversed(self.agent.state.context):
            if isinstance(message, Msg) and message.role == "assistant":
                return "".join(
                    block.text
                    for block in message.content
                    if isinstance(block, TextBlock)
                )
        return ""

    def _append_tool_audit(self, payload: dict[str, Any]) -> None:
        assert self.run_dir is not None
        with (self.run_dir / "tool_audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_report(self) -> None:
        if self.run_dir is None:
            return
        model = self.agent.model if self.agent is not None else None
        _atomic_write_json(
            self.run_dir / "run_report.json",
            {
                "schema_version": 1,
                "status": self._status,
                "started_at": self._started_at,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "model": {
                    "name": str(getattr(model, "model", "")),
                    "context_size": int(getattr(model, "context_size", M3_CONTEXT_SIZE)),
                },
                "compression": {
                    "trigger_tokens": self.compression_policy.trigger_tokens,
                    "reserve_tokens": self.compression_policy.reserve_tokens,
                    "keep_recent_tokens": self.compression_policy.keep_recent_tokens,
                },
                "turns": self._turns,
            },
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.workspace is not None:
                await self.workspace.close()
        finally:
            model = self.agent.model if self.agent is not None else None
            closer = getattr(model, "aclose", None) or getattr(model, "close", None)
            if closer is not None:
                result = closer()
                if inspect.isawaitable(result):
                    await result
            if self._transcript_handle is not None:
                self._transcript_handle.close()
