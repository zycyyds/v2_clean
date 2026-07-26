from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk, ToolMiddlewareBase

from .context import EngineerToolContext
from .tools import EngineerTools, _tool_payload, atomic_write_text


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _text(value: str, description: str, *, default: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if default is not None:
        schema["default"] = default
    return schema


def _integer(description: str, default: int, minimum: int, maximum: int) -> dict[str, Any]:
    return {
        "type": "integer",
        "description": description,
        "default": default,
        "minimum": minimum,
        "maximum": maximum,
    }


def _boolean(description: str, default: bool) -> dict[str, Any]:
    return {"type": "boolean", "description": description, "default": default}


class ToolAuditMiddleware(ToolMiddlewareBase):
    """Persist compact, append-only tool traces without logging file contents."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    async def on_tool_call(
        self,
        tool: ToolBase,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[ToolChunk, None]],
    ) -> AsyncGenerator[ToolChunk, None]:
        started = time.monotonic()
        state = "success"
        output_bytes = 0
        output_chunks = 0
        error_parts: list[str] = []
        try:
            async for chunk in next_handler(**input_kwargs):
                output_chunks += 1
                if hasattr(chunk, "model_dump_json"):
                    serialized = chunk.model_dump_json(exclude_none=True)
                else:
                    serialized = str(chunk)
                output_bytes += len(serialized.encode("utf-8", errors="replace"))
                if chunk.state == ToolResultState.ERROR:
                    state = "error"
                    error_parts.extend(
                        str(getattr(block, "text", "")).strip()
                        for block in chunk.content
                        if str(getattr(block, "text", "")).strip()
                    )
                yield chunk
        except Exception as exc:
            state = "exception"
            error_parts.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "tool": tool.name,
                "state": state,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "input_keys": sorted(input_kwargs),
                "output_chunks": output_chunks,
                "output_bytes": output_bytes,
                "error_summary": " | ".join(error_parts)[:1000],
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ScopedMethodTool(ToolBase):
    """AgentScope 2.x ToolBase adapter around the retained scoped implementations."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        method: Callable[..., Any],
        is_read_only: bool,
        is_concurrency_safe: bool,
        middlewares: list[ToolMiddlewareBase] | None = None,
    ) -> None:
        super().__init__(middlewares=middlewares)
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.method = method
        self.is_read_only = is_read_only
        self.is_concurrency_safe = is_concurrency_safe
        self.context = method.__self__.context

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        try:
            if self.name in {"Read", "InspectDataFile"}:
                self.context.resolve_read_path(tool_input.get("file_path", ""))
            elif self.name in {"Glob", "Grep"}:
                self.context.resolve_read_path(
                    tool_input.get("path") or str(self.context.workspace_dir),
                )
            elif self.name == "CompareArtifact":
                self.context.resolve_read_path(tool_input.get("expected_path", ""))
                self.context.resolve_read_path(tool_input.get("actual_path", ""))
            elif self.name in {"Write", "Edit", "RunAnalysisPython"}:
                self.context.resolve_write_path(tool_input.get("file_path", ""))
        except Exception as exc:
            return PermissionDecision(
                behavior=PermissionBehavior.DENY,
                message=f"Scoped path permission denied: {exc}",
            )
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="Path is inside the tool's explicit experiment scope.",
        )

    async def call(self, **kwargs: Any) -> ToolChunk:
        response = self.method(**kwargs)
        payload = _tool_payload(response)
        status = str(payload.get("status") or "NEEDS_REPAIR").upper()
        summary = str(payload.get("summary") or self.name)
        issues = [str(item) for item in payload.get("issues") or []]
        if issues:
            summary += "\n" + "\n".join(f"- {item}" for item in issues[:20])
        return ToolChunk(
            content=[TextBlock(text=summary)],
            state=(ToolResultState.SUCCESS if status == "SUCCESS" else ToolResultState.ERROR),
            metadata=payload,
        )


@dataclass
class ValidationToolServices:
    """Host callbacks and draft state shared by validation boundary tools."""

    run_pipeline: Callable[[str], dict[str, Any]]
    validate_draft: Callable[[], dict[str, Any]]
    current_pipeline_hash: Callable[[], str]
    last_draft_hash: str = ""
    last_draft_valid: bool = False
    last_draft_report: dict[str, Any] = field(default_factory=dict)


class RunPipelineTool(ToolBase):
    name = "RunPipeline"
    description = (
        "Run workspace/pipeline/run.py through the host sandbox for train or validation. "
        "The host chooses raw, keys and output paths; arbitrary commands are not accepted."
    )
    input_schema = _object_schema(
        {
            "split_mode": {
                "type": "string",
                "enum": ["train", "validation"],
                "description": "The authorized split to process.",
            }
        },
        ["split_mode"],
    )
    is_read_only = False
    is_concurrency_safe = False

    def __init__(
        self,
        services: ValidationToolServices,
        middlewares: list[ToolMiddlewareBase] | None = None,
    ) -> None:
        super().__init__(middlewares=middlewares)
        self.services = services

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="Pipeline execution is host-scoped.")

    async def call(self, split_mode: str) -> ToolChunk:
        result = self.services.run_pipeline(split_mode)
        success = int(result.get("returncode", 1)) == 0
        return ToolChunk(
            content=[TextBlock(text=str(result.get("summary") or ("Pipeline completed." if success else "Pipeline failed.")))],
            state=ToolResultState.SUCCESS if success else ToolResultState.ERROR,
            metadata=result,
        )


class ValidateDraftTool(ToolBase):
    name = "ValidateDraft"
    description = (
        "Run hidden-reference-free structural, package, lineage and replay checks for the current "
        "workspace Pipeline. This does not create an attempt or compute a score."
    )
    input_schema = _object_schema({})
    is_read_only = False
    is_concurrency_safe = False

    def __init__(
        self,
        services: ValidationToolServices,
        middlewares: list[ToolMiddlewareBase] | None = None,
    ) -> None:
        super().__init__(middlewares=middlewares)
        self.services = services

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="Draft validation is host-scoped.")

    async def call(self) -> ToolChunk:
        report = self.services.validate_draft()
        valid = bool(report.get("valid"))
        self.services.last_draft_hash = self.services.current_pipeline_hash()
        self.services.last_draft_valid = valid
        self.services.last_draft_report = report
        issues = [str(item) for item in report.get("issues") or []]
        text = "Draft validation passed." if valid else "Draft validation failed."
        if issues:
            text += "\n" + "\n".join(f"- {item}" for item in issues[:30])
        return ToolChunk(
            content=[TextBlock(text=text)],
            state=ToolResultState.SUCCESS if valid else ToolResultState.ERROR,
            metadata=report,
        )


class SubmitCandidateTool(ToolBase):
    name = "SubmitCandidate"
    description = (
        "Submit the current validated Pipeline and complete result_package to the host. "
        "The Agent pauses while the host applies gates, hidden scoring and best promotion."
    )
    input_schema = _object_schema(
        {
            "summary": _text("", "Concise statement of the verified business change."),
            "changed_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workspace Pipeline files intentionally changed.",
                "default": [],
            },
        },
        ["summary"],
    )
    is_read_only = False
    is_concurrency_safe = False
    is_external_tool = True

    def __init__(
        self,
        services: ValidationToolServices,
        middlewares: list[ToolMiddlewareBase] | None = None,
    ) -> None:
        super().__init__(middlewares=middlewares)
        self.services = services

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        current_hash = self.services.current_pipeline_hash()
        if not self.services.last_draft_valid:
            return PermissionDecision(
                behavior=PermissionBehavior.DENY,
                message="Call ValidateDraft and fix all reported issues before SubmitCandidate.",
            )
        if not current_hash or current_hash != self.services.last_draft_hash:
            return PermissionDecision(
                behavior=PermissionBehavior.DENY,
                message="Pipeline changed after ValidateDraft; run ValidateDraft again.",
            )
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="The latest draft hash is valid and unchanged.",
        )


@dataclass
class TestRunnerSpecServices:
    runner_spec_path: Path
    expected_spec: dict[str, Any]
    validate_runner_spec: Callable[[], dict[str, Any]]


class WriteRunnerSpecTool(ToolBase):
    name = "WriteRunnerSpec"
    description = (
        "Write the host-controlled runner_spec.json for the frozen Pipeline. "
        "Only optional notes are supplied by the Agent; paths and hashes are injected by the host."
    )
    input_schema = _object_schema({"notes": _text("", "Optional path or environment observations.", default="")})
    is_read_only = False
    is_concurrency_safe = False

    def __init__(self, services: TestRunnerSpecServices) -> None:
        super().__init__()
        self.services = services

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="Runner spec fields are host-controlled.")

    async def call(self, notes: str = "") -> ToolChunk:
        payload = {**self.services.expected_spec, "notes": str(notes or "")}
        atomic_write_text(
            self.services.runner_spec_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        return ToolChunk(
            content=[TextBlock(text=f"Wrote {self.services.runner_spec_path}.")],
            state=ToolResultState.SUCCESS,
            metadata={"runner_spec": str(self.services.runner_spec_path), **payload},
        )


class ValidateRunnerSpecTool(ToolBase):
    name = "ValidateRunnerSpec"
    description = "Validate runner_spec.json and frozen Pipeline structure without reading test gold data."
    input_schema = _object_schema({})
    is_read_only = True
    is_concurrency_safe = False

    def __init__(self, services: TestRunnerSpecServices) -> None:
        super().__init__()
        self.services = services

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="Runner validation is hidden-reference-free.")

    async def call(self) -> ToolChunk:
        report = self.services.validate_runner_spec()
        valid = bool(report.get("valid"))
        issues = [str(item) for item in report.get("issues") or []]
        text = "Runner spec validation passed." if valid else "Runner spec validation failed."
        if issues:
            text += "\n" + "\n".join(f"- {item}" for item in issues[:20])
        return ToolChunk(
            content=[TextBlock(text=text)],
            state=ToolResultState.SUCCESS if valid else ToolResultState.ERROR,
            metadata=report,
        )


def build_scoped_atomic_tools(
    context: EngineerToolContext,
    *,
    audit_path: str | Path,
) -> list[ToolBase]:
    """Build the always-visible validation file and analysis tools."""
    impl = EngineerTools(context)
    audit = [ToolAuditMiddleware(audit_path)]
    return [
        ScopedMethodTool(
            name="Read",
            description="Read an authorized UTF-8 text file with one-based line pagination.",
            input_schema=_object_schema(
                {
                    "file_path": _text("", "Authorized file path."),
                    "offset": _integer("One-based first line.", 1, 1, 10_000_000),
                    "limit": _integer("Maximum lines.", 400, 1, 2000),
                },
                ["file_path"],
            ),
            method=impl.Read,
            is_read_only=True,
            is_concurrency_safe=True,
            middlewares=audit,
        ),
        ScopedMethodTool(
            name="Glob",
            description="Find files under an authorized directory. Directories are omitted.",
            input_schema=_object_schema(
                {
                    "pattern": _text("", "Glob pattern; use **/* recursively."),
                    "path": _text("", "Authorized base directory.", default=""),
                    "limit": _integer("Maximum returned files.", 200, 1, 10_000),
                },
                ["pattern"],
            ),
            method=impl.Glob,
            is_read_only=True,
            is_concurrency_safe=True,
            middlewares=audit,
        ),
        ScopedMethodTool(
            name="Grep",
            description="Regex search across authorized text files.",
            input_schema=_object_schema(
                {
                    "pattern": _text("", "Regular expression."),
                    "path": _text("", "Authorized file or directory.", default=""),
                    "glob": _text("", "Optional file glob.", default=""),
                    "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"], "default": "files_with_matches"},
                    "ignore_case": _boolean("Case-insensitive matching.", False),
                    "limit": _integer("Maximum returned entries.", 200, 1, 10_000),
                },
                ["pattern"],
            ),
            method=impl.Grep,
            is_read_only=True,
            is_concurrency_safe=True,
            middlewares=audit,
        ),
        ScopedMethodTool(
            name="InspectDataFile",
            description="Inspect schema, rows, samples and missingness of an authorized structured data file.",
            input_schema=_object_schema(
                {
                    "file_path": _text("", "CSV, TSV, gzip, parquet, JSON or spreadsheet path."),
                    "sample_rows": _integer("Number of sample rows.", 3, 0, 50),
                    "max_columns": _integer("Maximum reported columns.", 80, 1, 500),
                },
                ["file_path"],
            ),
            method=impl.InspectDataFile,
            is_read_only=True,
            is_concurrency_safe=True,
            middlewares=audit,
        ),
        ScopedMethodTool(
            name="CompareArtifact",
            description="Compare two authorized structured artifacts as a public train diagnostic.",
            input_schema=_object_schema(
                {
                    "expected_path": _text("", "Expected public train artifact."),
                    "actual_path": _text("", "Generated artifact."),
                    "key_columns_json": _text("", "JSON list of key columns.", default="[]"),
                    "numeric_tolerance": {"type": "number", "minimum": 0, "default": 1e-8},
                    "ignore_row_order": _boolean("Ignore source row ordering.", True),
                    "normalize_empty": _boolean("Normalize empty sentinels.", True),
                    "limit": _integer("Maximum examples.", 10, 1, 100),
                },
                ["expected_path", "actual_path"],
            ),
            method=impl.CompareArtifact,
            is_read_only=False,
            is_concurrency_safe=False,
            middlewares=audit,
        ),
        ScopedMethodTool(
            name="Write",
            description="Create or overwrite a UTF-8 file inside the current Agent workspace; existing files require Read first.",
            input_schema=_object_schema(
                {"file_path": _text("", "Workspace-relative or authorized writable path."), "content": _text("", "Complete UTF-8 content.")},
                ["file_path", "content"],
            ),
            method=impl.Write,
            is_read_only=False,
            is_concurrency_safe=False,
            middlewares=audit,
        ),
        ScopedMethodTool(
            name="Edit",
            description="Replace an exact string in a previously read workspace file.",
            input_schema=_object_schema(
                {
                    "file_path": _text("", "Workspace file path."),
                    "old_string": _text("", "Exact existing text."),
                    "new_string": _text("", "Replacement text."),
                    "replace_all": _boolean("Replace every match.", False),
                },
                ["file_path", "old_string", "new_string"],
            ),
            method=impl.Edit,
            is_read_only=False,
            is_concurrency_safe=False,
            middlewares=audit,
        ),
        ScopedMethodTool(
            name="RunAnalysisPython",
            description="Run a workspace Python analysis script in the host audit sandbox. Network and child processes are forbidden.",
            input_schema=_object_schema(
                {
                    "file_path": _text("", "Workspace Python script."),
                    "args": {"type": "array", "items": {"type": "string"}, "default": []},
                    "timeout_seconds": _integer("Execution timeout.", 300, 1, 600),
                    "task_id": _text("", "Optional diagnostic task id.", default=""),
                },
                ["file_path"],
            ),
            method=impl.ExecutePython,
            is_read_only=False,
            is_concurrency_safe=False,
            middlewares=audit,
        ),
    ]


def build_test_tools(
    context: EngineerToolContext,
    services: TestRunnerSpecServices,
    *,
    audit_path: str | Path,
) -> list[ToolBase]:
    """Expose a read-mostly Test toolkit with one host-controlled write tool."""
    allowed_read_tools = {"Read", "Glob", "Grep", "InspectDataFile"}
    tools = [
        tool
        for tool in build_scoped_atomic_tools(context, audit_path=audit_path)
        if tool.name in allowed_read_tools
    ]
    tools.extend([WriteRunnerSpecTool(services), ValidateRunnerSpecTool(services)])
    return tools
