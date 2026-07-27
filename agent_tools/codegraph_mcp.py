from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agentscope.mcp import MCPClient, StdioMCPConfig
from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk

from .agentscope2_tools import ToolAuditMiddleware


CODEGRAPH_VERSION = "1.5.0"
CODEGRAPH_TOOL_NAME = "codegraph_explore"
CODEGRAPH_SCOPE_VERSION = "reference_code_roots_v1"
DEFAULT_MAX_FILES = 5
MAX_FILES = 8
FALLBACK_INDEXED_SUFFIXES = {
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _normalized_roots(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path not in roots:
            roots.append(path)
    return tuple(roots)


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("text") or "")
    return str(getattr(block, "text", ""))


def _indexed_paths(project_root: Path) -> tuple[str, ...]:
    database = project_root / ".codegraph" / "codegraph.db"
    if database.is_file():
        uri = f"file:{database}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                return tuple(
                    str(row[0])
                    for row in connection.execute("SELECT path FROM files ORDER BY path")
                    if row and row[0]
                )
        except sqlite3.Error:
            pass
    return tuple(
        path.relative_to(project_root).as_posix()
        for path in sorted(project_root.rglob("*"))
        if path.is_file()
        and path.suffix.casefold() in FALLBACK_INDEXED_SUFFIXES
        and ".codegraph" not in path.parts
    )


def codegraph_source_bundle_sha256(
    project_root: str | Path,
    allowed_roots: Iterable[str | Path],
) -> str:
    root = Path(project_root).expanduser().resolve()
    allowed = _normalized_roots(allowed_roots)
    candidates = [
        (root / relative_path).resolve()
        for relative_path in _indexed_paths(root)
        if (root / relative_path).resolve().is_file()
        and any(_within((root / relative_path).resolve(), allowed_root) for allowed_root in allowed)
    ]
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def codegraph_manifest_identity(
    *,
    enabled: bool,
    project_root: str | Path,
    allowed_roots: Iterable[str | Path],
) -> dict[str, Any]:
    return {
        "codegraph_enabled": bool(enabled),
        "codegraph_version": CODEGRAPH_VERSION if enabled else "",
        "codegraph_tool": CODEGRAPH_TOOL_NAME if enabled else "",
        "codegraph_scope_version": CODEGRAPH_SCOPE_VERSION,
        "codegraph_source_bundle_sha256": (
            codegraph_source_bundle_sha256(project_root, allowed_roots) if enabled else ""
        ),
    }


def _resolve_codegraph_binary() -> Path:
    value = shutil.which("codegraph")
    if not value:
        raise RuntimeError(
            "CodeGraph is enabled but the codegraph executable is not available on PATH.",
        )
    return Path(value).expanduser().resolve()


def _installed_codegraph_version(binary: Path) -> str:
    completed = subprocess.run(
        [str(binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
    if match is None:
        raise RuntimeError("Unable to determine the installed CodeGraph version.")
    return match.group(1)


def _codegraph_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL")
        if os.environ.get(key)
    }
    environment["CODEGRAPH_TELEMETRY"] = "0"
    environment["CODEGRAPH_MCP_TOOLS"] = "explore"
    return environment


def validate_codegraph_installation(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    database = root / ".codegraph" / "codegraph.db"
    if not database.is_file():
        raise RuntimeError(
            f"CodeGraph is enabled but no initialized index exists at {root / '.codegraph'}; "
            "run codegraph init manually before starting the experiment.",
        )
    binary = _resolve_codegraph_binary()
    installed = _installed_codegraph_version(binary)
    if installed != CODEGRAPH_VERSION:
        raise RuntimeError(
            f"Data Cleaning Agent requires CodeGraph {CODEGRAPH_VERSION}; installed={installed}",
        )
    return {
        "project_root": str(root),
        "binary": str(binary),
        "version": installed,
    }


def _json_lines(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def summarize_codegraph_usage(root: str | Path) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    events = [
        event
        for path in sorted(base.rglob("codegraph_usage.jsonl"))
        for event in _json_lines(path)
    ]
    status_counts: dict[str, int] = {}
    returned_files: set[str] = set()
    returned_file_count = 0
    for event in events:
        status = str(event.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        files = [str(path) for path in event.get("returned_files") or []]
        returned_file_count += len(files)
        returned_files.update(files)

    tool_counts = {"Read": 0, "Grep": 0}
    for path in sorted(base.rglob("tool_audit.jsonl")):
        for event in _json_lines(path):
            tool = str(event.get("tool") or "")
            if tool in tool_counts:
                tool_counts[tool] += 1

    model_tokens = 0
    for path in sorted(base.rglob("skill_usage_report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_tokens += int((report.get("agent_usage") or {}).get("total_tokens") or 0)

    return {
        "call_count": len(events),
        "status_counts": dict(sorted(status_counts.items())),
        "returned_file_count": returned_file_count,
        "returned_files": sorted(returned_files),
        "output_characters": sum(int(event.get("output_characters") or 0) for event in events),
        "duration_seconds": round(
            sum(float(event.get("duration_seconds") or 0.0) for event in events),
            6,
        ),
        "stale_calls": sum(bool(event.get("stale")) for event in events),
        "fallback_calls": sum(bool(event.get("fallback")) for event in events),
        "read_calls": tool_counts["Read"],
        "grep_calls": tool_counts["Grep"],
        "model_tokens": model_tokens,
    }


@dataclass
class CodeGraphRuntime:
    client: Any
    tool: "RestrictedCodeGraphExplore"

    async def close(self) -> None:
        if getattr(self.client, "is_connected", False):
            await self.client.close()


async def open_codegraph_runtime(
    *,
    project_root: str | Path,
    allowed_roots: Iterable[str | Path],
    usage_path: str | Path,
    audit_path: str | Path | None = None,
    attempt: str,
) -> CodeGraphRuntime:
    root = Path(project_root).expanduser().resolve()
    installation = validate_codegraph_installation(root)
    binary = Path(installation["binary"])
    client = MCPClient(
        name="codegraph",
        is_stateful=True,
        mcp_config=StdioMCPConfig(
            command=str(binary),
            args=["serve", "--mcp", "--path", str(root)],
            cwd=root,
            env=_codegraph_environment(),
        ),
        enable_tools=[CODEGRAPH_TOOL_NAME],
        execution_timeout=120.0,
    )
    try:
        await client.connect()
        native_tool = await client.get_tool(CODEGRAPH_TOOL_NAME)
        tool = RestrictedCodeGraphExplore(
            native_tool=native_tool,
            project_root=root,
            allowed_roots=allowed_roots,
            usage_path=usage_path,
            audit_path=audit_path,
            attempt=attempt,
        )
        return CodeGraphRuntime(client=client, tool=tool)
    except Exception:
        if getattr(client, "is_connected", False):
            await client.close()
        raise


async def probe_codegraph_connection(
    *,
    project_root: str | Path,
    allowed_roots: Iterable[str | Path],
) -> None:
    runtime = await open_codegraph_runtime(
        project_root=project_root,
        allowed_roots=allowed_roots,
        usage_path=Path(project_root) / ".codegraph" / "preflight_usage.jsonl",
        audit_path=Path(project_root) / ".codegraph" / "preflight_tool_audit.jsonl",
        attempt="preflight",
    )
    await runtime.close()


class RestrictedCodeGraphExplore(ToolBase):
    name = "CodeGraphExplore"
    description = (
        "Explore authorized project source, call paths, and impact using the local "
        "CodeGraph index. Use Read/Grep for data, configs, docs, current attempt "
        "files, denied results, or stale files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Question about authorized indexed project source.",
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum source files to return.",
                "default": DEFAULT_MAX_FILES,
                "minimum": 1,
                "maximum": MAX_FILES,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    is_read_only = True
    is_concurrency_safe = False

    def __init__(
        self,
        *,
        native_tool: Any,
        project_root: str | Path,
        allowed_roots: Iterable[str | Path],
        usage_path: str | Path,
        audit_path: str | Path | None = None,
        attempt: str,
    ) -> None:
        self.native_tool = native_tool
        self.project_root = Path(project_root).expanduser().resolve()
        self.allowed_roots = _normalized_roots(allowed_roots)
        self.usage_path = Path(usage_path).expanduser().resolve()
        self.audit_path = (
            Path(audit_path).expanduser().resolve()
            if audit_path is not None
            else self.usage_path.with_name("tool_audit.jsonl")
        )
        super().__init__(middlewares=[ToolAuditMiddleware(self.audit_path)])
        self.attempt = str(attempt)
        self.indexed_paths = _indexed_paths(self.project_root)

    async def check_permissions(
        self,
        _tool_input: dict[str, Any],
        _context: PermissionContext,
    ) -> PermissionDecision:
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="CodeGraph access is constrained to the host-selected source roots.",
        )

    async def call(
        self,
        query: str,
        max_files: int = DEFAULT_MAX_FILES,
        **unknown: Any,
    ) -> ToolChunk:
        started = time.monotonic()
        if unknown:
            return self._rejected(
                started,
                "CodeGraph request contains unsupported arguments; use Read/Grep instead.",
            )
        if not isinstance(query, str) or not query.strip():
            return self._rejected(started, "CodeGraph query must be a non-empty string.")
        if (
            not isinstance(max_files, int)
            or isinstance(max_files, bool)
            or not 1 <= max_files <= MAX_FILES
        ):
            return self._rejected(
                started,
                f"CodeGraph max_files must be between 1 and {MAX_FILES}.",
            )
        forbidden_path = self._forbidden_absolute_path(query)
        if forbidden_path is not None:
            return self._rejected(
                started,
                "CodeGraph query references a path outside authorized source roots.",
            )

        try:
            result = await self.native_tool.call(
                query=query.strip(),
                maxFiles=max_files,
                projectPath=str(self.project_root),
            )
        except Exception:
            self._record(
                query=query,
                max_files=max_files,
                status="error",
                started=started,
                returned_files=[],
                output_characters=0,
                stale=False,
                fallback=True,
            )
            return ToolChunk(
                content=[
                    TextBlock(
                        text=(
                            "CodeGraph is unavailable; continue with authorized "
                            "Read/Grep tools."
                        ),
                    ),
                ],
                state=ToolResultState.ERROR,
            )

        text = "\n".join(_block_text(block) for block in result.content)
        returned_files = self._returned_files(text)
        unauthorized = [path for path in returned_files if not self._path_allowed(path)]
        if unauthorized:
            self._record(
                query=query,
                max_files=max_files,
                status="denied",
                started=started,
                returned_files=[],
                output_characters=0,
                stale=False,
                fallback=True,
            )
            return ToolChunk(
                content=[
                    TextBlock(
                        text=(
                            "CodeGraph response crossed the authorized source boundary; "
                            "use Read/Grep instead."
                        ),
                    ),
                ],
                state=ToolResultState.DENIED,
            )

        stale = bool(
            re.search(
                r"\bstale\b|staleness|pending re-index|auto-sync is disabled",
                text,
                re.I,
            ),
        )
        state = getattr(result, "state", ToolResultState.RUNNING)
        status = "error" if state is ToolResultState.ERROR else "success"
        self._record(
            query=query,
            max_files=max_files,
            status=status,
            started=started,
            returned_files=returned_files,
            output_characters=len(text),
            stale=stale,
            fallback=stale or status == "error",
        )
        return result

    def _forbidden_absolute_path(self, query: str) -> Path | None:
        for value in re.findall(r"(?<![\w.-])(/[^\s，。；;：:'\"<>]+)", query):
            path = Path(value).expanduser().resolve()
            if not any(_within(path, root) for root in self.allowed_roots):
                return path
        return None

    def _returned_files(self, text: str) -> list[str]:
        return [path for path in self.indexed_paths if path in text]

    def _path_allowed(self, relative_path: str) -> bool:
        path = (self.project_root / relative_path).resolve()
        return any(_within(path, root) for root in self.allowed_roots)

    def _rejected(self, started: float, message: str) -> ToolChunk:
        self._record(
            query="",
            max_files=0,
            status="denied",
            started=started,
            returned_files=[],
            output_characters=0,
            stale=False,
            fallback=True,
        )
        return ToolChunk(content=[TextBlock(text=message)], state=ToolResultState.DENIED)

    def _record(
        self,
        *,
        query: str,
        max_files: int,
        status: str,
        started: float,
        returned_files: list[str],
        output_characters: int,
        stale: bool,
        fallback: bool,
    ) -> None:
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "attempt": self.attempt,
            "query": query,
            "max_files": max_files,
            "project_root": str(self.project_root),
            "returned_files": returned_files,
            "status": status,
            "duration_seconds": round(time.monotonic() - started, 6),
            "output_characters": output_characters,
            "stale": stale,
            "fallback": fallback,
        }
        with self.usage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
