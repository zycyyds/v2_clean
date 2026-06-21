from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from lib.agent_artifacts import begin_step, record_step
from workflow.skill_adapter import parse_variant_result, write_variant_execution_receipt

from .context import EngineerToolContext


def _response(
    status: str,
    summary: str,
    artifacts: dict[str, Any] | None = None,
    issues: Iterable[str] = (),
) -> ToolResponse:
    payload = {
        "status": status,
        "summary": summary,
        "artifacts": artifacts or {},
        "issues": list(issues),
    }
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, indent=2),
            ),
        ],
    )


def _truncate(text: str, limit: int = 8_000) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = limit // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:], True


def _diff(before: str, after: str, path: Path) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        ),
    )


class EngineerTools:
    """Clawd-style atomic tools bound to one Engineer phase."""

    def __init__(self, context: EngineerToolContext) -> None:
        self.context = context

    def Read(self, file_path: str, offset: int = 1, limit: int = 400) -> ToolResponse:
        """Read a UTF-8 text file with one-based line pagination."""
        try:
            if offset < 1 or limit < 1 or limit > 2_000:
                raise ValueError("offset must be >= 1 and limit must be between 1 and 2000")
            path = self.context.resolve_read_path(file_path)
            if not path.exists() or not path.is_file():
                raise ValueError(f"File does not exist: {path}")
            raw = path.read_bytes()
            if b"\x00" in raw[:8192]:
                raise ValueError(f"Binary files are not supported by Read: {path}")
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            selected = lines[offset - 1 : offset - 1 + limit]
            content = "\n".join(
                f"{line_no}\t{line}"
                for line_no, line in enumerate(selected, start=offset)
            )
            content, content_truncated = _truncate(content, limit=12_000)
            self.context.mark_read(path)
            return _response(
                "SUCCESS",
                f"Read {len(selected)} lines from {path.name}.",
                {
                    "file_path": str(path),
                    "content": content,
                    "start_line": offset,
                    "num_lines": len(selected),
                    "total_lines": len(lines),
                    "truncated": content_truncated or offset - 1 + len(selected) < len(lines),
                    "content_truncated": content_truncated,
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Read failed.", issues=[str(exc)])

    def Glob(
        self,
        pattern: str,
        path: str = "",
        limit: int = 200,
    ) -> ToolResponse:
        """Find files under an authorized directory using a glob pattern."""
        try:
            if not pattern or limit < 1 or limit > 10_000:
                raise ValueError("pattern is required and limit must be between 1 and 10000")
            base = self.context.resolve_read_path(path or str(self.context.workspace_dir))
            if not base.exists() or not base.is_dir():
                raise ValueError(f"Glob path is not a directory: {base}")
            all_files: list[Path] = []
            for candidate in base.glob(pattern):
                if not candidate.is_file():
                    continue
                try:
                    all_files.append(self.context.resolve_read_path(candidate))
                except Exception:
                    continue
            all_files.sort(key=lambda candidate: (-candidate.stat().st_mtime_ns, str(candidate)))
            selected = all_files[:limit]
            return _response(
                "SUCCESS",
                f"Glob matched {len(all_files)} files.",
                {
                    "filenames": [str(candidate) for candidate in selected],
                    "num_files": len(selected),
                    "total_matches": len(all_files),
                    "truncated": len(all_files) > limit,
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Glob failed.", issues=[str(exc)])

    def Grep(
        self,
        pattern: str,
        path: str = "",
        glob: str = "",
        output_mode: str = "files_with_matches",
        ignore_case: bool = False,
        limit: int = 200,
    ) -> ToolResponse:
        """Search authorized text files using a regular expression."""
        try:
            if output_mode not in {"content", "files_with_matches", "count"}:
                raise ValueError("output_mode must be content, files_with_matches, or count")
            if limit < 1 or limit > 10_000:
                raise ValueError("limit must be between 1 and 10000")
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
            base = self.context.resolve_read_path(path or str(self.context.workspace_dir))
            if not base.exists():
                raise ValueError(f"Grep path does not exist: {base}")
            candidates = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
            if glob:
                candidates = [
                    candidate
                    for candidate in candidates
                    if fnmatch.fnmatch(candidate.name, glob)
                    or fnmatch.fnmatch(str(candidate), glob)
                ]

            matched_files: list[Path] = []
            matched_lines: list[str] = []
            total_matches = 0
            for candidate in candidates:
                try:
                    candidate = self.context.resolve_read_path(candidate)
                except Exception:
                    continue
                if any(part in {".git", ".svn", ".hg"} for part in candidate.parts):
                    continue
                try:
                    if candidate.stat().st_size > 10 * 1024 * 1024:
                        continue
                    raw = candidate.read_bytes()
                    if b"\x00" in raw[:8192]:
                        continue
                    text = raw.decode("utf-8", errors="replace")
                except OSError:
                    continue
                file_matches = list(regex.finditer(text))
                if not file_matches:
                    continue
                matched_files.append(candidate.resolve())
                total_matches += len(file_matches)
                if output_mode == "content":
                    for line_no, line in enumerate(text.splitlines(), start=1):
                        if regex.search(line):
                            matched_lines.append(f"{candidate.resolve()}:{line_no}:{line}")

            matched_files = matched_files[:limit]
            matched_lines = matched_lines[:limit]
            artifacts: dict[str, Any] = {
                "mode": output_mode,
                "filenames": [str(candidate) for candidate in matched_files],
                "num_files": len(matched_files),
                "num_matches": total_matches,
                "truncated": len(matched_lines) >= limit or len(matched_files) >= limit,
            }
            if output_mode == "content":
                artifacts["content"] = "\n".join(matched_lines)
            return _response("SUCCESS", f"Grep matched {len(matched_files)} files.", artifacts)
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Grep failed.", issues=[str(exc)])

    def Write(self, file_path: str, content: str) -> ToolResponse:
        """Create or overwrite a UTF-8 file inside the Engineer phase."""
        try:
            path = self.context.resolve_write_path(file_path)
            self._guard_variant_metadata(path)
            if path.suffix.lower() == ".py":
                self.context.require_python_plan(path)
            before = ""
            operation = "create"
            if path.exists():
                if not path.is_file():
                    raise ValueError(f"Write target is not a file: {path}")
                if not self.context.was_read_and_unchanged(path):
                    raise ValueError("Refusing to overwrite: call Read first and keep the file unchanged")
                before = path.read_text(encoding="utf-8", errors="replace")
                operation = "update"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self._invalidate_variant_after_source_change(path)
            self.context.forget_read(path)
            full_diff = _diff(before, content, path)
            response_diff, diff_truncated = _truncate(full_diff)
            record = record_step(
                "Write",
                f"{operation.title()}d {path.name}.",
                [path],
                metadata={"operation": operation, "diff": full_diff},
            )
            return _response(
                "SUCCESS",
                f"{operation.title()}d {path}.",
                {
                    "file_path": str(path),
                    "operation": operation,
                    "diff": response_diff,
                    "diff_truncated": diff_truncated,
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Write failed.", issues=[str(exc)])

    def Edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResponse:
        """Replace an exact string in a previously read Engineer file."""
        try:
            path = self.context.resolve_write_path(file_path)
            self._guard_variant_metadata(path)
            if not path.exists() or not path.is_file():
                raise ValueError(f"Edit target does not exist: {path}")
            if not self.context.was_read_and_unchanged(path):
                raise ValueError("Refusing to edit: call Read first and keep the file unchanged")
            before = path.read_text(encoding="utf-8", errors="replace")
            count = before.count(old_string)
            if count == 0:
                raise ValueError("old_string was not found")
            if count > 1 and not replace_all:
                raise ValueError("old_string is not unique; expand it or set replace_all=true")
            after = before.replace(old_string, new_string, -1 if replace_all else 1)
            path.write_text(after, encoding="utf-8")
            self._invalidate_variant_after_source_change(path)
            self.context.forget_read(path)
            replacements = count if replace_all else 1
            full_diff = _diff(before, after, path)
            response_diff, diff_truncated = _truncate(full_diff)
            record = record_step(
                "Edit",
                f"Edited {path.name} with {replacements} replacement(s).",
                [path],
                metadata={"replacements": replacements, "diff": full_diff},
            )
            return _response(
                "SUCCESS",
                f"Edited {path}.",
                {
                    "file_path": str(path),
                    "replacements": replacements,
                    "diff": response_diff,
                    "diff_truncated": diff_truncated,
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Edit failed.", issues=[str(exc)])

    def ExecutePython(
        self,
        file_path: str,
        args: list[str] | None = None,
        timeout_seconds: int = 300,
    ) -> ToolResponse:
        """Execute a Python script with writes restricted to a new step directory."""
        step_ctx = None
        try:
            if timeout_seconds < 1 or timeout_seconds > 600:
                raise ValueError("timeout_seconds must be between 1 and 600")
            script = self.context.resolve_write_path(file_path)
            if not script.exists() or not script.is_file() or script.suffix.lower() != ".py":
                raise ValueError(f"ExecutePython requires an existing .py file: {script}")
            self.context.require_python_plan(script)
            variant_metadata = self._validate_variant_execution(script, args or [])
            step_ctx = begin_step("ExecutePython")
            if step_ctx is None:
                output_dir = Path(self.context.engineer_phase_root) / "artifacts" / "execute_python"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(step_ctx["step_dir"])
            runner = output_dir / "_execute_runner.py"
            runner.write_text(
                _runner_source(
                    output_dir,
                    read_roots=self.context.read_roots,
                ),
                encoding="utf-8",
            )
            before_files = {p.resolve() for p in output_dir.rglob("*") if p.is_file()}

            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_DIR": str(output_dir),
                    "TMPDIR": str(output_dir),
                    "MPLCONFIGDIR": str(output_dir / ".mplconfig"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )
            command = [sys.executable, "-u", str(runner), str(script), *(args or [])]
            timed_out = False
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(output_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                returncode = completed.returncode
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = -1
                stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
                stderr += f"\nTimeoutError: execution exceeded {timeout_seconds} seconds."

            variant_result = None
            after_files = {p.resolve() for p in output_dir.rglob("*") if p.is_file()}
            output_files = sorted(
                str(path)
                for path in after_files - before_files
                if path != runner.resolve()
            )
            status = "SUCCESS" if returncode == 0 and not timed_out else "NEEDS_REPAIR"
            if variant_metadata is not None and status == "SUCCESS":
                self._verify_variant_source_hashes(variant_metadata)
                try:
                    variant_result = parse_variant_result(stdout)
                except Exception as exc:
                    status = "NEEDS_REPAIR"
                    stderr += f"\nVariant protocol error: {exc}"
                else:
                    if variant_result["status"] != "SUCCESS":
                        status = "NEEDS_REPAIR"
                        stderr += "\nVariant reported NEEDS_REPAIR: " + "; ".join(variant_result["issues"])
                if status == "SUCCESS" and variant_result is not None:
                    try:
                        receipt_path = write_variant_execution_receipt(
                            script.parent,
                            variant_result,
                            output_files=[Path(path) for path in output_files],
                        )
                    except Exception as exc:
                        status = "NEEDS_REPAIR"
                        stderr += f"\nVariant receipt error: {exc}"
                    else:
                        output_files.append(str(receipt_path))
            stdout, stdout_truncated = _truncate(stdout)
            stderr, stderr_truncated = _truncate(stderr)
            issues = [] if status == "SUCCESS" else [stderr or f"Python exited with {returncode}"]
            record = record_step(
                "ExecutePython",
                "Python execution succeeded." if status == "SUCCESS" else "Python execution failed.",
                [script, *output_files],
                metadata={
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": timed_out,
                    "script_path": str(script),
                    "variant_name": variant_metadata.get("variant_name", "") if variant_metadata else "",
                },
                status=status,
                issues=issues,
                _step_ctx=step_ctx,
            )
            return _response(
                status,
                "Python execution succeeded." if status == "SUCCESS" else "Python execution failed.",
                {
                    "script_path": str(script),
                    "output_dir": str(output_dir),
                    "output_files": output_files,
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "timed_out": timed_out,
                    "variant_result": variant_result,
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
                issues,
            )
        except Exception as exc:
            if step_ctx is not None:
                record_step(
                    "ExecutePython",
                    "Python execution setup failed.",
                    [],
                    status="NEEDS_REPAIR",
                    issues=[str(exc)],
                    _step_ctx=step_ctx,
                )
            return _response("NEEDS_REPAIR", "ExecutePython failed.", issues=[str(exc)])

    def publish_artifact(self, file_path: str, alias: str) -> ToolResponse:
        """Publish an Engineer output through the existing phase handoff system."""
        try:
            if not alias or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", alias):
                raise ValueError("alias must be a lowercase snake_case identifier")
            path = self.context.resolve_write_path(file_path)
            if not path.exists() or not path.is_file():
                raise ValueError(f"Artifact does not exist: {path}")
            if self.context.require_skill_plan and not self.context.was_read_and_unchanged(path):
                raise ValueError("Read and inspect the final artifact before publish_artifact")
            record = record_step(
                "publish_artifact",
                f"Published {path.name} as {alias}.",
                [path],
                handoffs={alias: path},
            )
            handoffs = (record or {}).get("handoffs", {})
            return _response(
                "SUCCESS",
                f"Published {path.name} as {alias}.",
                {
                    "file_path": str(path),
                    "alias": alias,
                    "handoff": handoffs.get(alias, {}),
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Artifact publication failed.", issues=[str(exc)])

    def _validate_variant_execution(
        self,
        script: Path,
        args: list[str],
    ) -> dict[str, Any] | None:
        variant_root = self.context.variant_root.resolve()
        if script != variant_root and variant_root not in script.parents:
            return None
        if script.name != "variant.py" or script.parent.parent != variant_root:
            raise ValueError("Derived Skill execution requires skill_variants/<name>/variant.py")
        metadata_path = script.parent / "variant.json"
        request_path = script.parent / "request.json"
        if not metadata_path.is_file() or not request_path.is_file():
            raise ValueError("Derived Skill is missing variant.json or request.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") not in {"ready", "validated"}:
            raise ValueError("Derived Skill must be ready before ExecutePython")
        if len(args) != 1 or Path(args[0]).expanduser().resolve() != request_path.resolve():
            raise ValueError("Derived Skill must run with args=[request.json]")
        metadata["variant_name"] = script.parent.name
        self._verify_variant_source_hashes(metadata)
        return metadata

    def _guard_variant_metadata(self, path: Path) -> None:
        variant_root = self.context.variant_root.resolve()
        if path.name == "variant.json" and path.parent.parent == variant_root:
            raise ValueError("variant.json is lifecycle-managed and cannot be changed with Write/Edit")

    def _invalidate_variant_after_source_change(self, path: Path) -> None:
        variant_root = self.context.variant_root.resolve()
        if path.name != "variant.py" or path.parent.parent != variant_root:
            return
        metadata_path = path.parent / "variant.json"
        if not metadata_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(metadata, dict):
            return
        metadata["status"] = "draft"
        metadata.pop("validated_at", None)
        metadata.pop("validated_artifact", None)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _verify_variant_source_hashes(metadata: dict[str, Any]) -> None:
        expected = metadata.get("base_source_hashes")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Derived Skill is missing base_source_hashes")
        current: dict[str, str] = {}
        for raw_path in expected:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise ValueError(f"Base Skill source disappeared: {path}")
            current[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != expected:
            raise ValueError("Base Skill source changed after derived Skill creation")


def register_engineer_atomic_tools(
    toolkit: Toolkit,
    context: EngineerToolContext,
) -> EngineerTools:
    """Register Engineer-only atomic tools directly on an AgentScope toolkit."""
    tools = EngineerTools(context)
    for tool in (
        tools.Read,
        tools.Glob,
        tools.Grep,
        tools.Write,
        tools.Edit,
        tools.ExecutePython,
        tools.publish_artifact,
    ):
        toolkit.register_tool_function(tool)
    return tools


def _runner_source(
    output_dir: Path,
    *,
    read_roots: Iterable[Path],
) -> str:
    output_literal = repr(str(output_dir.resolve()))
    allowed_dirs = [
        str(path.resolve())
        for path in read_roots
        if path.exists() and path.is_dir()
    ]
    allowed_files = [
        str(path.resolve())
        for path in read_roots
        if path.exists() and path.is_file()
    ]
    for runtime_root in (Path(sys.prefix), Path(sys.base_prefix), output_dir):
        value = str(runtime_root.resolve())
        if value not in allowed_dirs:
            allowed_dirs.append(value)
    for runtime_file in (Path("/etc/localtime"), Path("/dev/null"), Path("/dev/urandom")):
        if runtime_file.exists():
            allowed_files.append(str(runtime_file.resolve()))
    allowed_dirs_literal = repr(allowed_dirs)
    allowed_files_literal = repr(allowed_files)
    return f'''from __future__ import annotations

import os
import runpy
import socket
import sys
from pathlib import Path

WRITE_ROOT = Path({output_literal}).resolve()
READ_DIRS = tuple(Path(value).resolve() for value in {allowed_dirs_literal})
READ_FILES = frozenset(Path(value).resolve() for value in {allowed_files_literal})


def _resolve_path(value):
    if isinstance(value, int):
        return None
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    path = Path(os.fspath(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _ensure_write_path(value):
    path = _resolve_path(value)
    if path is None:
        return
    if path != WRITE_ROOT and WRITE_ROOT not in path.parents:
        raise PermissionError(f"write outside the execution output directory is forbidden: {{path}}")


def _ensure_read_path(value):
    path = _resolve_path(value)
    if path is None:
        return
    if path in READ_FILES or any(path == root or root in path.parents for root in READ_DIRS):
        return
    raise PermissionError(f"read outside authorized read roots is forbidden: {{path}}")


def _audit(event, args):
    if event == "open":
        path, mode, flags = args
        write_mode = isinstance(mode, str) and any(char in mode for char in "wax+")
        write_flags = isinstance(flags, int) and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write_mode or write_flags:
            _ensure_write_path(path)
        else:
            _ensure_read_path(path)
    elif event in {{"os.remove", "os.unlink", "os.rmdir", "os.mkdir"}}:
        _ensure_write_path(args[0])
    elif event in {{"os.rename", "os.replace"}}:
        _ensure_write_path(args[0])
        _ensure_write_path(args[1])
    elif event in {{"subprocess.Popen", "os.system", "pty.spawn"}}:
        raise PermissionError(f"child processes and shell execution are forbidden: {{event}}")
    elif event.startswith("socket.") and event not in {{"socket.__new__"}}:
        raise PermissionError(f"network access is forbidden: {{event}}")


sys.addaudithook(_audit)
script_path = Path(sys.argv[1]).resolve()
script_args = sys.argv[2:]
sys.argv = [str(script_path), *script_args]
runpy.run_path(
    str(script_path),
    run_name="__main__",
    init_globals={{"OUTPUT_DIR": str(WRITE_ROOT)}},
)
'''
