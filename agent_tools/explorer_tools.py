from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from agentscope.tool import Toolkit, ToolResponse

from lib.agent_artifacts import begin_step, record_step

from .context import ExplorerToolContext
from .tools import EngineerTools, _response, _runner_source, _truncate


class ExplorerTools:
    """Read, inspection, and sandboxed analysis tools for DataExplorer."""

    def __init__(self, context: ExplorerToolContext) -> None:
        self.context = context
        self._file_tools = EngineerTools(context)

    def Read(self, file_path: str, offset: int = 1, limit: int = 400) -> ToolResponse:
        """Read a UTF-8 text file with one-based line pagination."""
        return self._file_tools.Read(file_path, offset, limit)

    def Glob(self, pattern: str, path: str = "", limit: int = 200) -> ToolResponse:
        """Find files under an authorized directory using a glob pattern."""
        return self._file_tools.Glob(pattern, path, limit)

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
        return self._file_tools.Grep(pattern, path, glob, output_mode, ignore_case, limit)

    def InspectDataFile(
        self,
        file_path: str,
        sample_rows: int = 3,
        max_columns: int = 50,
    ) -> ToolResponse:
        """Inspect schema, samples, row count, and missing values of a data file."""
        try:
            if sample_rows < 0 or sample_rows > 20:
                raise ValueError("sample_rows must be between 0 and 20")
            if max_columns < 1 or max_columns > 200:
                raise ValueError("max_columns must be between 1 and 200")
            path = self.context.resolve_read_path(file_path)
            if not path.is_file():
                raise ValueError(f"Data file does not exist: {path}")
            suffix = path.suffix.lower()
            loaders = {
                ".csv": ("csv", lambda pd: pd.read_csv(path, low_memory=False)),
                ".tsv": ("tsv", lambda pd: pd.read_csv(path, sep="\t", low_memory=False)),
                ".xlsx": ("excel", lambda pd: pd.read_excel(path)),
                ".xls": ("excel", lambda pd: pd.read_excel(path)),
                ".parquet": ("parquet", lambda pd: pd.read_parquet(path)),
                ".jsonl": ("jsonl", lambda pd: pd.read_json(path, lines=True)),
                ".json": ("json", lambda pd: pd.read_json(path)),
            }
            if suffix not in loaders:
                raise ValueError(
                    "InspectDataFile supports CSV, TSV, Excel, JSON, JSONL, and Parquet files",
                )
            import pandas as pd

            format_name, loader = loaders[suffix]
            frame = loader(pd)
            selected_names = list(frame.columns[:max_columns])
            columns = []
            for name in selected_names:
                series = frame[name]
                columns.append(
                    {
                        "name": str(name),
                        "dtype": str(series.dtype),
                        "missing_count": int(series.isna().sum()),
                        "unique_count": int(series.nunique(dropna=True)),
                    }
                )
            samples = frame[selected_names].head(sample_rows).where(
                frame[selected_names].head(sample_rows).notna(),
                None,
            )
            sample_records = json.loads(
                json.dumps(samples.to_dict(orient="records"), ensure_ascii=False, default=str),
            )
            sample_records = [
                {
                    key: (value[:500] + "... [truncated]" if isinstance(value, str) and len(value) > 500 else value)
                    for key, value in record.items()
                }
                for record in sample_records
            ]
            self.context.mark_read(path)
            return _response(
                "SUCCESS",
                f"Inspected {path.name}: {len(frame)} rows x {len(frame.columns)} columns.",
                {
                    "file_path": str(path),
                    "format": format_name,
                    "row_count": int(len(frame)),
                    "column_count": int(len(frame.columns)),
                    "columns": columns,
                    "columns_truncated": len(selected_names) < len(frame.columns),
                    "sample_records": sample_records,
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Data-file inspection failed.", issues=[str(exc)])

    def ExecuteAnalysisPython(
        self,
        code: str,
        timeout_seconds: int = 120,
    ) -> ToolResponse:
        """Execute inline Python for analysis with sandboxed reads and step-local writes."""
        step_ctx = None
        try:
            if not code.strip():
                raise ValueError("code is required")
            if len(code) > 100_000:
                raise ValueError("code exceeds the 100000 character limit")
            if timeout_seconds < 1 or timeout_seconds > 300:
                raise ValueError("timeout_seconds must be between 1 and 300")
            step_ctx = begin_step("ExecuteAnalysisPython")
            if step_ctx is None:
                output_dir = self.context.explorer_phase_root / "artifacts" / "execute_analysis_python"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(step_ctx["step_dir"])
            script = output_dir / "analysis_task.py"
            runner = output_dir / "_execute_runner.py"
            script.write_text(code, encoding="utf-8")
            runner.write_text(
                _runner_source(output_dir, read_roots=self.context.read_roots),
                encoding="utf-8",
            )
            before_files = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_DIR": str(output_dir),
                    "TMPDIR": str(output_dir),
                    "MPLCONFIGDIR": str(output_dir / ".mplconfig"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                }
            )
            timed_out = False
            try:
                completed = subprocess.run(
                    [sys.executable, "-u", str(runner), str(script)],
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
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                stderr += f"\nTimeoutError: execution exceeded {timeout_seconds} seconds."
            stdout, stdout_truncated = _truncate(stdout)
            stderr, stderr_truncated = _truncate(stderr)
            after_files = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
            output_files = sorted(
                str(path)
                for path in after_files - before_files
                if path not in {script.resolve(), runner.resolve()}
            )
            status = "SUCCESS" if returncode == 0 and not timed_out else "NEEDS_REPAIR"
            issues = [] if status == "SUCCESS" else [stderr or f"Python exited with {returncode}"]
            record = record_step(
                "ExecuteAnalysisPython",
                "Analysis Python succeeded." if status == "SUCCESS" else "Analysis Python failed.",
                [script, *output_files],
                metadata={
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": timed_out,
                },
                status=status,
                issues=issues,
                _step_ctx=step_ctx,
            )
            return _response(
                status,
                "Analysis Python succeeded." if status == "SUCCESS" else "Analysis Python failed.",
                {
                    "output_dir": str(output_dir),
                    "output_files": output_files,
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "timed_out": timed_out,
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
                issues,
            )
        except Exception as exc:
            if step_ctx is not None:
                record_step(
                    "ExecuteAnalysisPython",
                    "Analysis Python setup failed.",
                    [],
                    status="NEEDS_REPAIR",
                    issues=[str(exc)],
                    _step_ctx=step_ctx,
                )
            return _response("NEEDS_REPAIR", "Analysis Python failed.", issues=[str(exc)])


def register_explorer_atomic_tools(
    toolkit: Toolkit,
    context: ExplorerToolContext,
) -> ExplorerTools:
    """Register Explorer atomic tools without generic file-write capabilities."""
    tools = ExplorerTools(context)
    for tool in (
        tools.Read,
        tools.Glob,
        tools.Grep,
        tools.InspectDataFile,
        tools.ExecuteAnalysisPython,
    ):
        toolkit.register_tool_function(tool)
    return tools
