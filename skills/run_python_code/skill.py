"""run_python_code skill — agent writes and executes Python code directly."""
from __future__ import annotations

import json
import logging
import sys
import traceback
from io import StringIO
from pathlib import Path

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

from lib.agent_artifacts import get_phase_context, record_step
from skills import _common  # noqa: F401

V2_DIR = Path(__file__).resolve().parents[2]
LEGACY_OUTPUT_DIR = V2_DIR / "output" / "code_outputs"
MAX_CAPTURE_CHARS = 20_000


def _truncate_capture(text: str, limit: int = MAX_CAPTURE_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = limit // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:], True


def _current_output_dir() -> Path:
    context = get_phase_context()
    if context is None:
        LEGACY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return LEGACY_OUTPUT_DIR
    # Use artifacts_dir as a stable parent; the exact step subdir is
    # created later by record_step/begin_step to avoid double-counting.
    artifacts_dir = Path(context["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def run_python_code_tool(code: str, output_path: str = "") -> ToolResponse:
    """执行 agent 编写的 Python 代码，返回 stdout 输出和产物路径。"""
    # Reserve a step slot first so the output dir matches the manifest entry.
    from lib.agent_artifacts import begin_step as _begin_step
    step_ctx = _begin_step("run_python_code")
    if step_ctx is not None:
        output_dir = Path(step_ctx["step_dir"])
    else:
        output_dir = _current_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_context = get_phase_context()
    manifest_path = Path(phase_context["manifest_path"]) if phase_context else None
    manifest_before = manifest_path.read_bytes() if manifest_path and manifest_path.exists() else None
    before_files = {str(p.resolve()) for p in output_dir.rglob("*") if p.is_file()}

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = captured_out = StringIO()
    sys.stderr = captured_err = StringIO()

    # Redirect logging StreamHandlers so exec'd code's log output is captured.
    _log_handlers: list[tuple[logging.StreamHandler, object]] = []
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and hasattr(handler, "stream"):
            _log_handlers.append((handler, handler.stream))
            handler.stream = captured_err

    exec_globals = {
        "OUTPUT_DIR": str(output_dir),
        "__builtins__": __builtins__,
    }

    error = None
    try:
        exec(code, exec_globals)  # noqa: S102
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        for handler, orig_stream in _log_handlers:
            handler.stream = orig_stream

    if manifest_path is not None:
        manifest_after = manifest_path.read_bytes() if manifest_path.exists() else None
        if manifest_after != manifest_before:
            if manifest_before is not None:
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_bytes(manifest_before)
            elif manifest_path.exists():
                manifest_path.unlink()
            protection_error = "PermissionError: run_python_code cannot create, delete, or overwrite manifest.json"
            error = f"{error}\n{protection_error}" if error else protection_error

    stdout_text, stdout_truncated = _truncate_capture(captured_out.getvalue())
    stderr_text, stderr_truncated = _truncate_capture(captured_err.getvalue())
    after_files = [str(p.resolve()) for p in sorted(output_dir.rglob("*")) if p.is_file()]
    new_output_files = [path for path in after_files if path not in before_files]
    if output_path:
        new_output_files.append(str(Path(output_path).resolve()))

    summary = "代码执行出错。" if error else "代码执行成功。"
    issues = [error] if error else []
    manifest_record = record_step(
        "run_python_code",
        summary,
        new_output_files,
        metadata={
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "requested_output_path": output_path,
        },
        status="NEEDS_REPAIR" if error else "SUCCESS",
        issues=issues,
        _step_ctx=step_ctx,
    )

    payload = {
        "status": "NEEDS_REPAIR" if error else "SUCCESS",
        "summary": summary,
        "artifacts": {
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "error": error,
            "output_dir": str(output_dir),
            "new_output_files": new_output_files,
            "output_files": after_files,
            "manifest_path": (manifest_record or {}).get("manifest_path", ""),
            "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", str(output_dir)),
        },
        "issues": issues,
    }

    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


SKILL = {
    "name": "run_python_code",
    "layer": "process",
    "description": "执行 agent 自己编写的 Python 代码，可用 pandas/numpy/sklearn 自由处理数据：合并表、计算特征、构建标签、输出 CSV。每次调用都会写入独立 step 产物目录并登记 manifest。",
    "when_to_use": (
        "需要自定义数据处理逻辑时：合并多张表、计算时间差、构建特征、处理缺失值、输出训练集等。"
        "代码中用 OUTPUT_DIR 变量（已自动注入）作为当前步骤的输出目录；每次调用只做一个步骤并保存中间产物。"
        "示例：\n"
        "  import pandas as pd\n"
        "  df = pd.read_csv('/path/to/icustays.csv')\n"
        "  df['los_hours'] = df['los'] * 24\n"
        "  df.to_csv(OUTPUT_DIR + '/features.csv', index=False)\n"
        "  print(df.shape)"
    ),
    "when_to_skip": "已有领域 Skill、参数调整、adapter 或 fork 能完成任务时必须跳过；不得用它重写整条流水线。",
    "capability_types": ["standalone_python", "custom_data_transform"],
    "inputs": ["code（Python 代码字符串）", "output_path?（可选输出路径）"],
    "outputs": ["stdout", "new_output_files", "manifest_path", "step_artifact_dir"],
    "prerequisites": ["no_suitable_domain_skill", "single_task_scope_declared"],
    "adapter_policy": "standalone_only",
    "preserves_raw_values": False,
    "lib_entrypoints": [],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(run_python_code_tool)
