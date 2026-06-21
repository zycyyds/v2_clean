from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

STEP1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP1_DIR.parent
if str(STEP1_DIR) in sys.path:
    sys.path.remove(str(STEP1_DIR))
sys.path.insert(0, str(STEP1_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step1_runtime import (  # noqa: E402
    TerminalProgress,
    build_records_parallel,
    run_reorganize_from_records,
    scan_source_files,
    validate_records,
    validate_step1_output,
    write_generated_reorganizer,
)


def _json_response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


def _progress_enabled() -> bool:
    raw = str(os.environ.get("STEP1_PROGRESS_ENABLED", "true")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _new_progress() -> TerminalProgress:
    return TerminalProgress(enabled=_progress_enabled())


def scan_source_files_tool(input_path: str, include_tasks: bool = False, max_sample_tasks: int = 20) -> ToolResponse:
    progress = _new_progress()
    progress.emit(f"[Step1][Tool] scan_source_files_tool 开始: {input_path}")
    tasks = [asdict(task) for task in scan_source_files(input_path)]
    progress.emit(f"[Step1][Tool] scan_source_files_tool 完成: source_files={len(tasks)}")
    sample_size = max(0, int(max_sample_tasks or 0))
    artifacts = {
        "input_path": str(input_path),
        "source_files": len(tasks),
        "sample_tasks": tasks[:sample_size],
    }
    if include_tasks:
        artifacts["tasks"] = tasks
    return _json_response(
        {
            "status": "SUCCESS",
            "summary": f"已扫描 {len(tasks)} 个文件。",
            "artifacts": artifacts,
            "issues": [],
        }
    )


def build_records_parallel_tool(
    input_path: str,
    records_path: str,
    max_workers: int | None = None,
    parallel_enabled: bool | None = None,
) -> ToolResponse:
    progress = _new_progress()
    progress.emit(f"[Step1][Tool] build_records_parallel_tool 开始: input={input_path}")
    result = build_records_parallel(
        input_path=input_path,
        records_path=records_path,
        max_workers=max_workers,
        parallel_enabled=parallel_enabled,
        progress=progress.progress,
        emit=progress.emit,
    )
    progress.emit(f"[Step1][Tool] build_records_parallel_tool 完成: records_count={result['records_count']}")
    return _json_response(
        {
            "status": "SUCCESS" if not result.get("worker_errors") else "NEEDS_REPAIR",
            "summary": f"records 已生成，共 {result['records_count']} 条。",
            "artifacts": result,
            "issues": list(result.get("worker_errors") or []),
        }
    )


def validate_records_tool(records_path: str, expected_count: int | None = None) -> ToolResponse:
    progress = _new_progress()
    progress.emit(f"[Step1][Tool] validate_records_tool 开始: {records_path}")
    result = validate_records(records_path, expected_count=expected_count)
    progress.emit(f"[Step1][Tool] validate_records_tool 完成: status={'SUCCESS' if result.passed else 'NEEDS_REPAIR'}")
    return _json_response(
        {
            "status": "SUCCESS" if result.passed else "NEEDS_REPAIR",
            "summary": "records 校验通过。" if result.passed else "records 校验失败。",
            "artifacts": result.details,
            "issues": result.issues,
        }
    )


def write_generated_reorganizer_tool(
    script_path: str,
    records_path: str,
    output_root: str,
    input_root: str | None = None,
) -> ToolResponse:
    progress = _new_progress()
    progress.emit(f"[Step1][Tool] write_generated_reorganizer_tool 开始: {script_path}")
    generated_path = write_generated_reorganizer(
        script_path=script_path,
        records_path=records_path,
        output_root=output_root,
        input_root=input_root,
    )
    progress.emit(f"[Step1][Tool] write_generated_reorganizer_tool 完成: {generated_path}")
    return _json_response(
        {
            "status": "SUCCESS",
            "summary": "已生成 Step1 重组脚本。",
            "artifacts": {"generated_script_path": generated_path},
            "issues": [],
        }
    )


def run_reorganize_from_records_tool(
    records_path: str,
    output_root: str,
    input_root: str | None = None,
    clean_output: bool = True,
    max_workers: int | None = None,
    parallel_enabled: bool | None = None,
) -> ToolResponse:
    progress = _new_progress()
    progress.emit(f"[Step1][Tool] run_reorganize_from_records_tool 开始: records={records_path}")
    result = run_reorganize_from_records(
        records_path=records_path,
        output_root=output_root,
        input_root=input_root,
        clean_output=clean_output,
        max_workers=max_workers,
        parallel_enabled=parallel_enabled,
        progress=progress.progress,
        emit=progress.emit,
    )
    progress.emit(f"[Step1][Tool] run_reorganize_from_records_tool 完成: status={result.get('status')}")
    return _json_response(
        {
            "status": result.get("status", "FAILED"),
            "summary": f"已写出 {result.get('written_files', 0)} 个文件。" if result.get("status") == "SUCCESS" else "重组执行失败。",
            "artifacts": result,
            "issues": list(result.get("errors") or []),
        }
    )


def validate_step1_output_tool(output_root: str, expected_min_files: int = 1) -> ToolResponse:
    progress = _new_progress()
    progress.emit(f"[Step1][Tool] validate_step1_output_tool 开始: {output_root}")
    result = validate_step1_output(output_root, expected_min_files=expected_min_files)
    progress.emit(f"[Step1][Tool] validate_step1_output_tool 完成: status={'SUCCESS' if result.passed else 'NEEDS_REPAIR'}")
    return _json_response(
        {
            "status": "SUCCESS" if result.passed else "NEEDS_REPAIR",
            "summary": "Step1 输出校验通过。" if result.passed else "Step1 输出校验失败。",
            "artifacts": result.details,
            "issues": result.issues,
        }
    )


def register_step1_tools(toolkit: Toolkit) -> Toolkit:
    for func in (
        scan_source_files_tool,
        build_records_parallel_tool,
        validate_records_tool,
        write_generated_reorganizer_tool,
        run_reorganize_from_records_tool,
        validate_step1_output_tool,
    ):
        toolkit.register_tool_function(func)
    return toolkit
