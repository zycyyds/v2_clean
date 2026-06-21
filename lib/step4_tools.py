from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

STEP4_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP4_DIR.parent
for path in (STEP4_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step4_runtime import (  # noqa: E402
    get_default_output_root,
    normalize_step4_outputs,
    resolve_step4_input,
    resolve_step4_task_text,
    run_step4_column_selection,
    validate_step4_output,
)


def _json_response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


def _progress_enabled() -> bool:
    raw = str(os.environ.get("STEP4_PROGRESS_ENABLED", "true")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _emit(message: str) -> None:
    if _progress_enabled():
        print(message, file=sys.stderr, flush=True)


def resolve_step4_input_tool(input_path: str) -> ToolResponse:
    _emit(f"[Step4][Tool] resolve_step4_input_tool 开始: {input_path}")
    try:
        artifacts = resolve_step4_input(input_path)
        _emit(f"[Step4][Tool] resolve_step4_input_tool 完成: {artifacts['input_csv_path']}")
        return _json_response(
            {
                "status": "SUCCESS",
                "summary": "Step4 输入表已解析。",
                "artifacts": artifacts,
                "issues": [],
            }
        )
    except Exception as exc:
        _emit(f"[Step4][Tool] resolve_step4_input_tool 失败: {exc}")
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step4 输入表解析失败。",
                "artifacts": {"input_path": input_path},
                "issues": [str(exc)],
            }
        )


def resolve_step4_task_text_tool(task_text: str | None = None) -> ToolResponse:
    _emit("[Step4][Tool] resolve_step4_task_text_tool 开始")
    try:
        artifacts = resolve_step4_task_text(task_text)
        _emit("[Step4][Tool] resolve_step4_task_text_tool 完成")
        return _json_response(
            {
                "status": "SUCCESS",
                "summary": "Step4 任务文本已确定。",
                "artifacts": artifacts,
                "issues": [],
            }
        )
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step4 任务文本解析失败。",
                "artifacts": {"task_text": task_text or ""},
                "issues": [str(exc)],
            }
        )


def run_step4_column_selection_tool(
    input_csv_path: str,
    task_text: str,
    output_root: str | None = None,
    memory_context: str | None = None,
) -> ToolResponse:
    target_output = output_root or get_default_output_root()
    _emit(f"[Step4][Tool] run_step4_column_selection_tool 开始: {input_csv_path}")
    try:
        artifacts = run_step4_column_selection(
            input_csv_path=input_csv_path,
            output_root=target_output,
            task_text=task_text,
            memory_context=memory_context,
            emit=_emit,
        )
        _emit(f"[Step4][Tool] run_step4_column_selection_tool 完成: final_columns={artifacts.get('final_column_count')}")
        return _json_response(
            {
                "status": "SUCCESS",
                "summary": "Step4 列筛选已完成。",
                "artifacts": artifacts,
                "issues": [],
            }
        )
    except Exception as exc:
        _emit(f"[Step4][Tool] run_step4_column_selection_tool 失败: {exc}")
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step4 列筛选执行失败。",
                "artifacts": {
                    "input_csv_path": input_csv_path,
                    "output_root": target_output,
                    "task_text": task_text,
                    "memory_context_used": bool(str(memory_context or "").strip()),
                },
                "issues": [str(exc)],
            }
        )


def normalize_step4_outputs_tool(
    filtered_csv_path: str,
    selection_report_path: str,
    output_root: str | None = None,
) -> ToolResponse:
    _emit("[Step4][Tool] normalize_step4_outputs_tool 开始")
    try:
        artifacts = normalize_step4_outputs(
            filtered_csv_path=filtered_csv_path,
            selection_report_path=selection_report_path,
            output_root=output_root,
        )
        _emit(f"[Step4][Tool] normalize_step4_outputs_tool 完成: {artifacts['next_input_dir']}")
        return _json_response(
            {
                "status": "SUCCESS",
                "summary": "Step4 稳定 next_input 产物已生成。",
                "artifacts": artifacts,
                "issues": [],
            }
        )
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step4 产物标准化失败。",
                "artifacts": {
                    "filtered_csv_path": filtered_csv_path,
                    "selection_report_path": selection_report_path,
                    "output_root": output_root or "",
                },
                "issues": [str(exc)],
            }
        )


def validate_step4_output_tool(
    filtered_csv_path: str,
    selection_report_path: str,
    next_input_csv: str | None = None,
    next_selection_report: str | None = None,
) -> ToolResponse:
    _emit("[Step4][Tool] validate_step4_output_tool 开始")
    result = validate_step4_output(
        filtered_csv_path=filtered_csv_path,
        selection_report_path=selection_report_path,
        next_input_csv=next_input_csv,
        next_selection_report=next_selection_report,
    )
    _emit(f"[Step4][Tool] validate_step4_output_tool 完成: status={'SUCCESS' if result.passed else 'NEEDS_REPAIR'}")
    return _json_response(
        {
            "status": "SUCCESS" if result.passed else "NEEDS_REPAIR",
            "summary": "Step4 输出校验通过。" if result.passed else "Step4 输出校验失败。",
            "artifacts": result.details,
            "issues": result.issues,
        }
    )


def register_step4_tools(toolkit: Toolkit) -> Toolkit:
    for func in (
        resolve_step4_input_tool,
        resolve_step4_task_text_tool,
        run_step4_column_selection_tool,
        normalize_step4_outputs_tool,
        validate_step4_output_tool,
    ):
        toolkit.register_tool_function(func)
    return toolkit
