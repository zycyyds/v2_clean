from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

STEP23_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP23_DIR.parent
for path in (STEP23_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step23_runtime import (  # noqa: E402
    get_default_output_root,
    normalize_step23_outputs,
    resolve_step23_input,
    run_step23_directory_pipeline,
    run_step23_extraction,
    run_step23_fill_table,
    run_step23_ocr,
    scan_step23_input,
    select_step23_pipeline,
    validate_step23_output,
)


def _json_response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


def _progress_enabled() -> bool:
    raw = str(os.environ.get("STEP23_PROGRESS_ENABLED", "true")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _emit(message: str) -> None:
    if _progress_enabled():
        print(message, file=sys.stderr, flush=True)


def resolve_step23_input_tool(input_path: str, mode: str = "auto") -> ToolResponse:
    _emit(f"[Step2-3][Tool] resolve_step23_input_tool 开始: {input_path}")
    try:
        artifacts = resolve_step23_input(input_path=input_path, mode=mode)
        _emit(f"[Step2-3][Tool] resolve_step23_input_tool 完成: mode={artifacts['mode']}")
        return _json_response({"status": "SUCCESS", "summary": "Step2-3 输入已解析。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step2-3 输入解析失败。",
                "artifacts": {"input_path": input_path, "mode": mode},
                "issues": [str(exc)],
            }
        )


def scan_step23_input_tool(input_path: str) -> ToolResponse:
    _emit(f"[Step2-3][Tool] scan_step23_input_tool 开始: {input_path}")
    try:
        artifacts = scan_step23_input(input_path)
        _emit(
            "[Step2-3][Tool] scan_step23_input_tool 完成: "
            f"patients={artifacts.get('patient_count')}, images={artifacts.get('image_count')}"
        )
        return _json_response({"status": "SUCCESS", "summary": "Step2-3 输入目录扫描完成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step2-3 输入目录扫描失败。",
                "artifacts": {"input_path": input_path},
                "issues": [str(exc)],
            }
        )


def select_step23_pipeline_tool(
    input_path: str,
    requested_mode: str = "auto",
    user_hint: str = "",
    scan_artifacts: dict[str, Any] | None = None,
) -> ToolResponse:
    _emit(f"[Step2-3][Tool] select_step23_pipeline_tool 开始: requested_mode={requested_mode}")
    try:
        artifacts = select_step23_pipeline(
            input_path=input_path,
            scan_artifacts=scan_artifacts if isinstance(scan_artifacts, dict) else None,
            requested_mode=requested_mode,
            user_hint=user_hint,
        )
        _emit(
            "[Step2-3][Tool] select_step23_pipeline_tool 完成: "
            f"resolved_mode={artifacts.get('resolved_mode')}"
        )
        return _json_response({"status": "SUCCESS", "summary": "Step2-3 pipeline 模式已选择。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step2-3 pipeline 模式选择失败。",
                "artifacts": {"input_path": input_path, "requested_mode": requested_mode},
                "issues": [str(exc)],
            }
        )


async def run_step23_directory_pipeline_tool(
    input_path: str,
    output_root: str | None = None,
    concurrency: int = 8,
    use_llm: bool = True,
) -> ToolResponse:
    target_output = output_root or get_default_output_root()
    _emit(f"[Step2-3][Tool] run_step23_directory_pipeline_tool 开始: {input_path}")
    try:
        artifacts = await run_step23_directory_pipeline(
            input_path=input_path,
            output_root=target_output,
            concurrency=concurrency,
            use_llm=use_llm,
            emit=_emit,
        )
        _emit(
            "[Step2-3][Tool] run_step23_directory_pipeline_tool 完成: "
            f"rows={artifacts.get('wide_row_count')}, columns={artifacts.get('wide_column_count')}"
        )
        return _json_response({"status": "SUCCESS", "summary": "Step2-3 directory pipeline 已完成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step2-3 directory pipeline 执行失败。",
                "artifacts": {"input_path": input_path, "output_root": target_output},
                "issues": [str(exc)],
            }
        )


def run_step23_ocr_tool(
    input_path: str,
    output_root: str | None = None,
    ocr_workers: int = 8,
    cache_path: str | None = None,
) -> ToolResponse:
    target_output = output_root or get_default_output_root()
    _emit(f"[Step2-3][Tool] run_step23_ocr_tool 开始: {input_path}")
    try:
        artifacts = run_step23_ocr(
            input_path=input_path,
            output_root=target_output,
            ocr_workers=ocr_workers,
            cache_path=cache_path,
            emit=_emit,
        )
        _emit(f"[Step2-3][Tool] run_step23_ocr_tool 完成: successful_ocr={artifacts.get('successful_ocr')}")
        return _json_response({"status": "SUCCESS", "summary": "Step2-3 OCR 已完成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step2-3 OCR 执行失败。",
                "artifacts": {"input_path": input_path, "output_root": target_output},
                "issues": [str(exc)],
            }
        )


async def run_step23_extraction_tool(
    ocr_cache_path: str,
    results_dir: str,
    concurrency: int = 8,
) -> ToolResponse:
    _emit(f"[Step2-3][Tool] run_step23_extraction_tool 开始: {ocr_cache_path}")
    try:
        artifacts = await run_step23_extraction(
            ocr_cache_path=ocr_cache_path,
            results_dir=results_dir,
            concurrency=concurrency,
            emit=_emit,
        )
        _emit(f"[Step2-3][Tool] run_step23_extraction_tool 完成: entities={artifacts.get('total_entities')}")
        return _json_response({"status": "SUCCESS", "summary": "Step2-3 LLM 抽取已完成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step2-3 LLM 抽取失败。",
                "artifacts": {"ocr_cache_path": ocr_cache_path, "results_dir": results_dir},
                "issues": [str(exc)],
            }
        )


async def run_step23_fill_table_tool(
    input_path: str,
    results_dir: str,
    use_llm: bool = True,
) -> ToolResponse:
    _emit(f"[Step2-3][Tool] run_step23_fill_table_tool 开始: {input_path}")
    try:
        artifacts = await run_step23_fill_table(
            input_path=input_path,
            results_dir=results_dir,
            use_llm=use_llm,
            emit=_emit,
        )
        _emit(f"[Step2-3][Tool] run_step23_fill_table_tool 完成: {artifacts.get('filled_merged_csv') or 'no merged csv'}")
        return _json_response({"status": "SUCCESS", "summary": "Step2-3 宽表回填已完成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step2-3 宽表回填失败。",
                "artifacts": {"input_path": input_path, "results_dir": results_dir},
                "issues": [str(exc)],
            }
        )


def normalize_step23_outputs_tool(
    output_root: str | None = None,
    filled_merged_csv: str | None = None,
    fallback_csv: str | None = None,
    summary: dict[str, Any] | None = None,
) -> ToolResponse:
    target_output = output_root or get_default_output_root()
    _emit("[Step2-3][Tool] normalize_step23_outputs_tool 开始")
    try:
        artifacts = normalize_step23_outputs(
            output_root=target_output,
            filled_merged_csv=filled_merged_csv,
            fallback_csv=fallback_csv,
            summary=summary if isinstance(summary, dict) else None,
        )
        _emit(f"[Step2-3][Tool] normalize_step23_outputs_tool 完成: {artifacts['next_input_csv']}")
        return _json_response({"status": "SUCCESS", "summary": "Step2-3 next_input 已生成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step2-3 产物标准化失败。",
                "artifacts": {
                    "output_root": target_output,
                    "filled_merged_csv": filled_merged_csv or "",
                    "fallback_csv": fallback_csv or "",
                },
                "issues": [str(exc)],
            }
        )


def validate_step23_output_tool(next_input_csv: str, next_summary_json: str | None = None) -> ToolResponse:
    _emit("[Step2-3][Tool] validate_step23_output_tool 开始")
    result = validate_step23_output(next_input_csv=next_input_csv, next_summary_json=next_summary_json)
    _emit(f"[Step2-3][Tool] validate_step23_output_tool 完成: status={'SUCCESS' if result['passed'] else 'NEEDS_REPAIR'}")
    return _json_response(
        {
            "status": "SUCCESS" if result["passed"] else "NEEDS_REPAIR",
            "summary": "Step2-3 输出校验通过。" if result["passed"] else "Step2-3 输出校验失败。",
            "artifacts": result["details"],
            "issues": result["issues"],
        }
    )


def register_step23_tools(toolkit: Toolkit) -> Toolkit:
    for func in (
        resolve_step23_input_tool,
        scan_step23_input_tool,
        select_step23_pipeline_tool,
        run_step23_directory_pipeline_tool,
        run_step23_ocr_tool,
        run_step23_extraction_tool,
        run_step23_fill_table_tool,
        normalize_step23_outputs_tool,
        validate_step23_output_tool,
    ):
        toolkit.register_tool_function(func)
    return toolkit
