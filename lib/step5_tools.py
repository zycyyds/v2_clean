from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

STEP5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP5_DIR.parent
for path in (STEP5_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step5_runtime import (  # noqa: E402
    classify_step5_column_risks,
    get_default_output_root,
    normalize_step5_outputs,
    profile_step5_table,
    resolve_step5_input,
    run_step5_data_cleaning,
    validate_step5_output,
)


def _json_response(payload: dict[str, Any]) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


def _progress_enabled() -> bool:
    raw = str(os.environ.get("STEP5_PROGRESS_ENABLED", "true")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _emit(message: str) -> None:
    if _progress_enabled():
        print(message, file=sys.stderr, flush=True)


def resolve_step5_input_tool(input_path: str) -> ToolResponse:
    _emit(f"[Step5][Tool] resolve_step5_input_tool 开始: {input_path}")
    try:
        artifacts = resolve_step5_input(input_path)
        _emit(f"[Step5][Tool] resolve_step5_input_tool 完成: {artifacts['input_csv_path']}")
        return _json_response({"status": "SUCCESS", "summary": "Step5 输入表已解析。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        _emit(f"[Step5][Tool] resolve_step5_input_tool 失败: {exc}")
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step5 输入表解析失败。",
                "artifacts": {"input_path": input_path},
                "issues": [str(exc)],
            }
        )


def profile_step5_table_tool(input_csv_path: str, output_root: str | None = None, workers: int = 4) -> ToolResponse:
    target_output = output_root or get_default_output_root()
    _emit(f"[Step5][Tool] profile_step5_table_tool 开始: {input_csv_path}")
    try:
        artifacts = profile_step5_table(input_csv_path, output_root=target_output, workers=workers, emit=_emit)
        _emit(f"[Step5][Tool] profile_step5_table_tool 完成: columns={artifacts.get('column_count')}")
        return _json_response({"status": "SUCCESS", "summary": "Step5 列画像已完成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        _emit(f"[Step5][Tool] profile_step5_table_tool 失败: {exc}")
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step5 列画像失败。",
                "artifacts": {"input_csv_path": input_csv_path, "output_root": target_output},
                "issues": [str(exc)],
            }
        )


def classify_step5_column_risks_tool(
    input_csv_path: str,
    profile_report_path: str,
    output_root: str | None = None,
    workers: int = 4,
) -> ToolResponse:
    target_output = output_root or get_default_output_root()
    _emit("[Step5][Tool] classify_step5_column_risks_tool 开始")
    try:
        artifacts = classify_step5_column_risks(
            input_csv_path=input_csv_path,
            profile_report_path=profile_report_path,
            output_root=target_output,
            workers=workers,
            emit=_emit,
        )
        counts = artifacts.get("risk_counts") or {}
        _emit(
            "[Step5][Tool] classify_step5_column_risks_tool 完成: "
            f"low={counts.get('low')}, medium={counts.get('medium')}, high={counts.get('high')}"
        )
        return _json_response({"status": "SUCCESS", "summary": "Step5 风险列分类已完成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        _emit(f"[Step5][Tool] classify_step5_column_risks_tool 失败: {exc}")
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step5 风险列分类失败。",
                "artifacts": {
                    "input_csv_path": input_csv_path,
                    "profile_report_path": profile_report_path,
                    "output_root": target_output,
                },
                "issues": [str(exc)],
            }
        )


def run_step5_data_cleaning_tool(
    input_csv_path: str,
    column_risk_report_path: str,
    output_root: str | None = None,
    workers: int = 4,
    llm_workers: int = 2,
    enable_llm: bool = True,
) -> ToolResponse:
    target_output = output_root or get_default_output_root()
    _emit("[Step5][Tool] run_step5_data_cleaning_tool 开始")
    try:
        artifacts = run_step5_data_cleaning(
            input_csv_path=input_csv_path,
            column_risk_report_path=column_risk_report_path,
            output_root=target_output,
            workers=workers,
            llm_workers=llm_workers,
            enable_llm=enable_llm,
            emit=_emit,
        )
        _emit(f"[Step5][Tool] run_step5_data_cleaning_tool 完成: changed_cells={artifacts.get('changed_cell_count')}")
        return _json_response({"status": "SUCCESS", "summary": "Step5 数据清洗已完成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        _emit(f"[Step5][Tool] run_step5_data_cleaning_tool 失败: {exc}")
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step5 数据清洗执行失败。",
                "artifacts": {
                    "input_csv_path": input_csv_path,
                    "column_risk_report_path": column_risk_report_path,
                    "output_root": target_output,
                },
                "issues": [str(exc)],
            }
        )


def normalize_step5_outputs_tool(
    cleaned_csv_path: str,
    data_quality_report_path: str,
    column_risk_report_path: str,
    output_root: str | None = None,
    selection_report_path: str | None = None,
) -> ToolResponse:
    _emit("[Step5][Tool] normalize_step5_outputs_tool 开始")
    try:
        artifacts = normalize_step5_outputs(
            cleaned_csv_path=cleaned_csv_path,
            data_quality_report_path=data_quality_report_path,
            column_risk_report_path=column_risk_report_path,
            output_root=output_root,
            selection_report_path=selection_report_path,
        )
        _emit(f"[Step5][Tool] normalize_step5_outputs_tool 完成: {artifacts['next_input_dir']}")
        return _json_response({"status": "SUCCESS", "summary": "Step5 稳定 next_input 产物已生成。", "artifacts": artifacts, "issues": []})
    except Exception as exc:
        _emit(f"[Step5][Tool] normalize_step5_outputs_tool 失败: {exc}")
        return _json_response(
            {
                "status": "NEEDS_REPAIR",
                "summary": "Step5 产物标准化失败。",
                "artifacts": {
                    "cleaned_csv_path": cleaned_csv_path,
                    "data_quality_report_path": data_quality_report_path,
                    "column_risk_report_path": column_risk_report_path,
                    "output_root": output_root or "",
                },
                "issues": [str(exc)],
            }
        )


def validate_step5_output_tool(
    input_csv_path: str,
    cleaned_csv_path: str,
    data_quality_report_path: str | None = None,
    column_risk_report_path: str | None = None,
    next_input_csv: str | None = None,
    next_data_quality_report: str | None = None,
    next_column_risk_report: str | None = None,
) -> ToolResponse:
    _emit("[Step5][Tool] validate_step5_output_tool 开始")
    result = validate_step5_output(
        input_csv_path=input_csv_path,
        cleaned_csv_path=cleaned_csv_path,
        data_quality_report_path=data_quality_report_path,
        column_risk_report_path=column_risk_report_path,
        next_input_csv=next_input_csv,
        next_data_quality_report=next_data_quality_report,
        next_column_risk_report=next_column_risk_report,
    )
    _emit(f"[Step5][Tool] validate_step5_output_tool 完成: status={'SUCCESS' if result.passed else 'NEEDS_REPAIR'}")
    return _json_response(
        {
            "status": "SUCCESS" if result.passed else "NEEDS_REPAIR",
            "summary": "Step5 输出校验通过。" if result.passed else "Step5 输出校验失败。",
            "artifacts": result.details,
            "issues": result.issues,
        }
    )


def register_step5_tools(toolkit: Toolkit) -> Toolkit:
    for func in (
        resolve_step5_input_tool,
        profile_step5_table_tool,
        classify_step5_column_risks_tool,
        run_step5_data_cleaning_tool,
        normalize_step5_outputs_tool,
        validate_step5_output_tool,
    ):
        toolkit.register_tool_function(func)
    return toolkit
