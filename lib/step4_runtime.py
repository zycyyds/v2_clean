from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

STEP4_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP4_DIR.parent
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_4.medical_column_selector.main import (  # noqa: E402
    TaskDrivenColumnSelector,
    get_default_task_text,
    resolve_input_csv,
)


TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls"}
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class Step4ValidationResult:
    passed: bool
    issues: list[str]
    details: dict[str, Any]


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def get_default_output_root() -> str:
    return str(PROJECT_ROOT / "program" / "output" / "step4_results")


def resolve_step4_input(input_path: str | Path) -> dict[str, Any]:
    resolved_input = resolve_path(input_path)
    input_csv = Path(resolve_input_csv(str(resolved_input))).resolve()
    suffix = input_csv.suffix.lower()
    if suffix not in TABULAR_SUFFIXES:
        raise FileNotFoundError(f"Step4 输入文件不是支持的表格文件: {input_csv}")

    columns = _read_columns(input_csv)
    return {
        "input_path": str(resolved_input),
        "input_csv_path": str(input_csv),
        "suffix": suffix,
        "file_size": input_csv.stat().st_size if input_csv.exists() else 0,
        "column_count": len(columns),
        "sample_columns": columns[:30],
    }


def resolve_step4_task_text(task_text: str | None = None) -> dict[str, Any]:
    resolved = str(task_text or "").strip() or get_default_task_text()
    if not str(resolved or "").strip():
        raise ValueError("Step4 task_text 不能为空。")
    return {"task_text": resolved}


def run_step4_column_selection(
    input_csv_path: str | Path,
    output_root: str | Path | None = None,
    task_text: str | None = None,
    memory_context: str | None = None,
    emit: ProgressCallback | None = None,
) -> dict[str, Any]:
    input_csv = resolve_path(input_csv_path)
    output = resolve_path(output_root or get_default_output_root())
    output.mkdir(parents=True, exist_ok=True)
    task = resolve_step4_task_text(task_text)["task_text"]

    if emit:
        emit(f"[Step4][Tool] ColumnSelector 开始: input={input_csv}")
    selector = TaskDrivenColumnSelector()
    result = selector.run(
        input_csv_path=str(input_csv),
        task_text=task,
        output_dir=str(output),
        memory_context=memory_context,
    )
    filtered_csv = Path(result["filtered_csv_path"]).resolve()
    selection_report = Path(result["selection_report_path"]).resolve()
    report = _load_json(selection_report)
    counts = report.get("column_counts") if isinstance(report.get("column_counts"), dict) else {}
    if emit:
        emit(f"[Step4][Tool] ColumnSelector 完成: filtered={filtered_csv.name}")

    return {
        "input_csv_path": str(input_csv),
        "output_root": str(output),
        "filtered_csv_path": str(filtered_csv),
        "selection_report_path": str(selection_report),
        "task_text": task,
        "memory_context_used": bool(str(memory_context or "").strip()),
        "original_column_count": counts.get("original_column_count"),
        "final_column_count": counts.get("final_column_count"),
        "selector_candidate_column_count": counts.get("selector_candidate_column_count"),
    }


def normalize_step4_outputs(
    filtered_csv_path: str | Path,
    selection_report_path: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    filtered_csv = resolve_path(filtered_csv_path)
    selection_report = resolve_path(selection_report_path)
    output = resolve_path(output_root or filtered_csv.parent)
    next_input = output / "next_input"
    next_input.mkdir(parents=True, exist_ok=True)

    next_csv = next_input / "filtered.csv"
    next_report = next_input / "selection_report.json"
    if not filtered_csv.is_file():
        raise FileNotFoundError(f"Step4 filtered CSV 不存在: {filtered_csv}")
    if not selection_report.is_file():
        raise FileNotFoundError(f"Step4 selection report 不存在: {selection_report}")
    shutil.copy2(filtered_csv, next_csv)
    shutil.copy2(selection_report, next_report)
    return {
        "next_input_dir": str(next_input),
        "next_input_csv": str(next_csv),
        "next_selection_report": str(next_report),
        "filtered_csv_path": str(filtered_csv),
        "selection_report_path": str(selection_report),
    }


def validate_step4_output(
    filtered_csv_path: str | Path,
    selection_report_path: str | Path,
    next_input_csv: str | Path | None = None,
    next_selection_report: str | Path | None = None,
) -> Step4ValidationResult:
    issues: list[str] = []
    filtered_csv = resolve_path(filtered_csv_path)
    selection_report = resolve_path(selection_report_path)
    details: dict[str, Any] = {
        "filtered_csv_path": str(filtered_csv),
        "selection_report_path": str(selection_report),
    }

    if not filtered_csv.is_file():
        issues.append(f"Step4 filtered CSV 不存在: {filtered_csv}")
    if not selection_report.is_file():
        issues.append(f"Step4 selection_report 不存在: {selection_report}")

    report: dict[str, Any] = {}
    if selection_report.is_file():
        try:
            report = _load_json(selection_report)
        except Exception as exc:
            issues.append(f"Step4 selection_report 不是合法 JSON: {exc}")

    output_columns: list[str] = []
    if filtered_csv.is_file():
        try:
            output_columns = _read_columns(filtered_csv)
            details["output_column_count"] = len(output_columns)
            details["sample_output_columns"] = output_columns[:30]
        except Exception as exc:
            issues.append(f"Step4 filtered CSV 无法读取: {exc}")

    final_columns = report.get("final_columns") if isinstance(report.get("final_columns"), list) else []
    counts = report.get("column_counts") if isinstance(report.get("column_counts"), dict) else {}
    details.update(
        {
            "task_text": report.get("task_text", ""),
            "original_column_count": counts.get("original_column_count"),
            "final_column_count": counts.get("final_column_count") or len(final_columns),
        }
    )
    if report and not final_columns:
        issues.append("Step4 selection_report 缺少 final_columns 或 final_columns 为空。")
    if output_columns and final_columns and output_columns != final_columns:
        issues.append("Step4 filtered CSV 列顺序与 selection_report.final_columns 不一致。")

    if next_input_csv is not None:
        next_csv = resolve_path(next_input_csv)
        details["next_input_csv"] = str(next_csv)
        if not next_csv.is_file():
            issues.append(f"Step4 next_input filtered.csv 不存在: {next_csv}")
    if next_selection_report is not None:
        next_report = resolve_path(next_selection_report)
        details["next_selection_report"] = str(next_report)
        if not next_report.is_file():
            issues.append(f"Step4 next_input selection_report.json 不存在: {next_report}")

    return Step4ValidationResult(passed=not issues, issues=sorted(set(issues)), details=details)


def run_step4_pipeline(
    input_path: str | Path,
    output_root: str | Path | None = None,
    task_text: str | None = None,
    memory_context: str | None = None,
    emit: ProgressCallback | None = None,
) -> dict[str, Any]:
    input_info = resolve_step4_input(input_path)
    task_info = resolve_step4_task_text(task_text)
    run_info = run_step4_column_selection(
        input_csv_path=input_info["input_csv_path"],
        output_root=output_root,
        task_text=task_info["task_text"],
        memory_context=memory_context,
        emit=emit,
    )
    normalized = normalize_step4_outputs(
        filtered_csv_path=run_info["filtered_csv_path"],
        selection_report_path=run_info["selection_report_path"],
        output_root=run_info["output_root"],
    )
    validation = validate_step4_output(
        filtered_csv_path=run_info["filtered_csv_path"],
        selection_report_path=run_info["selection_report_path"],
        next_input_csv=normalized["next_input_csv"],
        next_selection_report=normalized["next_selection_report"],
    )
    return {
        **input_info,
        **task_info,
        **run_info,
        **normalized,
        **validation.details,
        "status": "SUCCESS" if validation.passed else "NEEDS_REPAIR",
        "issues": validation.issues,
    }


def _read_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix in {".xlsx", ".xls"}:
        return list(pd.read_excel(path, nrows=0).columns)
    raise FileNotFoundError(f"不支持的表格后缀: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}
