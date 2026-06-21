from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import textwrap
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

STEP5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP5_DIR.parent
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_loader import get_agent_config  # noqa: E402


TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls"}
NULL_LIKE_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "missing",
    "unknown",
    "未填写",
    "未知",
    "无",
    "空",
    "-",
    "--",
    "—",
}
HIGH_RISK_NAME_TOKENS = (
    "text",
    "note",
    "report",
    "finding",
    "impression",
    "diagnosis",
    "description",
    "summary",
    "history",
    "comment",
    "备注",
    "说明",
    "报告",
    "描述",
    "诊断",
    "病史",
    "主诉",
    "现病史",
    "影像",
    "结论",
    "所见",
    "意见",
)
ALLOWED_CLEANER_IMPORTS = {"pandas", "os", "re", "agentscope.tool"}
HIGH_RISK_SAMPLE_SIZE = 20
MAX_ANALYSIS_RETRIES = 3
MAX_GENERATION_ATTEMPTS = 4
MAX_RUNTIME_REPAIR_ATTEMPTS = 2
CATEGORY_CARDINALITY_LIMIT = 64
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class Step5ValidationResult:
    passed: bool
    issues: list[str]
    details: dict[str, Any]


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def get_default_output_root() -> str:
    return str(PROJECT_ROOT / "program" / "output" / "step5_results")


def resolve_step5_input(input_path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(input_path)
    if resolved.is_dir():
        for candidate in ("filtered.csv", "input.csv"):
            probe = resolved / candidate
            if probe.is_file():
                resolved = probe.resolve()
                break
    if not resolved.is_file():
        raise FileNotFoundError(f"Step5 输入表不存在: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in TABULAR_SUFFIXES:
        raise FileNotFoundError(f"Step5 输入不是支持的表格文件: {resolved}")
    df = _read_table(resolved)
    return {
        "input_path": str(resolve_path(input_path)),
        "input_csv_path": str(resolved),
        "suffix": suffix,
        "file_size": resolved.stat().st_size,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "sample_columns": [str(col) for col in list(df.columns)[:30]],
        "selection_report_path": _find_adjacent_selection_report(resolved),
    }


def profile_step5_table(
    input_csv_path: str | Path,
    output_root: str | Path | None = None,
    workers: int = 4,
    emit: ProgressCallback | None = None,
) -> dict[str, Any]:
    input_csv = resolve_path(input_csv_path)
    output = resolve_path(output_root or get_default_output_root())
    output.mkdir(parents=True, exist_ok=True)
    df = _read_table(input_csv)
    total = len(df.columns)
    if emit:
        emit(f"[Step5][Runtime] ColumnProfiler 开始: rows={len(df)}, columns={total}, workers={workers}")

    profiles: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as executor:
        futures = {executor.submit(_profile_column, str(col), df[col]): str(col) for col in df.columns}
        for index, future in enumerate(as_completed(futures), start=1):
            profiles.append(future.result())
            _emit_progress("ColumnProfiler", index, total, emit)

    profiles.sort(key=lambda item: list(df.columns).index(item["column_name"]))
    timestamp = _timestamp()
    profile_path = output / f"column_profile_{timestamp}.json"
    report = {
        "input_csv_path": str(input_csv),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "profiles": profiles,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(profile_path, report)
    if emit:
        emit(f"[Step5][Runtime] ColumnProfiler 完成: {profile_path}")
    return {
        "input_csv_path": str(input_csv),
        "output_root": str(output),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "profile_report_path": str(profile_path),
        "profile_sample": profiles[:10],
    }


def classify_step5_column_risks(
    input_csv_path: str | Path,
    profile_report_path: str | Path,
    output_root: str | Path | None = None,
    workers: int = 4,
    emit: ProgressCallback | None = None,
) -> dict[str, Any]:
    input_csv = resolve_path(input_csv_path)
    profile_path = resolve_path(profile_report_path)
    output = resolve_path(output_root or get_default_output_root())
    output.mkdir(parents=True, exist_ok=True)
    profile_report = _load_json(profile_path)
    profiles = list(profile_report.get("profiles") or [])
    total = len(profiles)
    if emit:
        emit(f"[Step5][Runtime] ColumnRiskClassifier 开始: columns={total}, workers={workers}")

    risk_items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as executor:
        futures = {executor.submit(_classify_profile_risk, profile): profile for profile in profiles}
        for index, future in enumerate(as_completed(futures), start=1):
            risk_items.append(future.result())
            _emit_progress("ColumnRiskClassifier", index, total, emit)

    order = {str(profile.get("column_name")): idx for idx, profile in enumerate(profiles)}
    risk_items.sort(key=lambda item: order.get(str(item.get("column_name")), 10**9))
    counts = _risk_counts(risk_items)
    timestamp = _timestamp()
    risk_path = output / f"column_risk_report_{timestamp}.json"
    report = {
        "input_csv_path": str(input_csv),
        "profile_report_path": str(profile_path),
        "risk_counts": counts,
        "columns": risk_items,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(risk_path, report)
    if emit:
        emit(f"[Step5][Runtime] ColumnRiskClassifier 完成: low={counts['low']}, medium={counts['medium']}, high={counts['high']}")
    return {
        "input_csv_path": str(input_csv),
        "column_risk_report_path": str(risk_path),
        "risk_counts": counts,
        "high_risk_columns": [item["column_name"] for item in risk_items if item["risk_level"] == "high"][:80],
        "medium_risk_columns": [item["column_name"] for item in risk_items if item["risk_level"] == "medium"][:80],
    }


def run_step5_data_cleaning(
    input_csv_path: str | Path,
    column_risk_report_path: str | Path,
    output_root: str | Path | None = None,
    workers: int = 4,
    llm_workers: int = 2,
    enable_llm: bool = True,
    emit: ProgressCallback | None = None,
) -> dict[str, Any]:
    input_csv = resolve_path(input_csv_path)
    risk_path = resolve_path(column_risk_report_path)
    output = resolve_path(output_root or get_default_output_root())
    output.mkdir(parents=True, exist_ok=True)
    scripts_dir = output / "generated_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    df = _read_table(input_csv)
    cleaned = df.copy()
    risk_report = _load_json(risk_path)
    risk_items = list(risk_report.get("columns") or [])
    by_column = {str(item.get("column_name")): item for item in risk_items}
    risk_counts = _risk_counts(risk_items)
    if emit:
        emit(
            "[Step5][Runtime] DataCleaningWorker 开始: "
            f"low={risk_counts['low']}, medium={risk_counts['medium']}, high={risk_counts['high']}, workers={workers}"
        )

    high_columns = [col for col in df.columns if by_column.get(str(col), {}).get("risk_level") == "high"]
    medium_columns = [col for col in df.columns if by_column.get(str(col), {}).get("risk_level") == "medium"]
    llm_columns = [col for col in df.columns if by_column.get(str(col), {}).get("risk_level") in {"medium", "high"}]
    low_columns = [col for col in df.columns if by_column.get(str(col), {}).get("risk_level") == "low"]

    changes: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_clean_targets = len(llm_columns)
    completed = 0

    def _run_llm(col: Any) -> tuple[str, pd.Series, dict[str, Any], list[str]]:
        risk_level = str(by_column.get(str(col), {}).get("risk_level") or "medium")
        original = df[col]
        cleaned_series, summary, item_warnings = _clean_high_risk_column(
            column_name=str(col),
            source_df=df,
            input_csv_path=input_csv,
            risk_item=by_column.get(str(col), {}),
            scripts_dir=scripts_dir,
            enable_llm=enable_llm,
            risk_level=risk_level,
        )
        return str(col), cleaned_series, summary, item_warnings

    with ThreadPoolExecutor(max_workers=max(1, min(int(llm_workers or 1), max(1, len(llm_columns))))) as executor:
        futures = {executor.submit(_run_llm, col): col for col in llm_columns}
        for future in as_completed(futures):
            col_name, result_series, summary, item_warnings = future.result()
            cleaned[col_name] = result_series
            changes.append(summary)
            warnings.extend(item_warnings)
            completed += 1
            _emit_progress("DataCleaningWorker", completed, total_clean_targets, emit)

    for col in low_columns:
        changes.append(
            {
                "column_name": str(col),
                "risk_level": "low",
                "method": "report_only",
                "changed_count": 0,
                "sample_changes": [],
            }
        )

    if list(cleaned.columns) != list(df.columns) or len(cleaned) != len(df):
        raise ValueError("Step5 清洗后行数或列结构发生变化，已阻止写出。")

    timestamp = _timestamp()
    cleaned_csv = output / f"cleaned_{timestamp}.csv"
    data_quality_report = output / f"data_quality_report_{timestamp}.json"
    changes_jsonl = output / f"cleaning_changes_{timestamp}.jsonl"
    _write_csv(cleaned, cleaned_csv)
    _write_jsonl(changes_jsonl, sorted(changes, key=lambda item: list(df.columns).index(item["column_name"])))

    changed_cells = int(sum(int(item.get("changed_count") or 0) for item in changes))
    llm_summaries = [item for item in changes if item.get("risk_level") in {"medium", "high"}]
    llm_generated = [item["column_name"] for item in llm_summaries if item.get("status") == "success"]
    llm_noop = [item["column_name"] for item in llm_summaries if item.get("status") == "noop_failed"]
    llm_failed = {
        item["column_name"]: {
            "risk_level": item.get("risk_level"),
            "generation_errors": item.get("generation_errors") or [],
            "runtime_errors": item.get("runtime_errors") or [],
        }
        for item in llm_summaries
        if item.get("status") == "noop_failed" and (item.get("generation_errors") or item.get("runtime_errors"))
    }
    high_summaries = [item for item in llm_summaries if item.get("risk_level") == "high"]
    high_generated = [item["column_name"] for item in high_summaries if item.get("status") == "success"]
    high_noop = [item["column_name"] for item in high_summaries if item.get("status") == "noop_failed"]
    high_failed = {
        item["column_name"]: {
            "generation_errors": item.get("generation_errors") or [],
            "runtime_errors": item.get("runtime_errors") or [],
        }
        for item in high_summaries
        if item.get("status") == "noop_failed" and (item.get("generation_errors") or item.get("runtime_errors"))
    }
    report = {
        "input_csv_path": str(input_csv),
        "cleaned_csv_path": str(cleaned_csv),
        "column_risk_report_path": str(risk_path),
        "cleaning_changes_path": str(changes_jsonl),
        "generated_scripts_dir": str(scripts_dir),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "risk_counts": risk_counts,
        "changed_cell_count": changed_cells,
        "llm_cleaner_enabled": bool(enable_llm),
        "llm_generated_cleaners": llm_generated,
        "llm_failed_columns": llm_failed,
        "llm_noop_columns": llm_noop,
        "high_risk_llm_enabled": bool(enable_llm),
        "high_risk_generated_cleaners": high_generated,
        "high_risk_failed_columns": high_failed,
        "high_risk_noop_columns": high_noop,
        "warnings": warnings,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(data_quality_report, report)
    if emit:
        emit(f"[Step5][Runtime] DataCleaningWorker 完成: changed_cells={changed_cells}")
    return {
        "input_csv_path": str(input_csv),
        "output_root": str(output),
        "cleaned_csv_path": str(cleaned_csv),
        "data_quality_report_path": str(data_quality_report),
        "column_risk_report_path": str(risk_path),
        "cleaning_changes_path": str(changes_jsonl),
        "generated_scripts_dir": str(scripts_dir),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "risk_counts": risk_counts,
        "changed_cell_count": changed_cells,
        "llm_generated_cleaners": llm_generated,
        "llm_failed_columns": llm_failed,
        "llm_noop_columns": llm_noop,
        "high_risk_generated_cleaners": high_generated,
        "high_risk_failed_columns": high_failed,
        "high_risk_noop_columns": high_noop,
        "warnings": warnings,
    }


def normalize_step5_outputs(
    cleaned_csv_path: str | Path,
    data_quality_report_path: str | Path,
    column_risk_report_path: str | Path,
    output_root: str | Path | None = None,
    selection_report_path: str | Path | None = None,
) -> dict[str, Any]:
    cleaned_csv = resolve_path(cleaned_csv_path)
    data_quality_report = resolve_path(data_quality_report_path)
    risk_report = resolve_path(column_risk_report_path)
    output = resolve_path(output_root or cleaned_csv.parent)
    next_input = output / "next_input"
    next_input.mkdir(parents=True, exist_ok=True)
    next_csv = next_input / "filtered.csv"
    next_quality = next_input / "data_quality_report.json"
    next_risk = next_input / "column_risk_report.json"
    for path in (cleaned_csv, data_quality_report, risk_report):
        if not path.is_file():
            raise FileNotFoundError(f"Step5 标准化缺少文件: {path}")
    shutil.copy2(cleaned_csv, next_csv)
    shutil.copy2(data_quality_report, next_quality)
    shutil.copy2(risk_report, next_risk)

    next_selection = ""
    if selection_report_path:
        selection_report = resolve_path(selection_report_path)
        if selection_report.is_file():
            next_selection_path = next_input / "selection_report.json"
            shutil.copy2(selection_report, next_selection_path)
            next_selection = str(next_selection_path)

    return {
        "next_input_dir": str(next_input),
        "next_input_csv": str(next_csv),
        "next_data_quality_report": str(next_quality),
        "next_column_risk_report": str(next_risk),
        "next_selection_report": next_selection,
        "cleaned_csv_path": str(cleaned_csv),
        "data_quality_report_path": str(data_quality_report),
        "column_risk_report_path": str(risk_report),
    }


def validate_step5_output(
    input_csv_path: str | Path,
    cleaned_csv_path: str | Path,
    data_quality_report_path: str | Path | None = None,
    column_risk_report_path: str | Path | None = None,
    next_input_csv: str | Path | None = None,
    next_data_quality_report: str | Path | None = None,
    next_column_risk_report: str | Path | None = None,
) -> Step5ValidationResult:
    issues: list[str] = []
    input_csv = resolve_path(input_csv_path)
    cleaned_csv = resolve_path(cleaned_csv_path)
    details: dict[str, Any] = {
        "input_csv_path": str(input_csv),
        "cleaned_csv_path": str(cleaned_csv),
    }
    if not input_csv.is_file():
        issues.append(f"Step5 输入表不存在: {input_csv}")
    if not cleaned_csv.is_file():
        issues.append(f"Step5 cleaned CSV 不存在: {cleaned_csv}")

    if input_csv.is_file() and cleaned_csv.is_file():
        try:
            input_df = _read_table(input_csv)
            output_df = _read_table(cleaned_csv)
            details.update(
                {
                    "input_row_count": int(len(input_df)),
                    "input_column_count": int(len(input_df.columns)),
                    "output_row_count": int(len(output_df)),
                    "output_column_count": int(len(output_df.columns)),
                    "sample_output_columns": [str(col) for col in list(output_df.columns)[:30]],
                }
            )
            if len(input_df) != len(output_df):
                issues.append(f"Step5 清洗后行数变化: {len(input_df)} -> {len(output_df)}")
            if list(input_df.columns) != list(output_df.columns):
                issues.append("Step5 清洗后列名或列顺序变化。")
        except Exception as exc:
            issues.append(f"Step5 cleaned CSV 无法读取或对账: {exc}")

    for label, value in (
        ("data_quality_report_path", data_quality_report_path),
        ("column_risk_report_path", column_risk_report_path),
        ("next_input_csv", next_input_csv),
        ("next_data_quality_report", next_data_quality_report),
        ("next_column_risk_report", next_column_risk_report),
    ):
        if value is None:
            continue
        path = resolve_path(value)
        details[label] = str(path)
        if not path.is_file():
            issues.append(f"Step5 产物不存在: {path}")

    return Step5ValidationResult(passed=not issues, issues=sorted(set(issues)), details=details)


def run_step5_pipeline(
    input_path: str | Path,
    output_root: str | Path | None = None,
    workers: int = 4,
    llm_workers: int = 2,
    enable_llm: bool = True,
    emit: ProgressCallback | None = None,
) -> dict[str, Any]:
    input_info = resolve_step5_input(input_path)
    profile = profile_step5_table(input_info["input_csv_path"], output_root=output_root, workers=workers, emit=emit)
    risks = classify_step5_column_risks(
        input_info["input_csv_path"],
        profile["profile_report_path"],
        output_root=output_root,
        workers=workers,
        emit=emit,
    )
    cleaned = run_step5_data_cleaning(
        input_info["input_csv_path"],
        risks["column_risk_report_path"],
        output_root=output_root,
        workers=workers,
        llm_workers=llm_workers,
        enable_llm=enable_llm,
        emit=emit,
    )
    normalized = normalize_step5_outputs(
        cleaned["cleaned_csv_path"],
        cleaned["data_quality_report_path"],
        cleaned["column_risk_report_path"],
        output_root=cleaned["output_root"],
        selection_report_path=input_info.get("selection_report_path") or None,
    )
    validation = validate_step5_output(
        input_csv_path=input_info["input_csv_path"],
        cleaned_csv_path=cleaned["cleaned_csv_path"],
        data_quality_report_path=cleaned["data_quality_report_path"],
        column_risk_report_path=cleaned["column_risk_report_path"],
        next_input_csv=normalized["next_input_csv"],
        next_data_quality_report=normalized["next_data_quality_report"],
        next_column_risk_report=normalized["next_column_risk_report"],
    )
    return {
        **input_info,
        **profile,
        **risks,
        **cleaned,
        **normalized,
        **validation.details,
        "status": "SUCCESS" if validation.passed else "NEEDS_REPAIR",
        "issues": validation.issues,
    }


def apply_builtin_safe_cleaning(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value)
    stripped = text.strip()
    if stripped.lower() in NULL_LIKE_TOKENS or stripped in NULL_LIKE_TOKENS:
        return ""
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", stripped):
        return stripped.replace(",", "")
    normalized_date = _normalize_obvious_date(stripped)
    if normalized_date is not None:
        return normalized_date
    return stripped


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return pd.read_csv(path, dtype=object, keep_default_na=False, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, dtype=object, keep_default_na=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=object, keep_default_na=False)
    raise FileNotFoundError(f"不支持的表格后缀: {path}")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _find_adjacent_selection_report(path: Path) -> str:
    for name in ("selection_report.json",):
        candidate = path.parent / name
        if candidate.is_file():
            return str(candidate)
    matches = sorted(path.parent.glob("*selection_report*.json"))
    return str(matches[-1]) if matches else ""


def _sample_values(series: pd.Series, max_items: int = 12, max_chars: int = 160) -> list[str]:
    samples: list[str] = []
    for value in series.tolist():
        text = "" if value is None else str(value).strip()
        if not text or text.lower() in NULL_LIKE_TOKENS:
            continue
        samples.append(text[:max_chars])
        if len(samples) >= max_items:
            break
    return samples


def _profile_column(column_name: str, series: pd.Series) -> dict[str, Any]:
    values = ["" if value is None else str(value) for value in series.tolist()]
    non_empty = [value.strip() for value in values if value.strip() and value.strip().lower() not in NULL_LIKE_TOKENS]
    lengths = [len(value) for value in non_empty]
    safe_change_count = sum(1 for value in values if apply_builtin_safe_cleaning(value) != value)
    text_like_count = sum(1 for value in non_empty if _looks_like_free_text(value))
    return {
        "column_name": column_name,
        "row_count": int(len(series)),
        "non_empty_count": int(len(non_empty)),
        "missing_count": int(len(series) - len(non_empty)),
        "unique_count": int(len(set(non_empty))),
        "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "max_length": max(lengths) if lengths else 0,
        "text_like_count": int(text_like_count),
        "safe_change_count": int(safe_change_count),
        "sample_values": _sample_values(series),
    }


def _classify_profile_risk(profile: dict[str, Any]) -> dict[str, Any]:
    column_name = str(profile.get("column_name") or "")
    name_lower = column_name.lower()
    non_empty = int(profile.get("non_empty_count") or 0)
    text_like = int(profile.get("text_like_count") or 0)
    max_length = int(profile.get("max_length") or 0)
    avg_length = float(profile.get("avg_length") or 0)
    safe_change_count = int(profile.get("safe_change_count") or 0)
    reasons: list[str] = []

    if any(token in name_lower or token in column_name for token in HIGH_RISK_NAME_TOKENS):
        reasons.append("column_name_indicates_semantic_text")
    if max_length >= 300 or avg_length >= 80:
        reasons.append("long_free_text_values")
    if non_empty and text_like / max(1, non_empty) >= 0.35:
        reasons.append("high_text_like_ratio")

    if reasons:
        risk = "high"
    elif safe_change_count > 0:
        risk = "medium"
        reasons.append("safe_deterministic_cleaning_available")
    else:
        risk = "low"
        reasons.append("no_safe_change_detected")

    return {
        "column_name": column_name,
        "risk_level": risk,
        "reasons": reasons,
        "profile": profile,
    }


def _risk_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"low": 0, "medium": 0, "high": 0}
    for item in items:
        risk = str(item.get("risk_level") or "low")
        if risk in counts:
            counts[risk] += 1
    return counts


def _looks_like_free_text(value: str) -> bool:
    if len(value) >= 80:
        return True
    punctuation_count = sum(1 for char in value if char in "，。；：,.!?;:")
    if punctuation_count >= 2:
        return True
    return bool(re.search(r"\s{2,}|\b(the|and|with|without|patient|history|finding)\b", value.lower()))


def _normalize_obvious_date(value: str) -> str | None:
    if re.fullmatch(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", value):
        try:
            return pd.to_datetime(value, errors="raise").strftime("%Y-%m-%d")
        except Exception:
            return None
    if re.fullmatch(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2}(:\d{2})?", value):
        try:
            return pd.to_datetime(value, errors="raise").strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    return None


def _parse_numeric_like(value: str) -> tuple[str, bool]:
    text = str(value).strip()
    if not text:
        return text, False
    lowered = text.lower()
    if lowered in NULL_LIKE_TOKENS or text in NULL_LIKE_TOKENS:
        return "", True
    comma_number = re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", text)
    if comma_number:
        return text.replace(",", ""), True
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return str(int(float(text))), True
    return text, False


def _normalize_category_value(value: str) -> tuple[str, bool]:
    text = str(value).strip()
    if not text:
        return text, False
    lowered = text.lower()
    if lowered in NULL_LIKE_TOKENS or text in NULL_LIKE_TOKENS:
        return "", True
    if text == "??":
        return "UNKNOWN", True
    normalized = re.sub(r"\s+", " ", text).strip()
    upper_value = normalized.upper()
    if upper_value in {"N/A", "NA", "NONE", "NULL", "UNKNOWN", "UNK"}:
        return "UNKNOWN", upper_value != "UNKNOWN"
    if upper_value == "MIXED-CASE":
        return "MIXED CASE", True
    if normalized != text:
        return normalized, True
    if normalized.islower():
        return normalized.upper(), True
    return upper_value, upper_value != text


def _normalize_code_value(value: str) -> tuple[str, bool]:
    text = str(value).strip()
    if not text:
        return text, False
    lowered = text.lower()
    if lowered in NULL_LIKE_TOKENS or text in NULL_LIKE_TOKENS:
        return "", True
    normalized = re.sub(r"\s+", "", text).upper()
    normalized = normalized.replace("|", ",")
    normalized = re.sub(r",+", ",", normalized).strip(",")
    if normalized in {"???", "UNK", "UNKNOWN"}:
        return "", True
    return normalized, normalized != text


def _apply_profile_driven_cleaning(
    column_name: str,
    series: pd.Series,
    profile: dict[str, Any],
) -> tuple[pd.Series, str, str]:
    column_type = str(profile.get("column_type") or "")
    sample_values = profile.get("sample_values") or []
    unique_count = int(profile.get("unique_count") or 0)
    text_like_count = int(profile.get("text_like_count") or 0)
    row_count = int(profile.get("row_count") or len(series))
    max_length = int(profile.get("max_length") or 0)
    safe_change_count = int(profile.get("safe_change_count") or 0)

    if safe_change_count <= 0 and column_type not in {"numeric", "categorical", "code", "datetime"}:
        return series.copy(), "", ""

    values = series.fillna("").astype(str)
    cleaned = values.copy()
    changed = False

    if column_type == "numeric":
        for idx, value in values.items():
            new_value, did_change = _parse_numeric_like(value)
            if did_change:
                cleaned.at[idx] = new_value
                changed = True
        if changed:
            return cleaned, "profile_numeric_fallback", f"Used profile-driven numeric normalization for {column_name}."
        return series.copy(), "", ""

    if column_type == "datetime":
        for idx, value in values.items():
            text = str(value).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in NULL_LIKE_TOKENS or text in NULL_LIKE_TOKENS:
                cleaned.at[idx] = ""
                changed = True
                continue
            normalized = _normalize_obvious_date(text)
            if normalized is not None and normalized != value:
                cleaned.at[idx] = normalized
                changed = True
        if changed:
            return cleaned, "profile_datetime_fallback", f"Used profile-driven datetime normalization for {column_name}."
        return series.copy(), "", ""

    numeric_like_ratio = sum(bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", str(v).strip())) for v in sample_values) / len(sample_values) if sample_values else 0.0
    looks_like_count_metric = any(token in column_name.lower() for token in ["count", "nunique", "mean", "std", "min", "max", "flag"]) and numeric_like_ratio >= 0.5
    if looks_like_count_metric:
        for idx, value in values.items():
            new_value, did_change = _parse_numeric_like(value)
            if did_change:
                cleaned.at[idx] = new_value
                changed = True
        if changed:
            return cleaned, "profile_metric_numeric_fallback", f"Used metric-style numeric normalization for {column_name}."

    if column_type == "categorical":
        if unique_count and unique_count <= CATEGORY_CARDINALITY_LIMIT and text_like_count <= max(2, row_count // 20):
            for idx, value in values.items():
                new_value, did_change = _normalize_category_value(value)
                if did_change:
                    cleaned.at[idx] = new_value
                    changed = True
            if changed:
                return cleaned, "profile_category_fallback", f"Used profile-driven categorical normalization for {column_name}."
        return series.copy(), "", ""

    if column_type == "code":
        if max_length <= 120:
            for idx, value in values.items():
                new_value, did_change = _normalize_code_value(value)
                if did_change:
                    cleaned.at[idx] = new_value
                    changed = True
            if changed:
                return cleaned, "profile_code_fallback", f"Used profile-driven code normalization for {column_name}."
        return series.copy(), "", ""

    return series.copy(), "", ""


def _column_change_summary(
    column_name: str,
    original: pd.Series,
    cleaned: pd.Series,
    risk_level: str,
    method: str,
    script_path: str = "",
    llm_explanation: str = "",
) -> dict[str, Any]:
    changed_mask = original.astype(str) != cleaned.astype(str)
    sample_changes = []
    for old, new in zip(original[changed_mask].head(10).tolist(), cleaned[changed_mask].head(10).tolist()):
        sample_changes.append({"before": str(old)[:160], "after": str(new)[:160]})
    return {
        "column_name": column_name,
        "risk_level": risk_level,
        "method": method,
        "changed_count": int(changed_mask.sum()),
        "sample_changes": sample_changes,
        "script_path": script_path,
        "llm_explanation": llm_explanation,
    }


def _clean_high_risk_column(
    column_name: str,
    source_df: pd.DataFrame,
    input_csv_path: Path,
    risk_item: dict[str, Any],
    scripts_dir: Path,
    enable_llm: bool,
    risk_level: str = "high",
) -> tuple[pd.Series, dict[str, Any], list[str]]:
    original = source_df[column_name]
    risk = risk_level if risk_level in {"medium", "high"} else "high"
    risk_label = f"{risk}-risk"
    cleaner_dir = scripts_dir / f"{_safe_file_stem(column_name)}_{_short_hash(column_name)}"
    if cleaner_dir.exists():
        shutil.rmtree(cleaner_dir)
    cleaner_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    generation_errors: list[str] = []
    runtime_errors: list[str] = []
    explanation = ""
    script_path = cleaner_dir / "implementation.py"
    profile = _infer_high_risk_column_profile(
        column_name,
        _high_risk_sample_values(original),
        risk_item,
    )

    fallback_series, fallback_method, fallback_explanation = _apply_profile_driven_cleaning(
        column_name=column_name,
        series=original,
        profile=profile,
    )

    if not enable_llm:
        method = fallback_method or "llm_disabled_noop"
        status = "success" if fallback_method else "noop_failed"
        explanation = fallback_explanation or f"LLM disabled; {risk_label} column kept unchanged."
        cleaned_series = fallback_series if fallback_method else original.copy()
        _write_noop_cleaner(cleaner_dir, column_name, explanation)
        if not fallback_method:
            warnings.append(f"LLM {risk_label} cleaner disabled for {column_name}; kept original values.")
    else:
        cleaner_info, generation_errors = _build_high_risk_cleaner(
            column_name=column_name,
            source_df=source_df,
            input_csv_path=input_csv_path,
            risk_item=risk_item,
            cleaner_dir=cleaner_dir,
        )
        if cleaner_info is None:
            method = fallback_method or "llm_cleaner_generation_failed_noop"
            status = "success" if fallback_method else "noop_failed"
            explanation = fallback_explanation or f"LLM cleaner generation failed; {risk_label} column kept unchanged."
            _write_noop_cleaner(cleaner_dir, column_name, explanation)
            cleaned_series = fallback_series if fallback_method else original.copy()
            warning_prefix = "used deterministic fallback" if fallback_method else "kept original values"
            warnings.append(
                f"LLM {risk_label} cleaner generation failed for {column_name}; {warning_prefix}: "
                f"{'; '.join(generation_errors[:3])}"
            )
        else:
            explanation = str(cleaner_info.get("summary") or "")
            cleaned_series, runtime_errors = _execute_high_risk_cleaner_with_repair(
                column_name=column_name,
                source_df=source_df,
                input_csv_path=input_csv_path,
                cleaner_dir=cleaner_dir,
                cleaner_info=cleaner_info,
            )
            method = "llm_generated_cleaner"
            status = "success"
            if runtime_errors:
                method = fallback_method or "llm_cleaner_runtime_failed_noop"
                status = "success" if fallback_method else "noop_failed"
                cleaned_series = fallback_series if fallback_method else original.copy()
                if fallback_method and fallback_explanation:
                    explanation = fallback_explanation
                warning_prefix = "used deterministic fallback" if fallback_method else "kept original values"
                warnings.append(
                    f"LLM {risk_label} cleaner runtime failed for {column_name}; {warning_prefix}: "
                    f"{'; '.join(runtime_errors[:3])}"
                )

    summary = _column_change_summary(
        column_name=column_name,
        original=original,
        cleaned=cleaned_series,
        risk_level=risk,
        method=method,
        script_path=str(script_path),
        llm_explanation=explanation,
    )
    summary.update(
        {
            "status": status,
            "cleaner_dir": str(cleaner_dir),
            "generation_errors": generation_errors,
            "runtime_errors": runtime_errors,
            "profile_column_type": profile.get("column_type"),
        }
    )
    return cleaned_series, summary, warnings


def _build_high_risk_cleaner(
    column_name: str,
    source_df: pd.DataFrame,
    input_csv_path: Path,
    risk_item: dict[str, Any],
    cleaner_dir: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    values = _high_risk_sample_values(source_df[column_name])
    profile = _infer_high_risk_column_profile(column_name, values, risk_item)
    summary = _default_high_risk_summary(column_name, profile)
    last_errors: list[str] = []

    if not summary:
        analysis_prompt = _medical_analysis_prompt(column_name, values, profile)
        for _ in range(MAX_ANALYSIS_RETRIES):
            try:
                summary = _call_high_risk_llm(analysis_prompt)
                break
            except Exception as exc:
                last_errors = [f"analysis failed: {type(exc).__name__}: {exc}"]
        if not summary:
            return None, last_errors or ["analysis failed"]

    cleaner_name = _safe_file_stem(column_name)
    prompt = _cleaner_generation_prompt(column_name, summary, cleaner_name, profile)
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        raw_text = ""
        try:
            raw_text = _call_high_risk_llm(prompt)
            (cleaner_dir / f"RAW_RESPONSE_attempt_{attempt}.txt").write_text(raw_text, encoding="utf-8")
            package = _parse_cleaner_package_response(raw_text)
        except Exception as exc:
            last_errors = [f"输出格式错误: {type(exc).__name__}: {exc}"]
            prompt = _build_cleaner_retry_prompt(column_name, summary, cleaner_name, profile, last_errors)
            continue

        last_errors = _validate_cleaner_package(package)
        if last_errors:
            prompt = _build_cleaner_retry_prompt(column_name, summary, cleaner_name, profile, last_errors)
            continue

        _write_cleaner_package(cleaner_dir, package)
        probe_path = cleaner_dir / f"probe_{cleaner_name}.csv"
        shutil.copyfile(input_csv_path, probe_path)
        output_file, run_errors = _load_run_and_validate_cleaner(
            impl_path=cleaner_dir / "implementation.py",
            input_path=probe_path,
            module_name=f"probe_{_short_hash(column_name)}_{attempt}",
            reference_df=source_df,
            target_column=column_name,
        )
        for path in (probe_path, Path(output_file) if output_file else None):
            if path and path.exists():
                path.unlink()
        if not run_errors:
            return {"cleaner_name": cleaner_name, "summary": summary, "profile": profile}, []

        last_errors = run_errors
        prompt = _build_cleaner_retry_prompt(column_name, summary, cleaner_name, profile, run_errors)

    return None, last_errors


def _execute_high_risk_cleaner_with_repair(
    column_name: str,
    source_df: pd.DataFrame,
    input_csv_path: Path,
    cleaner_dir: Path,
    cleaner_info: dict[str, Any],
) -> tuple[pd.Series, list[str]]:
    impl_path = cleaner_dir / "implementation.py"
    run_input = cleaner_dir / f"run_{cleaner_info['cleaner_name']}.csv"
    shutil.copyfile(input_csv_path, run_input)
    output_file: str | None = None
    errors: list[str] = []

    try:
        for attempt in range(1, MAX_RUNTIME_REPAIR_ATTEMPTS + 2):
            output_file, errors = _load_run_and_validate_cleaner(
                impl_path=impl_path,
                input_path=run_input,
                module_name=f"run_{_short_hash(column_name)}_{attempt}",
                reference_df=source_df,
                target_column=column_name,
            )
            if not errors and output_file:
                output_df = _read_table(Path(output_file))
                return output_df[column_name], []
            if attempt > MAX_RUNTIME_REPAIR_ATTEMPTS:
                break
            prompt = _build_cleaner_retry_prompt(
                column_name,
                str(cleaner_info.get("summary") or ""),
                str(cleaner_info.get("cleaner_name") or _safe_file_stem(column_name)),
                dict(cleaner_info.get("profile") or {}),
                errors,
            )
            try:
                raw_text = _call_high_risk_llm(prompt)
                (cleaner_dir / f"RUNTIME_REPAIR_attempt_{attempt}.txt").write_text(raw_text, encoding="utf-8")
                package = _parse_cleaner_package_response(raw_text)
                package_errors = _validate_cleaner_package(package)
                if package_errors:
                    errors = package_errors
                    continue
                _write_cleaner_package(cleaner_dir, package)
            except Exception as exc:
                errors = [f"修复 cleaner 失败: {type(exc).__name__}: {exc}"]
    finally:
        for path in (run_input, Path(output_file) if output_file else None):
            if path and path.exists():
                path.unlink()

    return source_df[column_name].copy(), errors or ["runtime failed"]


def _call_high_risk_llm(prompt: str) -> str:
    from openai import OpenAI

    cfg = get_agent_config("agent_5")
    api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key")
    if not api_key:
        raise RuntimeError("agent_5 api_key is not configured")
    base_url = os.environ.get("OPENAI_API_BASE") or cfg.get("base_url") or cfg.get("api_base") or "https://api.openai.com/v1"
    model = os.environ.get("MODEL_NAME") or cfg.get("model_name") or cfg.get("model") or "gpt-4.1-mini"
    timeout = float(cfg.get("timeout") or 120)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    response = client.chat.completions.create(
        model=str(model),
        messages=[
            {
                "role": "system",
                "content": "你是保守的结构化医疗 CSV 数据清洗专家，只生成安全、可验证的清洗逻辑。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=float(cfg.get("temperature") or 0.0),
    )
    return response.choices[0].message.content or ""


def _parse_cleaner_package_response(text: str) -> dict[str, str]:
    md = re.search(r"===CLEANER_MD===\s*(.*?)\s*===END_CLEANER_MD===", text, re.DOTALL)
    py = re.search(r"===IMPLEMENTATION_PY===\s*(.*?)\s*===END_IMPLEMENTATION_PY===", text, re.DOTALL)
    if not md or not py:
        raise ValueError("未找到完整的 cleaner 分段输出。")
    return {
        "cleaner_md": md.group(1).strip(),
        "implementation_py": _strip_code_fence(py.group(1).strip()),
    }


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _validate_cleaner_package(package: dict[str, str]) -> list[str]:
    errors: list[str] = []
    cleaner_md = str(package.get("cleaner_md") or "")
    code = str(package.get("implementation_py") or "")
    if not cleaner_md.strip():
        errors.append("缺少 cleaner_md。")
    elif len(cleaner_md) > 2500:
        errors.append("cleaner_md 过长。")
    if not code.strip():
        return errors + ["缺少 implementation_py。"]
    try:
        tree = ast.parse(code)
        compile(code, "<cleaner>", "exec")
    except SyntaxError as exc:
        return errors + [f"Python 语法错误: {exc.msg} (line {exc.lineno})"]

    has_entry = any(isinstance(node, ast.FunctionDef) and node.name == "data_cleaning" for node in tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_CLEANER_IMPORTS:
                    errors.append(f"导入了未允许模块: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_CLEANER_IMPORTS:
                errors.append(f"from-import 了未允许模块: {node.module}")
    if not has_entry:
        errors.append("缺少 data_cleaning 函数。")
    if "ToolResponse" not in code:
        errors.append("未使用 ToolResponse。")
    return errors


def _write_cleaner_package(cleaner_dir: Path, package: dict[str, str]) -> None:
    (cleaner_dir / "implementation.py").write_text(package["implementation_py"].strip() + "\n", encoding="utf-8")
    (cleaner_dir / "CLEANER.md").write_text(package["cleaner_md"].strip() + "\n", encoding="utf-8")


def _write_noop_cleaner(cleaner_dir: Path, column_name: str, explanation: str) -> None:
    package = {
        "cleaner_md": f"---\nname: {_safe_file_stem(column_name)}\ndescription: llm noop fallback\n---\n{explanation}",
        "implementation_py": textwrap.dedent(
            """
            import pandas as pd
            from agentscope.tool import ToolResponse

            def data_cleaning(input_path: str, index=False) -> ToolResponse:
                df = pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])
                output_path = input_path.replace(".csv", "_cleaned.csv")
                df.to_csv(output_path, index=index)
                return ToolResponse(content="llm target column kept unchanged", metadata={"output_file": output_path})
            """
        ).strip(),
    }
    _write_cleaner_package(cleaner_dir, package)


def _load_run_and_validate_cleaner(
    impl_path: Path,
    input_path: Path,
    module_name: str,
    reference_df: pd.DataFrame,
    target_column: str,
) -> tuple[str | None, list[str]]:
    output_file, errors = _load_and_run_cleaner(impl_path, input_path, module_name)
    if errors or not output_file:
        return output_file, errors
    try:
        output_df = _read_table(Path(output_file))
    except Exception as exc:
        return output_file, [f"输出文件无法读取: {exc}"]
    return output_file, _validate_cleaner_output(reference_df, output_df, target_column)


def _load_and_run_cleaner(impl_path: Path, input_path: Path, module_name: str) -> tuple[str | None, list[str]]:
    try:
        _ensure_tool_response_available()
        spec = importlib.util.spec_from_file_location(module_name, impl_path)
        if spec is None or spec.loader is None:
            return None, [f"无法加载 cleaner: {impl_path}"]
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "data_cleaning"):
            return None, ["缺少 data_cleaning 函数。"]
        response = module.data_cleaning(str(input_path))
        metadata = getattr(response, "metadata", None)
        if metadata is None and isinstance(response, dict):
            metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            return None, ["未返回 metadata。"]
        output_file = metadata.get("output_file")
        if not output_file:
            return None, ["未返回 metadata.output_file。"]
        if not Path(output_file).exists():
            return None, [f"输出文件不存在: {output_file}"]
        return str(output_file), []
    except Exception as exc:
        return None, [f"运行失败: {type(exc).__name__}: {exc}"]


def _validate_cleaner_output(reference_df: pd.DataFrame, output_df: pd.DataFrame, target_column: str) -> list[str]:
    errors: list[str] = []
    if len(reference_df) != len(output_df):
        errors.append(f"probe 输出行数变化: {len(reference_df)} -> {len(output_df)}")
    if list(reference_df.columns) != list(output_df.columns):
        errors.append("probe 输出列名或列顺序变化。")
        return errors
    changed_other = [
        str(col)
        for col in reference_df.columns
        if str(col) != target_column and not _series_values_equal(reference_df[col], output_df[col])
    ]
    if changed_other:
        errors.append(f"probe 修改了非目标列: {changed_other[:8]}")
    return errors


def _series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    return left.fillna("").astype(str).tolist() == right.fillna("").astype(str).tolist()


class _ToolResponseFallback:
    def __init__(self, content: str = "", metadata: dict[str, Any] | None = None, **_: Any) -> None:
        self.content = content
        self.metadata = metadata or {}


def _ensure_tool_response_available() -> None:
    try:
        from agentscope.tool import ToolResponse as _ToolResponse  # noqa: F401
        return
    except Exception:
        pass
    agentscope_module = sys.modules.get("agentscope") or types.ModuleType("agentscope")
    tool_module = types.ModuleType("agentscope.tool")
    tool_module.ToolResponse = _ToolResponseFallback
    setattr(agentscope_module, "tool", tool_module)
    sys.modules["agentscope"] = agentscope_module
    sys.modules["agentscope.tool"] = tool_module


def _medical_analysis_prompt(column_name: str, sample_values: list[str], column_profile: dict[str, Any]) -> str:
    return f"""
你是一位结构化 CSV 数据分析师。

列名: {column_name}
列画像: {column_profile}
样本: {sample_values}

请输出一段简短分析，内容只包括：
1. 这列大致是什么类型
2. 应采取什么清洗策略
3. 哪些操作应该避免
尽量控制在 120 字以内。
""".strip()


def _cleaner_generation_prompt(column_name: str, auto_desc: str, cleaner_name: str, column_profile: dict[str, Any]) -> str:
    return f"""你是一个结构化 CSV 数据清洗专家，请为字段 `{column_name}` 生成一个 AgentScope cleaner。

只生成一个单列 cleaner，只能处理 `{column_name}`，不能修改其他列。

列画像:
{column_profile}

分析摘要:
{auto_desc}

严格按下面格式输出，不要加解释：

===CLEANER_MD===
这里放 YAML+Markdown 描述
===END_CLEANER_MD===

===IMPLEMENTATION_PY===
这里放完整 Python 代码
===END_IMPLEMENTATION_PY===

cleaner_md 模板:
---
name: {cleaner_name}
description: 仅清洗 `{column_name}` 列

---
1. 简短步骤
2. 简短步骤

implementation_py 要求:
- 导入只能使用 `pandas`、`os`、`agentscope.tool`、`re`
- 必须定义 `def data_cleaning(input_path: str, index=False) -> ToolResponse:`
- 读取 CSV 时使用 `pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=[""])`
- 只在列存在时处理 `{column_name}`
- 保持其他列、行数、索引不变
- 优先做轻量规范化，不确定就保留原值
- identifier/code 列不要删重、不要填缺失、不要重编号、不要强制转 int
- 保存到 `input_path.replace('.csv', '_cleaned.csv')`
- 成功时只能返回 `ToolResponse(content="...", metadata={{"output_file": output_path}})`
- 失败时只能返回 `ToolResponse(content="...")`
- 不要使用 `ToolResponse(success=...)`、`ToolResponse(status=...)`、`message=` 这类参数
"""


def _build_cleaner_retry_prompt(
    column_name: str,
    summary: str,
    cleaner_name: str,
    profile: dict[str, Any],
    errors: list[str],
) -> str:
    base = _cleaner_generation_prompt(column_name, summary, cleaner_name, profile)
    error_text = "\n".join(f"- {error}" for error in errors[:6])
    return f"{base}\n\n【修正要求】\n请根据以下错误重写 cleaner，仅输出规定分段格式。\n{error_text}\n"


def _high_risk_sample_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.dropna().tolist():
        text = str(value).strip()
        if not text or text.lower() in NULL_LIKE_TOKENS or text in NULL_LIKE_TOKENS or text in seen:
            continue
        seen.add(text)
        values.append(text[:240])
        if len(values) >= HIGH_RISK_SAMPLE_SIZE:
            break
    return values


def _infer_high_risk_column_profile(column_name: str, values: list[str], risk_item: dict[str, Any]) -> dict[str, Any]:
    existing = dict(risk_item.get("profile") or {})
    lower = column_name.lower()

    def ratio(fn: Callable[[str], bool]) -> float:
        return sum(1 for value in values if fn(value)) / len(values) if values else 0.0

    is_num = lambda value: bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", value))
    is_dt = lambda value: any(
        re.fullmatch(pattern, value)
        for pattern in (
            r"\d{4}-\d{1,2}-\d{1,2}",
            r"\d{4}/\d{1,2}/\d{1,2}",
            r"\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{2}(:\d{2})?",
            r"\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}(:\d{2})?",
        )
    )
    profile_non_empty = int(existing.get("non_empty_count") or len(values) or 0)
    profile_unique = int(existing.get("unique_count") or len(set(values)) or 0)
    profile_text_like = int(existing.get("text_like_count") or 0)
    profile_avg_len = float(existing.get("avg_length") or 0.0)
    profile_max_len = int(existing.get("max_length") or 0)
    metric_hint = any(token in lower for token in ["count", "nunique", "mean", "std", "min", "max", "flag", "ratio"])
    numeric_ratio = ratio(is_num)
    datetime_ratio = ratio(is_dt)
    low_cardinality = profile_unique <= min(CATEGORY_CARDINALITY_LIMIT, max(12, profile_non_empty // 2 or 12))
    text_heavy = profile_text_like >= max(3, profile_non_empty // 3) or profile_avg_len >= 80 or profile_max_len >= 300

    if metric_hint and numeric_ratio >= 0.5:
        column_type, reason = "numeric", "列名像聚合指标且样本多数为数值"
    elif any(token in lower for token in ["id", "index", "编号", "序号"]) and numeric_ratio < 0.5:
        column_type, reason = "identifier", "列名像标识列"
    elif any(token in lower for token in ["time", "date", "日期", "时间"]) or datetime_ratio >= 0.6:
        column_type, reason = "datetime", "列名或样本像时间列"
    elif any(token in lower for token in ["code", "icd", "编码", "代码"]):
        column_type, reason = "code", "列名像编码列"
    elif numeric_ratio >= 0.8:
        column_type, reason = "numeric", "大部分样本是数值"
    elif low_cardinality and not text_heavy:
        column_type, reason = "categorical", "样本重复度高且不像自由文本"
    elif text_heavy:
        column_type, reason = "free_text", "样本较长或文本特征明显"
    else:
        column_type, reason = "text", "按普通文本列处理"
    existing.update(
        {
            "column_type": existing.get("column_type") or column_type,
            "profile_reason": existing.get("profile_reason") or reason,
            "risk_reasons": risk_item.get("reasons") or [],
            "sample_values": values,
        }
    )
    return existing


def _default_high_risk_summary(column_name: str, profile: dict[str, Any]) -> str:
    profile_type = profile.get("column_type")
    if profile_type in {"identifier", "code"}:
        return f"{column_name} 像标识/编码列，应保持原值语义，不删重、不补缺失、不重编号，只做轻量格式规范化。"
    if profile_type == "datetime":
        return f"{column_name} 像时间列，应统一时间格式，避免补造时间。"
    if profile_type == "numeric":
        return f"{column_name} 像数值列，应仅做安全格式标准化，不改变数值意义。"
    return ""


def _safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", value).strip("_")
    return stem[:60] or "column"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _emit_progress(label: str, done: int, total: int, emit: ProgressCallback | None) -> None:
    if not emit or total <= 0:
        return
    step = max(1, total // 20)
    if done == 1 or done == total or done % step == 0:
        emit(f"[Step5][Progress] {label}: {done}/{total} ({done / total * 100:.1f}%)")
