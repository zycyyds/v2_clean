# -*- coding: utf-8 -*-
"""STEP 6: patient consistency validation.

This module is intentionally standalone.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent  # lib/ → v2/
AGENT_ROOT = PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config_loader import get_agent_config
except Exception:  # pragma: no cover - standalone fallback
    get_agent_config = None

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope_compat import CharTokenCounter
from agentscope.tool import Toolkit, ToolResponse

_ENV_PATH = PROJECT_ROOT / ".env"
if _ENV_PATH.is_file():
    with open(_ENV_PATH, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

AGENT_CFG = get_agent_config("agent_6_7") if get_agent_config else {}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEYS") or str(AGENT_CFG.get("api_key") or "")
OPENAI_BASE_URL = (
    os.environ.get("YUNWU_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or str(AGENT_CFG.get("base_url") or AGENT_CFG.get("api_base") or "https://api.openai.com/v1")
)
MODEL_NAME = os.environ.get("MODEL_NAME") or str(AGENT_CFG.get("model") or AGENT_CFG.get("model_name") or "gpt-4.1-mini")

_FORMATTER_MAX_CHARS = 375_000
CONFIDENCE_THRESHOLD = float(os.environ.get("STEP6_CONFIDENCE_THRESHOLD", "0.70"))

DATA_SOURCE = os.environ.get("DATA_SOURCE", "original").lower()
_IS_MIMIC = DATA_SOURCE == "mimic"
PATIENT_ID_COL = os.environ.get("PATIENT_ID_COL", "record_index" if _IS_MIMIC else "patient_id")

INPUT_CSV = ""
RAW_CSV = ""
OUTPUT_STEP6 = ""
_LAST_SAVED_ARTIFACTS: dict[str, str] = {}
_LAST_ANALYSIS_PASSED_IDS: list[str] | None = None

_SKIP_COLS_STATIC = {
    "ocr_text",
    "preprocessed_text",
    "file_path",
    "error",
    "relations",
    "temporal_info",
    "quantity_info",
    "impression",
    "indication",
}
_MIMIC_EXTRA_SKIP_COLS = {
    "所有用药",
    "实验室检验_摘要",
    "所有手术操作_titles",
    "所有手术操作_codes",
    "所有诊断_icd_codes",
    "抽取_Disease",
    "抽取_Drug",
    "抽取_Symptom",
    "抽取_Test",
    "抽取_Treatment",
    "抽取_LabValue",
    "抽取_Finding",
}
_SKIP_COLS: set[str] = set()
_LONG_TEXT_RATIO_THRESHOLD = 0.3
_LONG_TEXT_LEN_THRESHOLD = 200
_PATIENT_ID_CANDIDATES = [
    "patient_id",
    "record_index",
    "subject_id",
    "hadm_id",
    "patientid",
    "patient",
    "id",
    "pid",
]


def _install_retry_interceptor(max_retries: int = 5, base_delay: float = 10.0) -> None:
    try:
        import openai as _openai
        from openai._base_client import AsyncAPIClient

        if getattr(AsyncAPIClient.request, "_step6_retry_patched", False):
            return
        _orig_request = AsyncAPIClient.request
        _PermissionDeniedError = _openai.PermissionDeniedError
        _APIConnectionError = _openai.APIConnectionError

        async def _patched_request(self, cast_to, options, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await _orig_request(self, cast_to, options, **kwargs)
                except (_PermissionDeniedError, _APIConnectionError) as exc:
                    if attempt >= max_retries - 1:
                        raise
                    delay = base_delay * (2**attempt)
                    err_type = "403" if isinstance(exc, _PermissionDeniedError) else "连接错误"
                    print(f"\n[Retry] {err_type}，{delay:.0f}s 后重试 (第{attempt + 1}/{max_retries}次)...")
                    await asyncio.sleep(delay)

        _patched_request._step6_retry_patched = True  # type: ignore[attr-defined]
        AsyncAPIClient.request = _patched_request
    except Exception as exc:
        print(f"[Retry] 安装重试拦截器失败: {exc}")


_install_retry_interceptor()


def get_project_root() -> str:
    return str(PROJECT_ROOT)


def get_default_input_dir() -> str:
    return str(AGENT_ROOT / "data_input")


def get_default_output_step6_dir() -> str:
    return str(PROJECT_ROOT / "program" / "output" / "step6_results")


def _reset_skip_cols() -> None:
    global _SKIP_COLS
    _SKIP_COLS = set(_SKIP_COLS_STATIC)
    if _IS_MIMIC:
        _SKIP_COLS |= _MIMIC_EXTRA_SKIP_COLS


def _resolve_default_input_csv(input_dir: str) -> str:
    directory = Path(input_dir)
    if not directory.is_dir():
        return ""
    for name in ("filtered.csv", "input.csv", "raw.csv"):
        path = directory / name
        if path.is_file():
            return str(path)
    candidates = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else ""


def configure_runtime(runtime_overrides: dict[str, Any] | None = None) -> dict[str, str]:
    global DATA_SOURCE, _IS_MIMIC, PATIENT_ID_COL, INPUT_CSV, RAW_CSV, OUTPUT_STEP6, _LAST_SAVED_ARTIFACTS, _LAST_ANALYSIS_PASSED_IDS

    overrides = runtime_overrides or {}
    data_source = str(overrides.get("data_source") or os.environ.get("DATA_SOURCE") or DATA_SOURCE or "original").lower()
    DATA_SOURCE = data_source
    _IS_MIMIC = DATA_SOURCE == "mimic"
    PATIENT_ID_COL = str(
        overrides.get("patient_id_col")
        or os.environ.get("PATIENT_ID_COL")
        or ("record_index" if _IS_MIMIC else "patient_id")
    )

    input_dir = str(overrides.get("input_dir") or get_default_input_dir())
    input_csv = str(overrides.get("input_csv") or overrides.get("input") or _resolve_default_input_csv(input_dir) or "")
    raw_csv = str(overrides.get("raw_csv") or input_csv or "")
    output_step6 = str(overrides.get("output_step6_dir") or overrides.get("output") or get_default_output_step6_dir())

    INPUT_CSV = str(Path(input_csv).expanduser().resolve()) if input_csv else ""
    RAW_CSV = str(Path(raw_csv).expanduser().resolve()) if raw_csv else ""
    OUTPUT_STEP6 = str(Path(output_step6).expanduser().resolve())
    Path(OUTPUT_STEP6).mkdir(parents=True, exist_ok=True)

    _LAST_SAVED_ARTIFACTS = {}
    _LAST_ANALYSIS_PASSED_IDS = None
    _reset_skip_cols()
    return {
        "input_csv": INPUT_CSV,
        "raw_csv": RAW_CSV,
        "output_step6": OUTPUT_STEP6,
        "data_source": DATA_SOURCE,
        "patient_id_col": PATIENT_ID_COL,
    }


configure_runtime(None)


def _validate_runtime_inputs() -> None:
    if not INPUT_CSV or not Path(INPUT_CSV).is_file():
        raise FileNotFoundError(f"Step6 输入 CSV 不存在: {INPUT_CSV}")
    if not RAW_CSV or not Path(RAW_CSV).is_file():
        raise FileNotFoundError(f"Step6 raw_csv 不存在: {RAW_CSV}")


def _trunc(val: Any, n: int = 80) -> str:
    s = str(val)
    return s if len(s) <= n else s[:n] + "..."


def _should_skip(col: str) -> bool:
    return col in _SKIP_COLS


def _build_skip_cols(df: pd.DataFrame) -> None:
    for col in df.columns:
        if col in _SKIP_COLS:
            continue
        if col.endswith(("_抽取", "_标准化", "_extracted", "_normalized")):
            _SKIP_COLS.add(col)
            continue
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        long_ratio = (sample.str.len() > _LONG_TEXT_LEN_THRESHOLD).mean()
        if long_ratio >= _LONG_TEXT_RATIO_THRESHOLD:
            _SKIP_COLS.add(col)


def _detect_patient_id_col(df: pd.DataFrame) -> str:
    if PATIENT_ID_COL and PATIENT_ID_COL in df.columns:
        return PATIENT_ID_COL
    for cand in _PATIENT_ID_CANDIDATES:
        if cand in df.columns:
            return cand
    for col in df.columns:
        if "id" in col.lower():
            return col
    return str(df.columns[0])


def _read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty and len(df.columns) == 0:
        raise ValueError(f"Step6 输入 CSV 没有表头: {path}")
    actual_id_col = _detect_patient_id_col(df)
    if actual_id_col != "patient_id":
        df = df.rename(columns={actual_id_col: "patient_id"})
        print(f"[数据加载] 患者ID列: '{actual_id_col}' -> 'patient_id'")
    _build_skip_cols(df)
    return df


def register_tool(toolkit: Toolkit):
    def decorator(func):
        toolkit.register_tool_function(func)
        return func

    return decorator


step6_toolkit = Toolkit()


@register_tool(step6_toolkit)
def load_data_overview() -> ToolResponse:
    """Load the dataset and return a compact overview."""
    raw = _read_data(RAW_CSV)
    n_patients = raw["patient_id"].nunique() if "patient_id" in raw.columns else "N/A"
    lines = [
        "=== 数据集概况 ===",
        f"原始CSV:  {raw.shape[0]} 行 x {raw.shape[1]} 列",
        f"患者数:   {n_patients}",
        f"跳过的长文本列数: {len(_SKIP_COLS)}",
        "",
        "每位患者的记录数:",
    ]
    if "patient_id" in raw.columns:
        rec_counts = raw.groupby("patient_id").size()
        lines.append(f"  记录数分布: min={rec_counts.min()} / max={rec_counts.max()} / mean={rec_counts.mean():.1f}")
        lines.append(f"  （患者列表已省略，共 {len(rec_counts)} 位）")
    else:
        lines.append("  （未检测到患者ID列，请检查数据）")

    lines.append("\n字段预览（第1条记录，长文本列已跳过，最多30个字段）:")
    if not raw.empty:
        row = raw.iloc[0]
        shown = 0
        for col in raw.columns:
            if _should_skip(col):
                continue
            val = row[col]
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                lines.append(f"  {col}: {_trunc(val)}")
                shown += 1
                if shown >= 30:
                    lines.append("  ... （更多字段已省略）")
                    break
    else:
        lines.append("  （CSV 无数据行）")

    lines.append(f"\n所有列名（共 {raw.shape[1]} 列，前50列）:")
    lines.append("  " + ", ".join(raw.columns.tolist()[:50]))
    if raw.shape[1] > 50:
        lines.append(f"  ... 还有 {raw.shape[1] - 50} 列")

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step6_toolkit)
def analyze_all_patients_consistency() -> ToolResponse:
    """Batch analyze patient-level record consistency."""
    global _LAST_ANALYSIS_PASSED_IDS

    raw = _read_data(RAW_CSV)
    meta_cols = frozenset({"patient_id", "filename", "file_path", "folder_category", "note_index", "note_type", "note_seq"})
    data_cols = [col for col in raw.columns if col not in meta_cols and not _should_skip(col)]

    results: list[dict[str, Any]] = []
    for pid, grp in raw.groupby("patient_id"):
        high = medium = low = 0
        for col in data_cols:
            uniq = grp[col].dropna().astype(str).str.strip()
            uniq = uniq[uniq != ""].unique()
            if len(uniq) <= 1:
                continue
            col_lower = col.lower()
            if any(key in col_lower for key in ("诊断", "diagnosis", "性别", "gender", "民族", "race")):
                high += 1
            elif any(key in col_lower for key in ("年龄", "age", "dob", "birth")):
                medium += 1
            else:
                low += 1

        filled = sum(1 for col in data_cols if grp[col].notna().any())
        completeness = filled / len(data_cols) * 100 if data_cols else 100.0
        score = 1.0 - high * 0.20 - medium * 0.10 - low * 0.03
        if completeness < 30:
            score -= 0.10
        score = max(0.0, round(score, 3))
        passed = score >= CONFIDENCE_THRESHOLD
        results.append(
            {
                "patient_id": str(pid),
                "n_records": int(len(grp)),
                "high": high,
                "medium": medium,
                "low": low,
                "completeness": round(completeness, 1),
                "score": score,
                "passed": passed,
            }
        )

    passed_ids = [item["patient_id"] for item in results if item["passed"]]
    failed_results = [item for item in results if not item["passed"]]
    _LAST_ANALYSIS_PASSED_IDS = list(passed_ids)

    lines = [
        "=== Step 6 批量一致性分析结果 ===",
        f"总患者数: {len(results)}  通过: {len(passed_ids)}  未通过: {len(failed_results)}",
        f"置信度阈值: {CONFIDENCE_THRESHOLD}",
        "",
        f"{'患者ID':<15} {'记录数':>5} {'HIGH':>5} {'MED':>5} {'LOW':>5} {'完整性':>7} {'置信度':>7} {'结果':>6}",
        "-" * 65,
    ]
    for item in failed_results:
        lines.append(
            f"{item['patient_id']:<15} {item['n_records']:>5} {item['high']:>5} {item['medium']:>5} "
            f"{item['low']:>5} {item['completeness']:>6.1f}% {item['score']:>7.3f}  未通过"
        )
    if passed_ids:
        pass_scores = [item["score"] for item in results if item["passed"]]
        lines.append(
            f"（通过的 {len(passed_ids)} 位患者已折叠，"
            f"置信度 min={min(pass_scores):.3f} / max={max(pass_scores):.3f}）"
        )
    lines.extend(
        [
            "-" * 65,
            "",
            f"PASSED_PATIENTS_COUNT: {len(passed_ids)}",
            "PASSED_PATIENTS: <stored_in_step6_tool_cache>",
            "",
            "请根据以上结果撰写摘要报告并调用 save_step6_report 保存。",
            "报告末尾保留 PASSED_PATIENTS: <stored_in_step6_tool_cache>，保存工具会使用缓存中的完整通过患者列表。",
        ]
    )
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step6_toolkit)
def save_step6_report(report_content: str) -> ToolResponse:
    """Save Step6 consistency report and passed patient IDs."""
    global _LAST_SAVED_ARTIFACTS

    passed_ids = _parse_passed_ids(report_content)
    if passed_ids is None:
        if "PASSED_PATIENTS:" in str(report_content) and _LAST_ANALYSIS_PASSED_IDS is not None:
            passed_ids = list(_LAST_ANALYSIS_PASSED_IDS)
        else:
            return ToolResponse(content=[TextBlock(type="text", text="Step 6 报告未保存：缺少 PASSED_PATIENTS: [...] 标记。")])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_STEP6)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"consistency_report_{ts}.txt"
    json_path = output_dir / f"consistency_report_{ts}.json"
    passed_json_path = output_dir / f"passed_patients_{ts}.json"

    txt_path.write_text(report_content, encoding="utf-8")

    raw = _read_data(RAW_CSV)
    struct = {
        "timestamp": datetime.now().isoformat(),
        "step": "STEP6_ConsistencyValidation",
        "data_file": RAW_CSV,
        "input_csv": INPUT_CSV,
        "summary": {
            "total_patients": int(raw["patient_id"].nunique()),
            "total_records": int(len(raw)),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "passed_count": len(passed_ids),
        },
        "report_text": report_content,
    }
    json_path.write_text(json.dumps(struct, ensure_ascii=False, indent=2), encoding="utf-8")
    passed_json_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "passed_count": len(passed_ids),
                "passed_patient_ids": passed_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _LAST_SAVED_ARTIFACTS = {
        "step6_report_txt": str(txt_path),
        "step6_report_json": str(json_path),
        "passed_patients_json": str(passed_json_path),
    }
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    "Step 6 报告已保存:\n"
                    f"  文本: {txt_path}\n"
                    f"  JSON: {json_path}\n"
                    f"  通过患者列表: {passed_json_path}  ({len(passed_ids)} 位患者)"
                ),
            )
        ]
    )


def create_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        stream=True,
        client_kwargs={"base_url": OPENAI_BASE_URL},
        generate_kwargs={"temperature": 0.2, "max_tokens": 4096},
    )


def create_consistency_agent() -> ReActAgent:
    if _IS_MIMIC:
        dataset_desc = "MIMIC 住院患者医疗数据集，每条记录对应一次住院或临床笔记记录。"
        analysis_hint = "请检查同一 patient_id 下不同记录之间的关键临床字段是否一致。"
    else:
        dataset_desc = "医疗结构化数据集，每位患者可能有多条记录。"
        analysis_hint = "请重点关注诊断、性别、年龄等关键字段的一致性。"

    agent = ReActAgent(
        name="ConsistencyAgent",
        sys_prompt=f"""你是医疗数据质量工程师，负责对医疗数据集执行 Step 6 一致性验证。

数据集说明：{dataset_desc}
注意事项：{analysis_hint}

## 工作流程（严格按顺序，共 3 步工具调用）

1. 调用 load_data_overview 了解数据集整体情况与字段结构。
2. 调用 analyze_all_patients_consistency 批量完成所有患者的一致性评分。
3. 根据第二步结果撰写简要中文摘要报告，然后调用 save_step6_report 保存报告。

报告末尾必须包含并原样保留第二步工具返回的 PASSED_PATIENTS 标记。
如果第二步返回 PASSED_PATIENTS: <stored_in_step6_tool_cache>，不要展开 ID 列表，保存工具会读取缓存。
""",
        model=create_model(),
        formatter=OpenAIChatFormatter(
            token_counter=CharTokenCounter(),
            max_tokens=_FORMATTER_MAX_CHARS,
        ),
        toolkit=step6_toolkit,
        memory=InMemoryMemory(),
        max_iters=8,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


def _extract_msg_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(str(block.text))
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _strip_thinking_blocks(text: str | None) -> str:
    """Remove model thinking blocks from user-facing terminal/result text."""
    if not text:
        return ""
    cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", str(text))
    cleaned = re.sub(r"(?is)<think>.*\Z", "", cleaned)
    return cleaned.strip()


def _parse_passed_ids(text: str | None) -> list[str] | None:
    """Parse PASSED_PATIENTS marker. Missing marker returns None."""
    if not text:
        return None
    match = re.search(r"PASSED_PATIENTS:\s*(\[[^\]]*\])", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    raw_list = match.group(1)
    try:
        parsed = ast.literal_eval(raw_list)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    inner = raw_list.strip()[1:-1].strip()
    if not inner:
        return []
    ids = re.findall(r"[^\s,\"']+", inner)
    return [item.strip().strip("\"'") for item in ids if item.strip().strip("\"'")]


def _latest_output_file(output_dir: str, prefix: str, suffix: str) -> str:
    directory = Path(output_dir)
    if not directory.is_dir():
        return ""
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.name.endswith(suffix)
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _existing_report_result(paths: dict[str, str]) -> dict[str, Any] | None:
    txt_path = _latest_output_file(OUTPUT_STEP6, "consistency_report_", ".txt")
    json_path = _latest_output_file(OUTPUT_STEP6, "consistency_report_", ".json")
    passed_path = _latest_output_file(OUTPUT_STEP6, "passed_patients_", ".json")
    if not txt_path:
        return None
    report_text = Path(txt_path).read_text(encoding="utf-8")
    passed_ids = _parse_passed_ids(report_text)
    if passed_ids is None and passed_path:
        payload = json.loads(Path(passed_path).read_text(encoding="utf-8"))
        passed_ids = [str(item) for item in payload.get("passed_patient_ids") or []]
    if passed_ids is None:
        return None
    return _success_result(
        paths=paths,
        step6_text=report_text,
        passed_ids=passed_ids,
        artifacts={
            "step6_report_txt": txt_path,
            "step6_report_json": json_path,
            "passed_patients_json": passed_path,
        },
    )


def _ensure_report_saved(step6_text: str) -> dict[str, str]:
    if _LAST_SAVED_ARTIFACTS:
        return dict(_LAST_SAVED_ARTIFACTS)
    save_step6_report(step6_text)
    return dict(_LAST_SAVED_ARTIFACTS)


def _passed_ids_from_artifacts(artifacts: dict[str, str]) -> list[str] | None:
    passed_path = artifacts.get("passed_patients_json", "")
    if not passed_path or not Path(passed_path).is_file():
        return None
    payload = json.loads(Path(passed_path).read_text(encoding="utf-8"))
    passed_ids = payload.get("passed_patient_ids")
    if not isinstance(passed_ids, list):
        return None
    return [str(item) for item in passed_ids]


def _success_result(paths: dict[str, str], step6_text: str, passed_ids: list[str], artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "success": True,
        "error": "",
        "input_csv": paths.get("input_csv", ""),
        "raw_csv": paths.get("raw_csv", ""),
        "output_step6": paths.get("output_step6", ""),
        "step6_text": step6_text,
        "step6_report_txt": artifacts.get("step6_report_txt", ""),
        "step6_report_json": artifacts.get("step6_report_json", ""),
        "passed_patients_json": artifacts.get("passed_patients_json", ""),
        "passed_patient_ids": passed_ids,
        "runtime_paths": paths,
    }


def _failure_result(paths: dict[str, str], error: str, step6_text: str = "") -> dict[str, Any]:
    return {
        "success": False,
        "error": error,
        "input_csv": paths.get("input_csv", ""),
        "raw_csv": paths.get("raw_csv", ""),
        "output_step6": paths.get("output_step6", ""),
        "step6_text": step6_text,
        "step6_report_txt": _latest_output_file(paths.get("output_step6", ""), "consistency_report_", ".txt"),
        "step6_report_json": _latest_output_file(paths.get("output_step6", ""), "consistency_report_", ".json"),
        "passed_patients_json": _latest_output_file(paths.get("output_step6", ""), "passed_patients_", ".json"),
        "passed_patient_ids": [],
        "runtime_paths": paths,
    }


async def run_step6_only(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = runtime_overrides or {}
    paths = configure_runtime(overrides)
    memory_context = str(overrides.get("memory_context") or "").strip()
    try:
        _validate_runtime_inputs()
    except Exception as exc:
        return _failure_result(paths, str(exc))

    if bool(overrides.get("reuse_existing", False)):
        existing = _existing_report_result(paths)
        if existing is not None:
            return existing

    data_label = "MIMIC" if _IS_MIMIC else "Original"
    print("=" * 65)
    print("  STEP 6: 患者数据一致性验证")
    print("=" * 65)
    print(f"  模式:    {data_label}（DATA_SOURCE={DATA_SOURCE}）")
    print(f"  模型:    {MODEL_NAME}")
    print(f"  输入:    {INPUT_CSV}")
    print(f"  数据:    {RAW_CSV}")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"  输出目录: {OUTPUT_STEP6}")
    print("=" * 65)

    prompt_parts = [
        "请对以下医疗数据集执行完整的 Step 6 一致性验证。\n",
        f"数据文件路径：{RAW_CSV}\n",
        "步骤：① load_data_overview → ② analyze_all_patients_consistency → "
        "③ 撰写摘要后 save_step6_report（末尾保留第二步返回的 PASSED_PATIENTS 标记）。\n",
        f"置信度阈值：{CONFIDENCE_THRESHOLD}",
        "注意：PASSED_PATIENTS 标记必须存在；若工具返回缓存标记，不要展开完整 ID 列表。",
    ]
    if memory_context:
        prompt_parts.extend(
            [
                "",
                "以下是可参考的 Step6 经验记忆。它只能作为报告组织和失败处理参考，不得覆盖工具计算结果：",
                memory_context,
            ]
        )

    msg = Msg(
        name="Pipeline",
        role="user",
        content="\n\n".join(prompt_parts),
    )

    try:
        print("\n[Step 6] 启动 ConsistencyAgent...")
        result = await create_consistency_agent()(msg)
        step6_text = _extract_msg_text(result)
        passed_ids = _parse_passed_ids(step6_text)
        if passed_ids is None:
            if "PASSED_PATIENTS:" in step6_text and _LAST_ANALYSIS_PASSED_IDS is not None:
                passed_ids = list(_LAST_ANALYSIS_PASSED_IDS)
        artifacts = _ensure_report_saved(step6_text)
        if passed_ids is None:
            passed_ids = _passed_ids_from_artifacts(artifacts)
        if passed_ids is None:
            return _failure_result(
                paths,
                "Step 6 output missing required PASSED_PATIENTS: [...] marker.",
                _strip_thinking_blocks(step6_text),
            )
        required = ("step6_report_txt", "step6_report_json", "passed_patients_json")
        missing = [name for name in required if not artifacts.get(name) or not Path(artifacts[name]).is_file()]
        if missing:
            return _failure_result(paths, f"Step 6 report artifacts missing: {missing}", _strip_thinking_blocks(step6_text))
        visible_step6_text = _strip_thinking_blocks(step6_text)
        print("\n" + "=" * 65)
        print("  Step 6 完成")
        print("=" * 65)
        print(visible_step6_text)
        print(f"\n[Step 6] 通过患者数量: {len(passed_ids)}")
        return _success_result(paths, visible_step6_text, passed_ids, artifacts)
    except Exception as exc:
        return _failure_result(paths, str(exc))


async def run_step6() -> None:
    result = await run_step6_only()
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Step6 failed")
    print("\n" + "=" * 65)
    print("  Step 6 全部完成")
    print(f"  报告目录: {result.get('output_step6')}/")
    print("=" * 65)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Agent6 Step6 consistency validation.")
    parser.add_argument("--input", "-i", default="", help="输入 CSV 路径。默认读取 step-6/data_input 下最新 CSV。")
    parser.add_argument("--raw-csv", default="", help="可选原始 CSV 路径；默认与 --input 相同。")
    parser.add_argument("--output", "-o", default="", help="输出目录。默认 program/output/step6_results。")
    parser.add_argument("--patient-id-col", default="", help="患者 ID 列名。默认自动检测。")
    parser.add_argument("--data-source", default="", choices=["", "original", "mimic"], help="数据源类型。")
    parser.add_argument("--reuse-existing", action="store_true", help="复用输出目录中最新 Step6 报告。")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    overrides = {
        "input_csv": args.input,
        "raw_csv": args.raw_csv,
        "output_step6_dir": args.output,
        "patient_id_col": args.patient_id_col,
        "data_source": args.data_source,
        "reuse_existing": args.reuse_existing,
    }
    result = asyncio.run(run_step6_only({key: value for key, value in overrides.items() if value not in ("", None)}))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
