# -*- coding: utf-8 -*-
"""
STEP 7: ML 训练数据生成
=======================
运行方式:
  python run_step7_ml_pipeline.py

前置条件:
  已运行 run_step6_ml_pipeline.py，output_step6/ 下存在
  passed_patients_<ts>.json 文件。

输出:
  output_step7/ml_dataset_<task>_<ts>.csv/.jsonl/.json
  output_step7/model_config_<task>_<ts>.json

ICD-10 查询链路
────────────────
  1. 精确 / 规范化匹配（ICD-10.xlsx）
  2. LLM 将英文缩写等转为标准中文后重试精确匹配
  3. BERT 向量检索：预建向量库 data/icd10_bert_index/
  4. 字符串模糊匹配（rapidfuzz）兜底
  5. LLM 在 BERT+模糊 Top 候选中择优

环境变量（可选）:
  ICD10_USE_BERT=1          启用 BERT 检索（默认 1）
  ICD10_BERT_MODEL=...      默认 BAAI/bge-small-zh-v1.5
  ICD10_BERT_MIN_SIM=0.70   余弦相似度 ≥ 此值则直接采纳 BERT Top1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent  # lib/ → v2/
AGENT_ROOT = PROJECT_ROOT.parent               # v2/ (ICD-10.xlsx, data/, models/ live here)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config_loader import get_agent_config
except Exception:  # pragma: no cover - standalone fallback
    get_agent_config = None

# 加载项目根目录的 .env 文件
_ENV_PATH = PROJECT_ROOT / ".env"
if _ENV_PATH.is_file():
    with open(_ENV_PATH, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import pandas as pd

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope_compat import CharTokenCounter
from agentscope.tool import Toolkit, ToolResponse

_FORMATTER_MAX_CHARS = 375_000


def _install_retry_interceptor(max_retries: int = 5, base_delay: float = 10.0) -> None:
    try:
        import openai as _openai
        from openai._base_client import AsyncAPIClient
        _orig_request = AsyncAPIClient.request
        _PermissionDeniedError = _openai.PermissionDeniedError
        _APIConnectionError = _openai.APIConnectionError

        async def _patched_request(self, cast_to, options, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await _orig_request(self, cast_to, options, **kwargs)
                except (_PermissionDeniedError, _APIConnectionError) as exc:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        err_type = "403" if isinstance(exc, _PermissionDeniedError) else "连接错误"
                        print(f"\n[Retry] {err_type}，{delay:.0f}s 后重试 (第{attempt+1}/{max_retries}次)...")
                        await asyncio.sleep(delay)
                    else:
                        raise

        AsyncAPIClient.request = _patched_request
    except Exception as _e:
        print(f"[Retry] 安装重试拦截器失败: {_e}")

_install_retry_interceptor()

# ══════════════════════════════════════════════════════════════════
# 0. 全局配置
# ══════════════════════════════════════════════════════════════════

AGENT_CFG = get_agent_config("agent_6_7") if get_agent_config else {}
OPENAI_API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEYS")
    or str(AGENT_CFG.get("api_key") or "")
)
OPENAI_BASE_URL = (
    os.environ.get("YUNWU_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or str(AGENT_CFG.get("base_url") or AGENT_CFG.get("api_base") or "https://api.openai.com/v1")
)
MODEL_NAME = os.environ.get("MODEL_NAME") or str(AGENT_CFG.get("model") or AGENT_CFG.get("model_name") or "gpt-4.1-mini")

_SCRIPT_DIR = str(AGENT_ROOT)

DATA_SOURCE = os.environ.get("DATA_SOURCE", "original").lower()
_IS_MIMIC = DATA_SOURCE == "mimic"
DATA_DIR = str(AGENT_ROOT / "data_input")
PATIENT_ID_COL = os.environ.get("PATIENT_ID_COL", "record_index" if _IS_MIMIC else "")
OUTPUT_STEP6 = ""
OUTPUT_STEP7 = ""
FILTERED_CSV = ""
SELECTION_REPORT = ""
PASSED_PATIENTS_JSON = ""
TASK_TEXT_OVERRIDE = ""

ICD10_XLSX = str(AGENT_ROOT / "ICD-10.xlsx")
ICD10_VECTOR_DIR = str(AGENT_ROOT / "data" / "icd10_bert_index")
ICD10_USE_BERT   = os.environ.get("ICD10_USE_BERT", "1").lower() not in ("0", "false", "no")
ICD10_BERT_MODEL = os.environ.get("ICD10_BERT_MODEL", str(AGENT_ROOT / "models" / "bge-small-zh-v1.5"))
ICD10_BERT_MIN_SIM = float(os.environ.get("ICD10_BERT_MIN_SIM", "0.70"))


def get_default_input_dir() -> str:
    return str(AGENT_ROOT / "data_input")


def get_default_input_csv() -> str:
    return str(PROJECT_ROOT / "program" / "output" / "step5_results" / "next_input" / "filtered.csv")


def get_default_selection_report() -> str:
    return str(PROJECT_ROOT / "program" / "output" / "step4_results" / "next_input" / "selection_report.json")


def get_default_output_step6_dir() -> str:
    return str(PROJECT_ROOT / "program" / "output" / "step6_results")


def get_default_output_step7_dir() -> str:
    return str(PROJECT_ROOT / "program" / "output" / "step7_results")


def _latest_file(directory: str, prefix: str, suffix: str) -> str:
    root = Path(directory)
    if not root.is_dir():
        return ""
    candidates = [
        path for path in root.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.name.endswith(suffix)
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def get_default_passed_patients_json() -> str:
    fixed = PROJECT_ROOT / "program" / "output" / "step6_results" / "passed_patients_20260519_183132.json"
    if fixed.is_file():
        return str(fixed)
    return _latest_file(get_default_output_step6_dir(), "passed_patients_", ".json")


def configure_runtime(runtime_overrides: dict[str, Any] | None = None) -> dict[str, str]:
    global DATA_SOURCE, _IS_MIMIC, DATA_DIR, PATIENT_ID_COL
    global FILTERED_CSV, SELECTION_REPORT, PASSED_PATIENTS_JSON, OUTPUT_STEP6, OUTPUT_STEP7, TASK_TEXT_OVERRIDE

    overrides = runtime_overrides or {}
    DATA_SOURCE = str(overrides.get("data_source") or os.environ.get("DATA_SOURCE") or DATA_SOURCE or "original").lower()
    _IS_MIMIC = DATA_SOURCE == "mimic"
    DATA_DIR = str(overrides.get("input_dir") or get_default_input_dir())
    PATIENT_ID_COL = str(
        overrides.get("patient_id_col")
        or os.environ.get("PATIENT_ID_COL")
        or ("record_index" if _IS_MIMIC else "")
    )

    input_csv = str(overrides.get("input_csv") or overrides.get("input") or get_default_input_csv())
    selection_report = str(overrides.get("selection_report") or get_default_selection_report())
    output_step6 = str(overrides.get("output_step6_dir") or overrides.get("step6_output") or get_default_output_step6_dir())
    passed_patients_json = str(
        overrides.get("passed_patients_json")
        or overrides.get("passed_patients")
        or get_default_passed_patients_json()
    )
    output_step7 = str(overrides.get("output_step7_dir") or overrides.get("output") or get_default_output_step7_dir())
    TASK_TEXT_OVERRIDE = str(
        overrides.get("task_text")
        or overrides.get("ml_task_goal")
        or overrides.get("task_goal")
        or ""
    ).strip()

    FILTERED_CSV = str(Path(input_csv).expanduser().resolve()) if input_csv else ""
    SELECTION_REPORT = str(Path(selection_report).expanduser().resolve()) if selection_report else ""
    OUTPUT_STEP6 = str(Path(output_step6).expanduser().resolve()) if output_step6 else ""
    PASSED_PATIENTS_JSON = str(Path(passed_patients_json).expanduser().resolve()) if passed_patients_json else ""
    OUTPUT_STEP7 = str(Path(output_step7).expanduser().resolve())
    Path(OUTPUT_STEP7).mkdir(parents=True, exist_ok=True)
    _reset_skip_cols()
    return {
        "input_csv": FILTERED_CSV,
        "selection_report": SELECTION_REPORT,
        "passed_patients_json": PASSED_PATIENTS_JSON,
        "output_step6": OUTPUT_STEP6,
        "output_step7": OUTPUT_STEP7,
        "data_source": DATA_SOURCE,
        "patient_id_col": PATIENT_ID_COL,
        "task_text": TASK_TEXT_OVERRIDE,
    }


def _validate_runtime_inputs() -> None:
    if not FILTERED_CSV or not Path(FILTERED_CSV).is_file():
        raise FileNotFoundError(f"Step7 输入 filtered.csv 不存在: {FILTERED_CSV}")
    if not SELECTION_REPORT or not Path(SELECTION_REPORT).is_file():
        raise FileNotFoundError(f"Step7 输入 selection_report.json 不存在: {SELECTION_REPORT}")
    if not PASSED_PATIENTS_JSON or not Path(PASSED_PATIENTS_JSON).is_file():
        raise FileNotFoundError(f"Step7 输入 passed_patients JSON 不存在: {PASSED_PATIENTS_JSON}")


def _find_latest(prefix: str, suffix: str, search_dir: str | None = None) -> str:
    d = search_dir or DATA_DIR
    candidates = [f for f in os.listdir(d)
                  if f.startswith(prefix) and f.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"找不到 {prefix}*{suffix}（目录：{d}）")
    return os.path.join(d, sorted(candidates)[-1])


def _find_latest_by_suffix(keyword: str, suffix: str, search_dir: str | None = None) -> str:
    d = search_dir or DATA_DIR
    candidates = [f for f in os.listdir(d)
                  if keyword in f and f.endswith(suffix)]
    if not candidates:
        fallback_name = keyword.lstrip("_") + suffix
        fallback_path = os.path.join(d, fallback_name)
        if os.path.isfile(fallback_path):
            return fallback_path
        raise FileNotFoundError(f"找不到 *{keyword}*{suffix}（目录：{d}）")
    return os.path.join(d, sorted(candidates)[-1])


# Step 6 → Step 7 传递通过患者列表（由 _load_passed_patient_ids 填充）
_passed_patient_ids: list[str] = []

# 长文本列黑名单
_SKIP_COLS_STATIC = {
    "ocr_text", "preprocessed_text", "file_path", "error",
    "relations", "temporal_info", "quantity_info",
    "impression", "indication",
}
_MIMIC_EXTRA_SKIP_COLS = {
    "所有用药", "实验室检验_摘要", "所有手术操作_titles", "所有手术操作_codes",
    "所有诊断_icd_codes",
    "抽取_Disease", "抽取_Drug", "抽取_Symptom", "抽取_Test",
    "抽取_Treatment", "抽取_LabValue", "抽取_Finding",
}
_SKIP_COLS: set[str] = set(_SKIP_COLS_STATIC)
if _IS_MIMIC:
    _SKIP_COLS |= _MIMIC_EXTRA_SKIP_COLS

_LONG_TEXT_RATIO_THRESHOLD = 0.3
_LONG_TEXT_LEN_THRESHOLD   = 200


def _reset_skip_cols() -> None:
    global _SKIP_COLS
    _SKIP_COLS = set(_SKIP_COLS_STATIC)
    if _IS_MIMIC:
        _SKIP_COLS |= _MIMIC_EXTRA_SKIP_COLS


configure_runtime(None)


def _build_skip_cols(df: "pd.DataFrame") -> None:
    global _SKIP_COLS
    for col in df.columns:
        if col in _SKIP_COLS:
            continue
        if col.endswith(("_抽取", "_标准化", "_extracted", "_normalized")):
            _SKIP_COLS.add(col)
            continue
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        long_ratio = (sample.str.len() > _LONG_TEXT_LEN_THRESHOLD).mean()
        if long_ratio >= _LONG_TEXT_RATIO_THRESHOLD:
            _SKIP_COLS.add(col)


def _trunc(val: Any, n: int = 80) -> str:
    s = str(val)
    return s if len(s) <= n else s[:n] + "…"


def _should_skip(col: str) -> bool:
    return col in _SKIP_COLS


_PATIENT_ID_CANDIDATES = [
    "patient_id", "record_index", "subject_id", "hadm_id",
    "patientid", "patient", "id", "pid",
]


def _detect_patient_id_col(df: "pd.DataFrame") -> str:
    if PATIENT_ID_COL and PATIENT_ID_COL in df.columns:
        return PATIENT_ID_COL
    for cand in _PATIENT_ID_CANDIDATES:
        if cand in df.columns:
            return cand
    for col in df.columns:
        if "id" in col.lower():
            return col
    return df.columns[0]


def _read_data(path: str) -> "pd.DataFrame":
    df = pd.read_csv(path)
    actual_id_col = _detect_patient_id_col(df)
    if actual_id_col != "patient_id":
        df = df.rename(columns={actual_id_col: "patient_id"})
        print(f"[数据加载] 患者ID列: '{actual_id_col}' → 'patient_id'")
    _build_skip_cols(df)
    return df


# ══════════════════════════════════════════════════════════════════
# 从 Step 6 输出读取通过患者列表
# ══════════════════════════════════════════════════════════════════

def _load_passed_patient_ids() -> list[str]:
    """
    从显式 passed_patients_<ts>.json 读取通过患者 ID 列表。
    独立版 Step7 不再回退到报告文本或全量患者，避免静默放行全部患者。
    """
    if not PASSED_PATIENTS_JSON:
        raise FileNotFoundError("Step7 缺少 passed_patients_json 路径。")
    path = Path(PASSED_PATIENTS_JSON)
    if not path.is_file():
        raise FileNotFoundError(f"Step7 输入 passed_patients JSON 不存在: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        raise ValueError(f"Step7 读取 passed_patients JSON 失败: {path}: {exc}") from exc
    ids = data.get("passed_patient_ids")
    if not isinstance(ids, list):
        raise ValueError(f"Step7 passed_patients JSON 缺少列表字段 passed_patient_ids: {path}")
    cleaned = [str(item).strip() for item in ids if str(item).strip()]
    if not cleaned:
        raise ValueError(f"Step7 passed_patients JSON 中没有可用患者 ID: {path}")
    print(f"[Step 7] 从 {path} 读取 {len(cleaned)} 个通过患者")
    return cleaned


# ══════════════════════════════════════════════════════════════════
# LLM 辅助缓存
# ══════════════════════════════════════════════════════════════════

_LLM_DIAG_FIELDS_CACHE: list[str] | None = None
_LLM_DISEASE_EXTRACT_CACHE: dict[tuple[str, str], str | None] = {}
_ICD10_CACHE: dict[str, tuple[str, str]] = {}

# ══════════════════════════════════════════════════════════════════
# 本地 ICD-10 查询库（精确 + BERT 向量 + 模糊 + LLM）
# ══════════════════════════════════════════════════════════════════

def _load_icd10_local() -> tuple["pd.DataFrame", dict[str, tuple[str, str]]]:
    try:
        df = pd.read_excel(ICD10_XLSX, dtype=str)
        df = df.dropna(subset=["诊断名称", "诊断编码"])
        df["诊断名称"] = df["诊断名称"].str.strip()
        df["诊断编码"] = df["诊断编码"].str.strip()
        name_to_entry: dict[str, tuple[str, str]] = {
            row["诊断名称"]: (row["诊断编码"], row["诊断名称"])
            for _, row in df.iterrows()
        }
        print(f"[ICD-10本地库] 已加载 {len(name_to_entry)} 条诊断记录（{ICD10_XLSX}）")
        return df, name_to_entry
    except Exception as exc:
        print(f"[ICD-10本地库] 加载失败: {exc}，ICD 查询将全部返回 NOT_FOUND")
        return pd.DataFrame(), {}


_ICD10_DF, _ICD10_NAME_TO_ENTRY = _load_icd10_local()
_ICD10_NAMES_LIST: list[str] = list(_ICD10_NAME_TO_ENTRY.keys())

_ICD10_EMB_MATRIX: np.ndarray | None = None
_ICD10_EMB_CODES: np.ndarray | None = None
_ICD10_EMB_NAMES: np.ndarray | None = None
_ICD10_EMB_META: dict[str, Any] | None = None
_ICD10_ST_MODEL: Any = None
_ICD10_VECTOR_WARNED: bool = False


def _try_load_icd10_vector_store() -> bool:
    global _ICD10_EMB_MATRIX, _ICD10_EMB_CODES, _ICD10_EMB_NAMES, _ICD10_EMB_META, _ICD10_VECTOR_WARNED

    if _ICD10_EMB_MATRIX is not None:
        return True

    emb_path   = os.path.join(ICD10_VECTOR_DIR, "embeddings.npy")
    codes_path = os.path.join(ICD10_VECTOR_DIR, "icd_codes.npy")
    names_path = os.path.join(ICD10_VECTOR_DIR, "icd_names.npy")
    meta_path  = os.path.join(ICD10_VECTOR_DIR, "meta.json")

    if not os.path.isfile(emb_path):
        if not _ICD10_VECTOR_WARNED:
            print(
                f"[ICD-BERT] 未找到向量库 {emb_path}，跳过 BERT 检索。"
                f"请运行: python {os.path.join(os.path.dirname(__file__), 'build_icd10_vector_store.py')}"
            )
            _ICD10_VECTOR_WARNED = True
        return False

    try:
        _ICD10_EMB_MATRIX = np.load(emb_path)
        _ICD10_EMB_CODES  = np.load(codes_path, allow_pickle=True)
        _ICD10_EMB_NAMES  = np.load(names_path, allow_pickle=True)
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                _ICD10_EMB_META = json.load(f)
            built_model = (_ICD10_EMB_META or {}).get("model", "")
            if built_model and built_model != ICD10_BERT_MODEL:
                print(
                    f"[ICD-BERT] 警告：向量库模型为「{built_model}」，"
                    f"当前 ICD10_BERT_MODEL=「{ICD10_BERT_MODEL}」。请重建向量库或对齐环境变量。"
                )
        else:
            _ICD10_EMB_META = {}
        n = int(_ICD10_EMB_MATRIX.shape[0])
        if _ICD10_EMB_CODES.shape[0] != n or _ICD10_EMB_NAMES.shape[0] != n:
            raise ValueError("embeddings 与 codes/names 行数不一致")
        print(f"[ICD-BERT] 已加载向量库 {n} 条 × dim={_ICD10_EMB_MATRIX.shape[1]}（{ICD10_VECTOR_DIR}）")
        return True
    except Exception as exc:
        _ICD10_EMB_MATRIX = None
        print(f"[ICD-BERT] 向量库加载失败: {exc}")
        return False


def _get_icd10_sentence_model() -> Any:
    global _ICD10_ST_MODEL
    if _ICD10_ST_MODEL is not None:
        return _ICD10_ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("使用 ICD BERT 检索需要安装: pip install sentence-transformers torch") from exc
    _ICD10_ST_MODEL = SentenceTransformer(ICD10_BERT_MODEL)
    return _ICD10_ST_MODEL


def _icd10_bert_search(query: str, top_k: int = 8) -> list[tuple[str, str, float]]:
    if not ICD10_USE_BERT or not _try_load_icd10_vector_store():
        return []
    if _ICD10_EMB_MATRIX is None:
        return []
    q = str(query).strip()
    if not q:
        return []
    try:
        model = _get_icd10_sentence_model()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        qv = np.asarray(qv, dtype=np.float32)
        sims = _ICD10_EMB_MATRIX @ qv
        k = min(top_k, int(sims.shape[0]))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out: list[tuple[str, str, float]] = []
        for i in idx:
            code = str(_ICD10_EMB_CODES[int(i)])
            name = str(_ICD10_EMB_NAMES[int(i)])
            out.append((code, name, float(sims[int(i)])))
        return out
    except Exception as exc:
        print(f"    [ICD-BERT-异常] {exc}")
        return []


def _merge_icd_candidates_for_llm(
    bert_cands: list[tuple[str, str, float]],
    fuzzy_cands: list[tuple[str, str, float]],
    max_total: int = 8,
) -> list[tuple[str, str, float]]:
    seen: set[str] = set()
    merged: list[tuple[str, str, float]] = []
    for code, name, sim in bert_cands:
        if code in seen:
            continue
        seen.add(code)
        merged.append((code, name, sim * 100.0))
    for code, name, ratio in fuzzy_cands:
        if code in seen:
            continue
        seen.add(code)
        merged.append((code, name, ratio))
        if len(merged) >= max_total:
            break
    return merged[:max_total]


def _normalize_diag(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'\([A-Za-z0-9 \-]+\)$', '', text).strip()
    return text


_ICD10_NORM_TO_ENTRY: dict[str, tuple[str, str]] = {
    _normalize_diag(name): entry
    for name, entry in _ICD10_NAME_TO_ENTRY.items()
}


def _local_lookup_icd10_fuzzy(diagnosis: str, top_k: int = 5) -> list[tuple[str, str, float]]:
    try:
        from rapidfuzz import process, fuzz as rfuzz
        wide_candidates = process.extract(
            diagnosis, _ICD10_NAMES_LIST, scorer=rfuzz.WRatio, limit=30,
        )
        reranked = sorted(
            [
                (_ICD10_NAME_TO_ENTRY[name][0], name, rfuzz.ratio(diagnosis, name))
                for name, _, _ in wide_candidates
                if name in _ICD10_NAME_TO_ENTRY
            ],
            key=lambda x: -x[2],
        )
        return reranked[:top_k]
    except ImportError:
        print("    [模糊匹配] rapidfuzz 未安装，跳过模糊匹配步骤")
        return []
    except Exception as exc:
        print(f"    [模糊匹配-异常] {exc}")
        return []


def _llm_select_icd10_candidate(
    diagnosis: str,
    candidates: list[tuple[str, str, float]],
) -> tuple[str, str] | None:
    if not candidates:
        return None
    options_text = "\n".join(
        f"  {i+1}. {name}（编码: {code}）"
        for i, (code, name, _) in enumerate(candidates)
    )
    prompt = (
        f"你是医疗编码专家。需要将以下诊断名称映射到 ICD-10 编码。\n\n"
        f"待映射诊断：{diagnosis}\n\n"
        f"候选 ICD-10 条目（来自中国国家标准库）：\n{options_text}\n\n"
        f"规则：\n"
        f"  1. 若某个候选与待映射诊断是同一疾病（允许叫法/缩写不同），返回对应序号（1-{len(candidates)}）。\n"
        f"  2. 若所有候选均与待映射诊断不是同一疾病，返回 0。\n"
        f"  3. 只返回数字，不加任何解释。\n\n"
        f"选择（数字）："
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        idx = int(re.search(r'\d+', raw).group())
        if 1 <= idx <= len(candidates):
            code, name, _ = candidates[idx - 1]
            print(f"    [LLM候选选择] '{diagnosis}' → 选择第{idx}项: {name}（{code}）")
            return code, name
        print(f"    [LLM候选选择] '{diagnosis}' → LLM 返回 {idx}，无合适候选")
        return None
    except Exception as exc:
        print(f"    [LLM候选选择-失败] {exc}")
        return None


_LLM_CN_NORMALIZE_CACHE: dict[str, str] = {}


def _llm_cn_normalize(term: str) -> str | None:
    if not term or not term.strip():
        return None
    cn_chars = sum(1 for c in term if '\u4e00' <= c <= '\u9fff')
    if cn_chars / max(len(term.strip()), 1) > 0.6 and len(term.strip()) >= 4:
        return None
    if term in _LLM_CN_NORMALIZE_CACHE:
        return _LLM_CN_NORMALIZE_CACHE[term]
    prompt = (
        "你是医学术语专家。请将以下诊断术语转换为中国 ICD-10 标准库中对应的标准中文名称。\n"
        "规则：\n"
        "  1. 若是英文缩写（如 bppv、UPVD、PPPD），输出其对应的标准中文全称。\n"
        "  2. 若是英文全称，直接翻译为中文标准术语。\n"
        "  3. 若无法确定对应的标准中文术语，输出 NULL。\n"
        "  4. 只输出中文术语本身，不加引号、不加标点、不做解释。\n\n"
        f"诊断术语：{term}\n"
        "标准中文名称（或 NULL）："
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip().strip('"\'').strip()
        result = None if raw.upper() == "NULL" or not raw else raw
        _LLM_CN_NORMALIZE_CACHE[term] = result
        if result:
            print(f"    [LLM标准化] '{term}' → '{result}'")
        return result
    except Exception as exc:
        print(f"    [LLM标准化-失败] '{term}': {exc}")
        return None


def _local_icd10_lookup(diagnosis: str) -> tuple[str, str, str]:
    """
    本地 ICD-10 查询主入口（精确 → 规范化 → LLM中文标准化 → BERT → 模糊 → LLM候选选择）。
    Returns: (source, icd10_code, preferred_name)
    """
    diag_clean = str(diagnosis).strip()
    if not diag_clean or diag_clean.lower() in ("nan", "none", "null"):
        return "NOT_FOUND", "NOT_FOUND", ""

    if diag_clean in _ICD10_CACHE:
        cached = _ICD10_CACHE[diag_clean]
        print(f"    [ICD-缓存] '{diag_clean}' → {cached[0]}")
        return "CACHE", cached[0], cached[1]

    try:
        def _store(src, code, name):
            _ICD10_CACHE[diag_clean] = (code, name)
            return src, code, name

        if diag_clean in _ICD10_NAME_TO_ENTRY:
            code, name = _ICD10_NAME_TO_ENTRY[diag_clean]
            print(f"    [ICD-精确] '{diag_clean}' → {code}")
            return _store("EXACT", code, name)

        norm = _normalize_diag(diag_clean)
        if norm in _ICD10_NORM_TO_ENTRY:
            code, name = _ICD10_NORM_TO_ENTRY[norm]
            print(f"    [ICD-规范化] '{diag_clean}' → {code}（{name}）")
            return _store("NORM", code, name)

        cn_term = _llm_cn_normalize(diag_clean)
        if cn_term:
            if cn_term in _ICD10_NAME_TO_ENTRY:
                code, name = _ICD10_NAME_TO_ENTRY[cn_term]
                print(f"    [ICD-CN标准化] '{diag_clean}' → '{cn_term}' → {code}")
                return _store("CN_NORM", code, name)
            norm2 = _normalize_diag(cn_term)
            if norm2 in _ICD10_NORM_TO_ENTRY:
                code, name = _ICD10_NORM_TO_ENTRY[norm2]
                print(f"    [ICD-CN标准化+规范化] '{diag_clean}' → '{cn_term}' → {code}")
                return _store("CN_NORM", code, name)

        query_for_match = (cn_term if cn_term else diag_clean).strip()

        bert_cands = _icd10_bert_search(query_for_match, top_k=8)
        if cn_term and cn_term.strip() != diag_clean.strip():
            bert_alt = _icd10_bert_search(diag_clean.strip(), top_k=8)
            if bert_alt and (not bert_cands or bert_alt[0][2] > bert_cands[0][2]):
                bert_cands = bert_alt

        if bert_cands:
            b_code, b_name, b_sim = bert_cands[0]
            if b_sim >= ICD10_BERT_MIN_SIM:
                print(f"    [ICD-BERT] '{diag_clean}' → {b_code}（{b_name}） cos_sim={b_sim:.3f}")
                return _store("BERT", b_code, b_name)

        candidates = _local_lookup_icd10_fuzzy(query_for_match, top_k=5)
        if not candidates and cn_term and cn_term != diag_clean:
            candidates = _local_lookup_icd10_fuzzy(diag_clean, top_k=5)

        if candidates:
            best_code, best_name, best_score = candidates[0]
            if best_score >= 85:
                print(f"    [ICD-模糊] '{diag_clean}' → {best_code}（{best_name}）ratio={best_score:.1f}")
                return _store("FUZZY", best_code, best_name)

        merged = _merge_icd_candidates_for_llm(bert_cands or [], candidates or [], max_total=8)
        if merged:
            _bs = bert_cands[0][2] if bert_cands else 0.0
            print(f"    [ICD-LLM选择] BERT最高={_bs:.3f}，合并候选 {len(merged)} 条，调用 LLM…")
            result = _llm_select_icd10_candidate(diag_clean, merged)
            if result:
                code, name = result
                return _store("LLM_SELECT", code, name)

        print(f"    [ICD-未找到] '{diag_clean}' 在本地库中无匹配")
        return _store("NOT_FOUND", "NOT_FOUND", "")

    except Exception as exc:
        print(f"    [ICD-严重错误] '{diag_clean}': {exc}")
        return "ERROR", f"ERROR: {exc}", ""


_umls_lookup_icd10 = _local_icd10_lookup


# ══════════════════════════════════════════════════════════════════
# LLM 辅助函数①：识别含诊断信息的字段
# ══════════════════════════════════════════════════════════════════

def _llm_identify_diagnosis_fields(columns_with_samples: dict[str, list]) -> list[str]:
    col_info_lines = []
    for col, samples in columns_with_samples.items():
        clean = [str(s)[:50] for s in samples if pd.notna(s) and str(s).strip() not in ("", "nan")]
        if clean:
            col_info_lines.append(f"  - {col}：示例值=[{', '.join(clean[:3])}]")

    if not col_info_lines:
        return []

    prompt = (
        "你是医疗数据专家。以下是一份医疗数据集的字段名称和示例值。\n"
        "请识别出所有可能包含疾病诊断信息的字段，包括但不限于：\n"
        "  - 主诊断、初步诊断、最终诊断、出院诊断\n"
        "  - 可能诊断（如可能诊断1、可能诊断2）\n"
        "  - 综合征类型、疾病类型\n"
        "  - 病因诊断、定位诊断\n"
        "请只返回字段名列表，每行一个完整字段名，不要解释，不要序号，不要其他内容。\n\n"
        "字段清单：\n"
        + "\n".join(col_info_lines)
        + "\n\n含诊断信息的字段（每行一个字段名）："
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )
        raw_output = response.choices[0].message.content.strip()
        diag_fields = []
        for line in raw_output.splitlines():
            field = line.strip().lstrip("-•").strip()
            if field and field in columns_with_samples:
                diag_fields.append(field)
        print(f"    [LLM识别诊断字段] 识别出 {len(diag_fields)} 个: {diag_fields}")
        return diag_fields
    except Exception as exc:
        print(f"    [LLM识别诊断字段-失败] {exc}，返回空列表")
        return []


# ══════════════════════════════════════════════════════════════════
# LLM 辅助函数②：从字段值中提取纯疾病内容
# ══════════════════════════════════════════════════════════════════

def _llm_extract_disease_content(field_name: str, field_value: str) -> str | None:
    val_clean = str(field_value).strip()
    if not val_clean or val_clean.lower() in ("nan", "none", "null", "无", "未填写", "—", "-"):
        return None

    cache_key = (field_name, val_clean)
    if cache_key in _LLM_DISEASE_EXTRACT_CACHE:
        result = _LLM_DISEASE_EXTRACT_CACHE[cache_key]
        print(f"    [LLM疾病提取-缓存] {field_name}='{val_clean}' → {result}")
        return result

    prompt = (
        f"字段名称：{field_name}\n"
        f"字段内容：{val_clean}\n\n"
        "请从上面的字段内容中提取疾病/诊断名称。\n"
        "规则：\n"
        "  1. 若内容是一个疾病/综合征名称，直接返回该名称。\n"
        "  2. 若内容包含多个疾病名称（逗号/空格/分号分隔，或混有ICD编码），"
        "     逐一提取每个疾病名称，用 || 分隔输出（如：高血压||糖尿病||心力衰竭）。\n"
        "  3. 若内容混有非疾病信息（状态描述、数值、编码等），只保留疾病名称部分。\n"
        "  4. 若内容完全不含疾病信息（纯数字、日期、'无'、'正常'、检查结论），返回 NULL。\n"
        "  5. 只输出疾病名称，不加引号、不加标点、不做解释。\n\n"
        "输出（单个疾病名称 或 疾病1||疾病2||... 或 NULL）："
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip().strip('"\'').strip()
        result = None if raw.upper() == "NULL" or not raw else raw
        _LLM_DISEASE_EXTRACT_CACHE[cache_key] = result
        print(f"    [LLM疾病提取] {field_name}='{val_clean[:40]}' → {result}")
        return result
    except Exception as exc:
        print(f"    [LLM疾病提取-失败] {exc}，返回原值降级")
        fallback = val_clean if val_clean else None
        _LLM_DISEASE_EXTRACT_CACHE[cache_key] = fallback
        return fallback


# ══════════════════════════════════════════════════════════════════
# 1. @register_tool 装饰器工厂
# ══════════════════════════════════════════════════════════════════

def register_tool(toolkit: Toolkit):
    def decorator(func):
        toolkit.register_tool_function(func)
        return func
    return decorator


# ══════════════════════════════════════════════════════════════════
# 2. Step 7 工具集
# ══════════════════════════════════════════════════════════════════

step7_toolkit = Toolkit()


def _load_selection_report() -> dict:
    try:
        with open(SELECTION_REPORT, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _report_final_columns(report: dict) -> list[str]:
    for key in ("final_columns", "selected_columns", "columns"):
        val = report.get(key)
        if isinstance(val, list) and val:
            return val
    return []


def _report_task_text(report: dict) -> str:
    if TASK_TEXT_OVERRIDE:
        return TASK_TEXT_OVERRIDE
    for key in ("task_text", "task_en", "task"):
        val = report.get(key)
        if val and str(val).strip() and str(val).strip() not in ("N/A", "????/????"):
            return str(val).strip()
    return "N/A"


def _report_reason_by_column(report: dict) -> dict:
    if isinstance(report.get("reason_by_column"), dict):
        return report["reason_by_column"]
    ad = report.get("agent_decisions", {})
    if isinstance(ad, dict) and isinstance(ad.get("reason_by_column"), dict):
        return ad["reason_by_column"]
    return {}


_NULL_THRESHOLD = 0.70

_FEATURE_EXCLUDE_PATTERNS = (
    "admittime", "dischtime", "deathtime", "dod", "edregtime", "edouttime",
    "hadm_id", "stay_id", "transfer_id", "icustay_id",
    "diagnoses_icd_codes", "prescriptions_drugs",
)


def _get_auto_feature_cols() -> list[str]:
    report = _load_selection_report()
    final_cols = _report_final_columns(report)
    filt = _read_data(FILTERED_CSV)
    if not final_cols:
        final_cols = filt.columns.tolist()

    null_rates: dict[str, float] = {}
    for c in final_cols:
        if c in filt.columns:
            null_rates[c] = filt[c].isnull().mean()

    def _should_exclude(col: str) -> bool:
        if col in ("patient_id", PATIENT_ID_COL):
            return True
        if _should_skip(col):
            return True
        if any(col == p or col.endswith("_" + p) for p in _FEATURE_EXCLUDE_PATTERNS):
            return True
        if col in _FEATURE_EXCLUDE_PATTERNS:
            return True
        if null_rates.get(col, 0) > _NULL_THRESHOLD:
            return True
        return False

    good_cols = [c for c in final_cols if not _should_exclude(c)]
    skipped_high_null = sum(1 for c in final_cols
                            if null_rates.get(c, 0) > _NULL_THRESHOLD and c in filt.columns)
    print(f"[_get_auto_feature_cols] 有效特征列: {len(good_cols)} / {len(final_cols)} "
          f"（过滤高缺失率: {skipped_high_null}）")
    return good_cols


@register_tool(step7_toolkit)
def load_task_context() -> ToolResponse:
    """
    读取列筛选报告，返回：
      1. 原始任务描述（task_text）
      2. 所有已筛选特征列及其入选原因
      3. 可能用于标签的候选列（诊断/结局相关列）及每位患者的取值分布
      4. 关键数值指标的每位患者聚合均值
    兼容多种 selection_report 字段命名约定。
    """
    report     = _load_selection_report()
    task_text  = _report_task_text(report)
    final_cols = _report_final_columns(report)

    filt      = _read_data(FILTERED_CSV)
    feat_cols = _get_auto_feature_cols()

    if not final_cols:
        final_cols = filt.columns.tolist()

    _DIAG_LABEL_KEYWORDS = (
        "诊断", "diagnosis", "icd", "disease", "condition",
        "outcome", "mortality", "death", "dod", "label",
        "综合征", "syndrome",
    )
    diag_cols = [
        c for c in final_cols
        if any(k in c.lower() for k in _DIAG_LABEL_KEYWORDS) and not _should_skip(c)
    ]

    good_feat_cols = _get_auto_feature_cols()

    icd_col_candidates = [c for c in filt.columns
                          if "icd" in c.lower() and "code" in c.lower()
                          and filt[c].notna().mean() > 0.5]

    lines = [
        "=" * 65,
        "  Step 7 任务上下文",
        "=" * 65,
        f"  任务描述: {task_text}",
        f"  筛选后数据: {filt.shape[0]} 行 × {filt.shape[1]} 列 / {filt['patient_id'].nunique()} 位患者",
        f"  有效特征列（缺失率<70%）: {len(good_feat_cols)} 列",
        f"  诊断/结局候选列: {len(diag_cols)} 列",
        "",
    ]

    if icd_col_candidates:
        lines += ["── ⚡ 发现 ICD 编码列（推荐用于创建二分类标签）──────────────"]
        for col in icd_col_candidates[:3]:
            sample_vals = [str(v)[:60] for v in filt[col].dropna().unique()[:2]]
            lines.append(f"  {col}  示例: {sample_vals}")
        lines += [
            "  -> 若任务是[是否患有某病]，调用 create_icd_binary_label(",
            "       icd_codes_column='...',",
            "       positive_icd_prefixes=['ICD前缀1', 'ICD前缀2', ...]",
            "    ) 创建二分类标签，无需 propose_label_mapping。",
            "",
        ]

    _SKIP_AS_LABEL = set(_FEATURE_EXCLUDE_PATTERNS) | {"patient_id", PATIENT_ID_COL}
    all_label_candidates = []
    for col in filt.columns:
        if col in _SKIP_AS_LABEL or _should_skip(col):
            continue
        miss = filt[col].isnull().mean()
        if miss > 0.8:
            continue
        nuniq = filt[col].nunique()
        uniq_vals = [str(v) for v in filt[col].dropna().unique()[:4] if str(v).strip() not in ("", "nan")]
        all_label_candidates.append((col, miss, nuniq, uniq_vals))

    lines += ["── 全量标签候选列（缺失率<80%，供探索任务用）────────────"]
    for col, miss, nuniq, uniq_vals in all_label_candidates[:30]:
        lines.append(f"  {col:<55}  缺失={miss:.0%}  唯一值={nuniq}  示例={uniq_vals}")

    lines += [
        "",
        "=" * 65,
        "  ⚡ 标签策略建议:",
        "  A) 非空值二分类（如死亡预测）：某列有值=正例，空=负例",
        "     → create_notnull_binary_label(source_column=..., positive_condition='not_null')",
        "  B) ICD 编码二分类（如疾病诊断）：",
        "     → create_icd_binary_label(icd_codes_column=..., positive_icd_prefixes=[...])",
        "  C) 现有分类列多分类（如出院去向）：",
        "     → propose_label_mapping(label_column=..., mode='multiclass_enum')",
        "  ⚠ 禁止选择时间戳、ID、或缺失率>70% 的列作为标签",
        "=" * 65,
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def get_column_distribution(column_name: str) -> ToolResponse:
    """
    获取筛选后数据集中指定列的完整分布信息。

    Args:
        column_name (str): 列名（必须精确匹配）。
    """
    filt = _read_data(FILTERED_CSV)
    if column_name not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"列 '{column_name}' 不在筛选数据中。请参考 load_task_context 中的列清单。")])

    col   = filt[column_name]
    lines = [
        f"=== 列分布: {column_name} ===",
        f"  缺失: {col.isnull().sum()}/{len(col)} ({col.isnull().mean()*100:.1f}%)",
        f"  唯一值: {col.nunique()}",
        "  值分布（行级别）:",
    ]
    for val, cnt in col.value_counts(dropna=False).head(15).items():
        lines.append(f"    {str(val):<55} → {cnt} 行")

    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().sum() >= 3:
        lines.append(f"  数值统计: min={numeric.min():.3f}  "
                     f"max={numeric.max():.3f}  mean={numeric.mean():.3f}  "
                     f"std={numeric.std():.3f}")

    filt2 = filt[["patient_id", column_name]].copy()
    filt2[column_name] = pd.to_numeric(filt2[column_name], errors="coerce")
    per_patient_num = filt2.groupby("patient_id")[column_name].mean().dropna()
    if len(per_patient_num) > 0:
        lines.append(f"\n  按患者聚合均值（数值）:")
        for pid, val in per_patient_num.items():
            lines.append(f"    患者 {pid}: {val:.3f}")
    else:
        per_patient_cat = filt.groupby("patient_id")[column_name].apply(
            lambda x: list(x.dropna().unique())
        ).to_dict()
        lines.append(f"\n  按患者取值（分类）:")
        for pid, vals in per_patient_cat.items():
            lines.append(f"    患者 {pid}: {vals}")

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def discover_diagnosis_fields() -> ToolResponse:
    """
    使用 LLM 自动扫描原始数据集与筛选数据集的全部字段，
    识别哪些列可能包含疾病/诊断信息。
    """
    global _LLM_DIAG_FIELDS_CACHE

    filt = _read_data(FILTERED_CSV)
    all_cols = list(filt.columns)

    _DIAG_COL_KEYWORDS = (
        "诊断", "diagnosis", "disease", "syndrome", "综合征",
        "icd", "condition", "impression", "indication",
    )
    col_samples: dict[str, list] = {}
    for col in all_cols:
        if _should_skip(col) or col in ("patient_id", "filename", "file_path", "folder_category"):
            continue
        col_lower = col.lower()
        if not any(k in col_lower for k in _DIAG_COL_KEYWORDS):
            continue
        non_null = filt[col].dropna()
        if len(non_null) == 0:
            continue
        samples = [str(v).strip() for v in non_null.unique()[:5].tolist()
                   if str(v).strip() not in ("", "nan")]
        if samples:
            col_samples[col] = samples

    print(f"    [discover_diagnosis_fields] 启发式候选列: {len(col_samples)} 个")

    if 0 < len(col_samples) <= 50:
        diag_fields = _llm_identify_diagnosis_fields(col_samples)
    elif len(col_samples) > 50:
        sorted_cols = sorted(col_samples.keys(),
                             key=lambda c: filt[c].notna().mean() if c in filt.columns else 0,
                             reverse=True)
        diag_fields = sorted_cols[:30]
        print(f"    [discover_diagnosis_fields] 候选过多({len(col_samples)})，取非空率最高的 {len(diag_fields)} 个")
    else:
        diag_fields = list(col_samples.keys())
    _LLM_DIAG_FIELDS_CACHE = diag_fields

    lines = [
        "=" * 65,
        "  LLM 自动识别诊断相关字段",
        "=" * 65,
        f"  扫描字段总数:        {len(col_samples)}",
        f"  识别出的诊断字段数:  {len(diag_fields)}",
        "",
        "── 识别结果（字段名 + 所有唯一值）────────────────────────",
    ]

    for field in diag_fields:
        src = filt if field in filt.columns else None
        if src is None:
            lines.append(f"\n  [{field}]  ← 字段不存在，跳过")
            continue
        unique_vals = [str(v) for v in src[field].dropna().unique().tolist()
                       if str(v).strip() not in ("", "nan")]
        sample_str = ", ".join(f'"{v[:40]}"' for v in unique_vals[:3])
        lines.append(f"  {field}  ({len(unique_vals)} 个唯一值)  示例: [{sample_str}]")

    lines += [
        "",
        "  ⬇ 下一步：调用 extract_and_map_all_diagnosis_fields(",
        f"      diagnosis_fields={diag_fields}",
        "  ) 完成疾病内容提取与 ICD-10 标准化。",
        "=" * 65,
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def extract_and_map_all_diagnosis_fields(
    diagnosis_fields: list[str],
) -> ToolResponse:
    """
    对指定的所有诊断相关字段进行两阶段处理：
    阶段 1 — LLM 疾病内容过滤
    阶段 2 — ICD-10 本地库标准化

    Args:
        diagnosis_fields (list[str]): 含诊断信息的字段名列表。
    """
    filt = _read_data(FILTERED_CSV)

    icd10_mapping: dict[str, str] = {}
    field_details: dict[str, list[dict]] = {}

    lines = [
        "=" * 65,
        "  多字段诊断提取 + ICD-10 标准化",
        "=" * 65,
        f"  处理字段数: {len(diagnosis_fields)}",
        "",
    ]

    for field in diagnosis_fields:
        src = filt if field in filt.columns else None
        if src is None:
            lines.append(f"[跳过] 字段 '{field}' 在数据集中不存在")
            continue

        _MAX_UNIQUE_PER_FIELD = 200
        all_vals_raw = [str(v) for v in src[field].dropna().tolist()
                        if str(v).strip() not in ("", "nan")]
        from collections import Counter as _Counter
        _freq = _Counter(all_vals_raw)
        unique_vals = [v for v, _ in _freq.most_common(_MAX_UNIQUE_PER_FIELD)]
        truncated = len(_freq) > _MAX_UNIQUE_PER_FIELD
        lines.append(
            f"\n── 字段: {field}  ({len(_freq)} 个唯一值"
            + (f"，取频次最高的 {_MAX_UNIQUE_PER_FIELD} 个）──" if truncated else "）──")
        )

        field_records: list[dict] = []

        for val in unique_vals:
            disease_raw = _llm_extract_disease_content(field, val)

            if disease_raw is None:
                lines.append(f"  [非疾病] {val[:55]:<57} → 跳过 ICD 查询")
                field_records.append({
                    "original_value":    val,
                    "extracted_disease": None,
                    "source":            None,
                    "icd10":             None,
                    "umls_preferred":    None,
                })
                continue

            disease_list = [d.strip() for d in disease_raw.split("||") if d.strip()]

            for disease in disease_list:
                source, icd10_code, pref_name = _umls_lookup_icd10(disease)
                icd10_mapping[disease] = icd10_code
                field_records.append({
                    "original_value":    val,
                    "extracted_disease": disease,
                    "source":            source,
                    "icd10":             icd10_code,
                    "local_preferred":   pref_name,
                })

        field_details[field] = field_records
        found = sum(1 for r in field_records if r.get("icd10") and r["icd10"] != "NOT_FOUND")
        lines.append(f"  {field}: {len(field_records)} 条记录，{found} 个命中 ICD-10")

    lines += [
        "",
        "=" * 65,
        f"  汇总：共提取 {len(icd10_mapping)} 个疾病名称并完成 ICD-10 标准化",
        "=" * 65,
        "",
        "  icd10_mapping（可直接传入 format_and_save_ml_dataset）:",
        json.dumps(icd10_mapping, ensure_ascii=False, indent=2),
    ]

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ICD 标签缓存（create_icd_binary_label / create_notnull_binary_label → format_and_save_ml_dataset）
_icd_label_cache: dict = {}


@register_tool(step7_toolkit)
def create_icd_binary_label(
    icd_codes_column: str,
    positive_icd_prefixes: list[str],
    label_name_positive: str = "positive",
    label_name_negative: str = "negative",
) -> ToolResponse:
    """
    从含有 ICD 编码（逗号分隔）的列中创建二分类标签列，并注入数据集。

    Args:
        icd_codes_column (str): 含 ICD 编码列表的列名。
        positive_icd_prefixes (list[str]): 阳性编码前缀列表。
        label_name_positive (str): 阳性标签名（默认 'positive'）。
        label_name_negative (str): 阴性标签名（默认 'negative'）。
    """
    filt = _read_data(FILTERED_CSV)
    pids = _passed_patient_ids if _passed_patient_ids else [str(p) for p in filt["patient_id"].unique()]
    filt = filt[filt["patient_id"].astype(str).isin(pids)].reset_index(drop=True)

    if icd_codes_column not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"列 '{icd_codes_column}' 不存在。")])

    prefixes = [p.strip().upper().replace(".", "") for p in positive_icd_prefixes]

    def _is_positive(icd_str: Any) -> str:
        if pd.isna(icd_str) or str(icd_str).strip() in ("", "nan"):
            return label_name_negative
        codes = [c.strip().upper().replace(".", "") for c in str(icd_str).split(",")]
        for code in codes:
            if any(code.startswith(pfx) for pfx in prefixes):
                return label_name_positive
        return label_name_negative

    filt["_icd_label"] = filt[icd_codes_column].apply(_is_positive)
    dist = filt["_icd_label"].value_counts().to_dict()
    label_mapping = {label_name_negative: 0, label_name_positive: 1}

    lines = [
        "=== create_icd_binary_label 结果 ===",
        f"  ICD 编码列:    {icd_codes_column}",
        f"  阳性前缀:      {prefixes}",
        f"  标签分布:",
        f"    {label_name_positive} (y=1): {dist.get(label_name_positive, 0)} 行",
        f"    {label_name_negative} (y=0): {dist.get(label_name_negative, 0)} 行",
        f"  阳性率: {dist.get(label_name_positive, 0)/len(filt)*100:.1f}%",
        "",
        "  推荐的 label_mapping（直接传入 format_and_save_ml_dataset）:",
        f"  label_column = '_icd_label'",
        f"  label_mapping = {label_mapping}",
        "",
        "  ⚠ 注意：此工具已将 '_icd_label' 列注入内存数据集。",
        "    请用 label_column='_icd_label' 调用 format_and_save_ml_dataset。",
    ]

    _icd_label_cache["column"] = "_icd_label"
    _icd_label_cache["data"] = filt[["patient_id", "_icd_label"]].copy()

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def create_notnull_binary_label(
    source_column: str,
    positive_condition: str = "not_null",
    label_name_positive: str = "positive",
    label_name_negative: str = "negative",
) -> ToolResponse:
    """
    从现有列的取值条件创建二分类标签列，并注入数据集。

    Args:
        source_column (str): 作为判断依据的列名。
        positive_condition (str): 'not_null' / 'is_null' / 'gt_0' / 'eq_0'
        label_name_positive (str): 阳性标签名。
        label_name_negative (str): 阴性标签名。
    """
    filt = _read_data(FILTERED_CSV)
    pids = _passed_patient_ids if _passed_patient_ids else [str(p) for p in filt["patient_id"].unique()]
    filt = filt[filt["patient_id"].astype(str).isin(pids)].reset_index(drop=True)

    if source_column not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"列 '{source_column}' 不存在。")])

    col = filt[source_column]

    if positive_condition == "not_null":
        mask_positive = col.notna() & (col.astype(str).str.strip() != "")
    elif positive_condition == "is_null":
        mask_positive = col.isna() | (col.astype(str).str.strip() == "")
    elif positive_condition == "gt_0":
        numeric = pd.to_numeric(col, errors="coerce")
        mask_positive = numeric.notna() & (numeric > 0)
    elif positive_condition == "eq_0":
        numeric = pd.to_numeric(col, errors="coerce")
        mask_positive = numeric.notna() & (numeric == 0)
    else:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"不支持的 positive_condition='{positive_condition}'。"
                 "可选: not_null / is_null / gt_0 / eq_0")])

    filt["_notnull_label"] = mask_positive.map({True: label_name_positive, False: label_name_negative})
    dist = filt["_notnull_label"].value_counts().to_dict()
    pos_rate = dist.get(label_name_positive, 0) / len(filt) * 100
    label_mapping = {label_name_negative: 0, label_name_positive: 1}

    lines = [
        "=== create_notnull_binary_label 结果 ===",
        f"  来源列:    {source_column}",
        f"  条件:      {positive_condition}",
        f"  标签分布:",
        f"    {label_name_positive} (y=1): {dist.get(label_name_positive, 0)} 行",
        f"    {label_name_negative} (y=0): {dist.get(label_name_negative, 0)} 行",
        f"  阳性率: {pos_rate:.1f}%",
        "",
        "  推荐的参数（直接传入 format_and_save_ml_dataset）:",
        f"  label_column  = '_notnull_label'",
        f"  label_mapping = {label_mapping}",
        "",
        "  ⚠ 注意：此工具已将 '_notnull_label' 列注入内存数据集。",
        "    请用 label_column='_notnull_label' 调用 format_and_save_ml_dataset。",
    ]

    _icd_label_cache["column"] = "_notnull_label"
    _icd_label_cache["data"] = filt[["patient_id", "_notnull_label"]].copy()

    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def map_diagnoses_to_icd10(diagnosis_names: list[str]) -> ToolResponse:
    """
    将诊断名称列表映射为 ICD-10 编码（本地库查询）。

    Args:
        diagnosis_names (list[str]): 需要标准化的诊断名称列表。
    """
    lines = [
        "=== ICD-10 标准化映射（本地库：精确→模糊→LLM候选选择）===",
        f"查询 {len(diagnosis_names)} 个诊断名称...\n",
        f"{'诊断名称':<30} {'来源':<12} {'ICD-10 编码':<15} {'本地库首选名称'}",
        "-" * 90,
    ]

    mapping: dict[str, str] = {}

    for name in diagnosis_names:
        disease_raw = _llm_extract_disease_content("诊断字段", name)
        if disease_raw is None:
            lines.append(f"{name:<30} {'—':<8} {'[非疾病跳过]':<15} —")
            continue

        disease_list = [d.strip() for d in disease_raw.split("||") if d.strip()]
        for disease in disease_list:
            source, code, preferred = _umls_lookup_icd10(disease)
            mapping[disease] = code
            lines.append(f"{disease:<30} {source:<8} {code:<15} {preferred}")

    lines += [
        "-" * 90,
        "",
        "icd10_mapping（可直接传入 format_and_save_ml_dataset）:",
        json.dumps(mapping, ensure_ascii=False, indent=2),
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def propose_label_mapping(
    label_column: str,
    mode: str = "multiclass_enum",
    positive_values: list[str] | None = None,
) -> ToolResponse:
    """
    根据筛选后数据集中标签列的实际非空取值，自动生成 label_mapping。

    Args:
        label_column (str): 作为 ML 标签的列名。
        mode (str): 'multiclass_enum' 或 'binary_positive_vs_rest'
        positive_values (list[str] | None): 仅 binary_positive_vs_rest 时必填。
    """
    filt = _read_data(FILTERED_CSV)
    pids_to_use = _passed_patient_ids if _passed_patient_ids else \
                  [str(p) for p in filt["patient_id"].unique()]
    filt = filt[filt["patient_id"].astype(str).isin(pids_to_use)]

    if label_column not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"❌ 列 '{label_column}' 不在筛选数据中。")])

    def _norm(v: Any) -> str | None:
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return None
        return s

    raw_vals = [_norm(v) for v in filt[label_column].tolist()]
    uniques = sorted({v for v in raw_vals if v is not None})

    if len(uniques) < 2:
        return ToolResponse(content=[TextBlock(type="text",
            text=(
                f"❌ 列 '{label_column}' 在通过 Step6 的子集中有效类别数 < 2。\n"
                f"  非空唯一值: {uniques}\n"
                f"  请换用其他标签列或检查筛选数据。"
            ))])

    mapping: dict[str, int] = {}
    mode_l = (mode or "multiclass_enum").strip().lower()

    if mode_l == "binary_positive_vs_rest":
        pos = set(positive_values or [])
        if not pos:
            return ToolResponse(content=[TextBlock(type="text",
                text=(
                    "❌ mode=binary_positive_vs_rest 时必须提供 positive_values（非空列表），"
                    f"当前数据中的可取值为: {uniques}"
                ))])
        for u in uniques:
            mapping[u] = 1 if u in pos else 0
        if len(set(mapping.values())) < 2:
            return ToolResponse(content=[TextBlock(type="text",
                text=(
                    "❌ 按 positive_values 划分后只有 1 类，无法二分类。"
                    f" positive_values={sorted(pos)} 与数据交集请检查。"
                ))])
    else:
        mapping = {u: i for i, u in enumerate(uniques)}

    lines = [
        "=== propose_label_mapping 结果 ===",
        f"  标签列: {label_column}",
        f"  模式:   {mode_l}",
        f"  非空类别数: {len(uniques)}",
        "  各类别 → 整数:",
    ]
    for k, v in sorted(mapping.items(), key=lambda x: (x[1], x[0])):
        lines.append(f"    {repr(k)} → {v}")
    lines += [
        "",
        "  请将下方 JSON 原样用于 build_label_series 与 format_and_save_ml_dataset：",
        "",
        "label_mapping =",
        json.dumps(mapping, ensure_ascii=False, indent=2),
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def build_label_series(label_column: str, label_mapping: dict[str, int]) -> ToolResponse:
    """
    验证标签生成效果：按每条筛选记录（每行）用标签列与映射生成 y，不保存。

    Args:
        label_column (str): 标签列名。
        label_mapping (dict[str, int]): 类别值 → 整数标签。
    """
    filt = _read_data(FILTERED_CSV)
    pids_to_use = _passed_patient_ids if _passed_patient_ids else \
                  [str(p) for p in filt["patient_id"].unique()]
    filt = filt[filt["patient_id"].astype(str).isin(pids_to_use)].reset_index(drop=True)

    if label_column not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"列 '{label_column}' 不在筛选数据中。")])

    rows_out = []
    for idx, row in filt.iterrows():
        lv = row[label_column]
        if pd.isna(lv):
            label_raw, y = None, None
        else:
            label_raw = str(lv).strip()
            y         = label_mapping.get(label_raw)
        rows_out.append({
            "sample_row_id": int(idx),
            "patient_id":    str(row["patient_id"]),
            "label_raw":     label_raw,
            "y":             y,
        })

    n_valid   = sum(1 for r in rows_out if r["y"] is not None)
    n_invalid = len(rows_out) - n_valid
    dist: dict[str, int] = {}
    for r in rows_out:
        k = str(r["y"])
        dist[k] = dist.get(k, 0) + 1

    lines = [
        f"=== 标签验证（列: {label_column}，按记录逐行）===\n",
        f"{'行号':<8} {'患者ID':<12} {'原始标签':<35} {'y':>4}",
        "-" * 65,
    ]
    for r in rows_out:
        y_str = str(r["y"]) if r["y"] is not None else "—(无法映射)"
        lines.append(
            f"{r['sample_row_id']:<8} {r['patient_id']:<12} {str(r['label_raw']):<35} {y_str:>4}"
        )
    lines += [
        "-" * 65,
        f"总记录数: {len(rows_out)}  有效标签: {n_valid}  无法映射: {n_invalid}",
        f"标签分布: {dist}",
    ]

    unique_y = {r["y"] for r in rows_out if r["y"] is not None}
    if len(unique_y) < 2:
        lines.append(
            "\n⚠ [质量警告] 有效标签中只有 1 种类别值，该列不适合作为分类标签！"
            "\n  请重新思考任务定义，选择能区分不同患者状态的列或重新设计映射。"
        )
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


@register_tool(step7_toolkit)
def format_and_save_ml_dataset(
    label_column: str,
    label_mapping: dict[str, int],
    icd10_mapping: dict[str, str] | None = None,
    task_name: str = "",
) -> ToolResponse:
    """
    使用全量特征 + Agent 确定的标签生成完整 ML 数据集并保存。

    Args:
        label_column (str): 标签列名。
        label_mapping (dict[str, int]): 类别值 → 整数标签映射。
        icd10_mapping (dict[str, str] | None): 诊断名称 → ICD-10 编码映射。
        task_name (str): 任务名称，用于区分多任务输出文件。
    """
    filt = _read_data(FILTERED_CSV)
    pids_to_use = _passed_patient_ids if _passed_patient_ids else \
                  [str(p) for p in filt["patient_id"].unique()]
    filt = filt[filt["patient_id"].astype(str).isin(pids_to_use)].reset_index(drop=True)

    # 如果 label_column 是由 create_* 注入的虚拟列，merge 进来
    if label_column not in filt.columns and _icd_label_cache.get("column") == label_column:
        label_df = _icd_label_cache["data"]
        label_df = label_df[label_df["patient_id"].astype(str).isin(pids_to_use)]
        filt = filt.merge(
            label_df[["patient_id", label_column]].drop_duplicates("patient_id"),
            on="patient_id", how="left"
        )

    feat_cols = _get_auto_feature_cols()
    feat_cols = [c for c in feat_cols if c in filt.columns and c != label_column]

    if label_column not in filt.columns:
        return ToolResponse(content=[TextBlock(type="text",
            text=f"❌ 列 '{label_column}' 不在筛选数据中，请重新确认列名。"
                 f"\n提示：若已调用 create_icd_binary_label，请使用 label_column='_icd_label'")])

    def _cell_value(v: Any) -> Any:
        if pd.isna(v):
            return None
        num = pd.to_numeric(v, errors="coerce")
        if pd.notna(num):
            return round(float(num), 6)
        s = str(v).strip()
        return s if s and s.lower() != "nan" else None

    ml_rows: list[dict[str, Any]] = []
    for idx, row in filt.iterrows():
        rec: dict[str, Any] = {
            "sample_row_id": int(idx),
            "patient_id":    str(row["patient_id"]),
        }
        for col in feat_cols:
            rec[col] = _cell_value(row[col])
        lv = row[label_column]
        if pd.isna(lv):
            rec["label_name"], rec["y"] = None, None
        else:
            rec["label_name"] = str(lv).strip()
            rec["y"] = label_mapping.get(rec["label_name"])
        ml_rows.append(rec)

    df_ml = pd.DataFrame(ml_rows)
    df_train = df_ml.dropna(subset=["y"]).copy()

    unique_y = df_train["y"].nunique() if not df_train.empty else 0
    if unique_y < 2:
        preview = ""
        if not df_ml.empty and label_column in filt.columns:
            preview = df_ml[["sample_row_id", "patient_id", "label_name"]].head(15).to_string(index=False)
        err_lines = [
            "❌ [错误] 数据集质量不合格，已中止保存。",
            f"  label_column='{label_column}' 有效样本中只有 {unique_y} 种类别值。",
            f"  当前 label_mapping={label_mapping}",
            f"  筛选后总记录数: {len(df_ml)}，有标签记录数: {len(df_train)}",
            "",
            "  前若干条记录的 label_name 预览:",
            preview or "（无）",
            "",
            "  请重新思考标签策略后再调用此工具。",
        ]
        return ToolResponse(content=[TextBlock(type="text", text="\n".join(err_lines))])

    df_train["y"] = df_train["y"].astype(int)
    for col in feat_cols:
        if pd.api.types.is_numeric_dtype(df_train[col]):
            col_mean = df_train[col].mean()
            df_train[col] = df_train[col].fillna(col_mean)

    # 诊断字段增强
    _DIAG_KEYWORDS = ("诊断", "综合征", "疾病类型", "病因")
    diag_feat_cols = [
        c for c in feat_cols
        if (c in (_LLM_DIAG_FIELDS_CACHE or []))
        or any(k in c for k in _DIAG_KEYWORDS)
    ]

    _effective_icd_map: dict[str, str] = dict(icd10_mapping or {})

    def _get_disease_and_icd(col: str, raw_val: Any) -> tuple[str | None, str | None]:
        if pd.isna(raw_val) or str(raw_val).strip() in ("", "nan", "None"):
            return None, None
        val_str = str(raw_val).strip()
        disease = _LLM_DISEASE_EXTRACT_CACHE.get((col, val_str))
        if disease is None:
            disease = _llm_extract_disease_content(col, val_str)
        if not disease:
            return None, None
        code = _effective_icd_map.get(disease)
        if code is None:
            _, code, pref = _local_icd10_lookup(disease)
            if code and code != "NOT_FOUND":
                _effective_icd_map[disease] = code
        return disease, (code if code and code != "NOT_FOUND" else None)

    icd_summary_lines: list[str] = []
    for col in diag_feat_cols:
        disease_vals: list[str | None] = []
        icd_vals: list[str | None] = []
        for _, row in df_train.iterrows():
            d, c = _get_disease_and_icd(col, row.get(col))
            disease_vals.append(d)
            icd_vals.append(c)
        col_short = col.split("-")[-1] if "-" in col else col
        df_train[f"{col_short}_疾病名称"] = disease_vals
        df_train[f"{col_short}_ICD10"]   = icd_vals
        icd_summary_lines.append(f"  {col}")
        for d, c in zip(disease_vals, icd_vals):
            if d:
                icd_summary_lines.append(f"    → {d} | {c or 'NOT_FOUND'}")

    def _label_to_icd(label_name: Any) -> str | None:
        if pd.isna(label_name) or str(label_name).strip() in ("", "nan"):
            return None
        lbl = str(label_name).strip()
        code = _effective_icd_map.get(lbl)
        if code is None:
            _, code, _ = _local_icd10_lookup(lbl)
        return code if code and code != "NOT_FOUND" else None

    df_train["label_ICD10"] = df_train["label_name"].apply(_label_to_icd)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _slug = f"_{task_name}" if task_name else ""

    csv_path = os.path.join(OUTPUT_STEP7, f"ml_dataset{_slug}_{ts}.csv")
    df_train.to_csv(csv_path, index=False, encoding="utf-8-sig")

    jsonl_path = os.path.join(OUTPUT_STEP7, f"ml_dataset{_slug}_{ts}.jsonl")

    def _json_safe(v: Any) -> Any:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, (np.integer, np.floating)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df_train.iterrows():
            sample = {
                "sample_row_id": int(row["sample_row_id"]),
                "patient_id":    str(row["patient_id"]),
                "features":      {c: _json_safe(row[c]) for c in feat_cols},
                "label":         int(row["y"]),
                "label_name":    row.get("label_name", ""),
                "label_ICD10":   _json_safe(row.get("label_ICD10")),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    json_path = os.path.join(OUTPUT_STEP7, f"ml_dataset{_slug}_{ts}.json")
    sel_report = _load_selection_report()

    # 模型推荐
    label_dist = df_train["y"].value_counts().to_dict()
    n_samples  = len(df_train)
    n_features = len(feat_cols)
    n_classes  = len(label_mapping)
    counts     = list(label_dist.values())
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    is_binary  = n_classes == 2
    is_small   = n_samples < 200
    is_high_dim = n_features > n_samples

    def _rf_params() -> dict:
        return {"n_estimators": 100 if is_small else 300, "max_depth": None,
                "min_samples_leaf": 2 if is_small else 1,
                "class_weight": "balanced" if imbalance_ratio > 2 else None,
                "random_state": 42, "n_jobs": -1}

    def _lr_params() -> dict:
        return {"C": 1.0, "max_iter": 1000,
                "class_weight": "balanced" if imbalance_ratio > 2 else None,
                "solver": "lbfgs", "multi_class": "auto", "random_state": 42}

    def _svm_params() -> dict:
        return {"C": 1.0, "kernel": "rbf", "gamma": "scale",
                "class_weight": "balanced" if imbalance_ratio > 2 else None,
                "probability": True, "random_state": 42}

    def _xgb_params() -> dict:
        return {"n_estimators": 100 if is_small else 300, "max_depth": 4,
                "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                "eval_metric": "logloss" if is_binary else "mlogloss",
                "scale_pos_weight": (counts[0] / counts[1]) if (is_binary and imbalance_ratio > 2) else 1,
                "random_state": 42}

    if is_small or is_high_dim:
        recommended_models = [
            {"model": "SVC", "library": "sklearn.svm", "priority": 1,
             "reason": f"小样本({n_samples}条)高维特征，SVM泛化能力强", "params": _svm_params()},
            {"model": "LogisticRegression", "library": "sklearn.linear_model", "priority": 2,
             "reason": "可解释性强，适合医疗场景", "params": _lr_params()},
            {"model": "RandomForestClassifier", "library": "sklearn.ensemble", "priority": 3,
             "reason": "对缺失值和噪声鲁棒", "params": _rf_params()},
        ]
    else:
        recommended_models = [
            {"model": "RandomForestClassifier", "library": "sklearn.ensemble", "priority": 1,
             "reason": "对医疗数据混合特征鲁棒，无需归一化", "params": _rf_params()},
            {"model": "XGBClassifier", "library": "xgboost", "priority": 2,
             "reason": "梯度提升通常在表格数据上表现最优", "params": _xgb_params()},
            {"model": "LogisticRegression", "library": "sklearn.linear_model", "priority": 3,
             "reason": "基线模型，可解释性强", "params": _lr_params()},
        ]

    cv_strategy = {
        "method": "StratifiedKFold",
        "n_splits": 5 if n_samples >= 100 else 3,
        "shuffle": True, "random_state": 42,
        "note": "样本量不足时建议 LeaveOneOut" if n_samples < 50 else "",
    }
    eval_metrics = (["accuracy", "roc_auc", "f1"] if is_binary
                    else ["accuracy", "f1_macro", "f1_weighted"])
    preprocessing = {
        "numeric_imputation": "median（已在数据集中完成）",
        "scaling": "StandardScaler（建议在 LR/SVM 前使用）",
        "categorical_encoding": "LabelEncoder（已在数据集中完成）",
        "class_imbalance": "SMOTE 或 class_weight='balanced'" if imbalance_ratio > 2 else "无需特殊处理",
    }

    inv_map = {v: k for k, v in label_mapping.items()}
    pytorch = {
        "timestamp":      datetime.now().isoformat(),
        "step":           "STEP7_ML_DataPrep",
        "granularity":    "record",
        "task_text":      _report_task_text(sel_report),
        "feature_names":  feat_cols,
        "n_samples":      int(n_samples),
        "n_features":     int(n_features),
        "label_column":   label_column,
        "label_map":      label_mapping,
        "icd10_mapping":  _effective_icd_map,
        "X": [[_json_safe(row[c]) for c in feat_cols] for _, row in df_train.iterrows()],
        "y": df_train["y"].tolist(),
        "sample_row_ids": df_train["sample_row_id"].tolist(),
        "patient_ids":    df_train["patient_id"].tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pytorch, f, ensure_ascii=False, indent=2)

    model_config = {
        "timestamp": datetime.now().isoformat(),
        "dataset_summary": {
            "n_samples": n_samples, "n_features": n_features, "n_classes": n_classes,
            "class_distribution": {inv_map.get(k, str(k)): v for k, v in label_dist.items()},
            "imbalance_ratio": round(imbalance_ratio, 2),
            "task_type": "binary_classification" if is_binary else "multiclass_classification",
        },
        "recommended_models": recommended_models,
        "cross_validation": cv_strategy,
        "evaluation_metrics": eval_metrics,
        "preprocessing_notes": preprocessing,
        "data_files": {"csv": csv_path, "jsonl": jsonl_path, "json": json_path},
    }
    model_config_path = os.path.join(OUTPUT_STEP7, f"model_config{_slug}_{ts}.json")
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    dist_str = "  ".join(
        f"y={k}({inv_map.get(k,'?')}): {v}例"
        for k, v in sorted(label_dist.items())
    )
    added_cols = [c for c in df_train.columns if c.endswith("_疾病名称") or c.endswith("_ICD10")]

    lines = [
        "=" * 65,
        "  ML 训练数据集（按记录逐行 + ICD 编码增强）",
        "=" * 65,
        f"  任务: {pytorch['task_text']}",
        f"  训练样本: {n_samples} 条（筛选后共 {len(df_ml)} 行记录，"
        f"{len(df_ml) - n_samples} 行因标签无法映射已排除）",
        f"  涉及患者数: {df_train['patient_id'].nunique()}",
        f"  特征维数: {n_features}（自动提取）",
        f"  标签列:  {label_column}",
        f"  标签分布: {dist_str}",
        f"  新增ICD增强列: {added_cols}",
    ]
    if icd_summary_lines:
        lines.append("\n── 诊断字段 ICD 标准化结果 ────────────────────────────")
        lines.extend(icd_summary_lines)
    lines += [
        "",
        "── 诊断标签 ICD 编码 ─────────────────────────────────────",
        df_train[["sample_row_id", "patient_id", "label_name", "label_ICD10", "y"]].to_string(index=False),
        "",
        "── 文件保存路径 ───────────────────────────────────────────",
        f"  CSV:         {csv_path}",
        f"  JSONL:       {jsonl_path}",
        f"  JSON:        {json_path}",
        f"  模型配置:    {model_config_path}",
        "",
        "── 推荐模型（按优先级）────────────────────────────────────",
    ]
    for m in recommended_models:
        lines.append(f"  [{m['priority']}] {m['model']}  —  {m['reason']}")
    lines += [
        f"  交叉验证: {cv_strategy['method']} n_splits={cv_strategy['n_splits']}",
        f"  评估指标: {', '.join(eval_metrics)}",
    ]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(lines))])


# ══════════════════════════════════════════════════════════════════
# 3. 模型与 Agent
# ══════════════════════════════════════════════════════════════════

def create_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        stream=True,
        client_kwargs={"base_url": OPENAI_BASE_URL},
        generate_kwargs={"temperature": 0.2, "max_tokens": 4096},
    )


def create_dataprep_agent() -> ReActAgent:
    if _IS_MIMIC:
        _label_hint = (
            "   - MIMIC 任务聚焦于死亡风险与住院时长；建议优先考虑死亡时间（是否为空 → 是否死亡）\n"
            "     或住院天数分桶作为标签，而非仅依赖 ICD 诊断列"
        )
    else:
        _label_hint = (
            "   - 根据 task_text 与诊断列分布选定最合适的标签列"
        )

    agent = ReActAgent(
        name="DataPrepAgent",
        sys_prompt="""你是一位 ML 数据工程师，负责将高置信度医疗数据转化为机器学习训练集。
数据集字段命名可能是中文、英文或混合，需通过工具自主探索。

## 特征列规则
- 特征列由工具自动确定（缺失率<70%），**禁止**手动指定特征列
- **禁止**将时间戳列（admittime/dischtime/deathtime/dod）用作特征或标签
- **禁止**将 ID 列（hadm_id/subject_id/patient_id）用作特征

## 工作流程

### 第一步：加载任务上下文
调用 load_task_context，获取：
- 有效特征列数量
- 全量标签候选列（含缺失率、唯一值数、示例值）
- ICD 编码列提示

### 第二步：发现可行的 ML 任务
根据标签候选列的分布，自主判断哪些列适合作为标签，识别出 **2～4 个可行任务**。
评估标准：
- 缺失率 < 70%
- 类别数在 2～20 之间（二分类或适度多分类）
- 类别分布不极度不均（正例占比 > 5%）
- 业务含义明确（死亡、出院去向、诊断类别等）

若需要了解某列的详细分布，可调用 get_column_distribution(column_name=...)。

### 第三步：依次为每个任务生成数据集
对每个发现的任务，按以下步骤操作：

**选择合适的标签策略（三选一）：**

策略 A — 非空值二分类（适合"某列是否有值"，如死亡预测）：
  - create_notnull_binary_label(source_column=..., positive_condition='not_null', ...)
  - format_and_save_ml_dataset(label_column='_notnull_label', label_mapping={...}, task_name='<任务名>')

策略 B — ICD 编码二分类（适合"是否患有某病"）：
  - create_icd_binary_label(icd_codes_column=..., positive_icd_prefixes=[...])
  - format_and_save_ml_dataset(label_column='_icd_label', label_mapping={...}, task_name='<任务名>')

策略 C — 现有分类列多分类（适合出院去向、疾病分级等）：
  - propose_label_mapping(label_column=..., mode='multiclass_enum')
  - build_label_series(...)
  - format_and_save_ml_dataset(label_column=..., task_name='<任务名>')

⚠ 每个任务必须用不同的 task_name（如 'mortality'、'discharge'、'diagnosis' 等）。
⚠ 每个任务的 format_and_save_ml_dataset 调用前须重置状态（策略 A/B 调用对应 create_* 工具）。

### 第四步：汇总报告
列出所有已生成的数据集，每个注明：任务名、标签含义、样本量、类别分布。

⚠ 关键约束：
  - 数据集质量优先：特征列缺失率<70% 由工具自动保障
  - **工作流必须闭环**：每个任务都要成功调用 format_and_save_ml_dataset
  - **禁止向用户追问**，自主决策
""",
        model=create_model(),
        formatter=OpenAIChatFormatter(
            token_counter=CharTokenCounter(),
            max_tokens=_FORMATTER_MAX_CHARS,
        ),
        toolkit=step7_toolkit,
        memory=InMemoryMemory(),
        max_iters=60,
        print_hint_msg=False,
    )
    agent._disable_console_output = True
    return agent


# ══════════════════════════════════════════════════════════════════
# 4. 辅助函数
# ══════════════════════════════════════════════════════════════════

def _extract_msg_text(msg: Msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
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


# ══════════════════════════════════════════════════════════════════
# 5. Pipeline
# ══════════════════════════════════════════════════════════════════

def _outputs_since(prefix: str, suffix: str, started_at: float) -> list[str]:
    root = Path(OUTPUT_STEP7)
    if not root.is_dir():
        return []
    paths = [
        path for path in root.iterdir()
        if path.is_file()
        and path.name.startswith(prefix)
        and path.name.endswith(suffix)
        and path.stat().st_mtime >= started_at
    ]
    paths.sort(key=lambda path: path.name)
    return [str(path) for path in paths]


async def run_step7_only(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    global _passed_patient_ids, _icd_label_cache, _LLM_DIAG_FIELDS_CACHE

    paths = configure_runtime(runtime_overrides)
    try:
        _validate_runtime_inputs()
        _passed_patient_ids = _load_passed_patient_ids()
        selection_report_payload = _load_selection_report()
        task_text = _report_task_text(selection_report_payload)
        if not task_text or task_text == "N/A":
            raise ValueError("Step7 缺少明确 task_text / ml_task_goal，不能生成与任务目标无关的数据集。")
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "input_csv": FILTERED_CSV,
            "selection_report": SELECTION_REPORT,
            "passed_patients_json": PASSED_PATIENTS_JSON,
            "passed_patient_count": 0,
            "output_step7": OUTPUT_STEP7,
            "task_text": TASK_TEXT_OVERRIDE,
            "step7_dataset_csvs": [],
            "step7_dataset_jsonls": [],
            "step7_dataset_jsons": [],
            "model_config_jsons": [],
            "runtime_paths": paths,
        }

    # 重置 Step 7 状态
    _icd_label_cache = {}
    _LLM_DIAG_FIELDS_CACHE = None

    _data_label = "MIMIC" if _IS_MIMIC else "Original"
    print("=" * 65)
    print("  STEP 7: ML 训练数据生成")
    print("=" * 65)
    print(f"  模式:    {_data_label}（DATA_SOURCE={DATA_SOURCE}）")
    print(f"  模型:    {MODEL_NAME}")
    print(f"  筛选后:  {FILTERED_CSV}")
    print(f"  筛选报告: {SELECTION_REPORT}")
    print(f"  Step6通过名单: {PASSED_PATIENTS_JSON}")
    print(f"  通过患者数: {len(_passed_patient_ids)}")
    print(f"  任务目标: {task_text}")
    print(f"  输出目录: {OUTPUT_STEP7}")
    print("=" * 65)

    agent7 = create_dataprep_agent()
    msg7 = Msg(
        name="Pipeline",
        role="user",
        content=(
            "Step 6 一致性验证已完成。\n\n"
            f"通过验证的患者数量: {len(_passed_patient_ids)}\n\n"
            f"本轮 ML 任务目标: {task_text}\n\n"
            "请执行 Step 7：严格围绕本轮 ML 任务目标构造训练数据集，"
            "禁止生成与任务目标无关的探索性任务。\n\n"
            "工作流：\n"
            "① 调用 load_task_context 了解数据概况和全量标签候选列\n"
            "② 只选择与本轮任务目标直接对应的标签列或标签策略\n"
            "③ 对任务选择合适策略（非空二分类/ICD二分类/多分类），"
            "用不同 task_name 保存独立数据集\n"
            "④ 汇总已生成数据集的报告；如果无法构造任务目标对应标签，返回 NEEDS_REPAIR\n\n"
            "⚠ 禁止将 dod/admittime/dischtime/deathtime 等时间戳列用作特征。\n"
            "⚠ 禁止生成非本轮任务目标的数据集。\n"
            f"筛选后数据路径：{FILTERED_CSV}\n"
        ),
    )

    print("\n[Step 7] ▶ 启动 DataPrepAgent...")
    run_started_at = datetime.now().timestamp()
    try:
        result7 = await agent7(msg7)
        text7 = _extract_msg_text(result7)
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "input_csv": FILTERED_CSV,
            "selection_report": SELECTION_REPORT,
            "passed_patients_json": PASSED_PATIENTS_JSON,
            "passed_patient_count": len(_passed_patient_ids),
            "output_step7": OUTPUT_STEP7,
            "task_text": task_text,
            "step7_dataset_csvs": _outputs_since("ml_dataset", ".csv", run_started_at),
            "step7_dataset_jsonls": _outputs_since("ml_dataset", ".jsonl", run_started_at),
            "step7_dataset_jsons": _outputs_since("ml_dataset", ".json", run_started_at),
            "model_config_jsons": _outputs_since("model_config", ".json", run_started_at),
            "runtime_paths": paths,
        }

    dataset_csvs = _outputs_since("ml_dataset", ".csv", run_started_at)
    dataset_jsonls = _outputs_since("ml_dataset", ".jsonl", run_started_at)
    dataset_jsons = _outputs_since("ml_dataset", ".json", run_started_at)
    model_configs = _outputs_since("model_config", ".json", run_started_at)
    visible_text7 = _strip_thinking_blocks(text7)
    error = ""
    if not dataset_csvs or not dataset_jsonls or not dataset_jsons or not model_configs:
        error = "Step7 未保存完整的 ml_dataset CSV/JSONL/JSON 和 model_config JSON。"

    print("\n" + "=" * 65)
    print("  Step 7 完成")
    print("=" * 65)
    print(visible_text7)

    print("\n" + "=" * 65)
    print("  Step 7 全部完成")
    print(f"  数据目录: {OUTPUT_STEP7}/")
    print("=" * 65)
    return {
        "success": bool(visible_text7) and not error,
        "error": error,
        "input_csv": FILTERED_CSV,
        "selection_report": SELECTION_REPORT,
        "passed_patients_json": PASSED_PATIENTS_JSON,
        "passed_patient_count": len(_passed_patient_ids),
        "output_step7": OUTPUT_STEP7,
        "task_text": task_text,
        "step7_dataset_csvs": dataset_csvs,
        "step7_dataset_jsonls": dataset_jsonls,
        "step7_dataset_jsons": dataset_jsons,
        "model_config_jsons": model_configs,
        "step7_text": visible_text7,
        "runtime_paths": paths,
    }


async def run_step7() -> None:
    result = await run_step7_only()
    if not result.get("success"):
        raise RuntimeError(str(result.get("error") or "Step7 执行失败"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone Step7 ML dataset generation.")
    parser.add_argument("--input", dest="input_csv", default="", help="Path to Step5 next_input/filtered.csv.")
    parser.add_argument("--selection-report", default="", help="Path to Step4 next_input/selection_report.json.")
    parser.add_argument("--passed-patients-json", default="", help="Path to Step6 passed_patients_<ts>.json.")
    parser.add_argument("--step6-output", dest="output_step6_dir", default="", help="Step6 output directory, used for metadata only.")
    parser.add_argument("--output", dest="output_step7_dir", default="", help="Step7 output directory.")
    parser.add_argument("--patient-id-col", default="", help="Optional patient id column override.")
    parser.add_argument("--data-source", default="", choices=["", "original", "mimic"], help="Dataset flavor override.")
    parser.add_argument("--task-text", default="", help="Explicit ML task goal, overrides selection_report task text.")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    return _build_arg_parser()


def _main() -> int:
    args = _build_arg_parser().parse_args()
    overrides = {
        "input_csv": args.input_csv,
        "selection_report": args.selection_report,
        "passed_patients_json": args.passed_patients_json,
        "output_step6_dir": args.output_step6_dir,
        "output_step7_dir": args.output_step7_dir,
        "patient_id_col": args.patient_id_col,
        "data_source": args.data_source,
        "task_text": args.task_text,
    }
    overrides = {key: value for key, value in overrides.items() if value}
    result = asyncio.run(run_step7_only(overrides))
    print(json.dumps({k: v for k, v in result.items() if k != "step7_text"}, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
