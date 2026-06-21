# -*- coding: utf-8 -*-
"""
信息抽取工具（Extract Tools）

使用 LLM 从医学文本中提取结构化实体。
供 ReActAgent 直接调用，也可被 CSVProcessor 和目录处理流程使用。

工具列表：
  - extract_from_text          从纯文本中抽取医学实体（async）
  - extract_from_text_sync     同步版本（内部用 asyncio.run）
  - standardize_entities       对已抽取的实体进行标准化（async）
"""
import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_AS_SRC = os.path.abspath(os.path.join(_HERE, "../../src"))
if _AS_SRC not in sys.path:
    sys.path.insert(0, _AS_SRC)

_OLD_ROOT = os.path.join(os.path.dirname(_HERE), "medical_data_cleaner")
if _OLD_ROOT not in sys.path:
    sys.path.insert(0, _OLD_ROOT)

from config import get_api_config, get_embedding_config


# ---------------------------------------------------------------------------
# 内部 LLM 调用
# ---------------------------------------------------------------------------

_llm_client: Optional[Any] = None  # 复用单个 AsyncOpenAI chat client
_embedding_client: Optional[Any] = None  # 独立 embedding client


def _get_llm_client() -> Any:
    global _llm_client
    if _llm_client is None:
        import openai
        cfg = get_api_config()
        _llm_client = openai.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
    return _llm_client


def _get_embedding_client() -> Any:
    global _embedding_client
    if _embedding_client is None:
        import openai
        cfg = get_embedding_config()
        _embedding_client = openai.AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"])
    return _embedding_client


async def _call_llm(prompt: str, model_name: Optional[str] = None) -> Optional[str]:
    """通用 LLM 调用，返回原始文本响应。"""
    try:
        cfg = get_api_config()
        if not cfg["api_key"]:
            return None
        client = _get_llm_client()
        resp = await client.chat.completions.create(
            model=model_name or cfg["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip() if resp.choices else None
    except Exception:
        return None


def _parse_json_response(content: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON。"""
    if not content:
        return None
    for marker in ["```json", "```"]:
        if marker in content:
            parts = content.split(marker)
            if len(parts) >= 2:
                try:
                    return json.loads(parts[1].split("```")[0].strip())
                except json.JSONDecodeError:
                    pass
    try:
        start, end = content.find("{"), content.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# 提取 Prompt 模板
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are a medical information extraction expert. Extract structured entities from the clinical text below.

## Output language rule
ALWAYS use the SAME language as the input text. If the text is Chinese, output Chinese. If English, output English. Never translate.

## Entity Categories and Fields

**Diagnosis** — confirmed or suspected diagnoses / conditions
  name: condition name (≤6 words)
  status: "confirmed" | "suspected" | "history of" | "ruled out" | "" (leave empty if clearly active/confirmed)
  severity: severity descriptor if stated (e.g. "mild", "moderate", "severe") | ""
  etiology: cause or complication if stated (e.g. "due to cirrhosis", "complicated by ascites") | ""
  treatment_context: treatment or management mentioned (e.g. "on ART", "requiring paracentesis") | ""

**Symptom** — patient-reported symptoms or chief complaints
  name: symptom name (≤6 words)
  severity: severity if stated (e.g. "mild", "severe", "10/10") | ""
  onset: onset descriptor if stated (e.g. "new onset", "since last week") | ""
  trend: change over time if stated (e.g. "worsening", "improving", "stable") | ""
  character: quality/location/radiation if stated (e.g. "constant, epigastric, radiates to back") | ""

**LabResult** — any measurement with a numeric value
  name: test or measurement name (≤6 words)
  value: numeric value ONLY (digits, optional sign/decimal) — NO units, NO text
  unit: unit string | ""
  ⚠ value must be a pure number

**Medication** — drugs or medications mentioned
  name: drug name (≤6 words)
  dose: dose amount if stated (e.g. "1g", "10mg") | ""
  frequency: frequency if stated (e.g. "daily", "BID", "x1") | ""
  route: route if stated (e.g. "IV", "PO", "topical") | ""

**Procedure** — medical procedures or interventions
  name: procedure name (≤6 words)
  result: key result or finding if stated (≤8 words) | ""

**Finding** — qualitative observations from physical exam, imaging, or functional tests (non-numeric)
  name: anatomical location or test name (≤6 words)
  value: brief qualitative descriptor (≤8 words)
  ⚠ SKIP findings that are explicitly absent/negative (e.g. "no effusion", "without consolidation")

## Rules
1. Output language MUST match input text language — never translate
2. LabResult.value must be a pure number — no units, no text
3. Extract only explicitly stated information — do not infer
4. Do not extract reference ranges or normal value intervals
5. name must be ≤6 words; free-text fields must be ≤10 words — never paste full sentences
6. SKIP any Finding or Symptom whose value starts with "no", "without", "absent", "negative", "not seen", "not identified"

## Output — JSON only, no other text
{{
  "entities": [
    {{"category": "Diagnosis", "name": "...", "status": "...", "severity": "...", "etiology": "...", "treatment_context": "..."}},
    {{"category": "Symptom", "name": "...", "severity": "...", "onset": "...", "trend": "...", "character": "..."}},
    {{"category": "LabResult", "name": "...", "value": "...", "unit": "..."}},
    {{"category": "Medication", "name": "...", "dose": "...", "frequency": "...", "route": "..."}},
    {{"category": "Procedure", "name": "...", "result": "..."}},
    {{"category": "Finding", "name": "...", "value": "..."}}
  ],
  "impression": "one-sentence summary of key findings or diagnoses (same language as input text)",
  "indication": "reason for study/visit if mentioned, else empty string"
}}

Clinical text:
{text}
"""

# 各 category 的子字段定义（用于宽表列构建）
ENTITY_SUBFIELDS: Dict[str, List[str]] = {
    "Diagnosis": ["status", "severity", "etiology", "treatment_context"],
    "Symptom":   ["severity", "onset", "trend", "character"],
    "LabResult": ["value", "unit"],
    "Medication": ["dose", "frequency", "route"],
    "Procedure": ["result"],
    "Finding":   ["value"],
}


def _postprocess_entities(entities: List[Dict]) -> List[Dict]:
    """
    后处理：
    1. LabResult.value 强制为纯数字字符串
    2. 过滤参考范围实体
    3. 确保各 category 的子字段存在（缺失补空字符串）
    """
    import re
    cleaned = []
    for ent in entities:
        cat = ent.get("category", "")
        value = ent.get("value", "")

        # 过滤参考范围
        if isinstance(value, str) and re.search(r'\d+\s*[~～\-]\s*\d+', value):
            name_lower = str(ent.get("name", "")).lower()
            if any(kw in name_lower for kw in ("参考", "正常值", "reference", "normal range")):
                continue

        # LabResult.value 必须是纯数字
        if cat == "LabResult" and value is not None:
            if isinstance(value, (int, float)):
                ent["value"] = str(value)
            elif isinstance(value, str) and value.strip():
                match = re.search(r'[+-]?\d+\.?\d*', value)
                ent["value"] = match.group() if match else value

        # 补全子字段（缺失的填空字符串）
        for field in ENTITY_SUBFIELDS.get(cat, []):
            if field not in ent:
                ent[field] = ""
            elif ent[field] is None:
                ent[field] = ""

        cleaned.append(ent)
    return cleaned


async def extract_from_text(text: str, max_chars: int = 4000) -> Dict[str, Any]:
    """
    使用 LLM 从医学文本中提取结构化实体（异步）。

    Args:
        text: 医学文本（自由文本，如出院记录、检查报告等）
        max_chars: 最大处理字符数（超出则截断）

    Returns:
        {
          "success": bool,
          "entities": [{"category", "name", "value", "unit", "original_text"}],
          "temporal_info": [...],
          "quantity_info": [...],
          "relations": [...],
          "impression": str,
          "indication": str,
          "entity_count": int,
          "error": str | None,
        }
    """
    if not text or not text.strip():
        return {"success": False, "entities": [], "temporal_info": [], "quantity_info": [],
                "relations": [], "impression": "", "indication": "", "entity_count": 0,
                "error": "输入文本为空"}

    prompt = _EXTRACT_PROMPT.format(text=text[:max_chars])
    content = await _call_llm(prompt)
    parsed = _parse_json_response(content) if content else None

    if parsed is None:
        return {"success": False, "entities": [], "temporal_info": [], "quantity_info": [],
                "relations": [], "impression": "", "indication": "", "entity_count": 0,
                "error": "LLM 抽取失败或 API 不可用"}

    entities = parsed.get("entities", [])
    entities = _postprocess_entities(entities)

    return {
        "success": True,
        "entities": entities,
        "temporal_info": parsed.get("temporal_info", []),
        "quantity_info": parsed.get("quantity_info", []),
        "relations": parsed.get("relations", []),
        "impression": parsed.get("impression", ""),
        "indication": parsed.get("indication", ""),
        "entity_count": len(entities),
        "error": None,
    }


def extract_from_text_sync(text: str, max_chars: int = 4000) -> Dict[str, Any]:
    """
    同步版本的 extract_from_text，供非 async 上下文调用。
    """
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, extract_from_text(text, max_chars))
                return future.result()
        return asyncio.run(extract_from_text(text, max_chars))
    except Exception as e:
        return {"success": False, "entities": [], "temporal_info": [], "quantity_info": [],
                "relations": [], "impression": "", "indication": "", "entity_count": 0,
                "error": str(e)}


# ---------------------------------------------------------------------------
# 实体名称语义聚类（Embedding + 层次聚类）
# ---------------------------------------------------------------------------

_CLUSTER_THRESHOLD = 0.25  # 距离阈值，对应余弦相似度 > 0.75


async def _get_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """批量获取文本 embedding，返回向量列表。"""
    try:
        cfg = get_embedding_config()
        if not cfg["api_key"] or not cfg["api_base"]:
            return None
        client = _get_embedding_client()
        kwargs = {"model": cfg["model_name"], "input": texts}
        if cfg.get("dimensions"):
            kwargs["dimensions"] = cfg["dimensions"]
        resp = await client.embeddings.create(**kwargs)
        return [d.embedding for d in resp.data]
    except Exception:
        return None


def _cluster_by_embedding(
    names: List[str],
    embeddings: List[List[float]],
    threshold: float = _CLUSTER_THRESHOLD,
) -> Dict[str, str]:
    """
    对已有 embedding 的名称列表做层次聚类，返回 {name: canonical_name}。
    canonical_name 取组内出现频次最高的名称（频次相同取最短）。
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from collections import defaultdict, Counter

    vecs = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = (vecs / norms) @ (vecs / norms).T
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)

    Z = linkage(squareform(dist), method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")

    groups: Dict[int, List[str]] = defaultdict(list)
    for name, lbl in zip(names, labels):
        groups[int(lbl)].append(name)

    mapping: Dict[str, str] = {}
    for members in groups.values():
        freq = Counter(members)
        canonical = min(freq.keys(), key=lambda n: (-freq[n], len(n), n))
        for m in members:
            mapping[m] = canonical
    return mapping


async def cluster_entity_names(
    names_by_category: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    对各 category 下的实体名称列表做 Embedding 语义聚类。

    Args:
        names_by_category: {category: [name, ...]}  （名称可重复，内部去重）

    Returns:
        {original_name_lower: canonical_name}  全局映射
    """
    from collections import Counter

    mapping: Dict[str, str] = {}
    for cat, names in names_by_category.items():
        # 去重，保留频次信息用于 canonical 选取
        freq = Counter(n.strip().lower() for n in names if n.strip())
        unique = sorted(freq.keys())
        if not unique:
            continue
        if len(unique) == 1:
            mapping[unique[0]] = unique[0]
            continue

        embeddings = await _get_embeddings(unique)
        if embeddings is None or len(embeddings) != len(unique):
            # fallback：identity mapping
            for n in unique:
                mapping[n] = n
            continue

        try:
            cat_map = _cluster_by_embedding(unique, embeddings)
            # canonical 选取时用原始频次加权
            from collections import defaultdict
            groups: Dict[str, List[str]] = defaultdict(list)
            for orig, canon in cat_map.items():
                groups[canon].append(orig)
            for members in groups.values():
                canonical = min(members, key=lambda n: (-freq.get(n, 1), len(n), n))
                for m in members:
                    mapping[m] = canonical
        except Exception:
            for n in unique:
                mapping[n] = n

    return mapping


# ---------------------------------------------------------------------------
# 标准化 Prompt 模板（对已抽取实体批量标准化）
# ---------------------------------------------------------------------------

_STANDARDIZE_PROMPT = """\
你是医学术语标准化专家。请对以下已抽取的医学实体进行标准化：

1. 术语标准化：将实体名称映射到标准编码（Disease→ICD-10, Drug→ATC, Test/LabValue→LOINC, Symptom/Finding→SNOMED-CT）
2. 量纲统一：如果实体有 value+unit，将其转换为标准单位（血糖→mmol/L，血红蛋白→g/L，白细胞→×10⁹/L 等）

输入实体列表：
{entities_json}

请在每个实体中添加以下字段（无法确定则省略）：
- standard_code: 标准编码
- standard_name: 标准英文名
- standard_system: 编码系统（ICD-10/ATC/LOINC/SNOMED-CT）
- normalized_value: 标准化数值（float）
- normalized_unit: 标准化单位

只返回更新后的 entities JSON 数组，不要其他文字。
"""


async def standardize_entities(
    entities: List[Dict[str, Any]],
    batch_size: int = 20,
) -> Dict[str, Any]:
    """
    对已抽取的实体列表进行批量标准化（术语编码 + 量纲统一）。

    Args:
        entities: extract_from_text 返回的 entities 列表
        batch_size: 每批处理的实体数（避免 token 超限）

    Returns:
        {
          "success": bool,
          "entities": [标准化后的实体列表],
          "standardized_count": int,
          "error": str | None,
        }
    """
    if not entities:
        return {"success": True, "entities": [], "standardized_count": 0, "error": None}

    results: List[Dict] = []
    for i in range(0, len(entities), batch_size):
        batch = entities[i: i + batch_size]
        prompt = _STANDARDIZE_PROMPT.format(
            entities_json=json.dumps(batch, ensure_ascii=False, indent=2)
        )
        content = await _call_llm(prompt)
        parsed = None
        if content:
            # 响应是数组
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                parsed = json.loads(content.strip())
            except json.JSONDecodeError:
                pass

        if isinstance(parsed, list):
            results.extend(parsed)
        else:
            # 标准化失败，原样保留
            results.extend(batch)

    std_count = sum(
        1 for e in results if e.get("standard_code") or e.get("normalized_value") is not None
    )
    return {
        "success": True,
        "entities": results,
        "standardized_count": std_count,
        "error": None,
    }
