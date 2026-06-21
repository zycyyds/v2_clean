"""sample_table skill — deep inspection of a single CSV/Excel table."""
from __future__ import annotations

import json
from pathlib import Path

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

from skills import _common  # noqa: F401


def sample_table_tool(
    csv_path: str,
    max_unique: int = 20,
    text_sample_rows: int = 3,
    compact: bool = True,
) -> ToolResponse:
    """对单张 CSV/Excel 表做深度探索：数值列统计、分类列 top-N、文本列样本、高基数列唯一值数量。

    Args:
        csv_path: CSV 或 Excel 文件的绝对路径。
        max_unique: 分类列展示的最大唯一值数量（默认 20）。
        text_sample_rows: 文本列展示的样本行数（默认 3）。
        compact: 是否返回适合 Agent 上下文的紧凑摘要（默认 True）。

    Returns:
        每列的类型判断 + 统计摘要 + 样本值，帮助 agent 决定如何处理该列。
    """
    import numpy as np
    import pandas as pd

    p = Path(csv_path)
    if not p.exists():
        return ToolResponse(content=[TextBlock(type="text", text=json.dumps({
            "status": "FAILED",
            "summary": f"文件不存在: {csv_path}",
            "artifacts": {},
            "issues": [f"文件不存在: {csv_path}"],
        }, ensure_ascii=False, indent=2))])

    try:
        if p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        return ToolResponse(content=[TextBlock(type="text", text=json.dumps({
            "status": "FAILED",
            "summary": f"读取失败: {e}",
            "artifacts": {},
            "issues": [str(e)],
        }, ensure_ascii=False, indent=2))])

    columns_info = []
    for col in df.columns:
        series = df[col]
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        col_info: dict = {
            "column": col,
            "dtype": str(series.dtype),
            "total_rows": n_total,
            "missing": n_missing,
            "missing_pct": round(n_missing / n_total * 100, 1) if n_total else 0,
            "unique_count": n_unique,
        }

        non_null = series.dropna()

        # numeric
        if pd.api.types.is_numeric_dtype(series):
            col_info["col_type"] = "numeric"
            if len(non_null) > 0:
                stats = {
                    "min": round(float(non_null.min()), 4),
                    "max": round(float(non_null.max()), 4),
                    "mean": round(float(non_null.mean()), 4),
                }
                if not compact:
                    stats.update({
                        "median": round(float(non_null.median()), 4),
                        "std": round(float(non_null.std()), 4),
                    })
                col_info["stats"] = stats
            if not compact:
                col_info["sample_values"] = [str(v) for v in non_null.head(5).tolist()]

        # datetime-like
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["col_type"] = "datetime"
            col_info["sample_values"] = [str(v) for v in non_null.head(1 if compact else 5).tolist()]

        else:
            # try parse as datetime
            if n_unique < n_total * 0.8 and n_unique <= max_unique:
                col_info["col_type"] = "categorical"
                vc = series.value_counts(dropna=True).head(min(max_unique, 5) if compact else max_unique)
                col_info["top_values"] = {str(k): int(v) for k, v in vc.items()}
            else:
                # check if text (long strings)
                avg_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
                if avg_len > 50:
                    col_info["col_type"] = "text"
                    col_info["avg_text_length"] = round(float(avg_len), 1)
                    col_info["sample_values"] = [
                        str(v)[:80 if compact else 200]
                        for v in non_null.head(min(text_sample_rows, 1) if compact else text_sample_rows).tolist()
                    ]
                    if not compact:
                        col_info["note"] = "长文本列，可用 extract_text_features 或 run_python_code 抽取结构化信息"
                else:
                    col_info["col_type"] = "high_cardinality"
                    col_info["sample_values"] = [
                        str(v) for v in non_null.head(2 if compact else 10).tolist()
                    ]
                    if not compact:
                        col_info["note"] = f"高基数列（{n_unique} 个唯一值），考虑 pivot/groupby 展开或编码"

        columns_info.append(col_info)

    payload = {
        "status": "SUCCESS",
        "summary": f"已分析 {p.name}：{len(df)} 行 × {len(df.columns)} 列。",
        "artifacts": {
            "csv_path": str(p),
            "row_count": len(df),
            "col_count": len(df.columns),
            "detail_level": "compact" if compact else "detailed",
            "columns": columns_info,
        },
        "issues": [],
    }
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


SKILL = {
    "name": "sample_table",
    "layer": "explore",
    "description": (
        "对单张 CSV 做深度探索：数值列给出 min/max/mean/std，分类列给出 top-N 取值分布，"
        "文本列给出样本内容和平均长度，高基数列（如 itemid）给出唯一值数量和样本。"
    ),
    "when_to_use": (
        "list_data_files 之后，对每张与任务相关的表调用此 skill，"
        "了解列的实际内容后再决定如何构建特征。"
        "特别适合：labevents（了解有哪些 itemid）、notes（了解文本内容）、diagnoses_icd（了解 ICD 分布）。"
    ),
    "when_to_skip": "目标文件不是 CSV/Excel，或只需要目录级文件清单而不需要列值画像时跳过。",
    "capability_types": ["table_sampling", "column_profiling"],
    "inputs": ["csv_path", "max_unique?（默认20）", "text_sample_rows?（默认3）", "compact?（默认true）"],
    "outputs": ["columns（含 col_type、stats/top_values/sample_values）"],
    "prerequisites": ["tabular_file_identified"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": [],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(sample_table_tool)
