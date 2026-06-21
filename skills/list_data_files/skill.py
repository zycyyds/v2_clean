"""list_data_files skill — recursively find all CSV/Excel files and return their column names."""
from __future__ import annotations

import json
import os
from pathlib import Path

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

from skills import _common  # noqa: F401


def list_data_files_tool(
    input_path: str,
    compact: bool = True,
    max_columns: int = 12,
) -> ToolResponse:
    """递归扫描目录，列出所有 CSV/Excel 文件及其列名和行数。

    Args:
        input_path: 要扫描的目录路径（绝对路径）。
        compact: 是否限制每个文件返回的列名数量。
        max_columns: compact 模式下每个文件最多返回的列名数量。

    Returns:
        包含所有数据文件路径、列名、行数的结构化结果。
    """
    import pandas as pd

    root = Path(input_path)
    if not root.exists():
        return ToolResponse(content=[TextBlock(type="text", text=json.dumps({
            "status": "FAILED",
            "summary": f"路径不存在: {input_path}",
            "artifacts": {},
            "issues": [f"路径不存在: {input_path}"],
        }, ensure_ascii=False, indent=2))])

    suffixes = {".csv", ".xlsx", ".xls", ".tsv"}
    files = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in suffixes:
            continue
        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p, nrows=0)
            else:
                df = pd.read_excel(p, nrows=0)
            # get row count without loading all data
            if p.suffix.lower() == ".csv":
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    row_count = sum(1 for _ in f) - 1
            else:
                row_count = len(pd.read_excel(p))
            all_columns = list(df.columns)
            displayed_columns = all_columns[:max_columns] if compact else all_columns
            files.append({
                "path": str(p),
                "relative_path": str(p.relative_to(root)),
                "columns": displayed_columns,
                "column_count": len(all_columns),
                "columns_truncated": len(displayed_columns) < len(all_columns),
                "row_count": row_count,
            })
        except Exception as e:
            files.append({
                "path": str(p),
                "relative_path": str(p.relative_to(root)),
                "columns": [],
                "row_count": -1,
                "error": str(e),
            })

    payload = {
        "status": "SUCCESS",
        "summary": f"找到 {len(files)} 个数据文件。",
        "artifacts": {
            "input_path": str(root),
            "file_count": len(files),
            "detail_level": "compact" if compact else "detailed",
            "files": files,
        },
        "issues": [],
    }
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


SKILL = {
    "name": "list_data_files",
    "layer": "explore",
    "description": "递归扫描目录，列出所有 CSV/Excel 文件的完整路径、列名和行数。",
    "when_to_use": "explore_dataset 之后，需要知道具体有哪些表、每张表有哪些列时调用。这是了解多表数据集结构的关键步骤。",
    "when_to_skip": "输入中没有 CSV/Excel 文件，或已有未过期的完整表清单时跳过。",
    "capability_types": ["tabular_file_discovery", "schema_inspection"],
    "inputs": ["input_path", "compact?（默认true）", "max_columns?（默认12）"],
    "outputs": ["files（含 path、columns、row_count）"],
    "prerequisites": ["input_path_accessible"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": [],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(list_data_files_tool)
