# -*- coding: utf-8 -*-
"""
IO 工具（IO Tools）

文件读写、结果保存、CSV 处理、Excel 表格读取。
供 ReActAgent 调用，也被处理流水线直接使用。

工具列表：
  - read_csv_all          读取 CSV 全部数据
  - save_rows_to_csv      将行数据保存为 CSV
  - save_result_json      将处理结果保存为 JSON
  - load_excel_table      读取 Excel 表格并按患者 ID 索引
  - get_patient_row       从已加载的表格中获取指定患者的行数据
  - build_output_dir      创建带时间戳的输出目录
"""
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_OLD_ROOT = os.path.join(os.path.dirname(_HERE), "medical_data_cleaner")
if _OLD_ROOT not in sys.path:
    sys.path.insert(0, _OLD_ROOT)


# ---------------------------------------------------------------------------
# CSV 读写
# ---------------------------------------------------------------------------

def read_csv_all(
    file_path: str,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    读取 CSV 文件的全部数据（或前 max_rows 行）。

    Args:
        file_path: CSV 文件路径
        max_rows: 最大读取行数（None 表示全部）

    Returns:
        {
          "success": bool,
          "columns": [列名],
          "rows": [{列名: 值, ...}, ...],
          "row_count": int,
          "error": str | None,
        }
    """
    if not os.path.exists(file_path):
        return {"success": False, "columns": [], "rows": [], "row_count": 0,
                "error": f"文件不存在: {file_path}"}

    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]:
        try:
            rows: List[Dict] = []
            with open(file_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                for i, row in enumerate(reader):
                    if max_rows is not None and i >= max_rows:
                        break
                    rows.append(dict(row))
            return {"success": True, "columns": list(cols),
                    "rows": rows, "row_count": len(rows), "error": None}
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return {"success": False, "columns": [], "rows": [], "row_count": 0, "error": str(e)}

    return {"success": False, "columns": [], "rows": [], "row_count": 0,
            "error": "无法解析文件编码"}


def save_rows_to_csv(
    rows: List[Dict[str, Any]],
    output_path: str,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    将行数据列表保存为 CSV 文件。

    Args:
        rows: 行数据列表（dict），所有行的 key 集合即为列名
        output_path: 输出文件路径（会自动创建父目录）
        columns: 指定列顺序（None 则按第一行的 key 顺序）

    Returns:
        {"success": bool, "output_path": str, "row_count": int, "error": str|None}
    """
    if not rows:
        return {"success": False, "output_path": "", "row_count": 0, "error": "rows 为空"}
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if columns is None:
            # 保持所有键、保持顺序（Python 3.7+ dict 有序）
            seen: Dict[str, None] = {}
            for row in rows:
                for k in row:
                    seen[k] = None
            columns = list(seen.keys())

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})

        return {"success": True, "output_path": output_path,
                "row_count": len(rows), "error": None}
    except Exception as e:
        return {"success": False, "output_path": "", "row_count": 0, "error": str(e)}


def save_result_json(
    result: Dict[str, Any],
    output_path: str,
) -> Dict[str, Any]:
    """
    将处理结果字典保存为 JSON 文件。

    Args:
        result: 任意可 JSON 序列化的字典
        output_path: 输出 JSON 文件路径

    Returns:
        {"success": bool, "output_path": str, "error": str|None}
    """
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        return {"success": True, "output_path": output_path, "error": None}
    except Exception as e:
        return {"success": False, "output_path": "", "error": str(e)}


# ---------------------------------------------------------------------------
# Excel 表格读取
# ---------------------------------------------------------------------------

def load_excel_table(xlsx_path: str) -> Dict[str, Any]:
    """
    读取 Excel 文件，返回以第一列（通常为患者 ID）为键的数据字典。

    Args:
        xlsx_path: Excel 文件路径（.xlsx / .xls）

    Returns:
        {
          "success": bool,
          "columns": [列名],
          "data": {患者ID字符串: {列名: 值, ...}},
          "row_count": int,
          "id_column": str,   # 用作 key 的列名
          "error": str|None,
        }
    """
    if not os.path.exists(xlsx_path):
        return {"success": False, "columns": [], "data": {}, "row_count": 0,
                "id_column": "", "error": f"文件不存在: {xlsx_path}"}
    try:
        # CSV 格式直接用 csv 模块
        if xlsx_path.lower().endswith(".csv"):
            data: Dict[str, Dict] = {}
            for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    with open(xlsx_path, "r", encoding=enc, newline="") as f:
                        reader = csv.DictReader(f)
                        header = list(reader.fieldnames or [])
                        if not header:
                            continue
                        id_col = header[0]
                        for row in reader:
                            key = str(row.get(id_col, "")).strip()
                            if key:
                                data[key] = dict(row)
                    return {"success": True, "columns": header, "data": data,
                            "row_count": len(data), "id_column": id_col, "error": None}
                except UnicodeDecodeError:
                    continue
            return {"success": False, "columns": [], "data": {}, "row_count": 0,
                    "id_column": "", "error": "无法解析 CSV 文件编码"}
        try:
            from tools.table_reader import load_table_data
            data = load_table_data(xlsx_path)
            # load_table_data 返回 {id: {col: val}}
            cols: List[str] = []
            if data:
                first = next(iter(data.values()))
                cols = list(first.keys())
            return {"success": True, "columns": cols, "data": data,
                    "row_count": len(data), "id_column": "id", "error": None}
        except ImportError:
            pass
        # 回退：直接用 openpyxl
        import openpyxl
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = iter(ws.iter_rows(values_only=True))
        header = [str(c) if c is not None else "" for c in next(rows_iter, [])]
        if not header:
            return {"success": False, "columns": [], "data": {}, "row_count": 0,
                    "id_column": "", "error": "Excel 文件无表头"}
        id_col = header[0]
        data: Dict[str, Dict] = {}
        for raw in rows_iter:
            row = dict(zip(header, [str(v) if v is not None else "" for v in raw]))
            key = str(row.get(id_col, "")).strip()
            if key:
                data[key] = row
        return {"success": True, "columns": header, "data": data,
                "row_count": len(data), "id_column": id_col, "error": None}
    except Exception as e:
        return {"success": False, "columns": [], "data": {}, "row_count": 0,
                "id_column": "", "error": str(e)}


def get_patient_row(
    table_data: Dict[str, Dict[str, Any]],
    patient_id: str,
) -> Dict[str, Any]:
    """
    从已加载的表格数据中获取指定患者的行数据。

    Args:
        table_data: load_excel_table 返回的 data 字段
        patient_id: 患者 ID 字符串

    Returns:
        该患者的行数据 dict（找不到则返回空 dict）
    """
    pid = str(patient_id).strip()
    # 精确匹配
    if pid in table_data:
        return dict(table_data[pid])
    # 模糊匹配（去除前导零）
    pid_norm = pid.lstrip("0") or pid
    for key, val in table_data.items():
        if key.lstrip("0") == pid_norm:
            return dict(val)
    return {}


# ---------------------------------------------------------------------------
# 目录管理
# ---------------------------------------------------------------------------

def read_excel_all(
    file_path: str,
    sheet_index: int = 0,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    读取 Excel 文件的全部数据（或前 max_rows 行），返回与 read_csv_all 相同的格式。

    Args:
        file_path: Excel 文件路径（.xlsx / .xls）
        sheet_index: 工作表索引（默认 0）
        max_rows: 最大读取行数（None 表示全部）

    Returns:
        {
          "success": bool,
          "columns": [列名],
          "rows": [{列名: 值, ...}, ...],
          "row_count": int,
          "sheet_name": str,
          "error": str | None,
        }
    """
    if not os.path.exists(file_path):
        return {"success": False, "columns": [], "rows": [], "row_count": 0,
                "sheet_name": "", "error": f"文件不存在: {file_path}"}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet_name = wb.sheetnames[sheet_index] if sheet_index < len(wb.sheetnames) else wb.sheetnames[0]
        ws = wb[sheet_name]

        rows_iter = ws.iter_rows(values_only=True)
        header_raw = next(rows_iter, None)
        if not header_raw:
            return {"success": False, "columns": [], "rows": [], "row_count": 0,
                    "sheet_name": sheet_name, "error": "Excel 文件无表头"}

        columns = [str(c) if c is not None else f"col_{i}" for i, c in enumerate(header_raw)]
        rows: List[Dict] = []
        for i, raw in enumerate(rows_iter):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(dict(zip(columns, [str(v) if v is not None else "" for v in raw])))

        return {"success": True, "columns": columns, "rows": rows,
                "row_count": len(rows), "sheet_name": sheet_name, "error": None}
    except Exception as e:
        return {"success": False, "columns": [], "rows": [], "row_count": 0,
                "sheet_name": "", "error": str(e)}


def build_output_dir(base_dir: str, prefix: str = "results") -> str:
    """
    在 base_dir 下创建带时间戳的子目录，避免覆盖旧结果。

    Args:
        base_dir: 父目录
        prefix: 子目录名前缀（默认 "results"）

    Returns:
        创建好的子目录路径
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_dir, f"{prefix}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out
