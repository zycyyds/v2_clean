#!/usr/bin/env python3
"""eval_n18_native_cells.py —— N18 的 **MIMIC 原生错误格**专项判定。

N18 与 i48 同为 **cohort 型**任务（reference = cohort 筛选后的输出表），
所以对齐方式与 i48 完全一致：**按 5 段 key 定位**
（subject_id / hadm_id / stay_id / charttime / itemid），
不能用 log 的 raw_row_index（那是源 raw 的行号空间）。

⇒ 直接复用 eval_i48_native_cells.py 的实现（同一套判定口径，避免两处漂移），
   这里只做一层薄包装，把 --label 设成 n18 并强制 --string-compare
   （原生错误维度自 2026-11-20 起只保留 string 口径）。

只输出聚合，不含病人级标识。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_i48_native_cells as BASE  # noqa: E402


def main():
    argv = sys.argv[1:]
    if "--label" not in argv:
        argv += ["--label", "n18"]
    sys.argv = [sys.argv[0]] + argv
    BASE.main()


if __name__ == "__main__":
    main()
