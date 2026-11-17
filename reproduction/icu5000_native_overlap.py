#!/usr/bin/env python3
"""icu5000_native_overlap.py —— 算 icu5000 两档的原生错误格**交集/差集**。

为什么需要它：icu5000 两档的原生错误**候选层**完全相同（reference 是同一份干净的
原始 MIMIC，注入器只改 raw），但**最终登记进 native 层的格数不同**
（1.6pct 2,450 / 4.7pct 2,174）—— 差别来自「itemid 也被注入器改过」被划归注入层的
那批格（141 / 417）。

⇒ 直接跨档比「原生错误修复率」是**分母不同**的比法，不严格。
  要严格可比，只看**两档都登记了的交集格**。

对齐键：log 的 `raw_row_index` = **reference 的行号**（两档的 reference 相同 ⇒ 行号可比）。
只输出聚合（格数 / itemid 分布），不打印病人级信息。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)


def load(ds: Path, split: str):
    p = ds / split / "mimic_native_error_log.csv"
    if not p.is_file():
        return {}
    out = {}
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["raw_row_index"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="1.6pct mimicraw 数据集目录")
    ap.add_argument("--b", required=True, help="4.7pct mimicraw 数据集目录")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A = load(Path(args.a), args.split)
    B = load(Path(args.b), args.split)
    ka, kb = set(A), set(B)
    both = ka & kb
    only_a = ka - kb
    only_b = kb - ka

    def items(keys, src):
        return Counter(src[k]["itemid"] for k in keys)

    res = {
        "split": args.split,
        "a_n": len(ka), "b_n": len(kb),
        "intersect": len(both),
        "only_a": len(only_a), "only_b": len(only_b),
        "intersect_itemids": dict(items(both, A).most_common()),
        "only_a_itemids": dict(items(only_a, A).most_common()),
        "only_b_itemids": dict(items(only_b, B).most_common()),
        # 交集格里两档的修复值是否一致（reference 相同 ⇒ 应当完全一致）
        "intersect_repaired_mismatch": sum(
            1 for k in both
            if A[k]["repaired_value"] != B[k]["repaired_value"]),
        "intersect_native_value_mismatch": sum(
            1 for k in both
            if A[k]["mimic_native_value"] != B[k]["mimic_native_value"]),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print("写出:", args.out)


if __name__ == "__main__":
    main()
