#!/usr/bin/env python3
"""i48_native_overlap.py —— 算 i48 两档的原生错误格**交集/差集**。

为什么需要：两档的原生错误**候选层相同**（reference 都源自同一份 MIMIC 清洗结果，
原生错误只取决于 MIMIC 自己的数据），但**最终登记进 native 层的格数不同**
（1.6pct 1,264 / 4.7pct 1,238）—— 差别来自「itemid 也被注入器改过」的格被划归注入层
（7 vs 33），以及叠脏格的处理。
⇒ 直接跨档比「原生修复率」是**分母不同**的比法，不严格；
  要严格可比，只看**两档都登记了的交集格**。

对齐键：log 的 `raw_row_index`（两档同源 ⇒ 行号可比）。
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
    ap.add_argument("--a", required=True, help="1.6pct mimicraw2 数据集目录")
    ap.add_argument("--b", required=True, help="4.7pct mimicraw2 数据集目录")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A = load(Path(args.a), args.split)
    B = load(Path(args.b), args.split)
    ka, kb = set(A), set(B)
    both = ka & kb
    only_a, only_b = ka - kb, kb - ka

    def items(keys, src):
        return Counter(src[k].get("itemid", "") for k in keys)

    res = {
        "split": args.split,
        "a_n": len(ka), "b_n": len(kb),
        "intersect": len(both),
        "only_a": len(only_a), "only_b": len(only_b),
        "intersect_itemids": dict(items(both, A).most_common()),
        "only_a_itemids": dict(items(only_a, A).most_common()),
        "only_b_itemids": dict(items(only_b, B).most_common()),
        # 交集格里两档的修复值是否一致（应当完全一致 —— 同一份 MIMIC 数据）
        "intersect_repaired_mismatch": sum(
            1 for k in both
            if A[k].get("repaired_value") != B[k].get("repaired_value")),
        "intersect_native_value_mismatch": sum(
            1 for k in both
            if A[k].get("mimic_native_value") != B[k].get("mimic_native_value")),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print("写出:", args.out)


if __name__ == "__main__":
    main()
