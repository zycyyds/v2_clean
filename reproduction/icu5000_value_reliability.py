#!/usr/bin/env python3
"""icu5000_value_reliability.py —— 查清「raw 的 value 文本列 vs log 的 clean_value」
在 icu5000 上到底差在哪。

背景：K29 的 prompt 里写着「注入类错误的 raw `value` 文本列记的就是那个正确值，
实测 unit_scale_error 99.6% 的格里 value 直接等于 clean_value ⇒ 修这类错误时直接
采用 value 比 ÷100 可靠」。

icu5000 的抽样统计只给出 **29.8%** ⇒ 那句结论在 icu5000 上**可能不成立**，
必须查清剩下的 70% 是什么形态，否则 prompt 会误导 agent。

分类：
  eq_str      : value == clean_value（字符串完全相同）
  eq_num      : 数值相等、字符串不同（如 "87" vs "87.0"）
  num_x100    : value 的数值 == clean_value × 100
  num_x010    : value 的数值 == clean_value ÷ 100
  other_num   : 数值上是别的数（脏值本身）
  not_num     : value 不是数值（带单位 / 文本 / 符号）

只输出聚合分类计数与若干**去标识化**的形态示例（数值形态，不含任何病人标识）。
"""
from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)
CE = "icu/chartevents.csv"
SAMPLE = 20000


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=SAMPLE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds = Path(args.dataset)
    log_p = ds / "test" / "modification_log.csv"
    raw_p = ds / "test" / "raw" / CE

    want = {}
    tot = Counter()
    picked = Counter()
    ins_pos = []
    with log_p.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e.get("operation") in ("insert", "duplicate"):
                try:
                    ins_pos.append(int(e["raw_row_index"]))
                except (TypeError, KeyError):
                    pass
            if e.get("column") != "valuenum":
                continue
            st = e.get("error_subtype", "")
            if st not in ("unit_scale_error", "sentinel_chart_value"):
                continue
            tot[st] += 1
            if picked[st] >= args.sample:
                continue
            try:
                L = int(e["raw_row_index"])
            except (TypeError, KeyError):
                continue
            picked[st] += 1
            want.setdefault(L, []).append((st, e.get("clean_value", ""),
                                          e.get("dirty_value", "")))
    ins_pos.sort()

    need = {}
    for L in want:
        need[L + bisect_left(ins_pos, L)] = L

    vals = {}
    with raw_p.open(newline="", encoding="utf-8") as f:
        for ri, row in enumerate(csv.DictReader(f)):
            L = need.get(ri)
            if L is not None:
                vals[L] = row.get("value")

    res = {}
    shapes = {}
    for st in ("unit_scale_error", "sentinel_chart_value"):
        cls = Counter()
        ex = []
        for L, lst in want.items():
            for s, cv, dv in lst:
                if s != st:
                    continue
                v = vals.get(L)
                if v is None:
                    cls["行没定位到"] += 1
                    continue
                v = v.strip()
                if v == "":
                    cls["空字符串"] += 1
                    continue
                if v == cv.strip():
                    cls["eq_str（value == clean_value）"] += 1
                    continue
                fv, fc = fnum(v), fnum(cv)
                if fv is None:
                    cls["not_num（value 不是数值）"] += 1
                    if len(ex) < 8:
                        ex.append({"kind": "not_num", "value": v[:40]})
                    continue
                if fc is not None and abs(fv - fc) < 1e-9:
                    cls["eq_num（数值相同、写法不同）"] += 1
                    if len(ex) < 8:
                        ex.append({"kind": "eq_num", "value": v[:40],
                                   "clean": cv[:40]})
                    continue
                if fc:
                    if abs(fv - fc * 100) < 1e-6:
                        cls["num_x100（value == clean×100）"] += 1
                        if len(ex) < 8:
                            ex.append({"kind": "num_x100", "value": v[:40],
                                       "clean": cv[:40]})
                        continue
                    if fc != 0 and abs(fv - fc / 100) < 1e-6:
                        cls["num_x010（value == clean÷100）"] += 1
                        if len(ex) < 8:
                            ex.append({"kind": "num_x010", "value": v[:40],
                                       "clean": cv[:40]})
                        continue
                cls["other_num（别的数值）"] += 1
                if len(ex) < 8:
                    ex.append({"kind": "other_num", "value": v[:40],
                               "clean": cv[:40], "dirty": dv[:40]})
        res[st] = {"total_in_log": tot[st], "sampled": picked[st],
                   "classes": dict(cls.most_common())}
        shapes[st] = ex

    print(json.dumps(res, ensure_ascii=False, indent=1))
    for st, ex in shapes.items():
        print("\n=== %s 的形态示例（去标识化，只看数值形态）===" % st)
        for e in ex:
            print("  ", e)
    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": res, "examples": shapes},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print("写出:", args.out)


if __name__ == "__main__":
    main()
