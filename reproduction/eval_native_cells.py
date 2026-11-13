#!/usr/bin/env python3
"""eval_native_cells.py —— 只针对「MIMIC 原生错误格」统计修复情况。

v29 的 compute_error_metrics.py 是**按文件**汇总的，混在 1414 万条注入错误里，
看不出「MIMIC 原生错误这一层修了多少」。这个脚本把 modification_log 里
error_subtype 以 mimic_native 开头的条目单独拎出来，逐格判定：

  b_clean  ✓ agent 的值 == 修复值（同患者同 itemid 的 R2 中位数）⇒ 修对了
  b_wrong  ✗ agent 改了，但改的不是修复值
  b_dirty  ✗ agent 没动这一格（值仍是脏值）⇒ 漏检
  b_missing✗ agent 输出里找不到这一行

只输出聚合，不含病人级内容。

用法：
  python3 eval_native_cells.py --dataset <数据集目录> --result <agent 结果包> \
      [--split test] [--out <json>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)
CE = "icu/chartevents.csv"
OUT_CE = "icu/chartevents.csv"
KEY = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def same(a, b):
    if a is None or b is None:
        return str(a) == str(b)
    fa, fb = fnum(a), fnum(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-9
    return str(a) == str(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result", required=True, help="agent 的结果包目录")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds = Path(args.dataset)
    res = Path(args.result)
    sp = args.split

    # ---- 1. log 里的原生错误条目 ----
    nat = {}
    with (ds / sp / "modification_log.csv").open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if (e.get("raw_file") == CE and e.get("column") == "valuenum"
                    and e.get("operation") == "cell_update"
                    and e.get("error_subtype", "").startswith("mimic_native")):
                nat[int(e["raw_row_index"])] = {
                    "clean": e.get("clean_value"),
                    "dirty": e.get("dirty_value"),
                    "subtype": e.get("error_subtype"),
                }
    if not nat:
        sys.exit("log 里没有 mimic_native_* 条目")
    print("原生错误条目（log）: %d" % len(nat))

    # ---- 2. gold：reference 里对应行的 key + 值 ----
    gold = {}
    with (ds / sp / "reference" / CE).open(newline="", encoding="utf-8") as f:
        for L, r in enumerate(csv.DictReader(f)):
            if L in nat:
                gold[L] = {
                    "key": tuple(r.get(k, "") for k in KEY),
                    "itemid": r.get("itemid"),
                    "value": r.get("valuenum"),
                }
    missing_gold = [L for L in nat if L not in gold]
    print("gold 行定位: %d / %d（缺失 %d）" % (len(gold), len(nat), len(missing_gold)))

    # ---- 3. agent 输出：按 key 匹配 ----
    want = {}
    for L, g in gold.items():
        want.setdefault(g["key"], []).append(L)
    agent_val = {}
    n_rows = 0
    res_ce = res / OUT_CE
    if not res_ce.is_file():
        cands = list(res.rglob("chartevents.csv"))
        if not cands:
            sys.exit("结果包里找不到 chartevents.csv")
        res_ce = cands[0]
        print("（按 glob 找到）", res_ce)
    with res_ce.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            k = tuple(r.get(c, "") for c in KEY)
            for L in want.get(k, ()):
                if L not in agent_val:
                    agent_val[L] = r.get("valuenum")
    print("agent 输出行数 %d，命中 %d / %d 格" % (n_rows, len(agent_val), len(gold)))

    # ---- 4. 逐格判定 ----
    buckets = Counter()
    per_item = defaultdict(Counter)
    per_sub = defaultdict(Counter)
    for L, g in gold.items():
        e = nat.get(L)
        if e is None:
            continue
        if L not in agent_val:
            b = "b_missing"
        elif same(agent_val[L], e["clean"]):
            b = "b_clean"
        elif same(agent_val[L], e["dirty"]):
            b = "b_dirty"
        else:
            b = "b_wrong"
        buckets[b] += 1
        per_item[g["itemid"]][b] += 1
        per_sub[e["subtype"]][b] += 1

    n = sum(buckets.values())
    fixed = buckets["b_clean"]
    report = {
        "dataset": str(ds),
        "result": str(res),
        "split": sp,
        "n_native_cells": n,
        "buckets": dict(buckets),
        "repair_rate": round(fixed / n, 6) if n else None,
        "detection_rate": round((n - buckets["b_dirty"] - buckets["b_missing"]) / n, 6) if n else None,
        "per_itemid": {k: dict(v) for k, v in sorted(per_item.items(), key=lambda x: -sum(x[1].values()))},
        "per_subtype": {k: dict(v) for k, v in per_sub.items()},
    }
    print("\n=== MIMIC 原生错误格修复情况 ===")
    print("  总格数 %d" % n)
    for b in ("b_clean", "b_wrong", "b_dirty", "b_missing"):
        print("  %-10s %5d  (%.1f%%)" % (b, buckets[b], 100.0 * buckets[b] / max(1, n)))
    print("  修复率 (== clean) : %.1f%%" % (100.0 * fixed / max(1, n)))
    print("\n  按 itemid:")
    for it, c in list(report["per_itemid"].items())[:14]:
        tot = sum(c.values())
        print("    %-8s 合计 %4d  修对 %4d  改错 %4d  未动 %4d  缺行 %4d"
              % (it, tot, c.get("b_clean", 0), c.get("b_wrong", 0),
                 c.get("b_dirty", 0), c.get("b_missing", 0)))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n写出:", args.out)


if __name__ == "__main__":
    main()
