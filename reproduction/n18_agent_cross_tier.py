#!/usr/bin/env python3
"""n18_agent_cross_tier.py —— N18 两档的 **agent 级**交叉验证。

比的是什么：两档数据集的原生错误格**交集**（按完整 5 段 key 对齐）上，
**两个独立跑出来的 agent**（1.6pct 与 4.7pct，两次独立的 val 训练 + test 重放）
对同一批格给出的**判断是否一致**。

为什么必须看交集而不是直接比修复率：
两档的原生层登记格数不同（1.6pct 715 / 4.7pct 586）——
分母不同（有些格因「itemid 也被注入器改过」被划归注入层）⇒ 直接比是**分母不同的比法**。
⇒ 交集才是严格可比的。

判定口径与 `eval_n18_native_cells.py` 完全一致：
  b_clean  = agent 的值 == 修复值（repaired_value）
  b_dirty  = agent 的值 == 原生错值（mimic_native_value）
  b_wrong  = 其它
  b_missing= agent 输出里按这个 key 找不到这一行

只输出聚合，不含病人级标识。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)

KEY5 = ("subject_id", "hadm_id", "stay_id", "charttime", "itemid")


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_of(r):
    return tuple(r.get(c, "") for c in KEY5)


def native_map(ds: Path, split: str):
    """{完整5段key: 溯源记录}"""
    p = ds / split / "mimic_native_error_log.csv"
    out = {}
    for r in load(p):
        out.setdefault(key_of(r), r)
    return out


def agent_map(res: Path):
    """{完整5段key: [agent 的 valuenum, ...]}"""
    p = res / "icu" / "chartevents.csv"
    out = {}
    for r in load(p):
        out.setdefault(key_of(r), []).append(r.get("valuenum"))
    return out


def verdict(vals, rec):
    """与 eval_n18_native_cells.py 同口径（主口径 = 该 key 的第一行）。"""
    if not vals:
        return "b_missing"
    gv = num(vals[0])
    if gv is None:
        return "b_missing"
    rep, dirty = num(rec.get("repaired_value")), num(rec.get("mimic_native_value"))
    if rep is not None and abs(gv - rep) < 1e-9:
        return "b_clean"
    if dirty is not None and abs(gv - dirty) < 1e-9:
        return "b_dirty"
    return "b_wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-ds", required=True, help="1.6pct mimicraw 数据集")
    ap.add_argument("--a-res", required=True, help="1.6pct 的 agent test 结果包")
    ap.add_argument("--b-ds", required=True, help="4.7pct mimicraw 数据集")
    ap.add_argument("--b-res", required=True, help="4.7pct 的 agent test 结果包")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A_ds, A_res = Path(args.a_ds), Path(args.a_res)
    B_ds, B_res = Path(args.b_ds), Path(args.b_res)
    sp = args.split

    na, nb = native_map(A_ds, sp), native_map(B_ds, sp)
    aa, ab = agent_map(A_res), agent_map(B_res)
    ka, kb = set(na), set(nb)
    both = ka & kb

    va = {k: verdict(aa.get(k), na[k]) for k in ka}
    vb = {k: verdict(ab.get(k), nb[k]) for k in kb}

    pair = Counter()
    for k in both:
        pair[(va[k], vb[k])] += 1

    both_clean = sum(1 for k in both if va[k] == "b_clean" and vb[k] == "b_clean")
    a_only = sum(1 for k in both if va[k] == "b_clean" and vb[k] != "b_clean")
    b_only = sum(1 for k in both if vb[k] == "b_clean" and va[k] != "b_clean")
    neither = len(both) - both_clean - a_only - b_only

    # 真值本身是否一致（两档数据集独立构建出来的修复值）
    rep_mismatch = sum(1 for k in both
                       if na[k].get("repaired_value") != nb[k].get("repaired_value"))
    dirty_mismatch = sum(1 for k in both
                         if na[k].get("mimic_native_value") != nb[k].get("mimic_native_value"))

    res = {
        "split": sp,
        "align_key": list(KEY5),
        "a_cells": len(ka), "b_cells": len(kb),
        "intersect": len(both),
        "a_only": len(ka - kb), "b_only": len(kb - ka),
        "verdict_pairs": {"%s|%s" % k: v for k, v in sorted(pair.items())},
        "both_clean": both_clean, "a_only_clean": a_only,
        "b_only_clean": b_only, "neither_clean": neither,
        "agreement_rate": round((both_clean + neither) / len(both), 6) if both else None,
        "truth_repaired_mismatch": rep_mismatch,
        "truth_native_value_mismatch": dirty_mismatch,
        "a_full": dict(Counter(va.values())),
        "b_full": dict(Counter(vb.values())),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print()
    print("=== 交集 %d 格上，两次独立 agent 的判断 ===" % len(both))
    print("  两档都修对           : %d" % both_clean)
    print("  只有 A 修对          : %d" % a_only)
    print("  只有 B 修对          : %d" % b_only)
    print("  两档都没修对         : %d" % neither)
    print("  判断一致率           : %.4f" % res["agreement_rate"])
    print("  真值不一致（修复值） : %d 格" % rep_mismatch)
    print("  真值不一致（原生错值）: %d 格" % dirty_mismatch)
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print("\n写出:", args.out)


if __name__ == "__main__":
    main()
