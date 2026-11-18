#!/usr/bin/env python3
"""i48_native_cross_tier.py —— i48 两档的**原生错误格**跨档对比（交集口径）。

为什么必须用交集：两档的原生错误**候选层相同**（reference 都源自同一份 MIMIC 清洗
结果，原生错误只取决于 MIMIC 自己的数据），但**登记层格数不同**
（1.6pct 1,264 / 4.7pct 1,238）—— 差别来自「itemid 也被注入器改过」的格被划归注入层
（7 vs 33）。
⇒ 直接跨档比「原生修复率」是**分母不同**的比法，不严格；
  要严格可比，只看**两档都登记了的交集格**。

对齐键：log 的 `raw_row_index`（两档同源 ⇒ 行号可比；实测交集 1,231 格，
且交集上两档的修复值/原生错值逐格完全一致）。
判定口径：与 `eval_i48_native_cells.py` 完全一致（按**干净 key** 在 agent 输出里查）。
只出聚合（格数 / itemid 分布 / 一致性矩阵），不打印病人级信息。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_i48_native_cells import judge, load_native, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-ds", required=True, help="1.6pct mimicraw2 数据集目录")
    ap.add_argument("--a-result", required=True, help="1.6pct 的 agent 结果包")
    ap.add_argument("--b-ds", required=True, help="4.7pct mimicraw2 数据集目录")
    ap.add_argument("--b-result", required=True, help="4.7pct 的 agent 结果包")
    ap.add_argument("--split", default="test")
    ap.add_argument("--string-compare", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    numeric = not args.string_compare

    dsA, dsB = Path(args.a_ds), Path(args.b_ds)
    print("=== 档 A（1.6pct）===")
    sA, sA_any, natA, metaA = judge(dsA, Path(args.a_result), args.split, numeric)
    print("=== 档 B（4.7pct）===")
    sB, sB_any, natB, metaB = judge(dsB, Path(args.b_result), args.split, numeric)

    ka, kb = set(sA), set(sB)
    both = sorted(ka & kb)

    bA, per_itemA, per_modeA, nA = summarize(sA, natA)
    bB, per_itemB, per_modeB, nB = summarize(sB, natB)

    # 交集上的一致性矩阵
    mat = Counter((sA[L], sB[L]) for L in both)
    mat_any = Counter((sA_any[L], sB_any[L]) for L in both)

    def rate(b, n):
        return round(b.get("b_clean", 0) / n, 6) if n else None

    rep = {
        "split": args.split,
        "compare": "numeric" if numeric else "string",
        "a": {"n": nA, "buckets": dict(bA), "repair_rate": rate(bA, nA),
              "ambiguous_keys": metaA["ambiguous_keys"],
              "agent_rows": metaA["agent_rows"]},
        "b": {"n": nB, "buckets": dict(bB), "repair_rate": rate(bB, nB),
              "ambiguous_keys": metaB["ambiguous_keys"],
              "agent_rows": metaB["agent_rows"]},
        "intersect": {
            "n": len(both),
            "only_a": len(ka - kb),
            "only_b": len(kb - ka),
            "matrix": {"%s|%s" % k: v for k, v in sorted(mat.items())},
            "matrix_any": {"%s|%s" % k: v for k, v in sorted(mat_any.items())},
            "a_repair_rate": rate(Counter(sA[L] for L in both), len(both)),
            "b_repair_rate": rate(Counter(sB[L] for L in both), len(both)),
            "repaired_value_mismatch": sum(
                1 for L in both
                if natA[L].get("repaired_value") != natB[L].get("repaired_value")),
            "native_value_mismatch": sum(
                1 for L in both
                if natA[L].get("mimic_native_value") != natB[L].get("mimic_native_value")),
            "both_b_clean": sum(1 for L in both if sA[L] == "b_clean" and sB[L] == "b_clean"),
            "both_b_dirty": sum(1 for L in both if sA[L] == "b_dirty" and sB[L] == "b_dirty"),
            "a_clean_b_not": sum(1 for L in both
                                 if sA[L] == "b_clean" and sB[L] != "b_clean"),
            "b_clean_a_not": sum(1 for L in both
                                 if sB[L] == "b_clean" and sA[L] != "b_clean"),
        },
        "per_detection_mode": {"a": {k: dict(v) for k, v in per_modeA.items()},
                               "b": {k: dict(v) for k, v in per_modeB.items()}},
    }

    print("\n=== 交集口径（%s，%s）===" % (args.split, rep["compare"]))
    print("  A 登记 %d 格 / B 登记 %d 格 / 交集 %d 格（只在A %d、只在B %d）"
          % (nA, nB, len(both), len(ka - kb), len(kb - ka)))
    print("  A 全档修复率 %.1f%%   B 全档修复率 %.1f%%"
          % (100 * (rep["a"]["repair_rate"] or 0), 100 * (rep["b"]["repair_rate"] or 0)))
    print("  交集上：A %.1f%%   B %.1f%%"
          % (100 * (rep["intersect"]["a_repair_rate"] or 0),
             100 * (rep["intersect"]["b_repair_rate"] or 0)))
    print("  交集一致性矩阵（A|B → 格数）:")
    for k, v in sorted(mat.items()):
        print("    %-24s %d" % (k, v))
    print("  两档都修对 %d / 两档都没动 %d / A修对B没修 %d / B修对A没修 %d"
          % (rep["intersect"]["both_b_clean"], rep["intersect"]["both_b_dirty"],
             rep["intersect"]["a_clean_b_not"], rep["intersect"]["b_clean_a_not"]))
    print("  交集上两档修复值不一致 %d / 原生错值不一致 %d"
          % (rep["intersect"]["repaired_value_mismatch"],
             rep["intersect"]["native_value_mismatch"]))

    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print("\n写出:", args.out)


if __name__ == "__main__":
    main()
