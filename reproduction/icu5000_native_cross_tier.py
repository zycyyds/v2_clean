#!/usr/bin/env python3
"""icu5000_native_cross_tier.py —— icu5000 两档的**原生错误格**跨档对比。

为什么必须用交集：两档的原生错误**候选层**相同（reference 是同一份干净 MIMIC，
注入器只改 raw），但**登记层格数不同**（1.6pct 2,450 / 4.7pct 2,174）——
差别来自「itemid 也被注入器改过」的格被划归注入层（141 vs 417）。
⇒ 直接比「原生修复率」是分母不同的比法，不严格；严格可比只看**两档都登记的交集格**。

对齐键：reference 的行号 L（两档 reference 相同 ⇒ L 可比）。

输出：
  * 交集规模 + 每档在交集上的 b_clean/b_wrong/b_dirty/b_missing
  * 两档逐格**一致性矩阵**（都修对 / 只有这档修对 / 都没修对 …）
  * 分歧格的 itemid 分布（聚合，不含病人级信息）

用法：
  python3 icu5000_native_cross_tier.py \
      --a-ds <1.6pct数据集> --a-result <1.6pct结果包> \
      --b-ds <4.7pct数据集> --b-result <4.7pct结果包> \
      [--split test] [--out out.json] [--string-compare]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_native_cells import judge_cells  # noqa: E402

BUCKETS = ("b_clean", "b_wrong", "b_dirty", "b_missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-ds", required=True, help="1.6pct mimicraw 数据集目录")
    ap.add_argument("--a-result", required=True, help="1.6pct agent 结果包")
    ap.add_argument("--b-ds", required=True, help="4.7pct mimicraw 数据集目录")
    ap.add_argument("--b-result", required=True, help="4.7pct agent 结果包")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--string-compare", action="store_true",
                    help="纯字符串比对（老口径）；默认数值感知")
    args = ap.parse_args()
    numeric = not args.string_compare
    sp = args.split

    print("=== A (1.6pct) ===")
    sa, ga, na = judge_cells(Path(args.a_ds), Path(args.a_result), sp, numeric)
    print("=== B (4.7pct) ===")
    sb, gb, nb = judge_cells(Path(args.b_ds), Path(args.b_result), sp, numeric)

    ka, kb = set(sa), set(sb)
    both = sorted(ka & kb)
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)

    def buckets_of(status, keys):
        c = Counter(status[k] for k in keys if k in status)
        return {b: c.get(b, 0) for b in BUCKETS}

    # ---- 交集上的一致性矩阵 ----
    joint = Counter((sa[k], sb[k]) for k in both)
    clean_a = {k for k in both if sa[k] == "b_clean"}
    clean_b = {k for k in both if sb[k] == "b_clean"}
    both_clean = clean_a & clean_b
    only_a_clean = clean_a - clean_b
    only_b_clean = clean_b - clean_a
    neither = set(both) - clean_a - clean_b

    # 分歧格的 itemid 分布（聚合）
    diff_items = Counter()
    for k in only_a_clean:
        diff_items[ga[k]["itemid"]] += 1
    for k in only_b_clean:
        diff_items[gb[k]["itemid"]] += 1

    # 两档在交集上的修复值是否一致（应当完全一致）
    val_mismatch = sum(1 for k in both if na[k]["clean"] != nb[k]["clean"])

    report = {
        "split": sp,
        "compare": "numeric" if numeric else "string",
        "a": {"dataset": str(args.a_ds), "n_cells": len(ka)},
        "b": {"dataset": str(args.b_ds), "n_cells": len(kb)},
        "intersect": len(both),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "intersect_clean_value_mismatch": val_mismatch,
        "a_on_intersect": buckets_of(sa, both),
        "b_on_intersect": buckets_of(sb, both),
        "a_full": buckets_of(sa, sorted(ka)),
        "b_full": buckets_of(sb, sorted(kb)),
        "agreement": {
            "both_clean": len(both_clean),
            "only_a_clean": len(only_a_clean),
            "only_b_clean": len(only_b_clean),
            "neither_clean": len(neither),
        },
        "joint_matrix": {"%s|%s" % k: v for k, v in sorted(joint.items())},
        "disagree_itemids": dict(diff_items.most_common()),
    }

    print("\n=== 跨档对比（交集 %d 格）===" % len(both))
    for k in ("a_on_intersect", "b_on_intersect"):
        b = report[k]
        n = len(both)
        print("  %-16s clean %5d (%.1f%%)  wrong %4d  dirty %4d  missing %4d"
              % (k, b["b_clean"], 100.0 * b["b_clean"] / max(1, n),
                 b["b_wrong"], b["b_dirty"], b["b_missing"]))
    ag = report["agreement"]
    n = len(both)
    print("  一致性：都修对 %d (%.1f%%) / 只 A 修对 %d / 只 B 修对 %d / 都没修对 %d"
          % (ag["both_clean"], 100.0 * ag["both_clean"] / max(1, n),
             ag["only_a_clean"], ag["only_b_clean"], ag["neither_clean"]))
    if diff_items:
        print("  分歧 itemid top:", dict(list(diff_items.items())[:10]))

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print("写出:", args.out)


if __name__ == "__main__":
    main()
