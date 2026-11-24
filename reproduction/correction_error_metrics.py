#!/usr/bin/env python3
"""correction_error_metrics.py —— 校正类（单表纠错）的**错误级指标**：
「检测」与「修复」的 P / R / F1，**口径与 MIMIC 那一族完全一致**。

## 校正类的评估为什么可以直接比对

MIMIC 那一族要靠 `modification_log.csv`（注入器的答案键）反推「哪一格本该被改」；
校正类**有 `host_private/cell_diff.jsonl`**（host 专用的逐 cell 差分档案）：
每条 = (index, column, clean_value, dirty_value, split)
⇒ 直接就是「这一格脏值是什么、正确值是什么」。

于是：
  b_clean  = agent 的值 == clean_value（**修对**）
  b_dirty  = agent 的值 == dirty_value（**没动**）
  b_wrong  = agent 的值既不是 clean 也不是 dirty（**改错**）
  b_missing= agent 输出里没有这一行（**缺行**）
  fp_det   = **干净 cell 被改坏**的格数（改坏干净 cell = 误报检测）

## 指标公式（与 score_only/error_metrics/compute_error_metrics.py 同款）

  tp_det = b_clean + b_wrong
  fn_det = b_dirty + b_missing
  tp_rep = b_clean
  fp_rep = b_wrong
  fp_det = 改坏的干净 cell 数
  P_det = tp_det/(tp_det+fp_det)   R_det = tp_det/(tp_det+fn_det)
  P_rep = tp_rep/(tp_det+fp_det)   R_rep = tp_rep/(tp_rep+fp_rep+fn_det)
  F1    = 2PR/(P+R)

只输出聚合统计（不打印任何单元格的值或行标识）。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)


def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True, help="host_private/cell_diff.jsonl")
    ap.add_argument("--result", required=True, help="agent 结果表")
    ap.add_argument("--gold", required=True, help="参考表")
    ap.add_argument("--split", required=True, help="diff 里的 split 名")
    ap.add_argument("--key", default="index")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    broken = []
    with open(args.diff, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("split") == args.split:
                    broken.append(d)

    res = {r[args.key]: r for r in load_csv(args.result)}
    gold = {r[args.key]: r for r in load_csv(args.gold)}
    cols = [c for c in next(iter(gold.values())).keys() if c != args.key]

    b_clean = b_dirty = b_wrong = b_missing = 0
    broken_cells = set()
    for c in broken:
        key = str(c.get(args.key))
        col = c.get("column", "")
        broken_cells.add((key, col))
        row = res.get(key)
        if row is None:
            b_missing += 1
            continue
        got = row.get(col, "")
        if got == c.get("clean_value", ""):
            b_clean += 1
        elif got == c.get("dirty_value", ""):
            b_dirty += 1
        else:
            b_wrong += 1

    # fp_det = 干净 cell 被改坏的格数
    fp_det = 0
    for key, g in gold.items():
        r = res.get(key)
        if r is None:
            continue
        for col in cols:
            if (key, col) in broken_cells:
                continue
            if (r.get(col) or "") != (g.get(col) or ""):
                fp_det += 1

    tp_det = b_clean + b_wrong
    fn_det = b_dirty + b_missing
    tp_rep = b_clean
    fp_rep = b_wrong
    P_det = tp_det / (tp_det + fp_det) if (tp_det + fp_det) else 0.0
    R_det = tp_det / (tp_det + fn_det) if (tp_det + fn_det) else 0.0
    P_rep = tp_rep / (tp_det + fp_det) if (tp_det + fp_det) else 0.0
    R_rep = tp_rep / (tp_rep + fp_rep + fn_det) if (tp_rep + fp_rep + fn_det) else 0.0

    print("=== 校正类错误级指标【%s】%s ===" % (args.label or args.split, args.split))
    print("  行数: agent %d / gold %d" % (len(res), len(gold)))
    print("  坏 cell %d 个（干净 cell %d 个）" % (len(broken), len(gold) * len(cols) - len(broken)))
    print("  桶: b_clean=%d  b_dirty=%d  b_wrong=%d  b_missing=%d  fp_det(改坏干净 cell)=%d"
          % (b_clean, b_dirty, b_wrong, b_missing, fp_det))
    print()
    print("  检测 Det : P %.6f  R %.6f  F1 %.6f" % (P_det, R_det, f1(P_det, R_det)))
    print("  修复 Rep : P %.6f  R %.6f  F1 %.6f" % (P_rep, R_rep, f1(P_rep, R_rep)))
    print()
    print("  (tp_det=%d fn_det=%d tp_rep=%d fp_rep=%d fp_det=%d)"
          % (tp_det, fn_det, tp_rep, fp_rep, fp_det))


if __name__ == "__main__":
    main()
