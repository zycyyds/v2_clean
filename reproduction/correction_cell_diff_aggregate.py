#!/usr/bin/env python3
"""correction_cell_diff_aggregate.py —— 读 `host_private/cell_diff.jsonl`，
**只输出聚合统计**（每列 / 每 split / 每种破坏形态的计数）。

⚠️ 这个文件是 host 专用的逐 cell 差分档案（agent 不可读），
里面含行级数据 ⇒ **只允许聚合计数进转录**，绝不打印任何单元格的值或行标识。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True, help="host_private/cell_diff.jsonl")
    ap.add_argument("--result", default=None, help="agent 结果表（可选，用于算修好率）")
    ap.add_argument("--gold", default=None, help="gold 表（可选）")
    ap.add_argument("--key", default="index")
    args = ap.parse_args()

    rows = []
    with open(args.diff, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("档案为空")
        return
    print("=== 差分档案结构 ===")
    print("  条目数: %d" % len(rows))
    print("  字段  : %s" % sorted(rows[0].keys()))

    def bucket(name):
        return Counter(str(r.get(name, "")) for r in rows)

    for name in sorted(rows[0].keys()):
        vals = bucket(name)
        if len(vals) <= 24:
            print()
            print("  %s 的取值分布:" % name)
            for k, v in vals.most_common():
                print("    %-34s %d" % (k[:34], v))

    # 若给了结果表与 gold，算「修好率」
    if args.result and args.gold:
        res = {r[args.key]: r for r in csv.DictReader(
            open(args.result, newline="", encoding="utf-8"))}
        gold = {r[args.key]: r for r in csv.DictReader(
            open(args.gold, newline="", encoding="utf-8"))}
        cols = [c for c in next(iter(gold.values())).keys() if c != args.key]
        per_col = Counter()
        total = wrong = 0
        for k, g in gold.items():
            r = res.get(k)
            if r is None:
                continue
            for c in cols:
                total += 1
                if (r.get(c) or "") != (g.get(c) or ""):
                    wrong += 1
                    per_col[c] += 1
        print()
        print("=== agent 结果 vs gold（逐格）===")
        print("  可比 cell %d / 错 %d（准确率 %.4f）" % (total, wrong, 1 - wrong / total if total else 0))
        for c, n in per_col.most_common():
            print("    %-24s %d" % (c, n))


if __name__ == "__main__":
    main()
