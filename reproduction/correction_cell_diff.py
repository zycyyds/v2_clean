#!/usr/bin/env python3
"""correction_cell_diff.py —— 把 agent 交付的表和 gold **逐格**比一遍，
看「哪些 cell 被修对了、哪些没修对」。

用途：校正类跑完后定位「val 分 vs 独立重放分」的差距来源
（交付表 vs pipeline 重跑出来的表 不一致 ⇒ 重放分低）。

只用 5 段 key（校正类是单表，key 就是 `index`）对齐；
只输出聚合统计 + **按列**的错格分布（不打印任何单元格的值）。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)


def load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="agent 的结果表（csv）")
    ap.add_argument("--gold", required=True, help="gold 表（csv）")
    ap.add_argument("--key", default="index", help="主键列")
    args = ap.parse_args()

    res = load(Path(args.result))
    gold = load(Path(args.gold))
    rk = {r[args.key]: r for r in res}
    gk = {r[args.key]: r for r in gold}

    print("=== 行数 ===")
    print("  agent: %d 行 / gold: %d 行" % (len(res), len(gold)))
    missing = set(gk) - set(rk)
    extra = set(rk) - set(gk)
    print("  agent 缺的行: %d / 多出的行: %d" % (len(missing), len(extra)))

    cols = [c for c in (gold[0].keys() if gold else []) if c != args.key]
    per_col = Counter()
    per_col_cells = Counter()
    total = 0
    wrong = 0
    for key, g in gk.items():
        r = rk.get(key)
        if r is None:
            continue
        for c in cols:
            total += 1
            if (r.get(c) or "") != (g.get(c) or ""):
                wrong += 1
                per_col[c] += 1
    print()
    print("=== 逐格（对齐到 key）===")
    print("  可比 cell: %d / 错: %d（准确率 %.4f）" % (total, wrong, 1 - wrong / total if total else 0))
    print()
    print("  按列的错格数（只列有错的列）:")
    for c, n in per_col.most_common():
        print("    %-24s %d" % (c, n))


if __name__ == "__main__":
    main()
