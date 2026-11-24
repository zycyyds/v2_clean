#!/usr/bin/env python3
"""correction_bucket_by_column.py —— 校正类的坏 cell **按列**拆三桶。

在 `correction_broken_cell_attribution.py` 的基础上再拆一层：
把「修对 / 没动 / 改错」按**列**摊开，并单独列出
「哪些列的**干净 cell 被改坏**」（fp_det）。

用途：判断 agent 的失效是「某一列整体没学会」还是「某一列过泛化」
（rayyan 的两个结论就是靠它出来的：author_list 改错 36、journal_issn 改坏 13 个干净 cell）。

⚠️ 只输出**列名 + 聚合计数**，绝不打印任何单元格的值或行标识。

用法:
  python3 correction_bucket_by_column.py <cell_diff.jsonl> <agent结果.csv> <gold.csv> <split> [key]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

csv.field_size_limit(2 ** 31)


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(2)
    diff_path, result_path, gold_path, split = sys.argv[1:5]
    key = sys.argv[5] if len(sys.argv) > 5 else "index"

    broken = []
    with open(diff_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("split") == split:
                    broken.append(d)

    res = {r[key]: r for r in load(result_path)}
    gold = {r[key]: r for r in load(gold_path)}
    cols = [c for c in next(iter(gold.values())).keys() if c != key]

    buckets = defaultdict(lambda: {"clean": 0, "dirty": 0, "wrong": 0})
    for c in broken:
        row = res.get(str(c.get(key)))
        if row is None:
            continue
        b = buckets[c.get("column", "")]
        got = row.get(c.get("column", ""), "")
        if got == c.get("clean_value", ""):
            b["clean"] += 1
        elif got == c.get("dirty_value", ""):
            b["dirty"] += 1
        else:
            b["wrong"] += 1

    print("=== %s：按列的三桶（修对 / 没动 / 改错）===" % split)
    print("%-22s %6s %6s %6s %6s" % ("column", "坏cell", "修对", "没动", "改错"))
    tot = {"clean": 0, "dirty": 0, "wrong": 0}
    for col in sorted(buckets, key=lambda c: -(buckets[c]["dirty"] + buckets[c]["wrong"])):
        b = buckets[col]
        for k in tot:
            tot[k] += b[k]
        print("%-22s %6d %6d %6d %6d"
              % (col, b["clean"] + b["dirty"] + b["wrong"], b["clean"], b["dirty"], b["wrong"]))
    print("%-22s %6d %6d %6d %6d"
          % ("【合计】", sum(tot.values()), tot["clean"], tot["dirty"], tot["wrong"]))

    # 干净 cell 被改坏（fp_det）的按列分布
    broken_cells = {(str(c.get(key)), c.get("column", "")) for c in broken}
    fp = defaultdict(int)
    for k, g in gold.items():
        r = res.get(k)
        if r is None:
            continue
        for col in cols:
            if (k, col) in broken_cells:
                continue
            if (r.get(col) or "") != (g.get(col) or ""):
                fp[col] += 1

    print("\n=== 干净 cell 被改坏（fp_det）按列 ===")
    if not fp:
        print("  0（没有任何干净 cell 被改坏）")
    for col, n in sorted(fp.items(), key=lambda kv: -kv[1]):
        print("  %-22s %d" % (col, n))
    print("  合计 fp_det = %d" % sum(fp.values()))


if __name__ == "__main__":
    main()
