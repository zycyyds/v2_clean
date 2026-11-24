#!/usr/bin/env python3
"""correction_broken_cell_attribution.py —— 校正类的**坏 cell 逐格归因**。

用 `host_private/cell_diff.jsonl`（host 专用的逐 cell 差分档案）里的
（index, column, clean_value, dirty_value, split），
把 agent 的输出表 join 上去，统计每个 split：

  修对（agent 值 == clean_value）/ 没动（== dirty_value）/ 改错（其它）/ 缺行

并按列给出「没修对」的分布。

⚠️ 只输出聚合计数与列名，**绝不打印任何单元格的值或行标识**。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)


def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True, help="host_private/cell_diff.jsonl")
    ap.add_argument("--key", default="index")
    ap.add_argument("--result", action="append", default=[],
                    help="split=路径 的 agent 结果表（可多次）")
    args = ap.parse_args()

    rows = []
    with open(args.diff, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_split = defaultdict(list)
    for r in rows:
        by_split[r.get("split", "?")].append(r)

    tables = {}
    for spec in args.result:
        split, _, path = spec.partition("=")
        tables[split] = {r[args.key]: r for r in load_csv(path)}

    print("=== 坏 cell 归因（聚合）===")
    for split in sorted(by_split):
        cells = by_split[split]
        tab = tables.get(split)
        if tab is None:
            print("\n【%s】%d 个坏 cell（没有该 split 的 agent 输出，跳过归因）"
                  % (split, len(cells)))
            continue
        fixed = untouched = wrong = missing = 0
        miss_by_col = Counter()
        for c in cells:
            row = tab.get(str(c.get(args.key)))
            if row is None:
                missing += 1
                miss_by_col[c.get("column", "?")] += 1
                continue
            got = row.get(c.get("column", ""), "")
            if got == c.get("clean_value", ""):
                fixed += 1
            elif got == c.get("dirty_value", ""):
                untouched += 1
                miss_by_col[c.get("column", "?")] += 1
            else:
                wrong += 1
                miss_by_col[c.get("column", "?")] += 1
        tot = len(cells)
        print("\n【%s】坏 cell %d 个" % (split, tot))
        print("   修对 %d（%.2f%%） / 没动 %d / 改错 %d / 缺行 %d"
              % (fixed, 100.0 * fixed / tot if tot else 0, untouched, wrong, missing))
        if miss_by_col:
            print("   没修对的按列分布:")
            for col, n in miss_by_col.most_common():
                print("     %-22s %d" % (col, n))


if __name__ == "__main__":
    main()
