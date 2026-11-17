#!/usr/bin/env python3
"""icu5000_enum_scan.py —— 全量扫 test reference，找出「同一 5 键多条不同取值」的
枚举型 itemid。

**为什么需要它**：prompt 里必须提醒 agent「枚举型项目同一时刻多条不同取值是合法的，
不要当重复行删掉」。K29 的 prompt 写死了 230033 / 229864 两个 —— 那是同一份 MIMIC-IV，
icu5000 是它的 subset，itemid 集合应当一致；但 icu5000 规模大 3.5 倍，可能有别的
枚举型 itemid 冒出来 ⇒ 扫一遍确认，别让 agent 删错行。

做法（两遍，内存可控）：
  ① 扫一遍统计每个 itemid 的行数（Counter，itemid 种类少，内存 O(种类数)）
  ② 只对行数 top-N 的 itemid 收集 (5 键 → 取值集合)，找多值组

只输出聚合（itemid / 行数 / 多值组数），不打印病人级信息。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

CE = "icu/chartevents.csv"
KEY = ("subject_id", "hadm_id", "stay_id", "charttime", "itemid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--min-rows", type=int, default=0,
                    help="只看行数 >= 这个值的 itemid（0 = 不限）")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="只看行数 <= 这个值的 itemid（0 = 不限）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = Path(args.dataset) / args.split / "reference" / CE
    print("扫:", ref, flush=True)

    cnt = Counter()
    with ref.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cnt[r["itemid"]] += 1
    print("itemid 种类: %d" % len(cnt), flush=True)

    if args.min_rows or args.max_rows:
        lo = args.min_rows
        hi = args.max_rows or float("inf")
        watch = {k for k, n in cnt.items() if lo <= n <= hi}
        desc = "%d ≤ 行数 ≤ %s" % (lo, args.max_rows or "∞")
    else:
        watch = {k for k, _ in cnt.most_common(args.top)}
        desc = "行数 top %d" % args.top
    print("重点盯 %d 个 itemid（%s）" % (len(watch), desc), flush=True)

    grp = defaultdict(set)
    with ref.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            it = r["itemid"]
            if it not in watch:
                continue
            grp[(r[KEY[0]], r[KEY[1]], r[KEY[2]], r[KEY[3]], it)].add(r["valuenum"])

    multi = Counter()
    for k, v in grp.items():
        if len(v) > 1:
            multi[k[4]] += 1

    out = {"rows": dict(cnt.most_common(args.top)),
           "multi_groups": dict(multi.most_common()),
           "n_items": len(cnt),
           "watched": len(watch)}
    Path(args.out or "/dev/null").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 多值组数 top 30（枚举型候选）===")
    for it, n in multi.most_common(30):
        print("  itemid %-10s 行数 %10s  多值组 %8s"
              % (it, "{:,}".format(cnt[it]), "{:,}".format(n)))
    if not multi:
        print("  （前 %d 个 itemid 里没有多值组）" % args.top)


if __name__ == "__main__":
    main()
