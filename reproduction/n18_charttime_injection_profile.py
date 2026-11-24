#!/usr/bin/env python3
"""n18_charttime_injection_profile.py —— 注入器对 chartevents 的 **charttime 列**
到底做了什么？（方向、偏移量分布）

从 modification_log.csv 里取 column=='charttime' 的条目，
算 delta = dirty_value − clean_value（分钟）。
只输出聚合统计（不输出病人标识、不输出逐格医学值）。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

csv.field_size_limit(2 ** 31)


def parse_ct(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ds, sp = Path(args.dataset), args.split

    mod = ds / sp / "modification_log.csv"
    n = 0
    sign = Counter()
    deltas = Counter()
    subtypes = Counter()
    for r in csv.DictReader(mod.open(newline="", encoding="utf-8")):
        if r.get("column") != "charttime" or r.get("raw_file") != "icu/chartevents.csv":
            continue
        n += 1
        subtypes[r.get("error_subtype", "")] += 1
        t0, t1 = parse_ct(r.get("clean_value", "")), parse_ct(r.get("dirty_value", ""))
        if t0 is None or t1 is None:
            sign["时间解析失败"] += 1
            continue
        d = int(round((t1 - t0).total_seconds() / 60.0))
        sign["dirty 更晚（往后挪）" if d > 0 else
             ("dirty 更早（往前挪）" if d < 0 else "没变")] += 1
        deltas[d] += 1

    print("=== %s：注入器对 chartevents.charttime 的改动 ===" % sp)
    print("  条目数: %s" % "{:,}".format(n))
    for k, v in subtypes.most_common():
        print("    subtype %-28s : %s" % (k, "{:,}".format(v)))
    print()
    for k, v in sign.most_common():
        print("    %-22s : %s" % (k, "{:,}".format(v)))
    print()
    print("  偏移量 top 12（分钟，dirty − clean）:")
    for d, c in deltas.most_common(12):
        print("    %+8d 分钟 (%+.2f 小时) : %s" % (d, d / 60.0, "{:,}".format(c)))
    if deltas:
        tot = sum(deltas.values())
        print()
        print("  ⇒ 不同偏移量的取值个数: %d；单一偏移量最大占比: %.1f%%"
              % (len(deltas), 100.0 * deltas.most_common(1)[0][1] / tot))


if __name__ == "__main__":
    main()
