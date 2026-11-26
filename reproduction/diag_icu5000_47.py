#!/usr/bin/env python3
"""ICU5000 4.7pct（我们 8 档里最低的一档）：逐表比行数（轻量版，不整表读进内存）。

只看行数差异 —— 行数对上但 key 分低，才是「对错行」；行数对不上则是「多筛/少筛」。
"""
from pathlib import Path

GOLD = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/icu5000/"
            "icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror/test/reference")
PKG = Path("/Users/mac/Downloads/ceshi1/ICU5000/4.7pct_我们的agent_pi+val_mimicraw/结果包")

print("=== ICU5000 4.7pct：我们 vs gold（行数）===")
tot_g = tot_o = 0
for g in sorted(GOLD.rglob("*.csv")):
    rel = g.relative_to(GOLD)
    o = PKG / rel
    if not o.exists():
        print("  %-30s 我们缺这个文件" % rel)
        continue
    gn = sum(1 for _ in open(g, "rb")) - 1
    on = sum(1 for _ in open(o, "rb")) - 1
    tot_g += gn
    tot_o += on
    mark = "" if gn == on else "   ← 行数不同 (%+d)" % (on - gn)
    print("  %-30s gold=%-10d ours=%-10d%s" % (rel, gn, on, mark))
print("  合计 gold=%d ours=%d (%+d)" % (tot_g, tot_o, tot_o - tot_g))
