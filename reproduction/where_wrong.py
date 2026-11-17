#!/usr/bin/env python3
"""两档在**纯字符串口径**下：哪些格被判成 b_wrong（"算错"）、哪些判成 b_clean（"没算错"）。

回答「为什么 4.7pct 有高有低、1.6pct 几乎全错」：错率由**列**决定，
根源是 log 里 clean_value 的写法分布 + agent 那一列是 int 列还是 float 列。
只输出聚合计数，不打印行级内容。
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)
R = Path(sys.argv[1])
DSROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/k29")

TIERS = [
    ("4.7pct", "k29_mimicraw_47_agent_error_metrics_v29_restore.json",
     "k29_mimicraw_47_agent_error_metrics_str_restore.json",
     DSROOT / "K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror"),
    ("1.6pct", "k29_mimicraw_16_agent_error_metrics_v29_rawform.json",
     "k29_mimicraw_16_agent_error_metrics_str_rawform.json",
     DSROOT / "K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"),
]


def shape(s):
    try:
        float(s)
    except (TypeError, ValueError):
        return "非数值"
    if "." in str(s) or "e" in str(s).lower():
        return "浮点写法"
    return "整数字面量"


def log_clean_forms(ds, raw_file):
    """log 里该文件各列的 clean_value 形态分布（注入器条目，排除原生）"""
    c = Counter()
    with (ds / "test/modification_log.csv").open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if (e.get("raw_file") == raw_file and e.get("operation") == "cell_update"
                    and not e.get("error_subtype", "").startswith("mimic_native")):
                c[(e.get("column"), shape(e.get("clean_value")))] += 1
    return c


for tier, pv, ps, ds in TIERS:
    dv = json.load(open(R / pv))
    dsx = json.load(open(R / ps))
    fv, fs = dv["per_file"], dsx["per_file"]
    print("=" * 96)
    print("### %s" % tier)
    tot_c = tot_w = tot_d = 0
    print("  逐文件（纯字符串口径）：")
    for f in sorted(fv):
        b = fs.get(f, {})
        bc, bw, bd = b.get("b_clean", 0), b.get("b_wrong", 0), b.get("b_dirty", 0)
        if bc + bw + bd == 0:
            continue
        tot_c += bc; tot_w += bw; tot_d += bd
        print("    %-26s b_clean=%10s  b_wrong=%10s  b_dirty=%8s   错%%=%5.1f%%"
              % (f, f"{bc:,}", f"{bw:,}", f"{bd:,}", 100.0 * bw / (bc + bw + bd)))
    n = tot_c + tot_w + tot_d
    print("    %-26s b_clean=%10s  b_wrong=%10s  b_dirty=%8s   错%%=%5.1f%%"
          % ("合计", f"{tot_c:,}", f"{tot_w:,}", f"{tot_d:,}", 100.0 * tot_w / n))
    print("    Rep F1: v29=%.6f   str=%.6f" % (dv["repair"]["F1"], dsx["repair"]["F1"]))
    print()
    print("  log 里 chartevents.valuenum 的 clean_value 形态（决定谁吃「写法税」）：")
    for (col, sh), n2 in sorted(log_clean_forms(ds, "icu/chartevents.csv").items()):
        if col != "valuenum":
            continue
        taxed = "吃税（log 记整数形式、agent 写 float repr）" if sh == "整数字面量" else \
                ("不吃税（两边同为浮点写法）" if sh == "浮点写法" else "不吃税（非数值，两边同为非数值）")
        print("    valuenum  %-10s %10s   %s" % (sh, f"{n2:,}", taxed))
    print()
