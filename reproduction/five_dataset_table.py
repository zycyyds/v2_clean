#!/usr/bin/env python3
"""五个数据集 × 七口径的横向对比（数字全部从 runs/*.json 读，不转抄）。"""
import json
from pathlib import Path

RUNS = Path("/Users/mac/PycharmProjects/v2_clean/runs")

# (显示名, {口径: 文件名})
SETS = [
    ("K29 1.6pct", "k29_mimicraw_16_agent", {
        "v29": "k29_mimicraw_16_agent_error_metrics_v29.json",
        "str": "k29_mimicraw_16_agent_error_metrics_ratio0p2629_str.json",
        "oldsw": "k29_mimicraw_16_agent_error_metrics_v29_oldsw.json",
        "old": "k29_mimicraw_16_agent_error_metrics_ratio0p2629_old.json",
        "goldv29": "k29_mimicraw_16_agent_error_metrics_goldv29.json",
        "goldstr": "k29_mimicraw_16_agent_error_metrics_goldstr.json",
        "goldold": "k29_mimicraw_16_agent_error_metrics_goldold.json",
    }),
    ("K29 4.7pct", "k29_mimicraw_47_agent", {
        "v29": "k29_mimicraw_47_agent_error_metrics_v29.json",
        "str": "k29_mimicraw_47_agent_error_metrics_v29_str.json",
        "oldsw": "k29_mimicraw_47_agent_error_metrics_v29_oldsw.json",
        "old": "k29_mimicraw_47_agent_error_metrics_v29_old.json",
        "goldv29": "k29_mimicraw_47_agent_error_metrics_goldv29.json",
        "goldstr": "k29_mimicraw_47_agent_error_metrics_goldstr.json",
        "goldold": "k29_mimicraw_47_agent_error_metrics_goldold.json",
    }),
    ("ICU5000 1.6pct", "icu5000_mimicraw_16_agent", {
        c: "icu5000_mimicraw_16_agent_error_metrics_%s_run20261118_021004.json" % c
        for c in ("v29", "str", "oldsw", "old", "goldv29", "goldstr", "goldold")
    }),
    ("ICU5000 4.7pct", "icu5000_mimicraw_47_agent", {
        c: "icu5000_mimicraw_47_agent_error_metrics_%s_run20261118_4p7.json" % c
        for c in ("v29", "str", "oldsw", "old", "goldv29", "goldstr")
    }),
    ("I48 1.6pct", "i48_mimicraw2_16_agent", {
        c: "i48_mimicraw2_16_agent_error_metrics_%s_run20261119_1p6.json" % c
        for c in ("v29", "str", "oldsw", "old", "goldv29", "goldstr", "goldold")
    }),
]
ORDER = ["v29", "str", "oldsw", "old", "goldv29", "goldstr", "goldold"]


def load(fn):
    p = RUNS / fn
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if "detection" not in d:
        return None
    b = d.get("buckets", {})
    return {
        "det": d["detection"]["F1"], "rep": d["repair"]["F1"],
        "Pd": d["detection"]["P"], "Rd": d["detection"]["R"],
        "Pr": d["repair"]["P"], "Rr": d["repair"]["R"],
        "clean": b.get("b_clean", 0), "wrong": b.get("b_wrong", 0),
        "dirty": b.get("b_dirty", 0), "miss": b.get("b_missing", 0),
        "fpdet": b.get("fp_det", 0), "noise": b.get("natural_noise", 0),
    }


for cal in ORDER:
    print("\n### 口径 %s" % cal)
    print("| 数据集 | Det F1 | Rep F1 | Det P/R | Rep P/R | b_clean | b_wrong | b_dirty | b_missing |")
    print("|---|---:|---:|---|---|---:|---:|---:|---:|")
    for name, _ds, files in SETS:
        r = load(files[cal]) if cal in files else None
        if r is None:
            print("| %s | — | — | — | — | — | — | — | — |" % name)
            continue
        print("| %s | **%.6f** | **%.6f** | %.4f / %.4f | %.4f / %.4f | %s | %s | %s | %s |"
              % (name, r["det"], r["rep"], r["Pd"], r["Rd"], r["Pr"], r["Rr"],
                 f"{r['clean']:,}", f"{r['wrong']:,}", f"{r['dirty']:,}", f"{r['miss']:,}"))
