#!/usr/bin/env python3
"""i48 4.7pct 七口径 + 原生格汇总，并与 1.6pct 对照。"""
import json
from pathlib import Path

RUNS = Path("/Users/mac/PycharmProjects/v2_clean/runs")
SFX = "run20261119_4p7"
ORDER = ["v29", "str", "oldsw", "old", "goldv29", "goldstr", "goldold"]
DESC = {
    "v29": "数值 + key还原 + 排除gold外行 + log判据",
    "str": "纯字符串 + key还原 + 排除gold外行 + log判据",
    "oldsw": "数值 + 关key还原 + 不排除gold外行 + log判据",
    "old": "纯字符串 + 关key还原 + 不排除gold外行 + log判据",
    "goldv29": "数值 + key还原 + 排除gold外行 + gold判据",
    "goldstr": "纯字符串 + key还原 + 排除gold外行 + gold判据",
    "goldold": "纯字符串 + 关key还原 + 不排除gold外行 + gold判据",
}


def load(dskey, cal, sfx=SFX):
    p = RUNS / ("%s_error_metrics_%s_%s.json" % (dskey, cal, sfx))
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def native(dskey, mode, sfx=SFX):
    p = RUNS / ("%s_native_cells_%s_%s.json" % (dskey, sfx, mode))
    return json.loads(p.read_text()) if p.is_file() else None


def fmt(n):
    return format(n, ",") if isinstance(n, int) else ("-" if n is None else str(n))


print("=" * 118)
print("i48 4.7pct —— 七口径错误级指标（test）")
print("=" * 118)
print("%-9s %-52s %10s %10s %12s %11s %11s %11s" % (
    "口径", "开关组合", "Det F1", "Rep F1", "b_clean", "b_wrong", "b_dirty", "b_missing"))
rows = {}
for cal in ORDER:
    d = load("i48_mimicraw2_47_agent", cal)
    if not d:
        print("%-9s %-52s   （缺）" % (cal, DESC[cal]))
        continue
    b = d["buckets"]
    rows[cal] = d
    print("%-9s %-52s %10.6f %10.6f %12s %11s %11s %11s" % (
        cal, DESC[cal], d["detection"]["F1"], d["repair"]["F1"],
        fmt(b.get("b_clean")), fmt(b.get("b_wrong")),
        fmt(b.get("b_dirty")), fmt(b.get("b_missing"))))

print()
print("=" * 118)
print("与 i48 1.6pct 对照（Det F1 / Rep F1）")
print("=" * 118)
print("%-9s %20s %20s" % ("口径", "1.6pct", "4.7pct"))
for cal in ORDER:
    a = load("i48_mimicraw2_16_agent", cal, "run20261119_1p6")
    b = rows.get(cal)
    fa = ("%.6f / %.6f" % (a["detection"]["F1"], a["repair"]["F1"])) if a else "—"
    fb = ("%.6f / %.6f" % (b["detection"]["F1"], b["repair"]["F1"])) if b else "—"
    print("%-9s %20s %20s" % (cal, fa, fb))

print()
print("=" * 118)
print("MIMIC 原生错误格（i48）")
print("=" * 118)
for mode in ("numeric", "string"):
    n = native("i48_mimicraw2_47_agent", mode)
    if not n:
        print("%-8s （缺）" % mode)
        continue
    print("--- %s ---" % mode)
    print(json.dumps({k: v for k, v in n.items()
                      if not isinstance(v, (dict, list))}, ensure_ascii=False, indent=1))
    for k in ("按类型", "按detection_mode", "buckets_any", "ambiguous_keys", "总格数"):
        if k in n:
            print("  %s: %s" % (k, json.dumps(n[k], ensure_ascii=False)))
