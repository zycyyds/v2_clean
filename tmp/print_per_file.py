"""打印错误级指标的逐文件明细（供台账 §2.9 用）。"""
import json
import sys

R = "/Users/mac/PycharmProjects/v2_clean/runs/"
for tag, em in (("cc N18 4.7pct", "n18_more_v1_cc_error_metrics_v29.json"),
                ("cc I48 1.6pct", "i48_less_cc_v1_cc_error_metrics_v29.json")):
    d = json.load(open(R + em))
    print(f"\n=== {tag} ===")
    print("det:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d["detection"].items()})
    print("rep:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d["repair"].items()})
    print("buckets:")
    for k in sorted(d["buckets"]):
        print(f"   {k:24s} {d['buckets'][k]:,}")
    print("per_file:")
    keys = ["b_clean", "b_dirty", "b_wrong", "b_missing", "dup_removed", "leak_removed"]
    print("   | " + " | ".join(keys) + " |")
    for f, v in d["per_file"].items():
        row = []
        for k in keys:
            val = v.get(k)
            row.append(f"{val:,}" if isinstance(val, int) else ("—" if val is None else str(val)))
        print(f"   | {f} | " + " | ".join(row) + " |")
