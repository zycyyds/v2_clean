#!/usr/bin/env python3
"""把 composite 的逐文件分数和错误级指标的逐文件账放一起对照。"""
import json
from pathlib import Path

RUNS = Path("/Users/mac/PycharmProjects/v2_clean/runs")
WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
E = WT / "i48_agent_run/experiments/i48_mimicraw2_4p7_test_seed666_v1"

em = json.loads((RUNS / "i48_mimicraw2_47_agent_error_metrics_v29_run20261119_4p7.json").read_text())
sr = json.loads((E / "host/score_report.json").read_text())
pf_em = em["per_file"]

rows = []
for fr in sr["file_reports"]:
    path = fr["relative_path"]
    m = fr.get("metrics", {})
    e = pf_em.get(path)
    if e:
        tot = e.get("b_clean", 0) + e.get("b_wrong", 0) + e.get("b_dirty", 0) + e.get("b_missing", 0)
        fix = e.get("b_clean", 0) / tot if tot else None
        wrong_rate = e.get("b_wrong", 0) / tot if tot else None
    else:
        fix = wrong_rate = None
    rows.append((path, fr["reference_rows"], fr["result_rows"],
                 fr["exact_row_matches"], fr["missing_reference_rows"], fr["extra_result_rows"],
                 m.get("row_aligned_cell_f1"), fix, wrong_rate, e))

print("%-30s %9s %9s %8s | %8s %8s | %7s %7s" % (
    "file", "ref_rows", "res_rows", "exact", "cell_f1", "err修复率", "b_wrong", "行数差"))
print("-" * 108)
for path, rr, res, ex, mr, er, cell, fix, wr, e in sorted(rows, key=lambda x: (x[7] is None, x[7] or 0)):
    d = res - rr
    print("%-30s %9s %9s %8s | %8s %8s | %7s %7s" % (
        path, format(rr, ","), format(res, ","), format(ex, ","),
        ("%.4f" % cell) if cell is not None else "-",
        ("%.4f" % fix) if fix is not None else "无注入",
        ("%.4f" % wr) if wr is not None else "-",
        ("%+d" % d) if d else "0"))
print()
print("composite = %.6f   (schema %.4f / key %.4f / cell %.4f / exact %.4f)" % (
    sr["metrics"]["composite_score"], sr["metrics"]["schema_f1"],
    sr["metrics"]["key_structure_f1"], sr["metrics"]["row_aligned_cell_f1"],
    sr["metrics"]["exact_row_f1"]))
print("错误级 v29: Det F1=%.6f  Rep F1=%.6f  fp_det=%d  natural_noise=%d" % (
    em["detection"]["F1"], em["repair"]["F1"],
    em["buckets"].get("fp_det", 0), em["buckets"].get("natural_noise", 0)))
print()
print("composite 覆盖文件 %d 个；错误级只覆盖有注入日志的 %d 个：%s" % (
    len(rows), len(pf_em), list(pf_em)))
print("无注入日志 ⇒ 错误级完全不看：", [p for p, *_ , e in rows if e is None])
