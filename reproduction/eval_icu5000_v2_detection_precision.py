"""给 icu_5000 v2 加检测 Precision 的 realistic 估算（Table 2）。

复用之前的 per_class 结果 + overall score 的 row_aligned_cell_f1 估算 FP。
"""
import json
from pathlib import Path

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
per_class = json.loads((ROOT / "runs" / "icu_5000_v2_error_metrics_per_class.json").read_text())
# 直接从之前的 score 报告 hardcode（JSON 文件被 tee 截断了）
overall = {"metrics": {"row_aligned_cell_f1": 0.954323}}

# icu_5000 cell counts (估算)
cell_counts = {
    "admissions.csv":       6249 * 13,
    "cohort.csv":           6000 * 4,
    "demographics.csv":     4855 * 6,
    "diagnoses.csv":        109546 * 5,
    "procedures.csv":       38146 * 6,  # rough
    "medications.csv":      200000 * 8,
    "labs.csv":             800000 * 6,
    "icu/icustays.csv":     7180 * 6,
    "icu/chartevents.csv":  32289217 * 7,
    "icu/inputevents.csv":  837909 * 7,
    "icu/outputevents.csv": 407537 * 6,
    "multimodal/ecg_manifest.csv": 100 * 2,
    "multimodal/notes.csv": 0,
}
total_cells = sum(cell_counts.values())
f1 = overall["metrics"]["row_aligned_cell_f1"]
diff_cells = int(total_cells * (1 - f1))

print(f"total_cells ≈ {total_cells:,}")
print(f"row_aligned_cell_f1 = {f1}")
print(f"diff_cells (agent vs ref) ≈ {diff_cells:,}")
print()

# Per-class detection P realistic (FP = diff_cells, since not all diffs are FP, this is upper bound)
print("=" * 100)
print("TABLE 2: REALISTIC DETECTION PRECISION (overall cell diff as FP upper bound)")
print("=" * 100)
print()
print(f"  {'Class':<28} {'TP':>6} {'FN':>6} | {'FP_est':>10} | {'P(conservative)':>15} | {'R':>7} | {'F1':>7}")
print("  " + "-" * 95)

cls_metrics = per_class["per_class"]
for cls in ["class1_intra_table", "class2_entity_alignment", "class3_cross_table", "class4_task_oriented"]:
    m = cls_metrics[cls]
    d_tp, d_fn = m["detect_TP"], m["detect_FN"]
    # Conservative P: treat all diff_cells as potential FP
    p_cons = d_tp / (d_tp + diff_cells) if (d_tp + diff_cells) else 0
    r = m["detect_R"]
    f1_cons = 2 * p_cons * r / (p_cons + r) if (p_cons + r) else 0
    print(f"  {cls:<28} {d_tp:>6} {d_fn:>6} | {diff_cells:>10,} | {p_cons:>15.4f} | {r:>7.4f} | {f1_cons:>7.4f}")

print()
print("=" * 100)
print("FINAL DETECTION TABLE  — by-construction P=1.0 (Table 1) vs realistic (Table 2)")
print("=" * 100)
print()
print(f"  {'Class':<28} {'TP':>6} {'FN':>6} | {'P(ideal)':>8} {'R':>7} {'F1':>7} | {'P(realistic)':>13} {'F1(realistic)':>14}")
print("  " + "-" * 100)
for cls in ["class1_intra_table", "class2_entity_alignment", "class3_cross_table", "class4_task_oriented", "TOTAL"]:
    if cls == "TOTAL":
        m = per_class["totals"]
        d_tp, d_fn = m["detect_TP"], m["detect_FN"]
        n_str = ""
    else:
        m = cls_metrics[cls]
        d_tp, d_fn = m["detect_TP"], m["detect_FN"]
        n_str = ""
    p_real = d_tp / (d_tp + diff_cells) if (d_tp + diff_cells) else 0
    r = d_tp / (d_tp + d_fn) if (d_tp + d_fn) else 0
    f1_ideal = 2 * 1.0 * r / (1.0 + r) if (1.0 + r) else 0
    f1_real = 2 * p_real * r / (p_real + r) if (p_real + r) else 0
    print(f"  {cls:<28} {d_tp:>6} {d_fn:>6} | {'1.0000':>8} {r:>7.4f} {f1_ideal:>7.4f} | {p_real:>13.4f} {f1_real:>14.4f}")

# Save to file
out_path = ROOT / "runs" / "icu_5000_v2_detection_precision_realistic.json"
out_path.write_text(json.dumps({
    "total_cells_estimate": total_cells,
    "row_aligned_cell_f1": f1,
    "diff_cells_estimate": diff_cells,
    "note": "P=1.0 by construction on injected-only data. Realistic P uses overall cell diff as upper-bound FP estimate.",
    "per_class": {
        cls: {
            "detect_TP": cls_metrics[cls]["detect_TP"],
            "detect_FN": cls_metrics[cls]["detect_FN"],
            "P_ideal": 1.0,
            "R": cls_metrics[cls]["detect_R"],
            "F1_ideal": round(2 * cls_metrics[cls]["detect_R"] / (1.0 + cls_metrics[cls]["detect_R"]), 4),
            "P_realistic": round(cls_metrics[cls]["detect_TP"] / (cls_metrics[cls]["detect_TP"] + diff_cells), 4),
        }
        for cls in cls_metrics
    }
}, indent=2, ensure_ascii=False))
print(f"\nSaved: {out_path}")