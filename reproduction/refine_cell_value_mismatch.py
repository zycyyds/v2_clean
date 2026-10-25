"""Rule-based refinement: 给 cell_value_mismatch 中"dirty_value 符合 7 个未识别 subtype 模式"的 entry 重新分类。

7 个未识别 subtype + 匹配规则:
1. unit_scale_error: raw valuenum ≈ 100×clean valuenum
2. implausible_negative_event_time: charttime 年份 = 1970
3. medication_itemid_alias_split: inputevents itemid 99 prefix
4. label_conflict_with_cohort: hospital_expire_flag 0↔1 flip
5. future_window_leakage: charttime year > 2200
6. stay_hadm_mismatch: inputevents hadm_id 改 (raw hadm != clean hadm)
7. label_leakage_feature: row insert 标志 (raw 多了行)
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

NEW_ROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/K29_clean_v2_rawerror_highrate_10pct_v2")


def classify_rule(r: dict) -> tuple[str, str] | None:
    """根据 (column, raw_value, clean_value) 推断 subtype。返回 (class, subtype) 或 None。"""
    col = r["column"]
    dirty = r["dirty_value"]
    clean = r["clean_value"]
    raw_file = r["raw_file"]

    # 0. baseline_precision_gap: 浮点精度差异 (clean=94.0 dirty=94) — 非注入错误
    if col in ("valuenum", "amount", "los"):
        try:
            d = float(dirty)
            c = float(clean)
            if abs(d - c) < 1e-9:
                return ("class0_baseline_gap", "baseline_precision_gap")
        except (ValueError, ZeroDivisionError):
            pass

    # 1. unit_scale_error: raw valuenum ≈ 100×clean
    if col == "valuenum" and raw_file == "icu/chartevents.csv":
        try:
            d = float(dirty)
            c = float(clean)
            if abs(c) > 0.001 and abs(d / c - 100) < 0.5:
                return ("class1_intra_table", "unit_scale_error")
        except (ValueError, ZeroDivisionError):
            pass

    # 1b. sentinel_chart_value: char valuenum 是 sentinel (--888/999/1000 等)
    if col == "valuenum" and raw_file == "icu/chartevents.csv":
        try:
            d = float(dirty)
            if d in (-999.0, -888.0, -1.0, -1000.0, 999.0, 888.0, 9999.0, -9999.0):
                return ("class1_intra_table", "sentinel_chart_value")
        except (ValueError, ZeroDivisionError):
            pass

    # 1c. sentinel_or_negative_med_amount: inputevents amount 是 sentinel/负值
    if col == "amount" and raw_file == "icu/inputevents.csv":
        try:
            d = float(dirty)
            if d < 0 or d in (999.0, 888.0, 9999.0, 100.0, 50.0, 200.0, 10.0, 1.0, 2.0, 1000.0):
                return ("class1_intra_table", "sentinel_or_negative_med_amount")
        except (ValueError, ZeroDivisionError):
            pass

    # 2. implausible_negative_event_time: charttime 年份 = 1970
    if col == "charttime" and raw_file in ("icu/chartevents.csv", "icu/outputevents.csv"):
        if dirty.startswith("1970"):
            return ("class1_intra_table", "implausible_negative_event_time")

    # 3. medication_itemid_alias_split: inputevents itemid 99 prefix
    if col == "itemid" and raw_file == "icu/inputevents.csv":
        if dirty.startswith("99") and dirty.isdigit() and clean.isdigit() and not clean.startswith("99"):
            return ("class2_entity_alignment", "medication_itemid_alias_split")

    # 4. label_conflict_with_cohort: hospital_expire_flag 0↔1 flip
    if col == "hospital_expire_flag" and raw_file == "hosp/admissions.csv":
        if dirty in ("0", "1") and clean in ("0", "1") and dirty != clean:
            return ("class3_cross_table", "label_conflict_with_cohort")

    # 5. future_window_leakage: charttime year > 2200
    if col == "charttime" and raw_file in ("icu/chartevents.csv", "icu/outputevents.csv"):
        try:
            d_year = int(dirty[:4])
            c_year = int(clean[:4]) if clean else 0
            if d_year > 2200 and c_year < 2200:
                return ("class4_task_oriented", "future_window_leakage")
        except (ValueError, IndexError):
            pass

    # 6. stay_hadm_mismatch: inputevents hadm_id 改
    if col == "hadm_id" and raw_file == "icu/inputevents.csv":
        if dirty and clean and dirty != clean and dirty.isdigit() and clean.isdigit():
            return ("class3_cross_table", "stay_hadm_mismatch")

    # 7. unclassified_real_inject: 已知是注入但模式无法识别 (int 3/4/13/2 等)
    #    - chart valuenum 是 clean 没 .0 (int 注入) → 真实注入但 mode 未识别
    #    - inputevents amount 是整数 → 真实注入但 mode 未识别
    if col == "valuenum" and raw_file == "icu/chartevents.csv":
        try:
            d = float(dirty)
            c = float(clean)
            # dirty 是整数（如 1, 2, 3），clean 是非整 (10.0 等)
            if d == int(d) and c != int(c) and abs(d - c) > 0.5:
                return ("class1_intra_table", "unclassified_real_inject")
        except (ValueError, ZeroDivisionError):
            pass

    if col == "amount" and raw_file == "icu/inputevents.csv":
        try:
            d = float(dirty)
            c = float(clean)
            # dirty 是整数 amount, clean 是浮点 amount
            if d == int(d) and c != int(c) and abs(d - c) > 0.5:
                return ("class1_intra_table", "unclassified_real_inject")
            # dirty 浮点但和 clean 偏差大 (e.g. 10.000000232830633 vs 655.78) → 真实注入
            if abs(d - c) > 1.0:
                return ("class1_intra_table", "unclassified_real_inject")
        except (ValueError, ZeroDivisionError):
            pass

    return None


def refine_split(split: str) -> dict:
    print(f"\n========== {split} ==========", flush=True)
    log_path = NEW_ROOT / split / "modification_log.csv"
    rows = list(csv.DictReader(log_path.open(newline="", encoding="utf-8")))
    print(f"  loaded {len(rows):,} entries", flush=True)

    n_cvm = sum(1 for r in rows if r["error_subtype"] == "cell_value_mismatch")
    n_refined = 0
    new_subtype_counts: Counter = Counter()
    for r in rows:
        if r["error_subtype"] != "cell_value_mismatch":
            continue
        result = classify_rule(r)
        if result:
            cls, subtype = result
            r["error_class"] = cls
            r["error_subtype"] = subtype
            n_refined += 1
            new_subtype_counts[subtype] += 1

    # 写回
    fieldnames = list(rows[0].keys())
    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"  cell_value_mismatch before: {n_cvm:,}", flush=True)
    print(f"  refined by rule: {n_refined:,}", flush=True)
    print(f"  rule-based new subtypes:", flush=True)
    for s, n in new_subtype_counts.most_common():
        print(f"    {s:<40} {n:>10,}", flush=True)

    return {
        "n_total": len(rows),
        "n_cvm_before": n_cvm,
        "n_refined": n_refined,
        "new_subtypes": dict(new_subtype_counts),
    }


def main():
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["train", "validation", "test"]
    for s in splits:
        refine_split(s)


if __name__ == "__main__":
    main()
