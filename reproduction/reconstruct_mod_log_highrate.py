"""Reconstruct K29 highrate_10pct modification_log.csv from raw vs reference diff.

Strategy: 对每个 split 的 7 个有错误注入的 raw 文件，逐 cell 与 reference 比较：
- raw cell ≠ reference cell → cell_update (clean=ref, dirty=raw)
- raw 行存在但 reference 找不到对应 key → duplicate/insert_row

输出：K29_clean_v2_rawerror_highrate_10pct/{split}/modification_log.csv

只对比 reference 中存在的列（raw 多的 admit_provider_id / caregiver_id / value 等忽略）。
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
DATASET = ROOT / "datasets/注入错误数据集/K29_clean_v2_rawerror_highrate_10pct_v2"

# raw 文件 → (reference 文件, key 列, error_subtype default, is_big)
FILES = [
    ("hosp/patients.csv",      "demographics.csv",    ["subject_id"],                                "enum_violation",          False),
    ("hosp/admissions.csv",    "admissions.csv",      ["hadm_id"],                                   "cell_value_mismatch",     False),
    ("hosp/diagnoses_icd.csv", "diagnoses.csv",       ["subject_id","hadm_id","seq_num"],            "cell_value_mismatch",     False),
    ("icu/icustays.csv",       "icu/icustays.csv",    ["subject_id","hadm_id","stay_id"],            "cell_value_mismatch",     False),
    ("icu/chartevents.csv",    "icu/chartevents.csv", ["subject_id","hadm_id","stay_id","charttime","itemid"], "cell_value_mismatch", True),
    ("icu/inputevents.csv",    "icu/inputevents.csv", ["subject_id","hadm_id","stay_id","starttime","itemid"], "cell_value_mismatch", False),
    ("icu/outputevents.csv",   "icu/outputevents.csv",["subject_id","hadm_id","stay_id","charttime","itemid"], "cell_value_mismatch", False),
]

LOG_HEADER = [
    "source_error_id", "canonical_stay_id", "error_class", "error_subtype",
    "target_effect_contract", "operation", "raw_file", "raw_row_index",
    "column", "clean_value", "dirty_value", "matching_columns",
]


def infer_subtype(col: str, raw_v: str, ref_v: str) -> tuple[str, str]:
    """根据列名和值推断 error_class + error_subtype。返回 (class, subtype)。"""
    raw_v = raw_v.strip()
    ref_v = ref_v.strip()
    if col == "anchor_age" and raw_v == "999":
        return "class1_intra_table", "schema_range_violation"
    if col == "gender" and raw_v == "UNKNOWN_GENDER_CODE":
        return "class1_intra_table", "enum_violation"
    if raw_v == "" or raw_v == "(error generating)":
        return "class1_intra_table", "unexpected_missingness"
    if col in ("endtime",) and raw_v and ref_v:
        return "class1_intra_table", "med_end_before_start"
    if col in ("outtime", "charttime", "admittime", "dischtime") and raw_v and ref_v:
        return "class1_intra_table", "temporal_order_violation"
    if col == "valuenum" and raw_v == "-999":
        return "class1_intra_table", "sentinel_chart_value"
    if col == "amount" and raw_v in ("-999", "-888"):
        return "class1_intra_table", "sentinel_or_negative_med_amount"
    if col == "itemid" and ref_v and raw_v and ref_v.isdigit() and raw_v.isdigit() and raw_v.startswith("99"):
        return "class2_entity_alignment", "chart_itemid_alias_split"
    if col == "icd_code" and ref_v and raw_v.startswith(("ALIAS_", "99")):
        return "class2_entity_alignment", "diagnosis_code_alias_split"
    if col == "stay_id" and ref_v and raw_v and ref_v != raw_v:
        return "class3_cross_table", "orphan_feature_stay_id"
    if col == "hadm_id" and ref_v and raw_v and ref_v != raw_v:
        return "class3_cross_table", "stay_hadm_mismatch"
    return "class1_intra_table", "cell_value_mismatch"


def diff_one_file(raw_path: Path, ref_path: Path, raw_rel: str, key_cols: list[str], split: str, error_id_prefix: str, error_subtype_default: str) -> list[dict]:
    """对单个文件做 diff，返回 mod_log rows。"""
    rows = []
    # load reference into dict by key
    print(f"  loading reference {ref_path.name}...", flush=True)
    ref_by_key: dict[tuple, dict] = {}
    with ref_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ref_fieldnames = reader.fieldnames or []
        for r in reader:
            kt = tuple(r.get(c, "") for c in key_cols)
            ref_by_key[kt] = r
    print(f"    reference has {len(ref_by_key)} rows, columns: {ref_fieldnames}", flush=True)

    print(f"  streaming raw {raw_path.name}...", flush=True)
    with raw_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, raw_row in enumerate(reader):
            kt = tuple(raw_row.get(c, "") for c in key_cols)
            ref_row = ref_by_key.get(kt)
            if ref_row is None:
                # raw 行不在 ref 中 → duplicate/insert
                # 用 stay_id (subject_id or hadm_id) 作 canonical_stay_id
                canonical = raw_row.get("stay_id", "") or raw_row.get("hadm_id", "") or raw_row.get("subject_id", "")
                rows.append({
                    "source_error_id": f"{split}-{error_id_prefix}-{len(rows):06d}",
                    "canonical_stay_id": canonical,
                    "error_class": "class1_intra_table",
                    "error_subtype": "duplicate_row",
                    "target_effect_contract": "(see raw_level_error_split)",
                    "operation": "duplicate",
                    "raw_file": raw_rel,
                    "raw_row_index": i,
                    "column": "(row)",
                    "clean_value": "",
                    "dirty_value": "(extra_row)",
                    "matching_columns": "|".join(key_cols),
                })
                continue
            # column by column compare (only reference columns)
            for col in ref_fieldnames:
                raw_v = raw_row.get(col, "")
                ref_v = ref_row.get(col, "")
                if raw_v != ref_v:
                    canonical = raw_row.get("stay_id", "") or raw_row.get("hadm_id", "") or raw_row.get("subject_id", "")
                    cls, subtype = infer_subtype(col, raw_v, ref_v)
                    rows.append({
                        "source_error_id": f"{split}-{error_id_prefix}-{len(rows):06d}",
                        "canonical_stay_id": canonical,
                        "error_class": cls,
                        "error_subtype": subtype,
                        "target_effect_contract": "(see raw_level_error_split)",
                        "operation": "cell_update",
                        "raw_file": raw_rel,
                        "raw_row_index": i,
                        "column": col,
                        "clean_value": ref_v,
                        "dirty_value": raw_v,
                        "matching_columns": "|".join(key_cols),
                    })
    return rows


def reconstruct_split(split: str) -> dict:
    print(f"\n========== {split} ==========", flush=True)
    raw_dir = DATASET / split / "raw"
    ref_dir = DATASET / split / "reference"
    out_path = DATASET / split / "modification_log.csv"

    all_rows = []
    file_stats = {}
    for raw_rel, ref_name, key_cols, subtype_default, is_big in FILES:
        raw_path = raw_dir / raw_rel
        ref_path = ref_dir / ref_name
        if not raw_path.exists():
            print(f"  SKIP {raw_rel}: raw not found", flush=True)
            continue
        if not ref_path.exists():
            print(f"  SKIP {raw_rel}: ref {ref_name} not found", flush=True)
            continue
        t0 = time.time()
        rows = diff_one_file(raw_path, ref_path, raw_rel, key_cols, split,
                              error_id_prefix=raw_rel.replace("/", "_").replace(".csv", ""),
                              error_subtype_default=subtype_default)
        t1 = time.time()
        n = len(rows)
        all_rows.extend(rows)
        file_stats[raw_rel] = {"n_entries": n, "seconds": round(t1 - t0, 1)}
        print(f"  {raw_rel}: {n:,} diff entries ({t1-t0:.1f}s)", flush=True)

    # write csv
    print(f"\n  writing {len(all_rows):,} entries to {out_path}...", flush=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADER)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    return {"n_entries": len(all_rows), "by_file": file_stats}


def main():
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["train", "validation", "test"]
    report = {"dataset": DATASET.name, "splits": {}}
    for split in splits:
        report["splits"][split] = reconstruct_split(split)

    report_path = DATASET / "reconstruct_mod_log_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[OK] report → {report_path}")


if __name__ == "__main__":
    main()
