"""N18 v2 — 错误检测 + 修复评估 (streaming for chartevents)

参考 icu_5000_v2 eval 但适配 N18：
- 18 个 output 文件
- subject_id 是 split key
- chartevents 同样可能很大 → 流式
- ed + note 多模态不在 raw（注入只到 hosp/icu），不在评估范围
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

MOD_LOG = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_10train_400_800_1800/test/modification_log.csv")
TEST_RAW = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_10train_400_800_1800/test/raw")
TEST_OUTPUT = Path("/Users/mac/PycharmProjects/review/N18_ckd_ED_v2/output/test/result")
REFERENCE = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_10train_400_800_1800/test/reference")

# raw_file → (output_rel, key_cols, dirty_in_key_cols, size_tag)
RAW_TO_OUTPUT = {
    "hosp/patients.csv":      ("demographics.csv",    ["subject_id"], [], "small"),
    "hosp/admissions.csv":    ("admissions.csv",      ["hadm_id"], [], "small"),
    "hosp/diagnoses_icd.csv": ("diagnoses.csv",       ["subject_id","hadm_id","seq_num","icd_code","icd_version"], ["icd_code"], "small"),
    "icu/icustays.csv":       ("icu/icustays.csv",    ["stay_id"], [], "small"),
    "icu/chartevents.csv":    ("icu/chartevents.csv", ["subject_id","hadm_id","stay_id","charttime","itemid","valuenum","valueuom"], ["itemid"], "big"),
    "icu/inputevents.csv":    ("icu/inputevents.csv", ["subject_id","hadm_id","stay_id","starttime","endtime","itemid","amount"], ["itemid"], "small"),
    "icu/outputevents.csv":   ("icu/outputevents.csv",["subject_id","hadm_id","stay_id","charttime","itemid","value"], ["itemid"], "small"),
}

CLASS_ORDER = ["class1_intra_table", "class2_entity_alignment", "class3_cross_table", "class4_task_oriented"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def stream_filter_csv(path: Path, key_tuples: set, key_cols: list[str], value_cols: list[str]) -> dict[tuple, list[dict]]:
    wanted: dict[tuple, list[dict]] = defaultdict(list)
    if not path.exists() or not key_tuples:
        return wanted
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kt = tuple(row.get(c, "") for c in key_cols)
            if kt in key_tuples:
                wanted[kt].append({c: row.get(c, "") for c in value_cols})
    return wanted


def main():
    if not MOD_LOG.exists():
        print(f"[ERROR] mod log not found: {MOD_LOG}")
        sys.exit(1)

    print(f"Loading mod log...", flush=True)
    mod_rows = list(csv.DictReader(MOD_LOG.open()))
    n_total = len(mod_rows)
    print(f"  {n_total} errors", flush=True)

    by_raw_idx: dict[str, set[int]] = defaultdict(set)
    for err in mod_rows:
        if err["dirty_value"] == "(error generating)" or err["clean_value"] == "(error generating)":
            continue
        by_raw_idx[err["raw_file"]].add(int(err["raw_row_index"]))

    dirty_by_file: dict[str, dict[int, dict]] = {}
    dirty_lookup_keys: dict[str, list[dict]] = defaultdict(list)
    print("Loading dirty raw rows...", flush=True)
    for raw_file, indices in by_raw_idx.items():
        if raw_file not in RAW_TO_OUTPUT:
            continue
        with (TEST_RAW / raw_file).open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i in indices:
                    dirty_by_file[raw_file] = dirty_by_file.get(raw_file, {})
                    dirty_by_file[raw_file][i] = row
                    dirty_lookup_keys[raw_file].append(row)

    out_lookups: dict[str, dict] = {}
    ref_lookups: dict[str, dict] = {}
    for raw_file, (out_rel, full_keys, dirty_in_key_cols, size_tag) in RAW_TO_OUTPUT.items():
        key_cols = [c for c in full_keys if c not in dirty_in_key_cols]
        dirty_keys_for_filter = set()
        for row in dirty_lookup_keys.get(raw_file, []):
            dirty_keys_for_filter.add(tuple(row.get(c, "") for c in key_cols))
        value_cols = list(full_keys)

        if size_tag == "big":
            print(f"  Streaming output {out_rel} (filter {len(dirty_keys_for_filter)} keys)...", flush=True)
            out_lookups[raw_file] = stream_filter_csv(TEST_OUTPUT / out_rel, dirty_keys_for_filter, key_cols, value_cols)
            print(f"    -> {sum(len(v) for v in out_lookups[raw_file].values())} matched rows", flush=True)
            print(f"  Streaming reference {out_rel}...", flush=True)
            ref_lookups[raw_file] = stream_filter_csv(REFERENCE / out_rel, dirty_keys_for_filter, key_cols, value_cols)
        else:
            out_path = TEST_OUTPUT / out_rel
            ref_path = REFERENCE / out_rel
            out_df = pd.DataFrame()
            ref_df = pd.DataFrame()
            if out_path.exists():
                out_df = pd.read_csv(out_path, dtype=str, keep_default_na=False, na_values=[], low_memory=False)
                if "Unnamed: 0" in out_df.columns:
                    out_df = out_df.drop(columns=["Unnamed: 0"])
            if ref_path.exists():
                ref_df = pd.read_csv(ref_path, dtype=str, keep_default_na=False, na_values=[], low_memory=False)
                if "Unnamed: 0" in ref_df.columns:
                    ref_df = ref_df.drop(columns=["Unnamed: 0"])
            keys_filter = [row.get(key_cols[0], "") for row in dirty_lookup_keys.get(raw_file, [])]
            out_sub = out_df[out_df[key_cols[0]].isin(keys_filter)] if key_cols[0] in out_df.columns else pd.DataFrame()
            ref_sub = ref_df[ref_df[key_cols[0]].isin(keys_filter)] if key_cols[0] in ref_df.columns else pd.DataFrame()
            for c in key_cols[1:]:
                if c in out_sub.columns and not out_sub.empty:
                    vals = [row.get(c, "") for row in dirty_lookup_keys.get(raw_file, [])]
                    out_sub = out_sub[out_sub[c].isin(vals)]
                if c in ref_sub.columns and not ref_sub.empty:
                    vals = [row.get(c, "") for row in dirty_lookup_keys.get(raw_file, [])]
                    ref_sub = ref_sub[ref_sub[c].isin(vals)]
            out_lookups[raw_file] = out_sub.to_dict("records") if not out_sub.empty else []
            ref_lookups[raw_file] = ref_sub.to_dict("records") if not ref_sub.empty else []
            print(f"  Loaded {out_rel}: output {len(out_lookups[raw_file])}, ref {len(ref_lookups[raw_file])} rows", flush=True)

    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {
        "detect_TP": 0, "detect_FN": 0, "detect_ambig": 0,
        "repair_TP": 0, "repair_FP": 0, "repair_FN": 0,
        "excluded": 0, "fixed": 0, "over_corrected": 0, "missing": 0, "not_fixed": 0,
    })

    n_eval = 0
    for err in mod_rows:
        cls = err["error_class"]
        col = err["column"]
        clean_v = err["clean_value"]
        dirty_v = err["dirty_value"]
        op = err["operation"]
        row_idx = int(err["raw_row_index"])
        raw_file = err["raw_file"]

        if dirty_v == "(error generating)" or clean_v == "(error generating)":
            per_class[cls]["excluded"] += 1
            continue
        n_eval += 1

        if raw_file not in RAW_TO_OUTPUT:
            per_class[cls]["detect_FN"] += 1
            per_class[cls]["repair_FN"] += 1
            per_class[cls]["not_fixed"] += 1
            continue

        _, full_keys, dirty_in_key_cols, size_tag = RAW_TO_OUTPUT[raw_file]
        dirty_row = dirty_by_file.get(raw_file, {}).get(row_idx)
        if dirty_row is None:
            per_class[cls]["detect_FN"] += 1
            per_class[cls]["repair_FN"] += 1
            per_class[cls]["not_fixed"] += 1
            continue

        key_cols = [c for c in full_keys if c not in dirty_in_key_cols]
        partial_key = tuple(dirty_row.get(c, "") for c in key_cols)
        out_data = out_lookups.get(raw_file)

        if size_tag == "big":
            out_rows = out_data.get(partial_key, []) if out_data else []
        else:
            out_rows = [r for r in (out_data or []) if all(r.get(c, "") == dirty_row.get(c, "") for c in key_cols)]

        out_values = [r.get(col, "") for r in out_rows]
        n_dirty = sum(1 for v in out_values if v == dirty_v)
        n_clean = sum(1 for v in out_values if v == clean_v)

        if op in ("duplicate", "insert"):
            ref_data = ref_lookups.get(raw_file, {})
            if size_tag == "big":
                n_in_ref = len(ref_data.get(partial_key, []))
            else:
                ref_rows = [r for r in ref_data if all(r.get(c, "") == dirty_row.get(c, "") for c in key_cols)]
                n_in_ref = len(ref_rows)
            if len(out_rows) > n_in_ref:
                per_class[cls]["detect_FN"] += 1
                per_class[cls]["repair_FN"] += 1
                per_class[cls]["not_fixed"] += 1
            else:
                per_class[cls]["detect_TP"] += 1
                per_class[cls]["repair_TP"] += 1
                per_class[cls]["fixed"] += 1
            continue

        if dirty_v == clean_v:
            per_class[cls]["detect_ambig"] += 1
            continue

        if n_dirty == 0:
            per_class[cls]["detect_TP"] += 1
        else:
            per_class[cls]["detect_FN"] += 1
            per_class[cls]["not_fixed"] += 1

        if n_clean > 0 and n_dirty == 0:
            per_class[cls]["repair_TP"] += 1
            per_class[cls]["fixed"] += 1
        elif n_dirty == 0:
            per_class[cls]["repair_FP"] += 1
            per_class[cls]["over_corrected"] += 1
        else:
            per_class[cls]["repair_FN"] += 1

    tot = Counter()
    for cls, m in per_class.items():
        for k, v in m.items():
            tot[k] += v

    print()
    print("=" * 100)
    print(f"N18 v2 ERROR DETECTION & REPAIR  (eval_n={n_eval}/{n_total}, excluded {n_total-n_eval})")
    print("=" * 100)
    print()
    print(f"  TOTAL DETECTION : TP={tot['detect_TP']} FN={tot['detect_FN']} ambig={tot['detect_ambig']}")
    print(f"  TOTAL REPAIR    : TP={tot['repair_TP']} FP={tot['repair_FP']} FN={tot['repair_FN']}")
    print()
    print(f"  TOTAL REPAIR DISTRIBUTION:")
    print(f"    fixed           = {tot['fixed']:>5}  ({100*tot['fixed']/n_eval:.2f}%)")
    print(f"    over_corrected  = {tot['over_corrected']:>5}  ({100*tot['over_corrected']/n_eval:.2f}%)")
    print(f"    missing         = {tot['missing']:>5}  ({100*tot['missing']/n_eval:.2f}%)")
    print(f"    not_fixed       = {tot['not_fixed']:>5}  ({100*tot['not_fixed']/n_eval:.2f}%)")

    print()
    print("=" * 100)
    print("TABLE: PER-CLASS METRICS  (1800 injected errors)")
    print("=" * 100)
    print()
    print(f"  {'Class':<28} {'N_eval':>6} | {'DETECTION':<32} | {'REPAIR':<32}")
    print(f"  {'':<28} {'':>6} | {'TP':>5} {'FN':>5} {'Precision':>9} {'Recall':>7} {'F1':>7} | {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>9} {'Recall':>7} {'F1':>7}")
    print("  " + "-" * 116)

    class_metrics = {}
    for cls in CLASS_ORDER:
        m = per_class.get(cls, {})
        d_tp = m.get("detect_TP", 0); d_fn = m.get("detect_FN", 0)
        det_p, det_r, det_f = prf(d_tp, 0, d_fn)
        r_tp = m.get("repair_TP", 0); r_fp = m.get("repair_FP", 0); r_fn = m.get("repair_FN", 0)
        rep_p, rep_r, rep_f = prf(r_tp, r_fp, r_fn)
        n_eval_cls = d_tp + d_fn + m.get("detect_ambig", 0)
        class_metrics[cls] = {
            "n_evaluated": n_eval_cls,
            "detect_TP": d_tp, "detect_FN": d_fn, "detect_ambig": m.get("detect_ambig", 0),
            "detect_P": det_p, "detect_R": det_r, "detect_F1": det_f,
            "repair_TP": r_tp, "repair_FP": r_fp, "repair_FN": r_fn,
            "repair_P": rep_p, "repair_R": rep_r, "repair_F1": rep_f,
            "fixed": m.get("fixed", 0),
            "over_corrected": m.get("over_corrected", 0),
            "missing": m.get("missing", 0),
            "not_fixed": m.get("not_fixed", 0),
        }
        print(f"  {cls:<28} {n_eval_cls:>6} | {d_tp:>5} {d_fn:>5} {'1.0000':>9} {det_r:>7.4f} {det_f:>7.4f} | {r_tp:>5} {r_fp:>5} {r_fn:>5} {rep_p:>9.4f} {rep_r:>7.4f} {rep_f:>7.4f}")

    print("  " + "-" * 116)
    d_tp, d_fn = tot["detect_TP"], tot["detect_FN"]
    r_tp, r_fp, r_fn = tot["repair_TP"], tot["repair_FP"], tot["repair_FN"]
    det_p, det_r, det_f = prf(d_tp, 0, d_fn)
    rep_p, rep_r, rep_f = prf(r_tp, r_fp, r_fn)
    print(f"  {'TOTAL':<28} {n_eval:>6} | {d_tp:>5} {d_fn:>5} {'1.0000':>9} {det_r:>7.4f} {det_f:>7.4f} | {r_tp:>5} {r_fp:>5} {r_fn:>5} {rep_p:>9.4f} {rep_r:>7.4f} {rep_f:>7.4f}")

    out_dir = Path("/Users/mac/PycharmProjects/v2_clean/runs")
    summary = {
        "n_total_in_mod_log": n_total,
        "n_evaluated": n_eval,
        "n_excluded_gen_fail": n_total - n_eval,
        "totals": {
            "detect_TP": d_tp, "detect_FN": d_fn, "detect_ambig": tot["detect_ambig"],
            "repair_TP": r_tp, "repair_FP": r_fp, "repair_FN": r_fn,
            "detect_P": 1.0, "detect_R": round(det_r, 4), "detect_F1": round(det_f, 4),
            "repair_P": round(rep_p, 4), "repair_R": round(rep_r, 4), "repair_F1": round(rep_f, 4),
            "fixed": tot["fixed"], "over_corrected": tot["over_corrected"],
            "missing": tot["missing"], "not_fixed": tot["not_fixed"],
        },
        "per_class": class_metrics,
    }
    p1 = out_dir / "n18_v2_error_metrics_per_class.json"
    p1.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {p1}")


if __name__ == "__main__":
    main()