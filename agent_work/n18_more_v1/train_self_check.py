"""Train self-check: run the pipeline on train/raw and diff against train/reference.

    python3 selfcheck_train.py [--out DIR] [--json FILE]

Two metrics are reported per output file:
  * positional  - the produced rows are compared with the reference rows in
                  file order (the reference preserves the pristine file order)
  * merged      - the produced rows are joined with the reference rows on the
                  natural key of the table (a row-aligned / key-aligned score)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pipeline"))
import n18_clean  # noqa: E402

DATASET = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more"
TRAIN_RAW = os.path.join(DATASET, "train", "raw")
TRAIN_REF = os.path.join(DATASET, "train", "reference")

FILES = n18_clean.OUTPUT_FILES

# natural join key of every output table
KEYS = {
    "admissions.csv": ["subject_id", "hadm_id"],
    "cohort.csv": ["subject_id", "hadm_id"],
    "demographics.csv": ["subject_id"],
    "diagnoses.csv": ["subject_id", "hadm_id", "seq_num"],
    "procedures.csv": ["subject_id", "hadm_id", "seq_num"],
    "medications.csv": ["subject_id", "hadm_id", "starttime", "drug"],
    "labs.csv": ["subject_id", "hadm_id", "itemid", "charttime"],
    "icu/icustays.csv": ["stay_id"],
    "icu/chartevents.csv": ["stay_id", "charttime", "itemid"],
    "icu/inputevents.csv": ["stay_id", "starttime", "itemid"],
    "icu/outputevents.csv": ["stay_id", "charttime", "itemid"],
    "multimodal/notes.csv": ["note_id"],
    "multimodal/ed_diagnosis.csv": ["stay_id", "seq_num"],
    "multimodal/ed_medrecon.csv": ["stay_id", "charttime", "name", "gsn"],
    "multimodal/ed_pyxis.csv": ["stay_id", "charttime", "name", "gsn_rn"],
    "multimodal/ed_triage.csv": ["stay_id"],
    "multimodal/ed_vitalsign.csv": ["stay_id", "charttime"],
    "multimodal/edstays.csv": ["stay_id"],
}


def norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v)
    if s in ("nan", "NaN", "<NA>"):
        return ""
    return s


def cell_eq(x, y, numeric=False):
    xs, ys = norm(x), norm(y)
    if numeric:
        try:
            return float(xs) == float(ys)
        except ValueError:
            return xs == ys
    return xs == ys


def positional(a, b):
    cols = [c for c in b.columns if c in a.columns]
    n = min(len(a), len(b))
    eq = tot = 0
    per = {}
    for c in cols:
        m = t = 0
        for i in range(n):
            t += 1
            if cell_eq(a[c].iat[i], b[c].iat[i], c in ("valuenum", "value", "amount")):
                m += 1
        per[c] = {"matched": m, "total": t, "acc": round(m / max(t, 1), 4)}
        eq += m
        tot += t
    return eq, tot, per


def merged(a, b, key):
    cols = [c for c in b.columns if c in a.columns]
    num = {"valuenum", "value", "amount", "gsn", "ndc", "etccode", "temperature", "heartrate",
           "resprate", "o2sat", "sbp", "dbp", "acuity", "hadm_id"}
    m = a.merge(b, on=key, how="inner", suffixes=("_out", "_ref"))
    per = {}
    eq = tot = 0
    for c in cols:
        if c in key:
            continue
        x = m[c + "_out"] if c + "_out" in m.columns else m[c]
        y = m[c + "_ref"] if c + "_ref" in m.columns else m[c]
        ok = [cell_eq(u, v, c in num) for u, v in zip(x, y)]
        per[c] = {"matched": int(sum(ok)), "total": len(ok),
                  "acc": round(sum(ok) / max(len(ok), 1), 4)}
        eq += int(sum(ok))
        tot += len(ok)
    ka = Counter(tuple(norm(v) for v in row) for row in a[key].itertuples(index=False))
    kb = Counter(tuple(norm(v) for v in row) for row in b[key].itertuples(index=False))
    inter = sum((ka & kb).values())
    prec = inter / max(sum(ka.values()), 1)
    rec = inter / max(sum(kb.values()), 1)
    return eq, tot, per, prec, rec, len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "work", "train_out"))
    ap.add_argument("--json", default=os.path.join(HERE, "analysis", "train_selfcheck.json"))
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    n18_clean.run(TRAIN_RAW, TRAIN_REF, out, workdir=os.path.join(out, "_work"))

    summary = {"files": {}, "totals": {"positional_matched": 0, "positional_total": 0,
                                        "merged_matched": 0, "merged_total": 0}}
    for rel in FILES:
        p_out = os.path.join(out, rel)
        p_ref = os.path.join(TRAIN_REF, rel)
        a = pd.read_csv(p_out, dtype=str)
        b = pd.read_csv(p_ref, dtype=str)
        peq, ptot, pper = positional(a, b)
        key = KEYS[rel]
        meq, mtot, mper, prec, rec, nmerged = merged(a, b, key)
        summary["files"][rel] = {
            "out_rows": len(a),
            "ref_rows": len(b),
            "positional": {"cells_matched": peq, "cells_total": ptot,
                           "cell_acc": round(peq / max(ptot, 1), 4), "per_col": pper},
            "merged": {"cells_matched": meq, "cells_total": mtot,
                       "cell_acc": round(meq / max(mtot, 1), 4), "per_col": mper,
                       "row_precision": round(prec, 4), "row_recall": round(rec, 4),
                       "row_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                       "merged_rows": nmerged},
        }
        summary["totals"]["positional_matched"] += peq
        summary["totals"]["positional_total"] += ptot
        summary["totals"]["merged_matched"] += meq
        summary["totals"]["merged_total"] += mtot

    t = summary["totals"]
    summary["overall_positional_cell_acc"] = round(t["positional_matched"] / max(t["positional_total"], 1), 4)
    summary["overall_merged_cell_acc"] = round(t["merged_matched"] / max(t["merged_total"], 1), 4)
    with open(args.json, "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    print(f"{'file':32s} {'rows':>16s} {'positional':>18s} {'merged':>18s}  rowF1")
    for rel, info in summary["files"].items():
        p = info["positional"]
        m = info["merged"]
        print(f"{rel:32s} {info['out_rows']:7d}/{info['ref_rows']:<8d} "
              f"{p['cell_acc']:8.4f} ({p['cells_matched']}/{p['cells_total']}) "
              f"{m['cell_acc']:8.4f} ({m['cells_matched']}/{m['cells_total']})  {m['row_f1']:.4f}")
    print(f"\nOVERALL positional cell accuracy: {summary['overall_positional_cell_acc']:.4f}")
    print(f"OVERALL merged     cell accuracy: {summary['overall_merged_cell_acc']:.4f}")
    print("wrote", args.json)


if __name__ == "__main__":
    main()
