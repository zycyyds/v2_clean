"""Compare chartevents strategies with the key-based evaluation."""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
import n18_clean  # noqa: E402
from eval_chartevents import evaluate  # noqa: E402

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
OUT = os.path.join(HERE, "..", "work", "train_out")

ref = pd.read_csv(f"{BASE}/reference/icu/chartevents.csv", dtype=str)
cur = pd.read_csv(f"{OUT}/icu/chartevents.csv", dtype=str)
print("current pipeline output:", len(cur), "ref:", len(ref))
evaluate(cur, ref, "pipeline (all rows)")

# variant: drop adjacent exact duplicates
raw = pd.read_csv(f"{BASE}/raw/icu/chartevents.csv", dtype=str)
ics = pd.read_csv(f"{BASE}/raw/icu/icustays.csv", dtype=str)
refics = pd.read_csv(f"{BASE}/reference/icu/icustays.csv", dtype=str).set_index("stay_id")
stays = set(ref.stay_id)
raw = raw[raw.stay_id.isin(stays)].reset_index(drop=True)
stay_meta = refics.loc[list(stays)][["subject_id", "hadm_id", "outtime"]]
ce = n18_clean.repair_chartevents(raw.copy(), stay_meta)
key = ["stay_id", "charttime", "itemid", "value", "valuenum", "valueuom", "storetime",
       "caregiver_id"]
for c in key:
    ce[c] = ce[c].fillna("~").astype(str)
dup_mask = ce.duplicated(subset=key, keep="first")
print("\nadjacent/other exact duplicates:", dup_mask.sum())
v = ce[~dup_mask]
v = v[["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"]]
evaluate(v.reset_index(drop=True), ref, "drop all exact duplicates")
