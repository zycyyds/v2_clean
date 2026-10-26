"""Probe: which rule best restores the charttime of the rows the corrupter
pushed outside the ICU task window (charttime := outtime + 1 day)?"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
RAW = f"{BASE}/raw"
REF = f"{BASE}/reference"

raw = pd.read_csv(f"{RAW}/icu/chartevents.csv", dtype=str)
ref = pd.read_csv(f"{REF}/icu/chartevents.csv", dtype=str)
ics = pd.read_csv(f"{RAW}/icu/icustays.csv", dtype=str).set_index("stay_id")
refics = pd.read_csv(f"{REF}/icu/icustays.csv", dtype=str).set_index("stay_id")

stays = list(set(ref.stay_id))
r = raw[raw.stay_id.isin(stays)].reset_index(drop=True)
r = r[r.itemid != "999001"].reset_index(drop=True)
r["it2"] = r.itemid.str.replace(r"^99(?=\d{6}$)", "", regex=True)
r["ct"] = pd.to_datetime(r.charttime, errors="coerce")
r["ot"] = pd.to_datetime(r.stay_id.map(refics.outtime))
r["bad"] = (r.ct - r.ot) == pd.Timedelta(days=1)
print("corrupted charttime rows:", r.bad.sum(), "of", len(r))

good = r[~r.bad]
# candidate A: (stay, storetime, itemid) -> charttime
mapA = defaultdict(Counter)
for _, x in good.iterrows():
    mapA[(x.stay_id, x.storetime, x.it2)][x.charttime] += 1
# candidate B: (stay, storetime) -> set of charttimes
mapB = defaultdict(Counter)
for _, x in good.iterrows():
    mapB[(x.stay_id, x.storetime)][x.charttime] += 1
# candidate C: (stay, storetime, itemid) from a *different* charttime relation
print("mapB group cardinality histogram:",
      Counter(len(c) for c in mapB.values()).most_common(8))


def repair(strategy):
    out = r.charttime.copy()
    # neighbour fallback: previous uncorrupted row of the same stay
    prev = None
    for i in r.index:
        if strategy in ("A", "AB", "ABC"):
            k = (r.at[i, "stay_id"], r.at[i, "storetime"], r.at[i, "it2"])
            c = mapA.get(k)
            if c and len(c) == 1:
                out.at[i] = list(c)[0]
                continue
        if strategy in ("B", "AB", "ABC"):
            c = mapB.get((r.at[i, "stay_id"], r.at[i, "storetime"]))
            if c and len(c) == 1:
                out.at[i] = list(c)[0]
                continue
        if r.at[i, "bad"]:
            out.at[i] = prev
        else:
            prev = r.at[i, "charttime"]
    return out


for strat in ["N", "A", "B", "AB", "ABC"]:
    ct = repair(strat)
    cr = Counter(zip(r.stay_id, ct, r.it2))
    cf = Counter(zip(ref.stay_id, ref.charttime, ref.itemid))
    onlyr = sum((cr - cf).values())
    onlyf = sum((cf - cr).values())
    print(f"strategy {strat:4s}: only-in-raw {onlyr:5d}  only-in-ref {onlyf:5d}")
