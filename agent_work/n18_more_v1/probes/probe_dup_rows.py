"""Probe: which exact-duplicate rows in the raw are the injected ones?"""
from __future__ import annotations

from collections import Counter

import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
RAW = f"{BASE}/raw"
REF = f"{BASE}/reference"

raw = pd.read_csv(f"{RAW}/icu/chartevents.csv", dtype=str)
ref = pd.read_csv(f"{REF}/icu/chartevents.csv", dtype=str)
refics = pd.read_csv(f"{REF}/icu/icustays.csv", dtype=str).set_index("stay_id")

stays = list(set(ref.stay_id))
r = raw[raw.stay_id.isin(stays)].reset_index(drop=True)
r = r[r.itemid != "999001"].reset_index(drop=True)
r["it2"] = r.itemid.str.replace(r"^99(?=\d{6}$)", "", regex=True)
r["ct"] = pd.to_datetime(r.charttime, errors="coerce")
r["ot"] = pd.to_datetime(r.stay_id.map(refics.outtime))
r["bad"] = (r.ct - r.ot) == pd.Timedelta(days=1)
out = r.charttime.copy()
prev = None
for i in r.index:
    if r.at[i, "bad"]:
        out.at[i] = prev
    else:
        prev = r.at[i, "charttime"]
r["ctfix"] = out
r["vn"] = pd.to_numeric(r.value, errors="coerce")

key = ["stay_id", "ctfix", "it2", "value", "vn", "valueuom"]
rkey = Counter(zip(*[r[k].fillna("<NA>") for k in key]))
fkey = Counter(
    zip(
        ref.stay_id,
        ref.charttime,
        ref.itemid,
        pd.to_numeric(ref.valuenum, errors="coerce").fillna(-999999),
        ref.valueuom.fillna("<NA>"),
    )
)
# rebuild the ref key in the same shape (value text is not in the ref, use valuenum)
fkey2 = Counter(
    zip(
        ref.stay_id,
        ref.charttime,
        ref.itemid,
        pd.to_numeric(ref.valuenum, errors="coerce").fillna(-999999),
        ref.valueuom.fillna("<NA>"),
    )
)
rkey2 = Counter(
    zip(
        r.stay_id,
        r.ctfix,
        r.it2,
        r.vn.fillna(-999999),
        r.valueuom.fillna("<NA>"),
    )
)
extra = rkey2 - fkey2
print("raw-only groups:", len(extra), "rows:", sum(extra.values()))
print("ref-only groups:", len(fkey2 - rkey2), "rows:", sum((fkey2 - rkey2).values()))

# for the groups where the raw has 2 identical rows: does the ref have 2 as well?
grp = r.groupby(key, dropna=False).size().rename("raw_n").reset_index()
grp["ref_n"] = [
    fkey2.get(
        (s, c, i, -999999 if pd.isna(v) else v, u), 0
    )
    for s, c, i, v, u in zip(grp.stay_id, grp.ctfix, grp.it2, grp.vn, grp.valueuom.fillna("<NA>"))
]
print("\ngroup size distribution (raw_n x ref_n):")
print(grp.groupby(["raw_n", "ref_n"]).size().to_string())

# adjacency: are the injected duplicates adjacent to their twin?
dups = grp[grp.raw_n > grp.ref_n]
print("\ninjected-duplicate groups:", len(dups))
idx_of = {}
for k, g in r.groupby(key, dropna=False):
    idx_of[k] = list(g.index)
adj = 0
tot = 0
for _, row in dups.iterrows():
    k = (row.stay_id, row.ctfix, row.it2, row.vn, row.valueuom.fillna("<NA>"))
    g = r.groupby(key, dropna=False)
    positions = []
    for kk, gg in g:
        if kk[0] == k[0] and kk[1] == k[1] and kk[2] == k[2]:
            positions = list(gg.index)
    positions = sorted(positions)
    tot += 1
    if len(positions) >= 2 and (positions[1] - positions[0]) <= 3:
        adj += 1
print(f"  injected dup twins adjacent (<=3 apart): {adj}/{tot}")

# how many rows in the ref are exact duplicates (ref_n>1)?
print("\nref groups with ref_n>1:", (grp.ref_n > 1).sum(), "rows involved:", grp.loc[grp.ref_n > 1, "ref_n"].sum())
