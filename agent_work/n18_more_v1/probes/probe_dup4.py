"""Probe: separate injected duplicates from legitimate exact duplicates."""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
RAW = f"{BASE}/raw"
REF = f"{BASE}/reference"

ce = pd.read_csv(f"{RAW}/icu/chartevents.csv", dtype=str)
ref = pd.read_csv(f"{REF}/icu/chartevents.csv", dtype=str)
refics = pd.read_csv(f"{REF}/icu/icustays.csv", dtype=str).set_index("stay_id")
stays = list(set(ref.stay_id))
ce = ce[ce.stay_id.isin(stays)].reset_index(drop=True)
ce = ce[ce.itemid != "999001"].reset_index(drop=True)
ce["it2"] = ce.itemid.str.replace(r"^99(?=\d{6}$)", "", regex=True)
ce["ct"] = pd.to_datetime(ce.charttime, errors="coerce")
ce["ot"] = pd.to_datetime(ce.stay_id.map(refics.outtime))
ce["bad"] = (ce.ct - ce.ot) == pd.Timedelta(days=1)
out = ce.charttime.copy()
prev = None
for i in ce.index:
    if ce.at[i, "bad"]:
        out.at[i] = prev
    else:
        prev = ce.at[i, "charttime"]
ce["ctfix"] = out

key = ["stay_id", "ctfix", "it2", "value", "valuenum", "valueuom", "storetime", "caregiver_id"]
ce["_k"] = list(zip(*[ce[c].fillna("<NA>").astype(str) for c in key]))

# reference side: how many rows does the ref have for each (stay, charttime, itemid)?
refkey = Counter(zip(ref.stay_id, ref.charttime, ref.itemid))
rawkey = Counter(zip(ce.stay_id, ce.ctfix, ce.it2))

# For each duplicate group: does the ref have as many rows with that
# (stay, charttime, itemid) key?  If fewer -> the extra copy is injected.
groups = []
for k, s in ce.groupby("_k", dropna=False):
    if len(s) < 2:
        continue
    idx = sorted(s.index)
    rk = (s.stay_id.iloc[0], s.ctfix.iloc[0], s.it2.iloc[0])
    groups.append((idx, rk, refkey.get(rk, 0), len(idx)))

# the number of rows in the ref with that (stay,charttime,itemid) minus the
# number of *distinct* (itemid,value,...) groups is the number of injected copies
print("dup groups:", len(groups))
# how many groups have refcount < group size
under = [g for g in groups if g[2] < g[3]]
print("groups where ref has fewer rows than the raw group:", len(under))
print("  total injected copies implied:", sum(g[3] - g[2] for g in under))

# distance between the copies
dists = []
for idx, rk, rc, n in groups:
    if n == 2:
        dists.append(idx[1] - idx[0])
d = Counter(min(x, 20) for x in dists)
print("\ndistance histogram (capped at 20):", sorted(d.items())[:22])

# positional: are the injected copies the ones that appear in a different
# "charttime block" than their twin?
print("\nsample groups where ref has fewer rows:")
for g in under[:10]:
    idx, rk, rc, n = g
    print("   positions", idx, "refcount", rc, "rawcount", n)
