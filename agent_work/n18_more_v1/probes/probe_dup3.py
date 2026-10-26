"""Probe: are the window-leaked chartevents rows injected duplicates?"""
from __future__ import annotations

from collections import Counter

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
sizes = ce.groupby("_k", dropna=False).size()
dup_keys = set(sizes[sizes > 1].index)
print("rows:", len(ce), " duplicate groups:", len(dup_keys),
      " extra duplicate rows:", sum(sizes[sizes > 1] - 1))

bad_idx = set(ce.index[ce.bad])
# for each duplicated group, how many members are 'bad' (window-leaked)?
cnt_bad_members = Counter()
groups_with_bad = 0
for k, s in ce.groupby("_k", dropna=False):
    if len(s) > 1:
        groups_with_bad += 1
        n_bad = sum(1 for i in s.index if i in bad_idx)
        cnt_bad_members[n_bad] += 1
print("dup groups by number of window-leaked members:", dict(cnt_bad_members))
print("dup groups containing >=1 leaked row:", sum(v for k, v in cnt_bad_members.items() if k >= 1))

# If the leaked rows are the injected duplicates, dropping one leaked member of
# every duplicated group should give exactly the reference row count.
leaked_in_dup = sum(v for k, v in cnt_bad_members.items() if k >= 1)
print("\nrows to drop = one leaked member per duplicated group ->",
      len(ce) - sum(1 for k, v in cnt_bad_members.items() if k >= 1))
print("reference rows:", len(ref))

# how many leaked rows are NOT part of a duplicated group?
leaked_only = sum(1 for i in bad_idx if ce.at[i, "_k"] not in dup_keys)
print("leaked rows that are unique after repair:", leaked_only)
print("leaked rows total:", len(bad_idx))
