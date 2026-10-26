"""Probe the remaining chartevents row-set mismatch."""
from __future__ import annotations

from collections import Counter, defaultdict

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

# neighbour repair
out = r.charttime.copy()
prev = None
for i in r.index:
    if r.at[i, "bad"]:
        out.at[i] = prev
    else:
        prev = r.at[i, "charttime"]
r["ctfix"] = out

# valuenum repair (numeric value wins)
r["vn"] = pd.to_numeric(r.value, errors="coerce")

print("== compare by (stay, itemid) ==")
cr = Counter(zip(r.stay_id, r.it2))
cf = Counter(zip(ref.stay_id, ref.itemid))
print("  only-in-raw", sum((cr - cf).values()), " only-in-ref", sum((cf - cr).values()))

print("== compare by (stay, itemid, valuenum) ==")
cr2 = Counter(zip(r.stay_id, r.it2, r.vn))
cf2 = Counter(zip(ref.stay_id, ref.itemid, pd.to_numeric(ref.valuenum, errors="coerce")))
print("  only-in-raw", sum((cr2 - cf2).values()), " only-in-ref", sum((cf2 - cr2).values()))
print("  sample only-in-ref:", list((cf2 - cr2).items())[:8])
print("  sample only-in-raw:", list((cr2 - cf2).items())[:8])

print("== compare by (stay, charttime-fixed, itemid) ==")
cr3 = Counter(zip(r.stay_id, r.ctfix, r.it2))
cf3 = Counter(zip(ref.stay_id, ref.charttime, ref.itemid))
print("  only-in-raw", sum((cr3 - cf3).values()), " only-in-ref", sum((cf3 - cr3).values()))

# Are the extra raw rows exact duplicates of other rows in the same stay?
key = ["stay_id", "ctfix", "it2", "value", "valuenum", "valueuom"]
dups = r.duplicated(subset=key, keep="first")
print("\nraw rows that are exact duplicates (after repair):", dups.sum())
print("  of which 999001-like? ", r.loc[dups, "it2"].eq("999001").sum())
grp = r.groupby(key, dropna=False).size()
print("  duplicate groups:", (grp > 1).sum(), "extra rows in groups:", (grp[grp > 1] - 1).sum())

# how many of the only-in-raw rows are exact duplicates of an only-in-ref... no.
# check: do the 'only-in-ref' rows exist in the raw at a different charttime?
onlyref = cf3 - cr3
missing_rows = []
for (st, ct, it), cnt in onlyref.items():
    cand = r[(r.stay_id == st) & (r.it2 == it)]
    missing_rows.append((st, ct, it, len(cand), sorted(cand.charttime.unique())[:4]))
print("\nsample only-in-ref rows and their candidates in the raw:")
for m in missing_rows[:12]:
    print("   ", m)
print("total only-in-ref rows:", len(missing_rows),
      " with 0 candidates:", sum(1 for m in missing_rows if m[3] == 0))
