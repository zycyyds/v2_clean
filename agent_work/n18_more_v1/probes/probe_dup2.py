"""Probe: identify the injected duplicate rows + validate the valuenum repair."""
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
out = r.charttime.copy()
prev = None
for i in r.index:
    if r.at[i, "bad"]:
        out.at[i] = prev
    else:
        prev = r.at[i, "charttime"]
r["ctfix"] = out

# ---- valuenum repair ------------------------------------------------------
SENT = {-999.0, -888.0, 9999.0, -1.0}
r["vn_num"] = pd.to_numeric(r.value, errors="coerce")
r["vn_raw"] = pd.to_numeric(r.valuenum, errors="coerce")
text = r.vn_num.isna()
# per (itemid, value) mode over trustworthy rows
mode_item = {}
tmp = r[text & ~r.vn_raw.isin(SENT) & (r.vn_raw >= 0)]
for (it, vl), s in tmp.groupby(["it2", "value"])["vn_raw"]:
    mode_item[(it, vl)] = s.mode().iat[0]
# global value_text mode
mode_glob = {}
tmp2 = r[text & ~r.vn_raw.isin(SENT) & (r.vn_raw >= 0)]
for vl, s in tmp2.groupby("value")["vn_raw"]:
    mode_glob[vl] = s.mode().iat[0]
rep = []
for i in r.index:
    if r.at[i, "vn_num"] == r.at[i, "vn_num"]:
        rep.append(r.at[i, "vn_num"])
        continue
    it, vl = r.at[i, "it2"], r.at[i, "value"]
    rep.append(mode_item.get((it, vl), mode_glob.get(vl, np.nan if False else None)))
r["vn_fix"] = pd.Series(rep, index=r.index, dtype="float64")

# accuracy of the valuenum repair against the reference (aligned by row index)
n = min(len(r), len(ref))
# align: use the ref order? do it per (stay, charttime, itemid) key instead
rkey = Counter(zip(r.stay_id, r.ctfix, r.it2))
fkey = Counter(zip(ref.stay_id, ref.charttime, ref.itemid))
print("row-set (stay,charttime,itemid): only-in-raw", sum((rkey - fkey).values()),
      "only-in-ref", sum((fkey - rkey).values()))

# valuenum agreement for the rows that match on (stay, charttime, itemid)
rk = list(zip(r.stay_id, r.ctfix, r.it2))
fk = list(zip(ref.stay_id, ref.charttime, ref.itemid))
from collections import defaultdict
buckets = defaultdict(list)
for i, k in enumerate(rk):
    buckets[k].append(i)
agree = tot = 0
for j, k in enumerate(fk):
    if k in buckets and buckets[k]:
        i = buckets[k].pop(0)
        a = r.at[i, "vn_fix"]
        b = pd.to_numeric(ref.at[j, "valuenum"], errors="coerce")
        if pd.isna(a) and pd.isna(b):
            agree += 1
        elif a == b:
            agree += 1
        tot += 1
print(f"valuenum agreement on matched rows: {agree}/{tot} = {agree/max(tot,1):.4f}")

# ---- duplicate detection ---------------------------------------------------
full = ["stay_id", "ctfix", "it2", "value", "vn_fix", "valueuom", "storetime", "caregiver_id"]
r["_k"] = list(zip(*[r[c].fillna("<NA>").astype(str) for c in full]))
grp = r.groupby("_k", dropna=False).size()
print("\nduplicate groups (full row incl storetime/caregiver):", (grp > 1).sum(),
      "extra rows:", (grp[grp > 1] - 1).sum())
# how many duplicate groups have a member with bad charttime
badset = set(r.index[r.bad])
cnt = 0
withbad = 0
for k, s in r.groupby("_k", dropna=False):
    if len(s) > 1:
        cnt += 1
        if any(i in badset for i in s.index):
            withbad += 1
print(f"  groups with a corrupted-charttime member: {withbad}/{cnt}")
