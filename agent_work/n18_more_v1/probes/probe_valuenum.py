"""Probe the valuenum <-> value text relationship (labs + chartevents)."""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
RAW = f"{BASE}/raw"
REF = f"{BASE}/reference"

# ---------------- labs ----------------
lab = pd.read_csv(f"{RAW}/hosp/labevents.csv", dtype=str)
reflab = pd.read_csv(f"{REF}/labs.csv", dtype=str)
coh = set(pd.read_csv(f"{REF}/cohort.csv", dtype=str).hadm_id)
lab = lab[lab.hadm_id.isin(coh)]
v = pd.to_numeric(lab.value, errors="coerce")
n = pd.to_numeric(lab.valuenum, errors="coerce")
print("labs rows:", len(lab), "value numeric:", v.notna().sum(), "valuenum null:", n.isna().sum())
print("  value numeric & valuenum != value:", ((v.notna()) & (n != v) & n.notna()).sum())
print("  value non-numeric & valuenum not null:", ((v.isna()) & n.notna()).sum())
print("  value numeric & valuenum null:", ((v.notna()) & n.isna()).sum())
print("  both null:", ((v.isna()) & n.isna()).sum())
# is the ref valuenum == float(value)?
m = reflab.merge(
    lab[["hadm_id", "itemid", "charttime", "value"]], on=["hadm_id", "itemid", "charttime"], how="left"
)
rv = pd.to_numeric(m.value, errors="coerce")
rn = pd.to_numeric(m.valuenum, errors="coerce")
both = rv.notna() & rn.notna()
print("  ref valuenum == float(value)?", (rv[both] == rn[both]).mean(), both.sum())
print("  ref valuenum null & value numeric:", (rn.isna() & rv.notna()).sum())

# ---------------- chartevents ----------------
ce = pd.read_csv(f"{RAW}/icu/chartevents.csv", dtype=str)
refce = pd.read_csv(f"{REF}/icu/chartevents.csv", dtype=str)
refics = pd.read_csv(f"{REF}/icu/icustays.csv", dtype=str).set_index("stay_id")
stays = list(set(refce.stay_id))
ce = ce[ce.stay_id.isin(stays)].reset_index(drop=True)
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
ce["vn_num"] = pd.to_numeric(ce.value, errors="coerce")
ce["vn_raw"] = pd.to_numeric(ce.valuenum, errors="coerce")
print("\nchartevents rows:", len(ce))
print("  value numeric:", ce.vn_num.notna().sum(), " valuenum null:", ce.vn_raw.isna().sum())
num = ce.vn_num.notna()
print("  numeric value: valuenum == value?", (ce.vn_raw[num] == ce.vn_num[num]).mean())
print("  numeric value: valuenum sentinel-ish:", ce.vn_raw[num].isin([-999, -888, 9999, -1]).sum())
print("  numeric value: valuenum == 100*value:", (ce.vn_raw[num] == 100 * ce.vn_num[num]).sum())
print("  numeric value: valuenum == value/100:", (ce.vn_raw[num] == ce.vn_num[num] / 100).sum())

# text values: per (itemid,value) mode vs ref
SENT = {-999.0, -888.0, 9999.0, -1.0}
text = ce.vn_num.isna()
good = text & ~ce.vn_raw.isin(SENT) & (ce.vn_raw >= 0)
mode_item = {}
for (it, vl), s in ce[good].groupby(["it2", "value"])["vn_raw"]:
    mode_item[(it, vl)] = s.mode().iat[0]
mode_glob = {}
for vl, s in ce[good].groupby("value")["vn_raw"]:
    mode_glob[vl] = s.mode().iat[0]
# align raw->ref on (stay, ctfix, itemid) and evaluate the text rows
buckets = defaultdict(list)
for i, k in enumerate(zip(ce.stay_id, ce.ctfix, ce.it2)):
    buckets[k].append(i)
ok_item = ok_glob = ok_raw = tot = 0
for j, k in enumerate(zip(refce.stay_id, refce.charttime, refce.itemid)):
    if k in buckets and buckets[k]:
        i = buckets[k].pop(0)
        if not text.at[i]:
            continue
        b = pd.to_numeric(refce.at[j, "valuenum"], errors="coerce")
        tot += 1
        mi = mode_item.get((ce.at[i, "it2"], ce.at[i, "value"]))
        mg = mode_glob.get(ce.at[i, "value"])
        rawv = ce.at[i, "vn_raw"]
        if mi is not None and mi == b:
            ok_item += 1
        if mg is not None and mg == b:
            ok_glob += 1
        if rawv == b or (pd.isna(rawv) and pd.isna(b)):
            ok_raw += 1
print(f"\ntext-value rows matched: {tot}  per-item mode {ok_item}  global mode {ok_glob}  raw as-is {ok_raw}")
