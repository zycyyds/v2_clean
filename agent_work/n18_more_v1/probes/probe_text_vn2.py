"""Probe: (itemid, value) -> valuenum mapping (NaN included) for text rows."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
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
ce["vn_num"] = pd.to_numeric(ce.value, errors="coerce")
ce["vn_raw"] = pd.to_numeric(ce.valuenum, errors="coerce")
SENT = {-999.0, -888.0, 9999.0, -1.0}

truth = defaultdict(list)
for _, x in ref.iterrows():
    truth[(x.stay_id, x.charttime, x.itemid)].append(pd.to_numeric(x.valuenum, errors="coerce"))

text = ce.vn_num.isna()
# ---- candidate mapping: (itemid, value) -> mode of the trustworthy valuenums
grp = ce[text].copy()
grp["_v"] = grp["vn_raw"]
grp["_k"] = list(zip(grp["it2"], grp["value"]))
mapping = {}
for k, s in grp.groupby("_k"):
    vals = s["_v"]
    trust = vals[~vals.isin(SENT)]
    # drop values that are a 100x multiple of another value in the group
    uniq = sorted({v for v in trust.dropna().unique()})
    mult = {v for v in uniq if any(abs(v - 100 * u) < 1e-9 for u in uniq if u != 0)}
    trust = trust[~trust.isin(mult)]
    if len(trust) == 0:
        trust = vals
    c = Counter([(None if pd.isna(v) else v) for v in trust])
    mapping[k] = c.most_common(1)[0][0]

ok = tot = 0
for i in ce.index[text]:
    k = (ce.at[i, "stay_id"], ce.at[i, "ctfix"], ce.at[i, "it2"])
    if k not in truth:
        continue
    vals = [v for v in truth[k] if not pd.isna(v)]
    b = Counter(vals).most_common(1)[0][0] if vals else None
    m = mapping.get((ce.at[i, "it2"], ce.at[i, "value"]))
    a = None if (m is None or (isinstance(m, float) and pd.isna(m))) else m
    tot += 1
    if (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9):
        ok += 1
print(f"mapping rule on text rows: {ok}/{tot} = {ok/max(tot,1):.4f}")

# global (value) fallback mapping
mapping_g = {}
for vl, s in grp.groupby("value"):
    vals = s["_v"]
    trust = vals[~vals.isin(SENT)]
    uniq = sorted({v for v in trust.dropna().unique()})
    mult = {v for v in uniq if any(abs(v - 100 * u) < 1e-9 for u in uniq if u != 0)}
    trust = trust[~trust.isin(mult)]
    if len(trust) == 0:
        trust = vals
    c = Counter([(None if pd.isna(v) else v) for v in trust])
    mapping_g[vl] = c.most_common(1)[0][0]
ok2 = 0
for i in ce.index[text]:
    k = (ce.at[i, "stay_id"], ce.at[i, "ctfix"], ce.at[i, "it2"])
    if k not in truth:
        continue
    vals = [v for v in truth[k] if not pd.isna(v)]
    b = Counter(vals).most_common(1)[0][0] if vals else None
    m = mapping.get((ce.at[i, "it2"], ce.at[i, "value"]))
    if m is None:
        m = mapping_g.get(ce.at[i, "value"])
    a = None if (m is None or (isinstance(m, float) and pd.isna(m))) else m
    tot += 1
    if (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9):
        ok2 += 1
print(f"mapping + global fallback: {ok2}/{tot} = {ok2/max(tot,1):.4f}")
