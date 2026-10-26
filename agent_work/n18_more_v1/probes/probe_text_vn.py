"""Probe: what does the corrupter do to `valuenum` of text-valued rows?"""
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

# ground truth per (stay, ctfix, itemid) group from the reference
truth = defaultdict(list)
for _, x in ref.iterrows():
    truth[(x.stay_id, x.charttime, x.itemid)].append(pd.to_numeric(x.valuenum, errors="coerce"))

# for each raw text row find the truth value of its group
text = ce.vn_num.isna()
pairs = Counter()
n_rows = 0
for i in ce.index[text]:
    k = (ce.at[i, "stay_id"], ce.at[i, "ctfix"], ce.at[i, "it2"])
    if k not in truth:
        continue
    vals = [v for v in truth[k] if not pd.isna(v)]
    b = Counter(vals).most_common(1)[0][0] if vals else None
    a = ce.at[i, "vn_raw"]
    n_rows += 1
    if b is None:
        pairs[("ref_NaN", "raw_NaN" if pd.isna(a) else "raw_num")] += 1
    elif pd.isna(a):
        pairs[("ref_num", "raw_NaN")] += 1
    elif a == b:
        pairs[("ref_num", "raw_same")] += 1
    elif a in SENT:
        pairs[("ref_num", "raw_sentinel")] += 1
    elif a == 100 * b:
        pairs[("ref_num", "raw_x100")] += 1
    elif a == b / 100:
        pairs[("ref_num", "raw_div100")] += 1
    else:
        pairs[("ref_num", "raw_other")] += 1
print("text rows with a truth:", n_rows)
for k, v in pairs.most_common():
    print(f"   {k}: {v}")

# for the 'raw_other' bucket: what are the (raw, ref) pairs?
other = Counter()
for i in ce.index[text]:
    k = (ce.at[i, "stay_id"], ce.at[i, "ctfix"], ce.at[i, "it2"])
    if k not in truth:
        continue
    vals = [v for v in truth[k] if not pd.isna(v)]
    b = Counter(vals).most_common(1)[0][0] if vals else None
    a = ce.at[i, "vn_raw"]
    if b is None or pd.isna(a) or a == b or a in SENT or a == 100 * b or a == b / 100:
        continue
    other[(ce.at[i, "it2"], ce.at[i, "value"], a, b)] += 1
print("\n(raw other) sample (itemid, value, raw, ref):")
for k, v in other.most_common(25):
    print("   ", k, v)
