"""Probe: recover the outputevents charttime (the corrupter set it to the epoch)."""
from __future__ import annotations

import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"

raw = pd.read_csv(f"{BASE}/raw/icu/outputevents.csv", dtype=str)
ref = pd.read_csv(f"{BASE}/reference/icu/outputevents.csv", dtype=str)
stays = list(set(ref.stay_id))
r = raw[raw.stay_id.isin(stays)].reset_index(drop=True)
f = ref.reset_index(drop=True)
print("rows:", len(r), "ref:", len(f))
print("per (stay,caregiver) groups:")
for k, g in r.groupby(["stay_id", "caregiver_id"]):
    idx = list(g.index)
    print("  ", k, "n=", len(idx), "ref charttimes:", list(f.charttime[idx]))

# hypothesis: the reference charttime is hour aligned, the raw storetime is
# charttime + jitter in [0,120) minutes
r["st"] = pd.to_datetime(r.storetime)
f["ct"] = pd.to_datetime(f.charttime)
jitter = (r.st.values - f.ct.values) / np.timedelta64(1, "m") if False else None
import numpy as np
j = (r.st.values - f.ct.values) / np.timedelta64(1, "m")
print("\njitter (storetime - charttime) minutes: min %.0f max %.0f" % (np.nanmin(j), np.nanmax(j)))
print("histogram:", pd.Series(j).describe())
print("negatives:", int((j < 0).sum()), "of", len(j))
print("values >=120:", int((j >= 120).sum()))
