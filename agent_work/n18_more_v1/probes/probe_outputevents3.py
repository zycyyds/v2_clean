"""Probe: DP over the output event charttime candidates (per stay x caregiver)."""
from __future__ import annotations

import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"

raw = pd.read_csv(f"{BASE}/raw/icu/outputevents.csv", dtype=str)
ref = pd.read_csv(f"{BASE}/reference/icu/outputevents.csv", dtype=str)
stays = list(set(ref.stay_id))
r = raw[raw.stay_id.isin(stays)].reset_index(drop=True)
f = ref.reset_index(drop=True)
S = pd.to_datetime(r.storetime)
C = pd.to_datetime(f.charttime)


def dp(weights, strict=True):
    """weights: dict offset->weight, offsets in minutes relative to floor(S)."""
    out = pd.Series(pd.NaT, index=r.index, dtype="datetime64[ns]")
    for _, g in r.groupby(["stay_id", "caregiver_id"], sort=False):
        idx = list(g.index)
        cands = []
        for i in idx:
            s = S.at[i]
            fl = s.floor("h")
            c = {}
            for off, w in weights.items():
                if off == "S":
                    continue
                c[fl + pd.Timedelta(minutes=off)] = w
            c[s] = max(c.get(s, -1e9), weights.get("S", 0.0))
            cands.append(c)
        # DP over the last chosen value
        states = {}
        for v, w in cands[0].items():
            states[v] = (w, [v])
        for k in range(1, len(idx)):
            nxt = {}
            for v, (sc, path) in states.items():
                for w, ws in cands[k].items():
                    if (w > v) if strict else (w >= v):
                        cand = (sc + ws, path + [w])
                        if w not in nxt or cand[0] > nxt[w][0]:
                            nxt[w] = cand
            states = nxt
            if not states:
                break
        if states:
            best = max(states.values(), key=lambda t: t[0])[1]
            for i, v in zip(idx, best):
                out.at[i] = v
    return out


def score(pred):
    return int((pred.values == C.values).sum())


variants = {
    "floor only": None,
    "S .21 floor .39 floor-60 .39": {0: 0.39, -60: 0.39, "S": 0.21},
    "S .21 floor .39 floor-60 .39 +60 .05": {0: 0.39, -60: 0.39, 60: 0.05, "S": 0.21},
    "equal floor/floor-60/S": {0: 0.34, -60: 0.33, "S": 0.33},
    "floor .5 floor-60 .4 S .1": {0: 0.5, -60: 0.4, "S": 0.1},
    "floor .45 floor-60 .45 S .1": {0: 0.45, -60: 0.45, "S": 0.1},
    "floor .6 floor-60 .3 S .1": {0: 0.6, -60: 0.3, "S": 0.1},
}
for name, w in variants.items():
    if w is None:
        pred = S.dt.floor("h")
    else:
        pred = dp(w)
    print(f"{name:34s} {score(pred)}/56 = {score(pred)/56:.4f}")
