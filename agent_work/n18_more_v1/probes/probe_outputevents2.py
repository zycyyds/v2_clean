"""Probe candidate rules for the outputevents charttime."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"

raw = pd.read_csv(f"{BASE}/raw/icu/outputevents.csv", dtype=str)
ref = pd.read_csv(f"{BASE}/reference/icu/outputevents.csv", dtype=str)
stays = list(set(ref.stay_id))
r = raw[raw.stay_id.isin(stays)].reset_index(drop=True)
f = ref.reset_index(drop=True)
S = pd.to_datetime(r.storetime)
C = pd.to_datetime(f.charttime)


def score(pred):
    return int((pred.values == C.values).sum()), len(C)


def rule_floor():
    return S.dt.floor("h")


def rule_floor_monotone():
    """floor to the hour, but keep the sequence strictly increasing per group."""
    out = S.dt.floor("h").copy()
    for _, g in r.groupby(["stay_id", "caregiver_id"], sort=False):
        idx = list(g.index)
        prev = None
        for i in idx:
            v = out.at[i]
            if prev is not None and v <= prev:
                v = prev + pd.Timedelta(hours=1)
            out.at[i] = v
            prev = v
    return out


def rule_dp(hb_bonus=1.0, s_bonus=0.0, window=(-60, 180)):
    """per group: choose a strictly increasing sequence from the candidate sets."""
    out = pd.Series(pd.NaT, index=r.index, dtype="datetime64[ns]")
    for _, g in r.groupby(["stay_id", "caregiver_id"], sort=False):
        idx = list(g.index)
        cands = []
        for i in idx:
            s = S.at[i]
            f = s.floor("h")
            c = {s: s_bonus}
            for d in (window[0], 0, window[1]):
                h = f + pd.Timedelta(minutes=d)
                c[h] = max(c.get(h, -1e9), hb_bonus)
            cands.append(c)
        # dynamic programming over the last chosen value
        states = {v: (score, path) for v, score in cands[0].items() for path in ([v],)}
        for k in range(1, len(idx)):
            nxt = {}
            for v, (sc, path) in states.items():
                for w, ws in cands[k].items():
                    if w > v:
                        key = w
                        val = (sc + ws, path + [w])
                        if key not in nxt or val[0] > nxt[key][0]:
                            nxt[key] = val
            states = nxt
        best = max(states.values(), key=lambda t: t[0])[1]
        for i, v in zip(idx, best):
            out.at[i] = v
    return out


for name, pred in [
    ("floor", rule_floor()),
    ("floor+monotone", rule_floor_monotone()),
    ("dp hb=1 s=0", rule_dp(1.0, 0.0)),
    ("dp hb=1 s=1", rule_dp(1.0, 1.0)),
    ("dp hb=2 s=1", rule_dp(2.0, 1.0)),
    ("dp hb=1 s=0.5", rule_dp(1.0, 0.5)),
    ("dp hb=1 s=0.3", rule_dp(1.0, 0.3)),
]:
    ok, n = score(pred)
    print(f"{name:20s} {ok}/{n} = {ok/n:.4f}")
