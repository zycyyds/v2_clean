"""Probe: how to restore the charttime of the rows the corrupter moved out of
the ICU task window."""
from __future__ import annotations

from collections import Counter

import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"

ce = pd.read_csv(f"{BASE}/raw/icu/chartevents.csv", dtype=str)
ref = pd.read_csv(f"{BASE}/reference/icu/chartevents.csv", dtype=str)
refics = pd.read_csv(f"{BASE}/reference/icu/icustays.csv", dtype=str).set_index("stay_id")
stays = list(set(ref.stay_id))
ce = ce[ce.stay_id.isin(stays)].reset_index(drop=True)
ce = ce[ce.itemid != "999001"].reset_index(drop=True)
ce["it2"] = ce.itemid.str.replace(r"^99(?=\d{6}$)", "", regex=True)
ce["ct"] = pd.to_datetime(ce.charttime, errors="coerce")
ce["ot"] = pd.to_datetime(ce.stay_id.map(refics.outtime))
ce["bad"] = (ce.ct - ce.ot) == pd.Timedelta(days=1)

# is the file order monotone in charttime inside a stay (ignoring leaked rows)?
for st in stays:
    g = ce[ce.stay_id == st]
    ok = g[~g.bad].ct.values
    inv = int((ok[1:] < ok[:-1]).sum())
    print(f"stay {st}: charttime inversions among the {len(ok)} non-leaked rows: {inv}")

refkey = Counter(zip(ref.stay_id, ref.charttime, ref.itemid))


def evaluate(name, fix):
    out = ce.charttime.copy()
    for st in stays:
        idx = list(ce.index[ce.stay_id == st])
        for i in idx:
            if ce.at[i, "bad"]:
                out.at[i] = fix(idx, i)
    cr = Counter(zip(ce.stay_id, out, ce.it2))
    onlyr = sum((cr - refkey).values())
    onlyf = sum((refkey - cr).values())
    print(f"{name:28s} only-in-raw {onlyr:5d}  only-in-ref {onlyf:5d}")


def make_fix(mode):
    def fix(idx, i):
        pos = idx.index(i)
        if mode == "prev":
            for j in reversed(idx[:pos]):
                if not ce.at[j, "bad"]:
                    return ce.at[j, "charttime"]
            return None
        if mode == "next":
            for j in idx[pos + 1:]:
                if not ce.at[j, "bad"]:
                    return ce.at[j, "charttime"]
            return None
        if mode == "prevnext":
            prev = next((ce.at[j, "charttime"] for j in reversed(idx[:pos]) if not ce.at[j, "bad"]), None)
            nxt = next((ce.at[j, "charttime"] for j in idx[pos + 1:] if not ce.at[j, "bad"]), None)
            if prev is not None and nxt is not None:
                return prev if prev == nxt else prev
            return prev or nxt
        raise ValueError(mode)

    return fix


for mode in ("prev", "next", "prevnext"):
    evaluate(mode, make_fix(mode))
