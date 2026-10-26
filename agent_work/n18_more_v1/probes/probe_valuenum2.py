"""Probe: best valuenum repair for text-valued chartevents rows (group-wise)."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
RAW = f"{BASE}/raw"
REF = f"{BASE}/reference"

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
SENT = {-999.0, -888.0, 9999.0, -1.0}

# group-wise ground truth: for each (stay, ctfix, itemid) the ref's valuenum set
refgroups = defaultdict(list)
for _, x in refce.iterrows():
    refgroups[(x.stay_id, x.charttime, x.itemid)].append(
        pd.to_numeric(x.valuenum, errors="coerce")
    )

text = ce.vn_num.isna()
print("text rows:", text.sum())


def evaluate(name, repair):
    rep = repair.copy()
    ok = tot = 0
    for i in ce.index[text]:
        k = (ce.at[i, "stay_id"], ce.at[i, "ctfix"], ce.at[i, "it2"])
        if k not in refgroups:
            continue
        vals = refgroups[k]
        truth = Counter([v for v in vals if not pd.isna(v)])
        b = truth.most_common(1)[0][0] if truth else None
        a = rep.at[i]
        tot += 1
        if (a is None or pd.isna(a)) and b is None:
            ok += 1
        elif a is not None and b is not None and a == b:
            ok += 1
    print(f"  {name:34s} {ok}/{tot} = {ok/max(tot,1):.4f}")


# repair variants
mode_item = {}
good = text & ~ce.vn_raw.isin(SENT) & (ce.vn_raw >= 0)
for (it, vl), s in ce[good].groupby(["it2", "value"])["vn_raw"]:
    mode_item[(it, vl)] = s.mode().iat[0]
mode_glob = {}
for vl, s in ce[good].groupby("value")["vn_raw"]:
    mode_glob[vl] = s.mode().iat[0]


def v_raw():
    return ce.vn_raw.copy()


def v_mode_always():
    return pd.Series(
        [mode_item.get((it, vl), mode_glob.get(vl, np.nan))
         for it, vl in zip(ce.it2, ce.value)],
        index=ce.index, dtype="float64",
    )


def v_sentinel_only():
    rep = ce.vn_raw.copy()
    bad = ce.vn_raw.isin(SENT) | (ce.vn_raw < 0)
    for i in ce.index[bad & text]:
        rep.at[i] = mode_item.get((ce.at[i, "it2"], ce.at[i, "value"]), mode_glob.get(ce.at[i, "value"], np.nan))
    return rep


def v_mode_unless_agree():
    rep = v_mode_always()
    # keep the raw value when the mode is missing
    rep = rep.fillna(ce.vn_raw)
    return rep


def v_sentinel_plus_100():
    rep = v_sentinel_only()
    for i in ce.index[text]:
        m = mode_item.get((ce.at[i, "it2"], ce.at[i, "value"]))
        if m is not None and not pd.isna(rep.at[i]) and rep.at[i] != m and rep.at[i] == 100 * m:
            rep.at[i] = m
    return rep


evaluate("raw as-is", v_raw())
evaluate("mode always", v_mode_always())
evaluate("sentinel -> mode", v_sentinel_only())
evaluate("mode unless missing", v_mode_unless_agree())
evaluate("sentinel->mode + x100 fix", v_sentinel_plus_100())

# how many text rows have a valuenum equal to 100*the mode (x100 corruption)
n100 = 0
for i in ce.index[text]:
    m = mode_item.get((ce.at[i, "it2"], ce.at[i, "value"]))
    if m and ce.at[i, "vn_raw"] == 100 * m:
        n100 += 1
print("text rows with valuenum == 100*mode:", n100)
