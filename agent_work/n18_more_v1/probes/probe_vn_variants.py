"""Tune the valuenum repair: mapping rule variants measured on train."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
RAW = f"{BASE}/raw"
REF = f"{BASE}/reference"
SENT = {-999.0, -888.0, 9999.0, -1.0}


def load(name, path):
    df = pd.read_csv(path, dtype=str)
    df["vnum"] = pd.to_numeric(df.value, errors="coerce")
    df["vn"] = pd.to_numeric(df.valuenum, errors="coerce")
    return df


def group_truth(ref, keycols, valcol):
    t = defaultdict(list)
    for _, x in ref.iterrows():
        t[tuple(x[c] for c in keycols)].append(pd.to_numeric(x[valcol], errors="coerce"))
    return t


def build_mapping(sub, skip_placeholder=False, threshold=None):
    mapping = {}
    for key, grp in sub.groupby(["it", "vl"], sort=False):
        if skip_placeholder and key[1] == "___":
            continue
        vals = grp["v"]
        trust = vals[~vals.isin(SENT)]
        uniq = sorted({v for v in trust.dropna().unique()})
        mult = {v for v in uniq if any(abs(v - 100.0 * u) < 1e-9 for u in uniq if u != 0.0)}
        if len(trust):
            trust = trust[~trust.isin(mult)]
        if not len(trust):
            trust = vals
        counts = Counter(np.nan if pd.isna(v) else v for v in trust)
        mode, hits = counts.most_common(1)[0]
        if threshold is not None and hits / max(len(vals), 1) < threshold:
            continue
        mapping[key] = mode
    return mapping


def score(df, truth, mapping, keycols, valcol="vn"):
    ok = tot = 0
    for i in df.index:
        k = tuple(df.at[i, c] for c in keycols)
        if k not in truth:
            continue
        vals = [v for v in truth[k] if not pd.isna(v)]
        b = Counter(vals).most_common(1)[0][0] if vals else None
        a = df.at[i, valcol]
        tot += 1
        if (a is None or pd.isna(a)) and b is None:
            ok += 1
        elif a is not None and b is not None and not pd.isna(a) and abs(a - b) < 1e-9:
            ok += 1
    return ok, tot


# ---------------- labs ----------------
lab = load("labs", f"{RAW}/hosp/labevents.csv")
reflab = pd.read_csv(f"{REF}/labs.csv", dtype=str)
coh = set(pd.read_csv(f"{REF}/cohort.csv", dtype=str).hadm_id)
lab = lab[lab.hadm_id.isin(coh)].reset_index(drop=True)
t_lab = group_truth(reflab, ["hadm_id", "itemid", "charttime"], "valuenum")
sub = lab[lab.vnum.isna()].copy()
sub["it"] = sub["itemid"]
sub["vl"] = sub["value"]
sub["v"] = sub["vn"]
for name, kw in [
    ("pure mapping", {}),
    ("skip '___'", {"skip_placeholder": True}),
    ("threshold 0.3", {"threshold": 0.3}),
    ("skip '___' + thr 0.2", {"skip_placeholder": True, "threshold": 0.2}),
]:
    s2 = sub.copy()
    mapping = build_mapping(s2, **kw)
    for i in s2.index:
        m = mapping.get((s2.at[i, "it"], s2.at[i, "vl"]))
        if m is not None:
            s2.at[i, "vn"] = np.nan if (isinstance(m, float) and np.isnan(m)) else float(m)
    ok, tot = score(s2, t_lab, mapping, ["hadm_id", "itemid", "charttime"])
    print(f"labs   {name:24s} {ok}/{tot} = {ok/max(tot,1):.4f}")

# ---------------- chartevents ----------------
ce = load("chartevents", f"{RAW}/icu/chartevents.csv")
refce = pd.read_csv(f"{REF}/icu/chartevents.csv", dtype=str)
refics = pd.read_csv(f"{REF}/icu/icustays.csv", dtype=str).set_index("stay_id")
stays = list(set(refce.stay_id))
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
t_ce = group_truth(refce, ["stay_id", "charttime", "itemid"], "valuenum")
subc = ce[ce.vnum.isna()].copy()
subc["it"] = subc["it2"]
subc["vl"] = subc["value"]
subc["v"] = subc["vn"]
for name, kw in [
    ("pure mapping", {}),
    ("skip '___'", {"skip_placeholder": True}),
    ("threshold 0.3", {"threshold": 0.3}),
    ("skip '___' + thr 0.2", {"skip_placeholder": True, "threshold": 0.2}),
]:
    s2 = subc.copy()
    s3 = s2.copy()
    mapping = build_mapping(s3, **kw)
    for i in s3.index:
        m = mapping.get((s3.at[i, "it"], s3.at[i, "vl"]))
        if m is not None:
            s3.at[i, "vn"] = np.nan if (isinstance(m, float) and np.isnan(m)) else float(m)
    ok, tot = score(s3, t_ce, mapping, ["stay_id", "ctfix", "it2"])
    print(f"ce     {name:24s} {ok}/{tot} = {ok/max(tot,1):.4f}")
