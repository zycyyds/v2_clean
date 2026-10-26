import numpy as np
import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
TEST = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/test"
pd.set_option("display.width", 400)

ics = pd.read_csv(f"{BASE}/raw/icu/icustays.csv", dtype=str)
refic = pd.read_csv(f"{BASE}/reference/icu/icustays.csv", dtype=str)
m = refic.merge(ics, on="stay_id", suffixes=("_ref", "_raw"))


def formulas(it, ot, los):
    d = ot - it
    ts = d.total_seconds()
    out = {}
    out["ts/86400"] = ts / 86400
    out["ts/3600/24"] = ts / 3600 / 24
    out["days+sec/86400"] = d.days + d.seconds / 86400
    out["td/1D"] = d / pd.Timedelta(days=1)
    out["td/86400s"] = d / pd.Timedelta(seconds=86400)
    out["td/24h"] = d / pd.Timedelta(hours=24)
    out["np_td64_D"] = d.to_timedelta64() / np.timedelta64(1, "D")
    out["np_td64_h/24"] = d.to_timedelta64() / np.timedelta64(24, "h")
    out["ns/86400e9"] = d.value / 86400e9
    out["days+sec/86400+us"] = d.days + d.seconds / 86400 + d.microseconds / 86400e6
    out["ts_int/86400"] = int(ts) / 86400
    out["round16"] = round(ts / 86400, 16)
    out["raw_los"] = los
    out["ts/86400 from los*24h"] = (
        (it + pd.Timedelta(hours=los * 24) - it).total_seconds() / 86400
    )
    out["td.total_seconds()"] = d.total_seconds()
    return out


for _, x in m.iterrows():
    it = pd.to_datetime(x.intime_ref)
    los = float(x.los_raw)
    ot = it + pd.Timedelta(seconds=los * 86400)
    ref = x.los_ref
    print(f"\n=== stay {x.stay_id} ref_los={ref}")
    for k, v in formulas(it, ot, los).items():
        if isinstance(v, float):
            print(f"    {k:28s} {v!r}  match={repr(v) == ref}")

# ---------------- test set structure ----------------
print("\n\n########## TEST SET STRUCTURE ##########")
keys = pd.read_csv(f"{TEST}/keys.csv")
print("keys.csv:", keys.shape, list(keys.columns))
print(keys.head(3).to_string())
subs = set(keys.iloc[:, 0].astype(str))
print("n subjects:", len(subs))

dx = pd.read_csv(f"{TEST}/raw/hosp/diagnoses_icd.csv", dtype=str)
dx["icd_code"] = dx.icd_code.str.replace("ALIAS_", "", regex=False)
n18 = dx[dx.icd_code.str.startswith("N18")].subject_id.unique()
i585 = dx[dx.icd_code.str.startswith("585")].subject_id.unique()
print("subjects with N18 codes:", len(n18), " with 585 codes:", len(i585))
ed = pd.read_csv(f"{TEST}/raw/ed/edstays.csv", dtype=str)
ed_subj = set(ed.subject_id)
print("subjects with ED rows:", len(ed_subj))
adm = pd.read_csv(f"{TEST}/raw/hosp/admissions.csv", dtype=str)
print("admissions rows:", len(adm), "subjects:", adm.subject_id.nunique())
n18_hadm = set(dx[dx.icd_code.str.startswith("N18")].hadm_id)
i585_hadm = set(dx[dx.icd_code.str.startswith("585")].hadm_id)
ed_hadm = set(ed.hadm_id.dropna())
coh = (n18_hadm | i585_hadm) & ed_hadm
coh_subj = set(adm[adm.hadm_id.isin(coh)].subject_id)
print("cohort hadms:", len(coh), "cohort subjects:", len(coh_subj))
print("all test subjects in cohort?", len(coh_subj), "of", len(subs))
print("subjects in keys but not in cohort:", len(subs - coh_subj))
print("ALIAS_ prefix present in test diagnoses?", dx.icd_code.str.startswith("ALIAS_").sum(), "of", len(dx))
