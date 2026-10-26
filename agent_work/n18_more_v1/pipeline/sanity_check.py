"""Sanity checks on a produced package: no residual corruption signatures."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from n18_clean import LEAK_ITEMID, SENTINEL_VALUENUM  # noqa: E402

ITEMID_99 = re.compile(r"^99\d{6}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    p = args.package
    res = {}

    ce = pd.read_csv(os.path.join(p, "icu/chartevents.csv"), dtype=str)
    res["chartevents_rows"] = len(ce)
    res["chartevents_itemid_99_prefixed"] = int(ce.itemid.astype(str).str.match(ITEMID_99).sum())
    res["chartevents_leak_rows"] = int((ce.itemid == LEAK_ITEMID).sum())
    res["chartevents_epoch_charttime"] = int(
        ce.charttime.astype(str).str.startswith("1970-01-01").sum()
    )
    vn = pd.to_numeric(ce.valuenum, errors="coerce")
    res["chartevents_sentinel_valuenum"] = int(vn.isin(list(SENTINEL_VALUENUM)).sum())
    res["chartevents_negative_valuenum"] = int((vn < 0).sum())

    icu = pd.read_csv(os.path.join(p, "icu/icustays.csv"), dtype=str)
    it = pd.to_datetime(icu.intime, errors="coerce")
    ot = pd.to_datetime(icu.outtime, errors="coerce")
    res["icustays_outtime_before_intime"] = int((ot < it).sum())

    ie = pd.read_csv(os.path.join(p, "icu/inputevents.csv"), dtype=str)
    s = pd.to_datetime(ie.starttime, errors="coerce")
    e = pd.to_datetime(ie.endtime, errors="coerce")
    res["inputevents_endtime_before_starttime"] = int((e < s).sum())
    res["inputevents_itemid_99_prefixed"] = int(ie.itemid.astype(str).str.match(ITEMID_99).sum())

    adm = pd.read_csv(os.path.join(p, "admissions.csv"), dtype=str)
    res["admissions_flag_values"] = sorted(adm.hospital_expire_flag.dropna().unique().tolist())
    res["admissions_flag_inverts_deathtime"] = int(
        ((adm.hospital_expire_flag == "1") & adm.deathtime.notna()).sum()
    )
    res["admissions_null_insurance"] = int(adm.insurance.isna().sum())

    dem = pd.read_csv(os.path.join(p, "demographics.csv"), dtype=str)
    res["demographics_gender_values"] = sorted(dem.gender.dropna().unique().tolist())
    res["demographics_anchor_age_sentinel"] = int(
        pd.to_numeric(dem.anchor_age, errors="coerce").isin([999]).sum()
    )

    dx = pd.read_csv(os.path.join(p, "diagnoses.csv"), dtype=str)
    res["diagnoses_alias_prefixed"] = int(dx.icd_code.astype(str).str.startswith("ALIAS_").sum())

    ed = pd.read_csv(os.path.join(p, "multimodal/edstays.csv"), dtype=str)
    res["edstays_non_iso_intime"] = int(
        ed.intime.astype(str).str.contains("/").sum()
    )
    for f in ("multimodal/ed_medrecon.csv", "multimodal/ed_pyxis.csv",
              "multimodal/ed_vitalsign.csv"):
        d = pd.read_csv(os.path.join(p, f), dtype=str)
        res[f"{f}_non_iso_charttime"] = int(d.charttime.astype(str).str.contains("/").sum())

    for k, v in res.items():
        print(f"{k:44s} {v}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
