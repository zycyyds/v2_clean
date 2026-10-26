"""N18 (CKD) MIMIC-IV cohort cleaning pipeline.

Reads an error-injected MIMIC-IV-style slice from {raw_root}, repairs the
injected cell/row level errors, selects the N18 cohort and emits the 18
reference result-package CSVs into {output_dir}.

    python3 n18_clean.py --raw-root RAW --train-reference TRAIN_REF \
                          --output-dir OUT [--workdir WORK] [--report REPORT]

Every rule below was derived from the train raw/reference comparison and is
documented (with its train evidence) in train_analysis.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# constants derived from the train raw/reference comparison
# --------------------------------------------------------------------------
COHORT_TYPE = "disease_with_required_modality"
COHORT_LABEL = "N18"
ICD_ROOT = "N18"

# cohort ICD selection: ICD-10 N18.x (CKD) and the ICD-9 equivalent 585.x
ICD10_PREFIXES = ("N18",)
ICD9_PREFIXES = ("585",)

# cells the corrupter used as sentinels / invalid enum values
SENTINEL_VALUENUM = {-999.0, -888.0, 9999.0, -1.0, -100.0, -200.0, -300.0, -400.0, -500.0}
ALIAS_PREFIX = "ALIAS_"
# itemid corruption: the corrupter prefixed the 6-digit itemid with "99"
ITEMID_99_RE = re.compile(r"^99(?=\d{6}$)")
# label-leakage inserted rows are flagged with itemid 999001
LEAK_ITEMID = "999001"
# the corrupter moved rows outside the ICU task window with this offset
WINDOW_LEAK_DAYS = 1
# patients.anchor_age is a sentinel in the raw and cannot be recovered from
# any other raw table; the train reference median (79.5 -> 80) is used.
ANCHOR_AGE_FALLBACK = 80
# MIMIC redacts a numeric measurement behind this placeholder in the value
# column; such rows keep their own valuenum
REDACTED_VALUE = "___" 
# admissions.insurance is null in the whole raw column; "Medicare" is the
# dominant value of the train reference (14/18 cohort rows).
INSURANCE_FALLBACK = "Medicare"

OUTPUT_FILES = [
    "admissions.csv",
    "cohort.csv",
    "demographics.csv",
    "diagnoses.csv",
    "procedures.csv",
    "medications.csv",
    "labs.csv",
    "icu/icustays.csv",
    "icu/chartevents.csv",
    "icu/inputevents.csv",
    "icu/outputevents.csv",
    "multimodal/notes.csv",
    "multimodal/ed_diagnosis.csv",
    "multimodal/ed_medrecon.csv",
    "multimodal/ed_pyxis.csv",
    "multimodal/ed_triage.csv",
    "multimodal/ed_vitalsign.csv",
    "multimodal/edstays.csv",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_csv(path, **kw):
    kw.setdefault("dtype", str)
    return pd.read_csv(path, **kw)


def read_csv_filtered(path, keep, chunksize=300_000, **kw):
    """Stream a large csv and keep only the rows whose `keep` column is in the
    given set.  Returns a single concatenated frame."""
    kw.setdefault("dtype", str)
    out = []
    for chunk in pd.read_csv(path, chunksize=chunksize, **kw):
        sel = chunk[chunk[keep[0]].isin(keep[1])]
        if len(sel):
            out.append(sel)
    if not out:
        return pd.DataFrame(columns=pd.read_csv(path, nrows=0).columns)
    return pd.concat(out, ignore_index=True)


def strip_itemid_alias(s: pd.Series) -> pd.Series:
    """`99220210` -> `220210` (the corrupter prefixed "99" to the itemid)."""
    return s.astype(str).str.replace(ITEMID_99_RE, "", regex=True)


def to_float(series: pd.Series) -> pd.Series:
    """Numeric column -> float column (the reference writes `97.0`, not `97`)."""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def to_int_str(series: pd.Series) -> pd.Series:
    """Numeric column -> integer written as a string (e.g. `29968`)."""
    v = pd.to_numeric(series, errors="coerce")
    return v.map(lambda x: "" if pd.isna(x) else str(int(x)))


def parse_ed_time(series: pd.Series) -> pd.Series:
    """ED timestamps use `Y/M/D H:M:S`; the reference writes `%Y-%m-%d %H:%M:%S`."""
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%Y/%m/%d %H:%M:%S", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce"))
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S")


def write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def repair_valuenum(value_col: pd.Series, valuenum_col: pd.Series, itemid_col: pd.Series) -> pd.Series:
    """Restore `valuenum` after the corrupter replaced it.

    * when the textual `value` column is numeric the true value is that number
      (the corrupter wrote sentinels or multiplied the value by 100)
    * for textual values the true value is a property of the (itemid, value)
      pair: the corrupter replaced it with a sentinel, with a 100x multiple or
      with an arbitrary number.  The dominant value of the whole
      (itemid, value) group - NaN included - is the pristine value.
    """
    val_num = pd.to_numeric(value_col, errors="coerce").to_numpy(dtype="float64")
    raw = pd.to_numeric(valuenum_col, errors="coerce").to_numpy(dtype="float64")
    res = raw.copy()
    numeric_text = ~np.isnan(val_num)
    res[numeric_text] = val_num[numeric_text]

    text_idx = np.where(~numeric_text)[0]
    if len(text_idx):
        it_all = itemid_col.to_numpy()
        vl_all = value_col.to_numpy()
        sub = pd.DataFrame({"it": it_all[text_idx], "vl": vl_all[text_idx], "v": raw[text_idx]})
        mapping = {}
        for key, grp in sub.groupby(["it", "vl"], sort=False, dropna=False):
            vals = grp["v"]
            trust = vals[~vals.isin(SENTINEL_VALUENUM)]
            uniq = sorted({v for v in trust.dropna().unique()})
            multiples = {
                v for v in uniq if any(abs(v - 100.0 * u) < 1e-9 for u in uniq if u != 0.0)
            }
            if len(trust):
                trust = trust[~trust.isin(multiples)]
            if not len(trust):
                # every row of the group carries a sentinel: the pristine value
                # is unknown, the reference keeps it empty
                mapping[key] = np.nan
                continue
            counts = Counter(np.nan if pd.isna(v) else v for v in trust)
            mode, _hits = counts.most_common(1)[0]
            # "___" is the MIMIC redaction placeholder: the value column hides a
            # per-row measurement, so the number has to be kept as it is.
            mapping[key] = mode if key[1] != REDACTED_VALUE else None
        for i in text_idx:
            m = mapping.get((it_all[i], vl_all[i]))
            if m is not None:
                res[i] = np.nan if (isinstance(m, float) and np.isnan(m)) else float(m)
    return pd.Series(res, index=valuenum_col.index, dtype="float64")


# --------------------------------------------------------------------------
# ICU repairs
# --------------------------------------------------------------------------
def repair_icustays(ics: pd.DataFrame) -> pd.DataFrame:
    """The corrupter rewrote `outtime` to `intime - 1h`; restore it from `los`."""
    ics = ics.copy()
    intime = pd.to_datetime(ics["intime"], errors="coerce")
    los = pd.to_numeric(ics["los"], errors="coerce")
    outtime = intime + pd.to_timedelta(los * 86400.0, unit="s")
    ics["outtime"] = outtime.dt.strftime("%Y-%m-%d %H:%M:%S")
    return ics


def repair_chartevents(ce: pd.DataFrame, stay_meta: pd.DataFrame) -> pd.DataFrame:
    """Repair the error-injected cells of icu/chartevents.

    * label-leakage rows (itemid 999001) are dropped
    * `itemid` lost its real value behind a "99" prefix
    * `charttime` of rows pushed outside the ICU task window (outtime + 1 day)
      is restored from the neighbouring rows of the same stay
    * `valuenum` is restored from the textual `value` column
    * `valueuom` is restored from the dominant unit of the itemid
    * `subject_id` / `hadm_id` are restored from the ICU stay registry
    """
    ce = ce.reset_index(drop=True)
    ce["itemid"] = strip_itemid_alias(ce["itemid"])
    ce = ce[ce["itemid"] != LEAK_ITEMID].reset_index(drop=True)
    # keep the pristine cell values for the duplicate detection below
    ce["_raw_valuenum"] = ce["valuenum"].fillna("~").astype(str)
    ce["_raw_valueuom"] = ce["valueuom"].fillna("~").astype(str)

    ce["subject_id"] = ce["stay_id"].map(stay_meta["subject_id"])
    ce["hadm_id"] = ce["stay_id"].map(stay_meta["hadm_id"])

    ct = pd.to_datetime(ce["charttime"], errors="coerce")
    outtime = pd.to_datetime(ce["stay_id"].map(stay_meta["outtime"]), errors="coerce")
    leaked = (ct - outtime) == pd.Timedelta(days=WINDOW_LEAK_DAYS)
    if leaked.any():
        fixed = ce["charttime"].copy()
        for _, grp in ce.groupby("stay_id", sort=False):
            prev = None
            for i in grp.index:
                if leaked.at[i]:
                    fixed.at[i] = prev
                else:
                    prev = ce.at[i, "charttime"]
        ce["charttime"] = fixed

    ce["valuenum"] = repair_valuenum(ce["value"], ce["valuenum"], ce["itemid"])

    uom = ce.groupby("itemid")["valueuom"].agg(
        lambda s: s.dropna().mode().iat[0] if len(s.dropna()) else np.nan
    )
    ce["valueuom"] = ce["itemid"].map(uom)
    ce["valueuom"] = ce["valueuom"].where(ce["valueuom"].notna(), ce["valueuom"])

    # whole-row duplicates injected by the corrupter: keep the first occurrence
    dup_key = [
        "stay_id", "charttime", "itemid", "value", "_raw_valuenum", "_raw_valueuom",
        "storetime", "caregiver_id",
    ]
    dup_key = [c for c in dup_key if c in ce.columns]
    ce = ce.drop_duplicates(subset=dup_key, keep="first").reset_index(drop=True)
    ce = ce.drop(columns=[c for c in ("_raw_valuenum", "_raw_valueuom") if c in ce.columns])
    return ce


def repair_inputevents(ie: pd.DataFrame, stay_meta: pd.DataFrame) -> pd.DataFrame:
    """Repair icu/inputevents.

    * `hadm_id` was overwritten with an unrelated admission -> ICU stay registry
    * `itemid` lost its value behind a "99" prefix
    * `amount` was replaced by a sentinel -> the real amount is `totalamount`
    * `endtime` was set to `starttime - 1min` (time inversion)
    """
    ie = ie.reset_index(drop=True)
    ie["itemid"] = strip_itemid_alias(ie["itemid"])
    ie["subject_id"] = ie["stay_id"].map(stay_meta["subject_id"])
    ie["hadm_id"] = ie["stay_id"].map(stay_meta["hadm_id"])

    amount = pd.to_numeric(ie["amount"], errors="coerce")
    original = pd.to_numeric(ie["originalamount"], errors="coerce")
    bad = amount.isin(SENTINEL_VALUENUM) | amount.isna() | (amount < 0)
    ie["amount"] = np.where(bad, original, amount)

    start = pd.to_datetime(ie["starttime"], errors="coerce")
    end = pd.to_datetime(ie["endtime"], errors="coerce")
    inverted = (end - start) == pd.Timedelta(minutes=-1)
    ie.loc[inverted, "endtime"] = (start[inverted] + pd.Timedelta(minutes=1)).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return ie


def repair_outputevents(oe: pd.DataFrame, stay_meta: pd.DataFrame) -> pd.DataFrame:
    """Repair icu/outputevents.

    * `charttime` was overwritten with the epoch sentinel `1970-01-01`.  The
      pristine event time is the hour aligned grid point of `storetime`; the
      grid point can be the hour of the storetime or the one before it (the
      reference grid lags the storetime by 0-120 minutes), and the sequence of
      the output events of one (stay, caregiver) is strictly increasing - which
      disambiguates the two candidates.
    * `subject_id` / `hadm_id` are restored from the ICU stay registry
    """
    oe = oe.reset_index(drop=True)
    oe["subject_id"] = oe["stay_id"].map(stay_meta["subject_id"])
    oe["hadm_id"] = oe["stay_id"].map(stay_meta["hadm_id"])
    st = pd.to_datetime(oe["storetime"], errors="coerce")
    epoch = pd.to_datetime(oe["charttime"], errors="coerce")
    broken = epoch.dt.year.eq(1970)
    if not broken.any():
        return oe

    hour = st.dt.floor("h")
    best = hour.copy()
    for _, grp in oe.groupby(["stay_id", "caregiver_id"], sort=False):
        idx = list(grp.index)
        cands = []
        for i in idx:
            if not broken.at[i]:
                cands.append({epoch.at[i]: 1.0})
                continue
            h = hour.at[i]
            c = {h: 0.39, h - pd.Timedelta(minutes=60): 0.39,
                 h + pd.Timedelta(minutes=60): 0.05, st.at[i]: 0.21}
            cands.append(c)
        # dynamic programme: strictly increasing sequence with maximal weight
        states = {v: (w, [v]) for v, w in cands[0].items()}
        for k in range(1, len(idx)):
            nxt = {}
            for v, (sc, path) in states.items():
                for w, ws in cands[k].items():
                    if w > v:
                        cand = (sc + ws, path + [w])
                        if w not in nxt or cand[0] > nxt[w][0]:
                            nxt[w] = cand
            states = nxt
            if not states:
                break
        if states:
            seq = max(states.values(), key=lambda t: t[0])[1]
            for i, v in zip(idx, seq):
                best.at[i] = v
    oe["charttime"] = best.dt.strftime("%Y-%m-%d %H:%M:%S")
    return oe


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------
def run(raw_root: str, train_reference: str, output_dir: str, workdir: str | None = None) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if workdir:
        os.makedirs(workdir, exist_ok=True)
    report: dict = {"raw_root": raw_root, "output_dir": output_dir, "steps": {}}

    hosp = os.path.join(raw_root, "hosp")
    icu = os.path.join(raw_root, "icu")
    ed = os.path.join(raw_root, "ed")
    note = os.path.join(raw_root, "note")

    # ------------------------------------------------------------------
    # 1. patients / admissions / diagnoses
    # ------------------------------------------------------------------
    patients = read_csv(os.path.join(hosp, "patients.csv"))
    admissions = read_csv(os.path.join(hosp, "admissions.csv"))
    diagnoses = read_csv(os.path.join(hosp, "diagnoses_icd.csv"))
    procedures = read_csv(os.path.join(hosp, "procedures_icd.csv"))

    # the training slice ships obfuscated icd codes ("ALIAS_<code>")
    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.replace(
        ALIAS_PREFIX, "", regex=False
    )

    # hospital_expire_flag was inverted by the corrupter (train evidence:
    # 38/39 raw rows carry 1 while the reference has 0, and the single raw row
    # with a deathtime carries 0 although the patient died in hospital)
    admissions["hospital_expire_flag"] = (
        1 - pd.to_numeric(admissions["hospital_expire_flag"], errors="coerce")
    )

    # ------------------------------------------------------------------
    # 2. cohort selection
    # ------------------------------------------------------------------
    icd = pd.concat(
        [
            diagnoses[diagnoses["icd_code"].str.startswith(ICD10_PREFIXES)]["hadm_id"],
            diagnoses[diagnoses["icd_code"].str.startswith(ICD9_PREFIXES)]["hadm_id"],
        ]
    )
    n18_hadm = set(icd.dropna())

    ed_stays_raw = read_csv(os.path.join(ed, "edstays.csv"))
    ed_hadm = set(ed_stays_raw["hadm_id"].dropna())
    cohort_hadm = sorted(n18_hadm & ed_hadm)
    report["steps"]["cohort"] = {
        "n18_hadm": len(n18_hadm),
        "ed_hadm": len(ed_hadm),
        "cohort_hadm": len(cohort_hadm),
    }

    coh = admissions[admissions["hadm_id"].isin(cohort_hadm)].copy()
    cohort = pd.DataFrame(
        {
            "subject_id": coh["subject_id"].values,
            "hadm_id": coh["hadm_id"].values,
            "cohort_type": COHORT_TYPE,
            "cohort_label": COHORT_LABEL,
            "icd_root": ICD_ROOT,
            "hospital_expire_flag": coh["hospital_expire_flag"].values,
        }
    )

    # ------------------------------------------------------------------
    # 3. ICU tables
    # ------------------------------------------------------------------
    icustays = read_csv(os.path.join(icu, "icustays.csv"))
    icustays = repair_icustays(icustays)
    icustays = icustays[icustays["hadm_id"].isin(cohort_hadm)].copy()
    stay_meta = icustays.set_index("stay_id")[["subject_id", "hadm_id", "outtime"]]
    cohort_stays = set(icustays["stay_id"])

    # ------------------------------------------------------------------
    # 4. demographics
    # ------------------------------------------------------------------
    gender_map = (
        ed_stays_raw.dropna(subset=["gender"])
        .groupby("subject_id")["gender"]
        .agg(lambda s: s.mode().iat[0])
    )
    patients = patients.copy()
    bad_gender = ~patients["gender"].isin(["M", "F"])
    patients.loc[bad_gender, "gender"] = patients.loc[bad_gender, "subject_id"].map(gender_map)
    patients["gender"] = patients["gender"].fillna("F")
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce")
    patients.loc[~patients["anchor_age"].between(0, 120), "anchor_age"] = ANCHOR_AGE_FALLBACK

    # ------------------------------------------------------------------
    # 5. ED tables
    # ------------------------------------------------------------------
    ed_stays = ed_stays_raw[ed_stays_raw["hadm_id"].isin(cohort_hadm)].copy()
    ed_stays["intime"] = parse_ed_time(ed_stays["intime"])
    ed_stays["outtime"] = parse_ed_time(ed_stays["outtime"])
    ed_stays["hadm_id"] = to_float(ed_stays["hadm_id"])
    cohort_ed_stays = set(ed_stays["stay_id"])

    def ed_subset(fname: str) -> pd.DataFrame:
        return read_csv_filtered(
            os.path.join(ed, fname), ("stay_id", cohort_ed_stays)
        )

    # ------------------------------------------------------------------
    # 6. emit
    # ------------------------------------------------------------------
    out = lambda rel: os.path.join(output_dir, rel)

    adm_out = coh[
        [
            "subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type",
            "admission_location", "discharge_location", "insurance", "language", "marital_status",
            "race", "hospital_expire_flag",
        ]
    ].copy()
    adm_out["insurance"] = adm_out["insurance"].fillna(INSURANCE_FALLBACK)
    write_csv(adm_out, out("admissions.csv"))
    write_csv(cohort, out("cohort.csv"))

    dem = patients[
        ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"]
    ].copy()
    write_csv(dem, out("demographics.csv"))

    dx = diagnoses[diagnoses["hadm_id"].isin(cohort_hadm)][
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    ].reset_index(drop=True)
    write_csv(dx, out("diagnoses.csv"))

    pr = procedures[procedures["hadm_id"].isin(cohort_hadm)][
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    ].reset_index(drop=True)
    write_csv(pr, out("procedures.csv"))

    rx = read_csv_filtered(
        os.path.join(hosp, "prescriptions.csv"), ("hadm_id", cohort_hadm)
    )
    rx = rx[
        ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx",
         "route"]
    ].reset_index(drop=True)
    write_csv(rx, out("medications.csv"))

    labs = read_csv_filtered(
        os.path.join(hosp, "labevents.csv"), ("hadm_id", cohort_hadm)
    )
    labs["valuenum"] = repair_valuenum(labs["value"], labs["valuenum"], labs["itemid"])
    labs = labs[
        ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    ].reset_index(drop=True)
    write_csv(labs, out("labs.csv"))

    ics_out = icustays[
        ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"]
    ].reset_index(drop=True)
    write_csv(ics_out, out("icu/icustays.csv"))

    ce = read_csv_filtered(
        os.path.join(icu, "chartevents.csv"), ("stay_id", cohort_stays)
    )
    ce = repair_chartevents(ce, stay_meta)
    ce_out = ce[
        ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"]
    ].reset_index(drop=True)
    write_csv(ce_out, out("icu/chartevents.csv"))

    ie = read_csv_filtered(
        os.path.join(icu, "inputevents.csv"), ("stay_id", cohort_stays)
    )
    ie = repair_inputevents(ie, stay_meta)
    ie_out = ie[
        ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount"]
    ].reset_index(drop=True)
    ie_out["amount"] = to_float(ie_out["amount"])
    write_csv(ie_out, out("icu/inputevents.csv"))

    oe = read_csv_filtered(
        os.path.join(icu, "outputevents.csv"), ("stay_id", cohort_stays)
    )
    oe = repair_outputevents(oe, stay_meta)
    oe_out = oe[["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"]].reset_index(
        drop=True
    )
    oe_out["value"] = to_float(oe_out["value"])
    write_csv(oe_out, out("icu/outputevents.csv"))

    notes = read_csv_filtered(
        os.path.join(note, "discharge.csv"), ("hadm_id", cohort_hadm)
    )
    rad = read_csv_filtered(
        os.path.join(note, "radiology.csv"), ("hadm_id", cohort_hadm)
    )
    notes = pd.concat([notes, rad], ignore_index=True)
    notes = notes[notes["hadm_id"].isin(cohort_hadm)]
    notes = notes[notes["note_type"] == "DS"]
    notes = notes[
        ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime",
         "text"]
    ].reset_index(drop=True)
    write_csv(notes, out("multimodal/notes.csv"))

    eds = ed_stays[
        ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender", "race",
         "arrival_transport", "disposition"]
    ].reset_index(drop=True)
    write_csv(eds, out("multimodal/edstays.csv"))

    ed_diag = ed_subset("diagnosis.csv")
    write_csv(
        ed_diag[["subject_id", "stay_id", "seq_num", "icd_code", "icd_version", "icd_title"]],
        out("multimodal/ed_diagnosis.csv"),
    )

    ed_med = ed_subset("medrecon.csv")
    ed_med["charttime"] = parse_ed_time(ed_med["charttime"])
    ed_med["gsn"] = to_int_str(ed_med["gsn"])
    ed_med["ndc"] = to_int_str(ed_med["ndc"])
    ed_med["etccode"] = to_float(ed_med["etccode"])
    write_csv(
        ed_med[["subject_id", "stay_id", "charttime", "name", "gsn", "ndc", "etc_rn", "etccode",
                "etcdescription"]],
        out("multimodal/ed_medrecon.csv"),
    )

    ed_pyx = ed_subset("pyxis.csv")
    ed_pyx["charttime"] = parse_ed_time(ed_pyx["charttime"])
    ed_pyx["gsn"] = to_float(ed_pyx["gsn"])
    ed_pyx["gsn_rn"] = to_int_str(ed_pyx["gsn_rn"])
    write_csv(
        ed_pyx[["subject_id", "stay_id", "charttime", "med_rn", "name", "gsn_rn", "gsn"]],
        out("multimodal/ed_pyxis.csv"),
    )

    ed_tri = ed_subset("triage.csv")
    for c in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity"]:
        ed_tri[c] = to_float(ed_tri[c])
    write_csv(
        ed_tri[["subject_id", "stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp",
                "dbp", "pain", "acuity", "chiefcomplaint"]],
        out("multimodal/ed_triage.csv"),
    )

    ed_vit = ed_subset("vitalsign.csv")
    ed_vit["charttime"] = parse_ed_time(ed_vit["charttime"])
    for c in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]:
        ed_vit[c] = to_float(ed_vit[c])
    write_csv(
        ed_vit[["subject_id", "stay_id", "charttime", "temperature", "heartrate", "resprate",
                "o2sat", "sbp", "dbp", "rhythm", "pain"]],
        out("multimodal/ed_vitalsign.csv"),
    )

    report["steps"]["rows"] = {f: int(len(pd.read_csv(out(f)))) for f in OUTPUT_FILES}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="N18 MIMIC-IV cohort cleaning pipeline")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--train-reference", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    report = run(args.raw_root, args.train_reference, args.output_dir, args.workdir)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)
    print(json.dumps(report["steps"], indent=1))


if __name__ == "__main__":
    main()
