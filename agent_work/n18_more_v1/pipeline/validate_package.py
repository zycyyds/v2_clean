"""Validate a produced result package: 18 csv files, schema and row sanity.

    python3 validate_package.py --package DIR [--expect-rows FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from n18_clean import OUTPUT_FILES  # noqa: E402

EXPECTED_COLUMNS = {
    "admissions.csv": ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                       "admission_type", "admission_location", "discharge_location", "insurance",
                       "language", "marital_status", "race", "hospital_expire_flag"],
    "cohort.csv": ["subject_id", "hadm_id", "cohort_type", "cohort_label", "icd_root",
                   "hospital_expire_flag"],
    "demographics.csv": ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group",
                         "dod"],
    "diagnoses.csv": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "procedures.csv": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "medications.csv": ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "dose_val_rx",
                        "dose_unit_rx", "route"],
    "labs.csv": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
    "icu/icustays.csv": ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
    "icu/chartevents.csv": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum",
                            "valueuom"],
    "icu/inputevents.csv": ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid",
                            "amount"],
    "icu/outputevents.csv": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"],
    "multimodal/notes.csv": ["note_id", "subject_id", "hadm_id", "note_type", "note_seq",
                             "charttime", "storetime", "text"],
    "multimodal/ed_diagnosis.csv": ["subject_id", "stay_id", "seq_num", "icd_code", "icd_version",
                                    "icd_title"],
    "multimodal/ed_medrecon.csv": ["subject_id", "stay_id", "charttime", "name", "gsn", "ndc",
                                   "etc_rn", "etccode", "etcdescription"],
    "multimodal/ed_pyxis.csv": ["subject_id", "stay_id", "charttime", "med_rn", "name", "gsn_rn",
                                "gsn"],
    "multimodal/ed_triage.csv": ["subject_id", "stay_id", "temperature", "heartrate", "resprate",
                                 "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"],
    "multimodal/ed_vitalsign.csv": ["subject_id", "stay_id", "charttime", "temperature",
                                    "heartrate", "resprate", "o2sat", "sbp", "dbp", "rhythm",
                                    "pain"],
    "multimodal/edstays.csv": ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender",
                               "race", "arrival_transport", "disposition"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    result = {"package": os.path.abspath(args.package), "files": {}, "ok": True}
    for rel in OUTPUT_FILES:
        path = os.path.join(args.package, rel)
        info = {"exists": os.path.exists(path)}
        if not info["exists"]:
            info["error"] = "missing"
            result["ok"] = False
            result["files"][rel] = info
            continue
        df = pd.read_csv(path, dtype=str)
        info["rows"] = len(df)
        info["columns"] = list(df.columns)
        exp = EXPECTED_COLUMNS[rel]
        info["schema_ok"] = list(df.columns) == exp
        info["empty_cells"] = int(df.isna().sum().sum())
        if not info["schema_ok"]:
            result["ok"] = False
        result["files"][rel] = info

    for rel, info in result["files"].items():
        if "error" in info:
            print(f"{rel:32s} MISSING")
        else:
            print(f"{rel:32s} rows={info['rows']:8d} schema_ok={info['schema_ok']} "
                  f"empty_cells={info['empty_cells']}")
    print("\nPACKAGE OK" if result["ok"] else "\nPACKAGE INCOMPLETE")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
