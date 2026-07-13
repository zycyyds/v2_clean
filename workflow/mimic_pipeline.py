"""MIMIC-IV reference-package primitives used by the retained pipeline skills."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

KEY_COLUMNS = {"subject_id", "hadm_id", "stay_id", "case_id"}
LABEL_COLUMNS = {"label", "mortality", "readmission", "los_label", "outcome"}
DEFAULT_FEATURE_GROUPS = ("diagnosis", "procedure", "lab", "medication", "icu")
DEFAULT_SPLIT_RATIOS = (0.6, 0.2, 0.2)

def split_teacher_reference(
    reference: str | Path,
    output_dir: str | Path,
    *,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    counts: tuple[int, int, int | None] | None = None,
    seed: int = 42,
    key_column: str = "",
    stratify_label: bool = True,
) -> dict[str, Any]:
    """Split a teacher reference table into train/validation/test references."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = _reference_wide_frame(reference)
    if frame is None or frame.empty:
        raise ValueError(f"teacher reference is empty or unreadable: {reference}")
    key = key_column if key_column and key_column in frame.columns else _infer_split_key(frame)
    if key not in frame.columns:
        key = "_row_id"
        frame = frame.copy()
        frame[key] = [f"row_{idx:08d}" for idx in range(len(frame))]
    frame = frame.drop_duplicates(subset=[key]).copy()
    label_col = _first_existing(["label", "mortality", "readmission", "los_label", "outcome"], frame.columns, frame.columns)
    split_mode = "counts" if counts else "ratios"
    if counts:
        train, validation, test = _split_frame_by_counts(
            frame,
            counts,
            label_col=label_col,
            stratify_label=stratify_label,
            seed=seed,
        )
        train_ratio, validation_ratio, test_ratio = _ratios_from_counts(len(train), len(validation), len(test))
    else:
        train_ratio, validation_ratio, test_ratio = _normalize_split_ratios(ratios)
        train, validation, test = _split_frame_by_ratios(
            frame,
            (train_ratio, validation_ratio, test_ratio),
            label_col=label_col,
            stratify_label=stratify_label,
            seed=seed,
        )
    paths = {
        "train_reference": output / "train_reference.csv",
        "validation_reference": output / "validation_reference.csv",
        "test_reference": output / "test_reference.csv",
        "train_keys": output / "train_keys.csv",
        "validation_keys": output / "validation_keys.csv",
        "test_keys": output / "test_keys.csv",
    }
    train.to_csv(paths["train_reference"], index=False)
    validation.to_csv(paths["validation_reference"], index=False)
    test.to_csv(paths["test_reference"], index=False)
    train[[key]].to_csv(paths["train_keys"], index=False)
    validation[[key]].to_csv(paths["validation_keys"], index=False)
    test[[key]].to_csv(paths["test_keys"], index=False)
    manifest = {
        "schema_version": 1,
        "source_reference": str(Path(reference).expanduser().resolve()),
        "split_mode": split_mode,
        "split_key": key,
        "label_column": label_col,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "requested_counts": {
            "train": counts[0] if counts else None,
            "validation": counts[1] if counts else None,
            "test": counts[2] if counts else None,
        },
        "seed": seed,
        "row_counts": {
            "source": int(len(frame)),
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "paths": {name: str(path) for name, path in paths.items()},
    }
    manifest_path = output / "split_manifest.json"
    _write_json(manifest_path, manifest)
    return {**{name: str(path) for name, path in paths.items()}, "split_manifest": str(manifest_path), "manifest": manifest}


def resolve_disease_icd(
    mimic_root: str | Path,
    disease_text: str = "",
    *,
    icd_prefixes: Iterable[str] = (),
    max_candidates: int = 25,
) -> dict[str, Any]:
    """Resolve a free-text disease or ICD prefix against MIMIC diagnosis dictionaries."""
    root = _normalize_mimic_root(mimic_root)
    requested = [value.strip().upper().replace(".", "") for value in icd_prefixes if str(value).strip()]
    disease = str(disease_text or "").strip()
    direct = _direct_icd_prefixes(disease)
    if direct:
        requested.extend(direct)
    if requested:
        return {
            "status": "supported",
            "input_text": disease,
            "icd_prefixes": sorted(set(requested)),
            "match_type": "explicit_icd_prefix",
            "candidates": [],
            "evidence": {"source": "task_text"},
        }
    if not disease:
        return {
            "status": "not_requested",
            "input_text": "",
            "icd_prefixes": [],
            "match_type": "none",
            "candidates": [],
            "evidence": {},
        }
    dictionary_path = _mimic_path(root, "hosp/d_icd_diagnoses.csv")
    if not dictionary_path.exists():
        return {
            "status": "unsupported",
            "input_text": disease,
            "icd_prefixes": [],
            "match_type": "dictionary_missing",
            "candidates": [],
            "evidence": {"missing_file": "hosp/d_icd_diagnoses.csv"},
        }
    dictionary = pd.read_csv(dictionary_path, dtype=object)
    terms = _disease_terms(disease)
    if not terms:
        return {
            "status": "ambiguous",
            "input_text": disease,
            "icd_prefixes": [],
            "match_type": "no_search_terms",
            "candidates": [],
            "evidence": {},
        }
    title = dictionary.get("long_title", pd.Series(dtype=object)).fillna("").astype(str).str.casefold()
    mask = pd.Series([False] * len(dictionary))
    for term in terms:
        mask = mask | title.str.contains(re.escape(term.casefold()), regex=True, na=False)
    matches = dictionary.loc[mask].copy()
    if matches.empty:
        return {
            "status": "unsupported",
            "input_text": disease,
            "icd_prefixes": [],
            "match_type": "no_title_match",
            "candidates": [],
            "evidence": {"terms": terms},
        }
    matches["icd_prefix"] = matches["icd_code"].fillna("").astype(str).str.upper().str.replace(".", "", regex=False).str[:3]
    prefix_counts = Counter(matches["icd_prefix"].dropna().astype(str))
    candidates = []
    for prefix, count in prefix_counts.most_common(max_candidates):
        example = matches.loc[matches["icd_prefix"] == prefix].head(3)
        candidates.append(
            {
                "icd_prefix": prefix,
                "match_count": int(count),
                "examples": [
                    {
                        "icd_code": str(row.get("icd_code") or ""),
                        "icd_version": str(row.get("icd_version") or ""),
                        "long_title": str(row.get("long_title") or ""),
                    }
                    for _, row in example.iterrows()
                ],
            }
        )
    selected = [item["icd_prefix"] for item in candidates[:5]]
    status = "supported" if selected else "unsupported"
    if len(candidates) > 8 and not _known_disease_synonyms(disease):
        status = "ambiguous"
    return {
        "status": status,
        "input_text": disease,
        "icd_prefixes": selected if status == "supported" else [],
        "match_type": "title_search",
        "candidates": candidates,
        "evidence": {"dictionary": str(dictionary_path), "terms": terms},
    }


def build_visit_base(
    mimic_root: str | Path,
    output_dir: str | Path,
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    root = _normalize_mimic_root(mimic_root)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    patients = _read_mimic_csv(root, "hosp/patients.csv")
    patients = patients[[column for column in ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"] if column in patients.columns]]
    if "dod" in patients.columns:
        patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    admissions = _read_mimic_csv(root, "hosp/admissions.csv")
    for column in ("admittime", "dischtime", "deathtime"):
        if column in admissions.columns:
            admissions[column] = pd.to_datetime(admissions[column], errors="coerce")
    setting = str(task_spec.get("care_setting") or "Non-ICU")
    if setting == "ICU":
        icu = _read_mimic_csv(root, "icu/icustays.csv")
        for column in ("intime", "outtime"):
            if column in icu.columns:
                icu[column] = pd.to_datetime(icu[column], errors="coerce")
        base = icu.merge(patients, on="subject_id", how="left").merge(
            admissions[[column for column in ["hadm_id", "race", "insurance", "hospital_expire_flag"] if column in admissions.columns]],
            on="hadm_id",
            how="left",
        )
        if "los" not in base.columns and {"intime", "outtime"} <= set(base.columns):
            base["los"] = (base["outtime"] - base["intime"]).dt.total_seconds() / 86400
    else:
        base = admissions.merge(patients, on="subject_id", how="left")
        if {"admittime", "dischtime"} <= set(base.columns):
            base["los"] = (base["dischtime"] - base["admittime"]).dt.total_seconds() / 86400
    if "anchor_age" in base.columns and "Age" not in base.columns:
        base["Age"] = base["anchor_age"]
    if "race" in base.columns and "ethnicity" not in base.columns:
        base["ethnicity"] = base["race"]
    key = str(task_spec.get("record_grain") or ("stay_id" if setting == "ICU" else "hadm_id"))
    base = base.dropna(subset=[key]).drop_duplicates(subset=[key])
    path = output / "visit_base.csv"
    base.to_csv(path, index=False)
    return {"status": "SUCCESS", "visit_base": str(path), "row_count": int(len(base)), "record_grain": key}


def build_outcome_label(
    visit_base_path: str | Path,
    output_dir: str | Path,
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(visit_base_path, dtype=object)
    outcome = str(task_spec.get("outcome_type") or "Mortality")
    admit_col = "intime" if str(task_spec.get("care_setting")) == "ICU" else "admittime"
    disch_col = "outtime" if str(task_spec.get("care_setting")) == "ICU" else "dischtime"
    for column in (admit_col, disch_col, "dod", "deathtime"):
        if column in cohort.columns:
            cohort[column] = pd.to_datetime(cohort[column], errors="coerce")
    if outcome == "Mortality":
        death = cohort["dod"] if "dod" in cohort.columns else cohort.get("deathtime", pd.Series([pd.NaT] * len(cohort)))
        cohort["label"] = ((death.notna()) & (death >= cohort[admit_col]) & (death <= cohort[disch_col])).astype(int)
    elif outcome == "Readmission":
        days = int(task_spec.get("time_window_days") or 30)
        cohort["label"] = 0
        grouped = cohort.sort_values(["subject_id", admit_col]).groupby("subject_id", dropna=False)
        for _, group in grouped:
            indices = list(group.index)
            for current, nxt in zip(indices, indices[1:]):
                current_discharge = cohort.at[current, disch_col]
                next_admit = cohort.at[nxt, admit_col]
                if pd.notna(current_discharge) and pd.notna(next_admit):
                    delta = (next_admit - current_discharge).total_seconds() / 86400
                    if 0 < delta <= days:
                        cohort.at[current, "label"] = 1
    elif outcome == "Length of Stay":
        days = int(task_spec.get("time_window_days") or 7)
        los = pd.to_numeric(cohort.get("los"), errors="coerce")
        cohort["label"] = (los > days).astype(int)
    else:
        cohort["label"] = 1
    path = output / "cohort.csv"
    cohort.to_csv(path, index=False)
    return {
        "status": "SUCCESS",
        "cohort": str(path),
        "row_count": int(len(cohort)),
        "label_distribution": {str(k): int(v) for k, v in cohort["label"].value_counts(dropna=False).to_dict().items()},
    }


def apply_case_key_filter(
    cohort_path: str | Path,
    case_keys_path: str | Path,
    output_dir: str | Path,
    *,
    preferred_key: str = "",
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_path, dtype=object)
    keys = pd.read_csv(case_keys_path, dtype=object)
    key = preferred_key if preferred_key and preferred_key in cohort.columns and preferred_key in keys.columns else ""
    if not key:
        key = _first_existing(["hadm_id", "stay_id", "subject_id", "case_id"], cohort.columns, keys.columns)
    if not key:
        raise ValueError(f"Could not find a shared split key between cohort and {case_keys_path}")
    allowed = set(keys[key].dropna().astype(str))
    filtered = cohort[cohort[key].astype(str).isin(allowed)].copy()
    path = output / "cohort_filtered_to_split.csv"
    filtered.to_csv(path, index=False)
    return {
        "status": "SUCCESS",
        "cohort": str(path),
        "split_key": key,
        "allowed_key_count": len(allowed),
        "row_count": int(len(filtered)),
        "source_keys": str(Path(case_keys_path).expanduser().resolve()),
    }


def filter_disease_cohort(
    cohort_path: str | Path,
    mimic_root: str | Path,
    output_dir: str | Path,
    disease_rule: dict[str, Any],
    *,
    primary_only: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_path, dtype=object)
    prefixes = [str(value).upper().replace(".", "") for value in disease_rule.get("icd_prefixes") or []]
    if not prefixes:
        path = output / "disease_filtered_cohort.csv"
        cohort.to_csv(path, index=False)
        return {"status": "SKIPPED", "cohort": str(path), "row_count": int(len(cohort)), "reason": "No ICD prefixes"}
    diagnoses = _read_mimic_csv(_normalize_mimic_root(mimic_root), "hosp/diagnoses_icd.csv", dtype=object)
    diagnoses["icd_clean"] = diagnoses["icd_code"].fillna("").astype(str).str.upper().str.replace(".", "", regex=False)
    mask = diagnoses["icd_clean"].apply(lambda value: any(str(value).startswith(prefix) for prefix in prefixes))
    if primary_only and "seq_num" in diagnoses.columns:
        mask = mask & (pd.to_numeric(diagnoses["seq_num"], errors="coerce") == 1)
    matched_hadm = set(diagnoses.loc[mask, "hadm_id"].dropna().astype(str))
    filtered = cohort[cohort["hadm_id"].astype(str).isin(matched_hadm)].copy()
    path = output / "disease_filtered_cohort.csv"
    filtered.to_csv(path, index=False)
    return {
        "status": "SUCCESS",
        "cohort": str(path),
        "row_count": int(len(filtered)),
        "matched_hadm_count": len(matched_hadm),
        "icd_prefixes": prefixes,
    }


def extract_diagnosis_features(
    mimic_root: str | Path,
    cohort_path: str | Path,
    output_dir: str | Path,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_path, dtype=object)
    hadm_ids = set(cohort["hadm_id"].dropna().astype(str))
    diag = _read_mimic_csv(_normalize_mimic_root(mimic_root), "hosp/diagnoses_icd.csv", dtype=object)
    diag = diag[diag["hadm_id"].astype(str).isin(hadm_ids)].copy()
    if diag.empty:
        return _empty_feature_output(output, "diagnosis_features.csv", cohort, "diagnosis")
    title = _maybe_read_dictionary(_normalize_mimic_root(mimic_root), "hosp/d_icd_diagnoses.csv")
    if title is not None:
        keys = [key for key in ("icd_code", "icd_version") if key in diag.columns and key in title.columns]
        if keys:
            diag = diag.merge(title, on=keys, how="left")
    diag["icd_clean"] = diag["icd_code"].fillna("").astype(str).str.upper().str.replace(".", "", regex=False)
    diag["icd_prefix3"] = diag["icd_clean"].str[:3]
    grouped = diag.groupby("hadm_id", as_index=False).agg(
        diagnosis_count=("icd_clean", "count"),
        unique_diagnosis_count=("icd_clean", "nunique"),
        all_icd_codes=("icd_clean", _join_unique),
    )
    if "seq_num" in diag.columns:
        primary = diag.sort_values(["hadm_id", "seq_num"]).drop_duplicates("hadm_id")
        primary = primary[["hadm_id", "icd_clean", "icd_prefix3", *([] if "long_title" not in primary.columns else ["long_title"])]]
        primary = primary.rename(
            columns={
                "icd_clean": "primary_icd_code",
                "icd_prefix3": "primary_icd_prefix3",
                "long_title": "primary_icd_title",
            }
        )
        grouped = grouped.merge(primary, on="hadm_id", how="left")
    wide = _pivot_top_flags(diag, "hadm_id", "icd_prefix3", "diag_icd3", top_n)
    result = grouped.merge(wide, on="hadm_id", how="left") if not wide.empty else grouped
    path = output / "diagnosis_features.csv"
    result.to_csv(path, index=False)
    return {"status": "SUCCESS", "features": str(path), "row_count": int(len(result)), "feature_group": "diagnosis"}


def extract_procedure_features(
    mimic_root: str | Path,
    cohort_path: str | Path,
    output_dir: str | Path,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_path, dtype=object)
    hadm_ids = set(cohort["hadm_id"].dropna().astype(str))
    proc_path = _mimic_path(_normalize_mimic_root(mimic_root), "hosp/procedures_icd.csv")
    if not proc_path.exists():
        return _empty_feature_output(output, "procedure_features.csv", cohort, "procedure")
    proc = pd.read_csv(proc_path, dtype=object)
    proc = proc[proc["hadm_id"].astype(str).isin(hadm_ids)].copy()
    if proc.empty:
        return _empty_feature_output(output, "procedure_features.csv", cohort, "procedure")
    title = _maybe_read_dictionary(_normalize_mimic_root(mimic_root), "hosp/d_icd_procedures.csv")
    if title is not None:
        keys = [key for key in ("icd_code", "icd_version") if key in proc.columns and key in title.columns]
        if keys:
            proc = proc.merge(title, on=keys, how="left")
    proc["procedure_code_clean"] = proc["icd_code"].fillna("").astype(str).str.upper().str.replace(".", "", regex=False)
    grouped = proc.groupby("hadm_id", as_index=False).agg(
        procedure_count=("procedure_code_clean", "count"),
        unique_procedure_count=("procedure_code_clean", "nunique"),
        all_procedure_codes=("procedure_code_clean", _join_unique),
    )
    if "long_title" in proc.columns:
        titles = proc.groupby("hadm_id", as_index=False).agg(procedure_titles=("long_title", _join_unique))
        grouped = grouped.merge(titles, on="hadm_id", how="left")
    wide = _pivot_top_flags(proc, "hadm_id", "procedure_code_clean", "proc_code", top_n)
    result = grouped.merge(wide, on="hadm_id", how="left") if not wide.empty else grouped
    path = output / "procedure_features.csv"
    result.to_csv(path, index=False)
    return {"status": "SUCCESS", "features": str(path), "row_count": int(len(result)), "feature_group": "procedure"}


def extract_lab_features(
    mimic_root: str | Path,
    cohort_path: str | Path,
    output_dir: str | Path,
    *,
    top_n: int = 30,
    chunksize: int = 250_000,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = _normalize_mimic_root(mimic_root)
    cohort = pd.read_csv(cohort_path, dtype=object)
    hadm_ids = set(cohort["hadm_id"].dropna().astype(str))
    path = _mimic_path(root, "hosp/labevents.csv")
    if not path.exists():
        return _empty_feature_output(output, "lab_features.csv", cohort, "lab")
    chunks = []
    usecols = _available_usecols(path, ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom", "flag"])
    for chunk in pd.read_csv(path, dtype=object, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["hadm_id"].astype(str).isin(hadm_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return _empty_feature_output(output, "lab_features.csv", cohort, "lab")
    labs = pd.concat(chunks, ignore_index=True)
    labs["valuenum"] = pd.to_numeric(labs.get("valuenum"), errors="coerce")
    labitems = _maybe_read_dictionary(root, "hosp/d_labitems.csv")
    if labitems is not None and "itemid" in labitems.columns:
        labs = labs.merge(labitems[[column for column in ["itemid", "label", "category"] if column in labitems.columns]], on="itemid", how="left")
    labs["lab_name"] = labs.get("label", labs["itemid"]).fillna(labs["itemid"]).astype(str)
    top_items = labs["lab_name"].value_counts().head(top_n).index.tolist()
    subset = labs[labs["lab_name"].isin(top_items)].copy()
    agg = subset.groupby(["hadm_id", "lab_name"])["valuenum"].agg(["count", "mean", "min", "max"]).reset_index()
    wide = agg.pivot(index="hadm_id", columns="lab_name")
    wide.columns = [f"lab_{_safe_name(label)}_{stat}" for stat, label in wide.columns]
    wide = wide.reset_index()
    global_stats = labs.groupby("hadm_id", as_index=False).agg(
        lab_event_count=("itemid", "count"),
        lab_item_count=("itemid", "nunique"),
    )
    result = global_stats.merge(wide, on="hadm_id", how="left")
    out = output / "lab_features.csv"
    result.to_csv(out, index=False)
    return {"status": "SUCCESS", "features": str(out), "row_count": int(len(result)), "feature_group": "lab"}


def extract_medication_features(
    mimic_root: str | Path,
    cohort_path: str | Path,
    output_dir: str | Path,
    *,
    top_n: int = 30,
    chunksize: int = 250_000,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = _normalize_mimic_root(mimic_root)
    cohort = pd.read_csv(cohort_path, dtype=object)
    hadm_ids = set(cohort["hadm_id"].dropna().astype(str))
    path = _mimic_path(root, "hosp/prescriptions.csv")
    if not path.exists():
        return _empty_feature_output(output, "medication_features.csv", cohort, "medication")
    chunks = []
    usecols = _available_usecols(path, ["subject_id", "hadm_id", "drug", "drug_type", "starttime", "stoptime", "dose_val_rx"])
    for chunk in pd.read_csv(path, dtype=object, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["hadm_id"].astype(str).isin(hadm_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return _empty_feature_output(output, "medication_features.csv", cohort, "medication")
    meds = pd.concat(chunks, ignore_index=True)
    meds["drug_norm"] = meds["drug"].fillna("").astype(str).map(_safe_name)
    global_stats = meds.groupby("hadm_id", as_index=False).agg(
        medication_order_count=("drug_norm", "count"),
        unique_medication_count=("drug_norm", "nunique"),
    )
    wide = _pivot_top_flags(meds, "hadm_id", "drug_norm", "drug", top_n)
    result = global_stats.merge(wide, on="hadm_id", how="left") if not wide.empty else global_stats
    out = output / "medication_features.csv"
    result.to_csv(out, index=False)
    return {"status": "SUCCESS", "features": str(out), "row_count": int(len(result)), "feature_group": "medication"}


def extract_icu_event_features(
    mimic_root: str | Path,
    cohort_path: str | Path,
    output_dir: str | Path,
    *,
    top_n: int = 20,
    chunksize: int = 250_000,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = _normalize_mimic_root(mimic_root)
    cohort = pd.read_csv(cohort_path, dtype=object)
    icu_path = _mimic_path(root, "icu/icustays.csv")
    if not icu_path.exists():
        return _empty_feature_output(output, "icu_features.csv", cohort, "icu")
    icu = pd.read_csv(icu_path, dtype=object)
    icu = icu[icu["hadm_id"].astype(str).isin(set(cohort["hadm_id"].dropna().astype(str)))].copy()
    if icu.empty:
        return _empty_feature_output(output, "icu_features.csv", cohort, "icu")
    icu["los"] = pd.to_numeric(icu.get("los"), errors="coerce")
    features = icu.groupby("hadm_id", as_index=False).agg(
        icu_stay_count=("stay_id", "nunique"),
        icu_los_mean=("los", "mean"),
        icu_los_max=("los", "max"),
    )
    chart_path = _mimic_path(root, "icu/chartevents.csv")
    if chart_path.exists() and "stay_id" in icu.columns:
        stay_to_hadm = icu[["stay_id", "hadm_id"]].dropna().drop_duplicates()
        stay_ids = set(stay_to_hadm["stay_id"].astype(str))
        usecols = _available_usecols(chart_path, ["stay_id", "itemid", "charttime", "valuenum"])
        chunks = []
        for chunk in pd.read_csv(chart_path, dtype=object, usecols=usecols, chunksize=chunksize):
            chunk = chunk[chunk["stay_id"].astype(str).isin(stay_ids)]
            if not chunk.empty:
                chunks.append(chunk)
        if chunks:
            chart = pd.concat(chunks, ignore_index=True).merge(stay_to_hadm, on="stay_id", how="left")
            chart["valuenum"] = pd.to_numeric(chart.get("valuenum"), errors="coerce")
            items = _maybe_read_dictionary(root, "icu/d_items.csv")
            if items is not None and "itemid" in items.columns:
                chart = chart.merge(items[[column for column in ["itemid", "label"] if column in items.columns]], on="itemid", how="left")
            chart["item_name"] = chart.get("label", chart["itemid"]).fillna(chart["itemid"]).astype(str)
            top_items = chart["item_name"].value_counts().head(top_n).index.tolist()
            agg = chart[chart["item_name"].isin(top_items)].groupby(["hadm_id", "item_name"])["valuenum"].agg(["count", "mean", "min", "max"]).reset_index()
            if not agg.empty:
                wide = agg.pivot(index="hadm_id", columns="item_name")
                wide.columns = [f"icu_{_safe_name(label)}_{stat}" for stat, label in wide.columns]
                features = features.merge(wide.reset_index(), on="hadm_id", how="left")
    out = output / "icu_features.csv"
    features.to_csv(out, index=False)
    return {"status": "SUCCESS", "features": str(out), "row_count": int(len(features)), "feature_group": "icu"}


def clean_feature_table(
    cohort_path: str | Path,
    feature_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    record_grain: str = "hadm_id",
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(cohort_path, dtype=object)
    if record_grain not in dataset.columns:
        record_grain = "hadm_id" if "hadm_id" in dataset.columns else dataset.columns[0]
    dataset = dataset.drop_duplicates(subset=[record_grain])
    merged_paths = []
    for raw_path in feature_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            continue
        frame = pd.read_csv(path, dtype=object)
        if frame.empty or record_grain not in frame.columns:
            continue
        frame = frame.drop_duplicates(subset=[record_grain])
        overlap = [column for column in frame.columns if column in dataset.columns and column != record_grain]
        frame = frame.drop(columns=overlap)
        dataset = dataset.merge(frame, on=record_grain, how="left")
        merged_paths.append(str(path))
    for column in dataset.columns:
        if column in KEY_COLUMNS or column in {"label"}:
            continue
        numeric = pd.to_numeric(dataset[column], errors="coerce")
        if numeric.notna().sum() > 0 and numeric.notna().sum() >= dataset[column].notna().sum() * 0.8:
            dataset[column] = numeric
    out = output / "features_wide.csv"
    dataset.to_csv(out, index=False)
    return {
        "status": "SUCCESS",
        "features_wide": str(out),
        "row_count": int(len(dataset)),
        "column_count": int(len(dataset.columns)),
        "merged_feature_paths": merged_paths,
    }


def _normalize_mimic_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    candidates = [root, root / "mimiciv" / "3.1", root / "3.1", root / "raw_mimic"]
    for candidate in candidates:
        if (candidate / "hosp").is_dir() or (candidate / "icu").is_dir():
            return candidate.resolve()
    raise ValueError(f"Could not find a MIMIC 3.1 hosp/icu layout under: {root}")


def _mimic_path(root: Path, relative: str) -> Path:
    path = root / relative
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    return gz if gz.exists() else path


def _read_mimic_csv(root: Path, relative: str, **kwargs) -> pd.DataFrame:
    path = _mimic_path(root, relative)
    if not path.exists():
        raise FileNotFoundError(f"MIMIC file not found: {relative}")
    return pd.read_csv(path, **kwargs)


def _write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _direct_icd_prefixes(value: str) -> list[str]:
    tokens = re.findall(r"\b[A-TV-Z][0-9][A-Z0-9]{0,4}\b|\b[0-9V][0-9]{2,4}\b", str(value).upper().replace(".", ""))
    return [token[:3] for token in tokens]


def _known_disease_synonyms(value: str) -> list[str]:
    lower = value.casefold()
    synonyms = {
        "肝": ["liver", "hepatic", "hepatitis", "cirrhosis"],
        "肝病": ["liver", "hepatic", "hepatitis", "cirrhosis"],
        "心衰": ["heart failure", "cardiac failure"],
        "heart failure": ["heart failure", "cardiac failure"],
        "肺炎": ["pneumonia"],
        "糖尿病": ["diabetes"],
    }
    for key, values in synonyms.items():
        if key in lower:
            return values
    return []


def _disease_terms(value: str) -> list[str]:
    synonyms = _known_disease_synonyms(value)
    if synonyms:
        return synonyms
    words = [word for word in re.split(r"[^A-Za-z0-9]+", value) if len(word) >= 3]
    return words[:5]


def _read_reference_frames(reference: Path) -> dict[str, pd.DataFrame]:
    files = _reference_files_with_package(reference)
    frames: dict[str, pd.DataFrame] = {}
    for path in files:
        frames.update(_read_reference_file_frames(path))
    return frames


def _read_reference_file_frames(path: Path) -> dict[str, pd.DataFrame]:
    name = path.name.lower()
    try:
        if name.endswith(".parquet"):
            return {path.stem: pd.read_parquet(path)}
        if name.endswith((".xlsx", ".xls")):
            return {
                f"{path.stem}:{sheet}": frame
                for sheet, frame in pd.read_excel(path, sheet_name=None, dtype=object).items()
            }
        sep = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
        return {path.stem: pd.read_csv(path, sep=sep, dtype=object)}
    except Exception:
        return {}


def _reference_files_with_package(reference: Path) -> list[Path]:
    if reference.is_file():
        files = [reference]
        package_manifest = reference.parent / "package_manifest.json"
        if package_manifest.is_file():
            try:
                payload = json.loads(package_manifest.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            for section in ("cohort_files", "feature_files"):
                for item in payload.get(section, []) if isinstance(payload, dict) else []:
                    path = Path(str(item.get("path") or ""))
                    if path.is_file() and path not in files:
                        files.append(path)
        return files
    return sorted(
        path for path in reference.rglob("*")
        if path.is_file() and path.name.lower().endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".xlsx", ".xls", ".parquet"))
    )


def _reference_wide_frame(reference: str | Path | None) -> pd.DataFrame | None:
    if reference is None:
        return None
    frames = _read_reference_frames(Path(reference).expanduser().resolve())
    if not frames:
        return None
    candidates = sorted(frames.items(), key=lambda item: (item[1].shape[1], item[1].shape[0]), reverse=True)
    return candidates[0][1].copy()


def _normalize_split_ratios(ratios: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("split ratios must contain train, validation and test ratios")
    values = tuple(float(value) for value in ratios)
    if any(value < 0 for value in values):
        raise ValueError("split ratios must be non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError("at least one split ratio must be positive")
    normalized = tuple(value / total for value in values)
    return (normalized[0], normalized[1], normalized[2])


def _split_frame_by_ratios(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float],
    *,
    label_col: str,
    stratify_label: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ratio, validation_ratio, _test_ratio = ratios
    groups = [frame]
    if stratify_label and label_col:
        groups = [group for _, group in frame.groupby(label_col, dropna=False)]
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for offset, group in enumerate(groups):
        shuffled = group.sample(frac=1.0, random_state=seed + offset).reset_index(drop=True)
        n = len(shuffled)
        train_n = int(round(n * train_ratio))
        validation_n = int(round(n * validation_ratio))
        if n >= 3:
            train_n = max(1, min(train_n, n - 2))
            validation_n = max(1, min(validation_n, n - train_n - 1))
        elif n == 2:
            train_n = 1
            validation_n = 1
        elif n == 1:
            train_n = 1
            validation_n = 0
        test_n = max(n - train_n - validation_n, 0)
        train_parts.append(shuffled.iloc[:train_n])
        validation_parts.append(shuffled.iloc[train_n:train_n + validation_n])
        test_parts.append(shuffled.iloc[train_n + validation_n:train_n + validation_n + test_n])
    return (
        _concat_or_empty(train_parts, frame.columns),
        _concat_or_empty(validation_parts, frame.columns),
        _concat_or_empty(test_parts, frame.columns),
    )


def _split_frame_by_counts(
    frame: pd.DataFrame,
    counts: tuple[int, int, int | None],
    *,
    label_col: str,
    stratify_label: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_requested = max(int(counts[0]), 0)
    validation_requested = max(int(counts[1]), 0)
    train_count = min(train_requested, len(frame))
    validation_count = min(validation_requested, max(len(frame) - train_count, 0))
    train, remaining = _take_split_rows(
        frame,
        train_count,
        label_col=label_col,
        stratify_label=stratify_label,
        seed=seed,
    )
    validation, test = _take_split_rows(
        remaining,
        validation_count,
        label_col=label_col,
        stratify_label=stratify_label,
        seed=seed + 10_000,
    )
    if counts[2] is not None:
        test_requested = max(int(counts[2]), 0)
        test_count = min(test_requested, len(test))
        test, _unused = _take_split_rows(
            test,
            test_count,
            label_col=label_col,
            stratify_label=stratify_label,
            seed=seed + 20_000,
        )
    else:
        test = test.sample(frac=1.0, random_state=seed + 20_000).reset_index(drop=True) if not test.empty else test.reset_index(drop=True)
    return train, validation, test


def _take_split_rows(
    frame: pd.DataFrame,
    count: int,
    *,
    label_col: str,
    stratify_label: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if count <= 0 or frame.empty:
        return pd.DataFrame(columns=frame.columns), frame.reset_index(drop=True)
    count = min(count, len(frame))
    if not stratify_label or not label_col or label_col not in frame.columns:
        shuffled = frame.sample(frac=1.0, random_state=seed)
        selected = shuffled.iloc[:count]
        remaining = shuffled.iloc[count:]
        return selected.reset_index(drop=True), remaining.reset_index(drop=True)
    groups = [(label, group) for label, group in frame.groupby(label_col, dropna=False)]
    total = len(frame)
    allocations: list[dict[str, Any]] = []
    for index, (_label, group) in enumerate(groups):
        exact = len(group) * count / max(total, 1)
        take = min(int(exact), len(group))
        allocations.append({"index": index, "group": group, "take": take, "fraction": exact - int(exact)})
    remaining_slots = count - sum(item["take"] for item in allocations)
    while remaining_slots > 0:
        progressed = False
        for item in sorted(allocations, key=lambda value: (value["fraction"], len(value["group"])), reverse=True):
            if item["take"] < len(item["group"]):
                item["take"] += 1
                remaining_slots -= 1
                progressed = True
                if remaining_slots == 0:
                    break
        if not progressed:
            break
    selected_parts = []
    for item in allocations:
        group = item["group"].sample(frac=1.0, random_state=seed + int(item["index"]))
        selected_parts.append(group.iloc[: int(item["take"])])
    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else pd.DataFrame(columns=frame.columns)
    if len(selected) < count:
        fill = frame.drop(index=selected.index, errors="ignore").sample(frac=1.0, random_state=seed + 999)
        selected = pd.concat([selected, fill.iloc[: count - len(selected)]], ignore_index=False)
    selected = selected.iloc[:count]
    remaining = frame.drop(index=selected.index, errors="ignore")
    return (
        selected.sample(frac=1.0, random_state=seed + 1_000).reset_index(drop=True),
        remaining.sample(frac=1.0, random_state=seed + 2_000).reset_index(drop=True) if not remaining.empty else remaining.reset_index(drop=True),
    )


def _ratios_from_counts(train_count: int, validation_count: int, test_count: int) -> tuple[float, float, float]:
    total = max(train_count + validation_count + test_count, 1)
    return (train_count / total, validation_count / total, test_count / total)


def _infer_split_key(frame: pd.DataFrame) -> str:
    for column in ("hadm_id", "stay_id", "subject_id", "case_id"):
        if column in frame.columns:
            return column
    return ""


def _concat_or_empty(parts: list[pd.DataFrame], columns: Iterable[str]) -> pd.DataFrame:
    non_empty = [part for part in parts if not part.empty]
    if not non_empty:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(non_empty, ignore_index=True)


def _available_usecols(path: Path, requested: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    return [column for column in requested if column in header.columns]


def _maybe_read_dictionary(root: Path, relative: str) -> pd.DataFrame | None:
    path = _mimic_path(root, relative)
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=object)


def _join_unique(values: Iterable[Any]) -> str:
    return "|".join(sorted({str(value) for value in values if pd.notna(value) and str(value).strip()}))


def _safe_name(value: Any) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^0-9A-Z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "UNK"


def _pivot_top_flags(frame: pd.DataFrame, key: str, column: str, prefix: str, top_n: int) -> pd.DataFrame:
    if frame.empty or key not in frame.columns or column not in frame.columns:
        return pd.DataFrame()
    top = frame[column].dropna().astype(str).value_counts().head(top_n).index.tolist()
    if not top:
        return pd.DataFrame()
    subset = frame[frame[column].astype(str).isin(top)][[key, column]].copy()
    subset["value"] = 1
    wide = subset.pivot_table(index=key, columns=column, values="value", aggfunc="max", fill_value=0)
    wide.columns = [f"{prefix}_{_safe_name(value)}" for value in wide.columns]
    return wide.reset_index()


def _empty_feature_output(output: Path, filename: str, cohort: pd.DataFrame, group: str) -> dict[str, Any]:
    key = "hadm_id" if "hadm_id" in cohort.columns else cohort.columns[0]
    frame = cohort[[key]].drop_duplicates().copy()
    path = output / filename
    frame.to_csv(path, index=False)
    return {"status": "SUCCESS", "features": str(path), "row_count": int(len(frame)), "feature_group": group, "empty": True}


def _first_existing(candidates: list[str], left: Iterable[str], right: Iterable[str]) -> str:
    left_set = set(left)
    right_set = set(right)
    return next((candidate for candidate in candidates if candidate in left_set and candidate in right_set), "")


