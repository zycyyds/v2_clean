from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def _safe_feature_name(value: object) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^0-9A-Z一-鿿]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "UNK"


HEPATIC_PREFIXES = ("570", "571", "572", "573", "070", "155", "156", "7895")
HEPATIC_DRUG_KEYWORDS = (
    "LACTULOSE",
    "RIFAXIMIN",
    "SPIRONOLACTONE",
    "FUROSEMIDE",
    "ENTECAVIR",
    "TENOFOVIR",
    "LAMIVUDINE",
    "URSODIOL",
)


def _slugify_task(task_text: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z一-鿿]+", "_", task_text.strip()).strip("_")
    return text[:40] or "dataset"


def _code_prefix(code: object, n: int) -> str | None:
    if pd.isna(code):
        return None
    cleaned = str(code).strip().upper().replace(".", "")
    if not cleaned or cleaned == "NAN":
        return None
    return cleaned[:n]


def _pivot_top_counts(df: pd.DataFrame, key: str, value: str, prefix: str, top_n: int) -> pd.DataFrame:
    counts = df[value].value_counts().head(top_n)
    top_values = counts.index.tolist()
    subset = df[df[value].isin(top_values)].copy()
    if subset.empty:
        return pd.DataFrame({key: df[key].drop_duplicates()})
    subset["flag"] = 1
    wide = subset.pivot_table(index=key, columns=value, values="flag", aggfunc="max", fill_value=0)
    wide.columns = [f"{prefix}_{_safe_feature_name(col)}" for col in wide.columns]
    wide = wide.reset_index()
    for col in wide.columns:
        if col != key:
            wide[col] = wide[col].astype(int)
    return wide


def build_mimic_liver_dataset(input_root: str, output_root: str, task_text: str) -> dict[str, object]:
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    structured = input_dir / "structured"
    notes_dir = input_dir / "notes"

    admissions = pd.read_csv(structured / "admissions.csv")
    patients = pd.read_csv(structured / "patients.csv")
    diagnoses = pd.read_csv(structured / "diagnoses_icd.csv")
    icustays = pd.read_csv(structured / "icustays.csv")
    labevents = pd.read_csv(structured / "labevents.csv")
    prescriptions = pd.read_csv(structured / "prescriptions.csv")
    discharge_notes = pd.read_csv(notes_dir / "discharge_notes.csv") if (notes_dir / "discharge_notes.csv").exists() else pd.DataFrame()
    radiology_notes = pd.read_csv(notes_dir / "radiology_notes.csv") if (notes_dir / "radiology_notes.csv").exists() else pd.DataFrame()

    diag = diagnoses.copy()
    diag["icd_code_clean"] = diag["icd_code"].map(lambda x: _code_prefix(x, 32))
    diag["icd_prefix3"] = diag["icd_code"].map(lambda x: _code_prefix(x, 3))
    diag["icd_prefix4"] = diag["icd_code"].map(lambda x: _code_prefix(x, 4))
    diag["is_hepatic"] = diag["icd_code_clean"].fillna("").str.startswith(HEPATIC_PREFIXES)

    label_df = diag.groupby("hadm_id", as_index=False)["is_hepatic"].max()
    label_df["hepatic_dx"] = label_df["is_hepatic"].astype(int)
    label_df = label_df[["hadm_id", "hepatic_dx"]]

    primary_diag = (
        diag.sort_values(["hadm_id", "seq_num"])
        .dropna(subset=["icd_code_clean"])
        .drop_duplicates("hadm_id")[["hadm_id", "icd_code_clean", "icd_prefix3", "icd_prefix4"]]
        .rename(columns={
            "icd_code_clean": "primary_icd_code",
            "icd_prefix3": "primary_icd_prefix3",
            "icd_prefix4": "primary_icd_prefix4",
        })
    )

    all_diag = diag.groupby("hadm_id", as_index=False).agg(
        diagnoses_count=("icd_code_clean", "count"),
        unique_diagnoses_count=("icd_code_clean", "nunique"),
        hepatic_code_count=("is_hepatic", "sum"),
        all_icd_codes=("icd_code_clean", lambda s: ",".join(sorted({x for x in s.dropna().astype(str) if x}))),
    )

    diag_prefix3 = _pivot_top_counts(diag.dropna(subset=["icd_prefix3"]), "hadm_id", "icd_prefix3", "icd3", top_n=20)
    diag_prefix4 = _pivot_top_counts(diag.dropna(subset=["icd_prefix4"]), "hadm_id", "icd_prefix4", "icd4", top_n=20)

    icu_features = icustays.groupby("hadm_id", as_index=False).agg(
        icu_flag=("stay_id", lambda s: int(s.notna().any())),
        icu_stay_count=("stay_id", "nunique"),
        icu_los_mean=("los", "mean"),
        icu_los_max=("los", "max"),
        first_careunit_nunique=("first_careunit", "nunique"),
    )

    lab = labevents.dropna(subset=["hadm_id"]).copy()
    lab["itemid"] = lab["itemid"].astype(str)
    lab_global = lab.groupby("hadm_id", as_index=False).agg(
        lab_event_count=("valuenum", "count"),
        lab_itemid_nunique=("itemid", "nunique"),
        lab_valuenum_mean=("valuenum", "mean"),
        lab_valuenum_std=("valuenum", "std"),
        lab_valuenum_min=("valuenum", "min"),
        lab_valuenum_max=("valuenum", "max"),
    )
    top_lab_items = lab["itemid"].value_counts().head(15).index.tolist()
    top_lab = lab[lab["itemid"].isin(top_lab_items)].copy()
    lab_pivot = top_lab.pivot_table(index="hadm_id", columns="itemid", values="valuenum", aggfunc=["mean", "max", "min"])
    lab_pivot.columns = [f"lab_{stat}_{itemid}" for stat, itemid in lab_pivot.columns]
    lab_pivot = lab_pivot.reset_index()

    rx = prescriptions.copy()
    rx["drug_norm"] = rx["drug"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    rx["hep_drug_flag"] = rx["drug_norm"].str.contains("|".join(HEPATIC_DRUG_KEYWORDS), regex=True, na=False)
    rx_global = rx.groupby("hadm_id", as_index=False).agg(
        prescription_count=("drug_norm", "count"),
        unique_drug_count=("drug_norm", "nunique"),
        hep_drug=("hep_drug_flag", "max"),
        hep_drug_count=("hep_drug_flag", "sum"),
        doses_per_24_hrs_mean=("doses_per_24_hrs", "mean"),
        doses_per_24_hrs_max=("doses_per_24_hrs", "max"),
    )
    top_drugs = rx["drug_norm"].value_counts().head(20).index.tolist()
    rx_top = _pivot_top_counts(rx[rx["drug_norm"].isin(top_drugs)], "hadm_id", "drug_norm", "drug", top_n=20)

    note_features = []
    if not discharge_notes.empty:
        ds = discharge_notes.copy()
        ds["text_len"] = ds["text"].astype(str).str.len()
        ds["mentions_liver"] = ds["text"].astype(str).str.contains(r"liver|hepatic|cirrho|ascites|jaundice|hepat", case=False, regex=True, na=False).astype(int)
        note_features.append(
            ds.groupby("hadm_id", as_index=False).agg(
                discharge_note_count=("note_id", "count"),
                discharge_note_text_len_mean=("text_len", "mean"),
                discharge_note_mentions_liver=("mentions_liver", "max"),
            )
        )
    if not radiology_notes.empty:
        rn = radiology_notes.dropna(subset=["hadm_id"]).copy()
        rn["text_len"] = rn["text"].astype(str).str.len()
        rn["mentions_liver"] = rn["text"].astype(str).str.contains(r"liver|hepatic|cirrho|ascites|jaundice|hepat", case=False, regex=True, na=False).astype(int)
        note_features.append(
            rn.groupby("hadm_id", as_index=False).agg(
                radiology_note_count=("note_id", "count"),
                radiology_note_text_len_mean=("text_len", "mean"),
                radiology_note_mentions_liver=("mentions_liver", "max"),
            )
        )

    dataset = admissions.merge(patients, on="subject_id", how="left")
    for frame in [label_df, primary_diag, all_diag, diag_prefix3, diag_prefix4, icu_features, lab_global, lab_pivot, rx_global, rx_top, *note_features]:
        dataset = dataset.merge(frame, on="hadm_id", how="left")

    dataset["hepatic_dx"] = dataset["hepatic_dx"].fillna(0).astype(int)
    for col in ["hep_drug", "hep_drug_count", "icu_flag", "icu_stay_count", "diagnoses_count", "unique_diagnoses_count", "hepatic_code_count", "hospital_expire_flag"]:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce").fillna(0).astype(int)

    drop_cols = [c for c in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime", "dod"] if c in dataset.columns]
    dataset = dataset.drop(columns=drop_cols)

    indicator_prefixes = ("icd3_", "icd4_", "drug_")
    for col in dataset.columns:
        if col.startswith(indicator_prefixes):
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce").fillna(0).astype(int)

    task_slug = _slugify_task(task_text)
    csv_path = output_dir / f"{task_slug}_rich.csv"
    dataset.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report = {
        "task": task_text,
        "dataset_path": str(csv_path),
        "report_path": str(output_dir / f"{task_slug}_report.json"),
        "row_count": int(dataset.shape[0]),
        "feature_count": int(max(dataset.shape[1] - 3, 0)),
        "label_column": "hepatic_dx",
        "label_distribution": {str(k): int(v) for k, v in dataset["hepatic_dx"].value_counts(dropna=False).to_dict().items()},
        "top_lab_items": top_lab_items,
        "top_drugs": top_drugs,
        "notes": [
            "标签来自 diagnoses_icd 的肝病相关 ICD 前缀聚合。",
            "保留了住院级 ICD 串、主诊断编码、top ICD 前缀 one-hot。",
            "labevents 按 top itemid 做 mean/max/min 聚合，而不是仅全局统计。",
            "prescriptions 保留了 top drug one-hot 和肝病相关药物聚合。",
        ],
    }
    report_path = output_dir / f"{task_slug}_report.json"
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    report_path.write_text(report_text, encoding="utf-8")
    (output_dir / "data_analysis_report.json").write_text(report_text, encoding="utf-8")
    return report
