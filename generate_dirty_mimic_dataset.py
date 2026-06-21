from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from lib.mimic_dataset_builder import build_mimic_liver_dataset

V2_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = V2_DIR / "input" / "mimic-mini"
DEFAULT_OUTPUT_ROOT = V2_DIR / "output" / "dirty_mimic_mini_benchmark"
DEFAULT_SEED = 20260613


def pick_indices(length: int, label: str, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    count = min(length, count)
    keyed = []
    for idx in range(length):
        digest = hashlib.sha256(f"{DEFAULT_SEED}:{label}:{idx}".encode("utf-8")).hexdigest()
        keyed.append((digest, idx))
    keyed.sort()
    return sorted(idx for _, idx in keyed[:count])


def assign_values(df: pd.DataFrame, rows: list[int], column: str, values: list[object], report: dict[str, list[dict[str, object]]], error_type: str) -> None:
    if column not in df.columns or not rows:
        return
    changes = report.setdefault(error_type, [])
    for offset, row in enumerate(rows):
        before = df.at[row, column]
        after = values[offset % len(values)]
        df.at[row, column] = after
        changes.append(
            {
                "row": int(row),
                "column": column,
                "before": "" if before is None else str(before)[:120],
                "after": "" if after is None else str(after)[:120],
            }
        )


def duplicate_rows(df: pd.DataFrame, rows: list[int], report: dict[str, list[dict[str, object]]], error_type: str, patch: dict[str, object] | None = None) -> pd.DataFrame:
    if not rows:
        return df
    patch = patch or {}
    copies = df.iloc[rows].copy()
    changes = report.setdefault(error_type, [])
    for idx in copies.index:
        for column, value in patch.items():
            if column in copies.columns:
                copies.at[idx, column] = value
        changes.append(
            {
                "source_row": int(idx),
                "patch": {key: str(value) for key, value in patch.items() if key in copies.columns},
            }
        )
    return pd.concat([df, copies], ignore_index=True)


def dirty_structured_tables(clean_root: Path, dirty_root: Path) -> dict[str, dict[str, list[dict[str, object]]]]:
    reports: dict[str, dict[str, list[dict[str, object]]]] = {}
    structured_in = clean_root / "structured"
    structured_out = dirty_root / "structured"
    structured_out.mkdir(parents=True, exist_ok=True)

    admissions = pd.read_csv(structured_in / "admissions.csv", dtype=object, keep_default_na=False)
    admission_report: dict[str, list[dict[str, object]]] = {}
    assign_values(admissions, pick_indices(len(admissions), "admissions_nulls", 18), "language", ["未知", "N/A", "  "], admission_report, "null_like_tokens")
    assign_values(admissions, pick_indices(len(admissions), "admissions_time", 16), "admittime", ["2180/05/06 22:23", "05-07-2180 17:15", "2180-6-7 8:5"], admission_report, "datetime_format_drift")
    assign_values(admissions, pick_indices(len(admissions), "admissions_flag", 12), "hospital_expire_flag", ["0 ", " 1", "unknown"], admission_report, "categorical_numeric_conflict")
    admissions.to_csv(structured_out / "admissions.csv", index=False, encoding="utf-8-sig")
    reports["admissions.csv"] = admission_report

    patients = pd.read_csv(structured_in / "patients.csv", dtype=object, keep_default_na=False)
    patients_report: dict[str, list[dict[str, object]]] = {}
    assign_values(patients, pick_indices(len(patients), "patients_gender", 10), "gender", ["female", "Male ", "UNK", ""], patients_report, "category_drift")
    assign_values(patients, pick_indices(len(patients), "patients_age", 12), "anchor_age", ["089", "-1", "300", " 52 "], patients_report, "numeric_outliers_and_formatting")
    assign_values(patients, pick_indices(len(patients), "patients_dod", 10), "dod", ["2180/09/09", "09-09-2180", ""], patients_report, "datetime_format_drift")
    patients.to_csv(structured_out / "patients.csv", index=False, encoding="utf-8-sig")
    reports["patients.csv"] = patients_report

    diagnoses = pd.read_csv(structured_in / "diagnoses_icd.csv", dtype=object, keep_default_na=False)
    diagnoses_report: dict[str, list[dict[str, object]]] = {}
    assign_values(diagnoses, pick_indices(len(diagnoses), "diag_icd", 80), "icd_code", ["572.3", " 78959 ", "v10.07", "???", "571-5"], diagnoses_report, "code_format_pollution")
    assign_values(diagnoses, pick_indices(len(diagnoses), "diag_seq", 50), "seq_num", ["01", "2 ", "first", ""], diagnoses_report, "mixed_numeric_tokens")
    assign_values(diagnoses, pick_indices(len(diagnoses), "diag_version", 40), "icd_version", ["9 ", "10", "nine", ""], diagnoses_report, "mixed_version_tokens")
    diagnoses = duplicate_rows(diagnoses, pick_indices(len(diagnoses), "diag_dupes", 24), diagnoses_report, "duplicate_rows", {"seq_num": "999"})
    diagnoses.to_csv(structured_out / "diagnoses_icd.csv", index=False, encoding="utf-8-sig")
    reports["diagnoses_icd.csv"] = diagnoses_report

    icustays = pd.read_csv(structured_in / "icustays.csv", dtype=object, keep_default_na=False)
    icustays_report: dict[str, list[dict[str, object]]] = {}
    assign_values(icustays, pick_indices(len(icustays), "icu_los", 12), "los", ["0.4102662037037037 ", "-0.5", "9999", "1,25"], icustays_report, "numeric_outliers_and_formatting")
    assign_values(icustays, pick_indices(len(icustays), "icu_unit", 8), "first_careunit", ["MICU  ", "medical intensive care unit (micu)", "??"], icustays_report, "category_drift")
    assign_values(icustays, pick_indices(len(icustays), "icu_intime", 8), "intime", ["2180/07/23 14:00", "23-07-2180 23:50:47"], icustays_report, "datetime_format_drift")
    icustays.to_csv(structured_out / "icustays.csv", index=False, encoding="utf-8-sig")
    reports["icustays.csv"] = icustays_report

    labevents = pd.read_csv(structured_in / "labevents.csv", dtype=object, keep_default_na=False)
    labevents_report: dict[str, list[dict[str, object]]] = {}
    assign_values(labevents, pick_indices(len(labevents), "lab_valuenum", 800), "valuenum", ["1,234.5", "<5", "error", "", " 98 "], labevents_report, "mixed_numeric_tokens")
    assign_values(labevents, pick_indices(len(labevents), "lab_value", 500), "value", ["___", "hemolyzed sample", "see note", "negative"], labevents_report, "semantic_text_noise")
    assign_values(labevents, pick_indices(len(labevents), "lab_flag", 300), "flag", ["abnormal", "HIGH", "low ", "?"], labevents_report, "category_drift")
    assign_values(labevents, pick_indices(len(labevents), "lab_charttime", 300), "charttime", ["2180/03/23 11:51", "03-23-2180 16:00", ""], labevents_report, "datetime_format_drift")
    labevents.to_csv(structured_out / "labevents.csv", index=False, encoding="utf-8-sig")
    reports["labevents.csv"] = labevents_report

    prescriptions = pd.read_csv(structured_in / "prescriptions.csv", dtype=object, keep_default_na=False)
    prescriptions_report: dict[str, list[dict[str, object]]] = {}
    assign_values(prescriptions, pick_indices(len(prescriptions), "rx_drug", 200), "drug", [" furosemide ", "LACTULOSE", "Spironolactone??", "UNKNOWN_DRUG"], prescriptions_report, "category_drift")
    assign_values(prescriptions, pick_indices(len(prescriptions), "rx_dose24", 180), "doses_per_24_hrs", ["4.0 ", "q6h", "1,0", ""], prescriptions_report, "mixed_numeric_tokens")
    assign_values(prescriptions, pick_indices(len(prescriptions), "rx_time", 180), "starttime", ["2180/05/08 08:00", "05-07-2180 22:00", ""], prescriptions_report, "datetime_format_drift")
    assign_values(prescriptions, pick_indices(len(prescriptions), "rx_route", 120), "route", ["po/ng", " IV ", "unknown", ""], prescriptions_report, "category_drift")
    prescriptions = duplicate_rows(prescriptions, pick_indices(len(prescriptions), "rx_dupes", 60), prescriptions_report, "duplicate_rows", {"dose_val_rx": "999"})
    prescriptions.to_csv(structured_out / "prescriptions.csv", index=False, encoding="utf-8-sig")
    reports["prescriptions.csv"] = prescriptions_report

    return reports


def dirty_note_tables(clean_root: Path, dirty_root: Path) -> dict[str, dict[str, list[dict[str, object]]]]:
    reports: dict[str, dict[str, list[dict[str, object]]]] = {}
    notes_in = clean_root / "notes"
    notes_out = dirty_root / "notes"
    notes_out.mkdir(parents=True, exist_ok=True)

    for name in ["discharge_notes.csv", "radiology_notes.csv"]:
        path = notes_in / name
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=object, keep_default_na=False)
        report: dict[str, list[dict[str, object]]] = {}
        if "text" in df.columns:
            rows = pick_indices(len(df), f"{name}_text_noise", min(120, max(20, len(df) // 50)))
            noisy_values = []
            for row in rows:
                original = str(df.at[row, "text"])
                noisy_values.append(f"[OCR_NOISE]  {original[:800]}\n\n<dup_section> {original[:200]}")
            assign_values(df, rows, "text", noisy_values or ["[OCR_NOISE] unreadable"], report, "high_risk_text_noise")
        if "charttime" in df.columns:
            assign_values(df, pick_indices(len(df), f"{name}_charttime", min(80, max(10, len(df) // 80))), "charttime", ["2180/01/02 03:04", "01-02-2180 03:04:05", ""], report, "datetime_format_drift")
        df.to_csv(notes_out / name, index=False, encoding="utf-8-sig")
        reports[name] = report

    return reports


def dirty_cxr_table(clean_root: Path, dirty_root: Path) -> dict[str, dict[str, list[dict[str, object]]]]:
    reports: dict[str, dict[str, list[dict[str, object]]]] = {}
    src = clean_root / "cxr_sampled_with_reports.csv"
    if not src.exists():
        return reports
    df = pd.read_csv(src, dtype=object, keep_default_na=False)
    report: dict[str, list[dict[str, object]]] = {}
    report_cols = [col for col in df.columns if "report" in col.lower() or "impression" in col.lower() or "findings" in col.lower()]
    for col in report_cols[:2]:
        rows = pick_indices(len(df), f"cxr_{col}", min(60, max(12, len(df) // 20)))
        noisy_values = []
        for row in rows:
            original = str(df.at[row, col])
            noisy_values.append(f"###AUTO_IMPORT###\n{original[:600]}\n\n[[duplicate sentence]] {original[:120]}")
        assign_values(df, rows, col, noisy_values or ["###AUTO_IMPORT###"], report, "high_risk_text_noise")
    time_cols = [col for col in df.columns if "time" in col.lower() or "date" in col.lower()]
    for col in time_cols[:2]:
        assign_values(df, pick_indices(len(df), f"cxr_{col}_time", min(30, max(8, len(df) // 40))), col, ["2180/01/02 03:04", "01-02-2180", ""], report, "datetime_format_drift")
    df.to_csv(dirty_root / "cxr_sampled_with_reports.csv", index=False, encoding="utf-8-sig")
    reports[src.name] = report
    return reports


def build_dirty_flat_dataset(clean_root: Path, output_root: Path) -> dict[str, object]:
    flat_root = output_root / "flat_clean_reference"
    report = build_mimic_liver_dataset(str(clean_root), str(flat_root), "mimic-mini 肝病诊断基准数据集")
    clean_csv = Path(report["dataset_path"])
    dirty_csv = output_root / "dirty_flat_liver_dataset.csv"
    df = pd.read_csv(clean_csv, dtype=object, keep_default_na=False)
    flat_report: dict[str, list[dict[str, object]]] = {}

    for col in [c for c in ["admission_type", "insurance", "language", "marital_status", "race", "gender"] if c in df.columns]:
        assign_values(df, pick_indices(len(df), f"flat_cat_{col}", 12), col, ["未知", " n/a ", "??", "mixed-case"], flat_report, "flat_category_drift")
    for col in [c for c in df.columns if c.startswith("lab_")][:8]:
        assign_values(df, pick_indices(len(df), f"flat_lab_{col}", 8), col, ["1,234.5", "<5", "", "-9999"], flat_report, "flat_numeric_noise")
    for col in [c for c in ["all_icd_codes", "primary_icd_code"] if c in df.columns]:
        assign_values(df, pick_indices(len(df), f"flat_code_{col}", 10), col, ["572.3|789.59", " ??? ", "571-5"], flat_report, "flat_code_pollution")

    note_cols = [c for c in df.columns if "note" in c.lower()]
    for col in note_cols[:4]:
        assign_values(df, pick_indices(len(df), f"flat_note_{col}", 8), col, ["[OCR] liver??\n\nDUPLICATE DUPLICATE", " ", "报告待补录"], flat_report, "flat_text_noise")

    df.to_csv(dirty_csv, index=False, encoding="utf-8-sig")
    return {
        "clean_reference_csv": str(clean_csv),
        "dirty_flat_csv": str(dirty_csv),
        "mutation_report": flat_report,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }


def summarize_reports(report_map: dict[str, dict[str, list[dict[str, object]]]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for filename, report in report_map.items():
        summary[filename] = {kind: len(items) for kind, items in report.items()}
    return summary


def generate_dirty_benchmark(input_root: Path, output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    dirty_root = output_root / "dirty_mimic_mini"
    if dirty_root.exists():
        shutil.rmtree(dirty_root)
    dirty_root.mkdir(parents=True, exist_ok=True)

    mutation_reports: dict[str, dict[str, list[dict[str, object]]]] = {}
    mutation_reports.update(dirty_structured_tables(input_root, dirty_root))
    mutation_reports.update(dirty_note_tables(input_root, dirty_root))
    mutation_reports.update(dirty_cxr_table(input_root, dirty_root))

    flat = build_dirty_flat_dataset(input_root, output_root)

    benchmark_report = {
        "seed": DEFAULT_SEED,
        "source_root": str(input_root),
        "dirty_root": str(dirty_root),
        "error_types": {
            "null_like_tokens": "把空值伪装成未知、N/A、空白等字符串。",
            "datetime_format_drift": "同一列混入多种日期时间格式或空串。",
            "mixed_numeric_tokens": "数值列混入逗号、小于号、文本、空串。",
            "categorical_numeric_conflict": "本应数值/布尔的列混入 unknown 等类别值。",
            "category_drift": "类别值大小写、空白、别名、未知值漂移。",
            "code_format_pollution": "诊断编码混入点号、连字符、空白、非法字符。",
            "duplicate_rows": "复制部分记录并打补丁，制造近重复或冲突记录。",
            "semantic_text_noise": "短文本列混入自由文本说明。",
            "high_risk_text_noise": "长文本列加入 OCR 噪声、重复段落、模板标记。",
            "numeric_outliers_and_formatting": "数值列同时出现极值、负值、零填充、格式漂移。",
            "flat_category_drift": "平铺训练表中的类别列污染。",
            "flat_numeric_noise": "平铺训练表中的数值特征污染。",
            "flat_code_pollution": "平铺训练表中的编码聚合列污染。",
            "flat_text_noise": "平铺训练表中的文本特征污染。",
        },
        "per_file_summary": summarize_reports(mutation_reports),
        "flat_dataset_summary": {
            "dirty_flat_csv": flat["dirty_flat_csv"],
            "row_count": flat["row_count"],
            "column_count": flat["column_count"],
            "mutation_counts": {kind: len(items) for kind, items in flat["mutation_report"].items()},
        },
        "suggested_agent_inputs": {
            "directory_cleaning": str(dirty_root),
            "single_table_cleaning": str(flat["dirty_flat_csv"]),
        },
        "mutation_details": mutation_reports,
        "flat_mutation_details": flat["mutation_report"],
    }

    report_path = output_root / "dirty_benchmark_report.json"
    report_path.write_text(json.dumps(benchmark_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dirty_root": str(dirty_root),
        "report_path": str(report_path),
        "dirty_flat_csv": flat["dirty_flat_csv"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible dirty-data benchmark from v2/input/mimic-mini.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    result = generate_dirty_benchmark(Path(args.input_root), Path(args.output_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
