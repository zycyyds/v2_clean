from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from reproduction.raw_level_error_split import RAW_TEMPLATE_OUTPUT_RELATIVES, REFERENCE_RELATIVES


def _write_csv(path: Path, rows: list[dict[str, object]], *, gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    if gzip_output:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_raw(root: Path, stay_count: int) -> Path:
    patients: list[dict[str, object]] = []
    admissions: list[dict[str, object]] = []
    icustays: list[dict[str, object]] = []
    diagnoses: list[dict[str, object]] = []
    chart: list[dict[str, object]] = []
    inputevents: list[dict[str, object]] = []
    outputevents: list[dict[str, object]] = []
    procedureevents: list[dict[str, object]] = []
    for offset in range(stay_count):
        subject, hadm, stay = 1000 + offset, 2000 + offset, 3000 + offset
        patients.append({"subject_id": subject, "gender": "M" if offset % 2 else "F", "anchor_age": 40 + offset, "anchor_year": 2015, "anchor_year_group": "2014 - 2016", "dod": ""})
        admissions.append({"subject_id": subject, "hadm_id": hadm, "admittime": "2020-01-01 00:00:00", "dischtime": "2020-01-10 00:00:00", "deathtime": "", "insurance": "Medicare", "race": "WHITE", "hospital_expire_flag": 0})
        icustays.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2020-01-02 00:00:00", "outtime": "2020-01-03 00:00:00", "los": 1.0})
        diagnoses.append({"subject_id": subject, "hadm_id": hadm, "seq_num": 1, "icd_code": "I500", "icd_version": 10})
        for event in range(3):
            chart.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "caregiver_id": 1, "charttime": f"2020-01-02 0{event}:00:00", "storetime": f"2020-01-02 0{event}:05:00", "itemid": 220045, "value": str(70 + event), "valuenum": 70 + event, "valueuom": "bpm", "warning": 0})
        inputevents.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "caregiver_id": 1, "starttime": "2020-01-02 01:00:00", "endtime": "2020-01-02 02:00:00", "storetime": "2020-01-02 01:05:00", "itemid": 225158, "amount": 1.0, "amountuom": "mg", "rate": "", "rateuom": "", "orderid": 5000 + offset})
        outputevents.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "caregiver_id": 1, "charttime": "2020-01-02 03:00:00", "storetime": "2020-01-02 03:05:00", "itemid": 226559, "value": 10.0, "valueuom": "mL"})
        procedureevents.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "caregiver_id": 1, "starttime": "2020-01-02 01:00:00", "endtime": "2020-01-02 02:00:00", "storetime": "2020-01-02 01:05:00", "itemid": 225441, "value": 1.0, "valueuom": "unit"})
    for relative, rows in {
        "hosp/patients.csv.gz": patients,
        "hosp/admissions.csv.gz": admissions,
        "hosp/diagnoses_icd.csv.gz": diagnoses,
        "hosp/d_icd_diagnoses.csv.gz": [{"icd_code": "I500", "icd_version": 10, "long_title": "Heart failure"}],
        "hosp/labevents.csv.gz": [{"subject_id": 1000, "hadm_id": 2000, "value": "unrelated"}],
        "icu/icustays.csv.gz": icustays,
        "icu/chartevents.csv.gz": chart,
        "icu/inputevents.csv.gz": inputevents,
        "icu/outputevents.csv.gz": outputevents,
        "icu/procedureevents.csv.gz": procedureevents,
        "icu/d_items.csv.gz": [{"itemid": 220045, "label": "Heart Rate"}],
    }.items():
        _write_csv(root / relative, rows, gzip_output=True)
    lookup_tables = (
        "hosp/d_hcpcs.csv.gz",
        "hosp/d_icd_procedures.csv.gz",
        "hosp/d_labitems.csv.gz",
    )
    subject_tables = (
        "hosp/drgcodes.csv.gz",
        "hosp/emar.csv.gz",
        "hosp/emar_detail.csv.gz",
        "hosp/hcpcsevents.csv.gz",
        "hosp/labevents.csv.gz",
        "hosp/microbiologyevents.csv.gz",
        "hosp/omr.csv.gz",
        "hosp/pharmacy.csv.gz",
        "hosp/poe.csv.gz",
        "hosp/poe_detail.csv.gz",
        "hosp/prescriptions.csv.gz",
        "hosp/procedures_icd.csv.gz",
        "hosp/services.csv.gz",
        "hosp/transfers.csv.gz",
        "icu/datetimeevents.csv.gz",
        "icu/ingredientevents.csv.gz",
    )
    for relative in lookup_tables:
        _write_csv(root / relative, [{"code": "lookup"}], gzip_output=True)
    for relative in subject_tables:
        _write_csv(root / relative, [{"subject_id": 1000}], gzip_output=True)
    return root


def _build_clean_package(root: Path, stay_count: int) -> tuple[Path, Path]:
    stays = [3000 + index for index in range(stay_count)]
    cohort_rows = [{"subject_id": 1000 + index, "hadm_id": 2000 + index, "stay_id": stay, "intime": "2020-01-02 00:00:00", "outtime": "2020-01-03 00:00:00", "Age": 40 + index, "gender": "M", "ethnicity": "WHITE", "insurance": "Medicare", "label": index % 2, "dod": ""} for index, stay in enumerate(stays)]
    _write_csv(root / "cohort/cohort_icu_mortality_0__.csv.gz", cohort_rows, gzip_output=True)
    _write_csv(root / "labels.csv", [{"stay_id": stay, "label": index % 2} for index, stay in enumerate(stays)])
    for name in ("preproc_chart_icu", "preproc_diag_icu", "preproc_med_icu", "preproc_out_icu", "preproc_proc_icu"):
        _write_csv(root / f"features/{name}.csv.gz", [{"stay_id": stay, "itemid": 220045, "value": index} for index, stay in enumerate(stays)], gzip_output=True)
    for relative in [item for item in REFERENCE_RELATIVES if item.startswith("summary/")]:
        _write_csv(root / relative, [{"itemid": 220045, "total_count": stay_count}])
    shutil.copytree(root, root.parent / "dirty")

    subtype_specs = (
        ("schema_range_violation", "cohort/cohort_icu_mortality_0__.csv.gz"),
        ("enum_violation", "cohort/cohort_icu_mortality_0__.csv.gz"),
        ("unexpected_missingness", "cohort/cohort_icu_mortality_0__.csv.gz"),
        ("temporal_order_violation", "cohort/cohort_icu_mortality_0__.csv.gz"),
        ("med_end_before_start", "features/preproc_med_icu.csv.gz"),
        ("sentinel_or_negative_med_amount", "features/preproc_med_icu.csv.gz"),
        ("sentinel_chart_value", "features/preproc_chart_icu.csv.gz"),
        ("unit_scale_error", "features/preproc_chart_icu.csv.gz"),
        ("duplicate_row", "features/preproc_chart_icu.csv.gz"),
        ("implausible_negative_event_time", "features/preproc_out_icu.csv.gz"),
        ("chart_itemid_alias_split", "features/preproc_chart_icu.csv.gz"),
        ("medication_itemid_alias_split", "features/preproc_med_icu.csv.gz"),
        ("diagnosis_code_alias_split", "features/preproc_diag_icu.csv.gz"),
        ("orphan_feature_stay_id", "features/preproc_chart_icu.csv.gz"),
        ("stay_hadm_mismatch", "features/preproc_med_icu.csv.gz"),
        ("label_conflict_with_cohort", "labels.csv"),
        ("future_window_leakage", "features/preproc_chart_icu.csv.gz"),
        ("label_leakage_feature", "features/preproc_chart_icu.csv.gz"),
    )
    classes = ("class1_intra_table", "class2_entity_alignment", "class3_cross_table", "class4_task_oriented")
    rows = []
    for index, (subtype, relative) in enumerate(subtype_specs):
        rows.append({"operation": "row_insert" if subtype in {"duplicate_row", "label_leakage_feature"} else "cell_update", "file": relative, "clean_row_index": "" if subtype in {"duplicate_row", "label_leakage_feature"} else index, "dirty_row_index": index, "column": "", "clean_value": "", "dirty_value": "1", "error_class": classes[index % len(classes)], "error_subtype": subtype, "repair_hint": "private", "stay_id": "999999" if subtype in {"duplicate_row", "label_leakage_feature"} else ""})
    log = root.parent / "modification_log.csv"
    _write_csv(log, rows)
    return root, log


def test_build_raw_level_split_copies_raw_and_keeps_reference_clean(tmp_path: Path) -> None:
    from reproduction.raw_level_error_split import build_raw_level_error_split

    source = _build_raw(tmp_path / "source_3_1", stay_count=30)
    clean, source_log = _build_clean_package(tmp_path / "clean", stay_count=30)
    source_before = _tree_digest(source)

    result = build_raw_level_error_split(
        source,
        clean,
        source_log,
        tmp_path / "raw_level_split",
        train_count=4,
        validation_count=6,
        selection_seed=8,
    )

    output = tmp_path / "raw_level_split"
    assert result["status"] == "SUCCESS"
    assert _tree_digest(source) == source_before
    assert {name: len(pd.read_csv(output / name / "keys.csv")) for name in ("train", "validation", "test")} == {"train": 4, "validation": 6, "test": 20}
    for split in ("train", "validation", "test"):
        reference_name = "reference" if split == "train" else "reference_private"
        assert (output / split / "raw/hosp").is_dir()
        assert (output / split / "raw/icu").is_dir()
        raw_files = {
            path.relative_to(output / split / "raw").as_posix()
            for path in (output / split / "raw").rglob("*")
            if path.is_file()
        }
        assert raw_files == RAW_TEMPLATE_OUTPUT_RELATIVES
        assert not any(path.suffix == ".gz" for path in (output / split / "raw").rglob("*"))
        assert len(list((output / split / reference_name).rglob("*.csv"))) == 17
        assert not any(path.suffix == ".gz" for path in (output / split / reference_name).rglob("*"))
        assert not any(path.is_symlink() for path in (output / split).rglob("*"))
        with (output / "host_private" / f"{split}_raw_modification_log.csv").open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        assert records
        assert all(record["raw_file"].startswith(("hosp/", "icu/")) for record in records)

    all_records = []
    for split in ("train", "validation", "test"):
        with (output / "host_private" / f"{split}_raw_modification_log.csv").open(newline="", encoding="utf-8") as handle:
            all_records.extend(csv.DictReader(handle))
    assert len(all_records) == 18
    assert {record["error_subtype"] for record in all_records} == {
        "schema_range_violation", "enum_violation", "unexpected_missingness", "temporal_order_violation",
        "med_end_before_start", "sentinel_or_negative_med_amount", "sentinel_chart_value", "unit_scale_error",
        "duplicate_row", "implausible_negative_event_time", "chart_itemid_alias_split", "medication_itemid_alias_split",
        "diagnosis_code_alias_split", "orphan_feature_stay_id", "stay_hadm_mismatch", "label_conflict_with_cohort",
        "future_window_leakage", "label_leakage_feature",
    }

    dirty_chart = pd.read_csv(output / "test/raw/icu/chartevents.csv", dtype=object)
    source_chart = pd.read_csv(source / "icu/chartevents.csv.gz", dtype=object)
    assert len(dirty_chart) >= len(source_chart[source_chart["stay_id"].isin(pd.read_csv(output / "test/keys.csv", dtype=str)["stay_id"])])
    clean_chart = pd.read_csv(output / "test/reference_private/features/preproc_chart_icu.csv", dtype=object)
    assert "999001" not in set(clean_chart["itemid"].astype(str))
    report = json.loads((output / "split_validation_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert all(item["passed"] for item in report["checks"])


def test_raw_level_split_rejects_nonempty_output_and_bad_counts(tmp_path: Path) -> None:
    from reproduction.raw_level_error_split import build_raw_level_error_split

    source = _build_raw(tmp_path / "source_3_1", stay_count=18)
    clean, source_log = _build_clean_package(tmp_path / "clean", stay_count=18)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        build_raw_level_error_split(source, clean, source_log, occupied, train_count=1, validation_count=1)
    with pytest.raises(ValueError, match="leave at least one Test"):
        build_raw_level_error_split(source, clean, source_log, tmp_path / "too_large", train_count=10, validation_count=8)
