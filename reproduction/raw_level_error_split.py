"""Build a raw-MIMIC error-injection benchmark without mutating its source data.

The source error package supplies cohort membership and error-class proportions only.
Its feature-level mutations are deliberately not reverse-mapped to raw rows: the
output lineage is many-to-one.  Instead, this module creates auditable,
semantically equivalent corruptions in copied ``hosp`` and ``icu`` source tables.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, TextIO

import pandas as pd

from workflow.reference_splits import copy_mimic_raw_subset


SPLITS = ("train", "validation", "test")
REFERENCE_RELATIVES = (
    "cohort/cohort_icu_mortality_0__.csv",
    "csv/labels.csv",
    "features/preproc_chart_icu.csv",
    "features/preproc_diag_icu.csv",
    "features/preproc_med_icu.csv",
    "features/preproc_out_icu.csv",
    "features/preproc_proc_icu.csv",
    "summary/chart_features.csv",
    "summary/chart_summary.csv",
    "summary/diag_features.csv",
    "summary/diag_summary.csv",
    "summary/med_features.csv",
    "summary/med_summary.csv",
    "summary/out_features.csv",
    "summary/out_summary.csv",
    "summary/proc_features.csv",
    "summary/proc_summary.csv",
)
RAW_TEMPLATE_RELATIVES = {
    "hosp/admissions.csv.gz",
    "hosp/d_hcpcs.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/d_icd_procedures.csv.gz",
    "hosp/d_labitems.csv.gz",
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
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/inputevents.csv.gz",
    "icu/outputevents.csv.gz",
    "icu/procedureevents.csv.gz",
    "icu/d_items.csv.gz",
    "icu/datetimeevents.csv.gz",
    "icu/ingredientevents.csv.gz",
}
RAW_TEMPLATE_OUTPUT_RELATIVES = frozenset(
    relative.removesuffix(".gz") for relative in RAW_TEMPLATE_RELATIVES
)

PRIVATE_LOG_FIELDS = (
    "source_error_id",
    "canonical_stay_id",
    "error_class",
    "error_subtype",
    "target_effect_contract",
    "operation",
    "raw_file",
    "raw_row_index",
    "column",
    "clean_value",
    "dirty_value",
)


def build_raw_level_error_split(
    source_raw_3_1: str | Path,
    clean_package: str | Path,
    source_modification_log: str | Path,
    output_root: str | Path,
    *,
    train_count: int = 10,
    validation_count: int = 20,
    selection_seed: int = 20260804,
) -> dict[str, Any]:
    """Create a 10:20:970-style physical raw MIMIC split with hidden Gold.

    ``clean_package`` and its modification log are inputs controlled by the host.
    They are never copied below an agent-visible ``raw`` directory.  A caller must
    keep ``host_private`` inaccessible to the Agent process.
    """
    source = Path(source_raw_3_1).expanduser().resolve()
    clean = Path(clean_package).expanduser().resolve()
    source_log = Path(source_modification_log).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    _validate_inputs(source, clean, source_log, output, train_count, validation_count)

    source_fingerprint_before = directory_metadata_fingerprint(source)
    all_stays = _read_clean_stays(clean)
    modifications = _load_source_modifications(clean, source_log, valid_stays=set(all_stays))
    assignments = _select_splits(
        all_stays,
        modifications,
        train_count=train_count,
        validation_count=validation_count,
        seed=selection_seed,
    )
    identity = {
        "workflow": "raw_level_error_split",
        "source_raw": str(source),
        "source_raw_fingerprint": source_fingerprint_before,
        "clean_package_sha256": directory_sha256(clean),
        "source_modification_log_sha256": file_sha256(source_log),
        "counts": {name: len(values) for name, values in assignments.items()},
        "selection_seed": selection_seed,
        "selection_method": "deterministic_error_coverage_then_stable_order",
        "raw_layout": "nested_icu_mortality_full_hosp_icu_csv_v1",
        "raw_template_files": sorted(RAW_TEMPLATE_OUTPUT_RELATIVES),
    }
    existing_manifest = output / "split_manifest.json"
    if existing_manifest.is_file():
        existing = _read_json(existing_manifest)
        if existing.get("identity") == identity and existing.get("status") == "SUCCESS":
            return {"status": "SUCCESS", "dataset_root": str(output), "manifest": existing}
        raise ValueError(f"existing raw-level split manifest does not match: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"raw-level split output is not empty: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        prepared = Path(temporary) / output.name
        prepared.mkdir()
        split_records: dict[str, Any] = {}
        injection_records: dict[str, list[dict[str, str]]] = {}
        for split, stays in assignments.items():
            split_dir = prepared / split
            _write_keys(split_dir / "keys.csv", stays)
            raw_dir = split_dir / "raw"
            raw_record = copy_mimic_raw_subset(
                source,
                raw_dir,
                split_dir / "keys.csv",
                split_key="stay_id",
                decompress_gzip=True,
                include_relative_paths=RAW_TEMPLATE_RELATIVES,
            )
            injection_records[split] = _inject_raw_errors(
                raw_dir,
                stays,
                [row for row in modifications if row["canonical_stay_id"] in stays],
            )
            reference_name = "reference" if split == "train" else "reference_private"
            reference_record = _materialize_clean_reference(clean, split_dir / reference_name, stays)
            split_records[split] = {
                "schema_version": 1,
                "split": split,
                "key_count": len(stays),
                "raw_root": str(output / split / "raw"),
                "reference_root": str(output / split / reference_name),
                "raw_record": raw_record,
                "reference_record": reference_record,
                "injected_error_count": len(injection_records[split]),
            }
            _write_json(split_dir / "split_record.json", split_records[split])

        host_private = prepared / "host_private"
        for split, rows in injection_records.items():
            _write_csv(host_private / f"{split}_raw_modification_log.csv", PRIVATE_LOG_FIELDS, rows)
        _write_json(
            host_private / "injection_summary.json",
            {
                "schema_version": 1,
                "by_split": {name: len(rows) for name, rows in injection_records.items()},
                "by_error_class": dict(Counter(row["error_class"] for rows in injection_records.values() for row in rows)),
                "by_error_subtype": dict(Counter(row["error_subtype"] for rows in injection_records.values() for row in rows)),
            },
        )
        source_fingerprint_after = directory_metadata_fingerprint(source)
        report = _validate_prepared_split(
            prepared,
            assignments=assignments,
            injection_records=injection_records,
            source_fingerprint_before=source_fingerprint_before,
            source_fingerprint_after=source_fingerprint_after,
        )
        _write_json(prepared / "split_validation_report.json", report)
        if not report["passed"]:
            failed = [item["name"] for item in report["checks"] if not item["passed"]]
            raise RuntimeError("raw-level split validation failed: " + ", ".join(failed))
        manifest = {
            "schema_version": 1,
            "status": "SUCCESS",
            "identity": identity,
            "split_key": "stay_id",
            "counts": {name: len(values) for name, values in assignments.items()},
            "paths": {
                split: {
                    "raw": str(output / split / "raw"),
                    "reference": str(output / split / ("reference" if split == "train" else "reference_private")),
                    "keys": str(output / split / "keys.csv"),
                }
                for split in SPLITS
            },
            "source_metadata_before": source_fingerprint_before,
            "source_metadata_after": source_fingerprint_after,
        }
        _write_json(prepared / "split_manifest.json", manifest)
        if output.exists():
            output.rmdir()
        os.replace(prepared, output)
    return {"status": "SUCCESS", "dataset_root": str(output), "manifest": manifest}


def _validate_inputs(
    source: Path,
    clean: Path,
    source_log: Path,
    output: Path,
    train_count: int,
    validation_count: int,
) -> None:
    if not (source / "hosp").is_dir() or not (source / "icu").is_dir():
        raise ValueError(f"source_raw_3_1 must contain hosp/ and icu/: {source}")
    for relative in ("hosp/admissions.csv.gz", "hosp/patients.csv.gz", "icu/icustays.csv.gz"):
        if not (source / relative).is_file():
            raise ValueError(f"required raw table is missing: {source / relative}")
    if not clean.is_dir():
        raise ValueError(f"clean_package does not exist: {clean}")
    if not source_log.is_file():
        raise ValueError(f"source_modification_log does not exist: {source_log}")
    if output == source or source in output.parents or output == clean or clean in output.parents:
        raise ValueError("output_root cannot be inside either input source")
    if train_count < 1 or validation_count < 1:
        raise ValueError("train_count and validation_count must be positive")


def _load_source_modifications(
    clean: Path,
    source_log: Path,
    *,
    valid_stays: set[str] | None = None,
) -> list[dict[str, str]]:
    with source_log.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("source modification log is empty")
    indexed: dict[str, dict[int, dict[str, str]]] = {}
    dirty_indexed: dict[str, dict[int, dict[str, str]]] = {}
    needed: dict[str, set[int]] = defaultdict(set)
    dirty_needed: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        file_name = str(row.get("file") or "").strip()
        index = _optional_index(row.get("clean_row_index"))
        if file_name and index is not None:
            needed[file_name].add(index)
        elif file_name:
            dirty_index = _optional_index(row.get("dirty_row_index"))
            if dirty_index is not None:
                dirty_needed[file_name].add(dirty_index)
    for relative, indices in needed.items():
        indexed[relative] = _rows_at_indices(clean / relative, indices)
    dirty_root = clean.parent / "dirty"
    for relative, indices in dirty_needed.items():
        candidate = dirty_root / relative
        if candidate.is_file():
            dirty_indexed[relative] = _rows_at_indices(candidate, indices)
    for index, row in enumerate(rows):
        clean_index = _optional_index(row.get("clean_row_index"))
        stay = ""
        if clean_index is not None:
            candidate = indexed.get(str(row.get("file") or ""), {}).get(
                clean_index,
                {},
            )
            stay = str(candidate.get("stay_id") or "").strip()
        if not stay:
            logged_stay = str(row.get("stay_id") or "").strip()
            if logged_stay and (valid_stays is None or logged_stay in valid_stays):
                stay = logged_stay
        if not stay:
            dirty_index = _optional_index(row.get("dirty_row_index"))
            candidate = dirty_indexed.get(str(row.get("file") or ""), {}).get(
                dirty_index if dirty_index is not None else -1,
                {},
            )
            stay = str(candidate.get("stay_id") or "").strip()
        if valid_stays is not None and stay not in valid_stays:
            stay = ""
        if not stay:
            raise ValueError(f"cannot resolve canonical stay_id for source error row {index}")
        row["source_error_id"] = str(index)
        row["canonical_stay_id"] = stay
    return rows


def _read_clean_stays(clean: Path) -> list[str]:
    cohort = clean / "cohort" / "cohort_icu_mortality_0__.csv.gz"
    if not cohort.is_file():
        cohort = clean / "cohort" / "cohort_icu_mortality_0__.csv"
    if not cohort.is_file():
        raise ValueError("clean package cohort file is missing")
    frame = pd.read_csv(cohort, dtype={"stay_id": "string"}, usecols=["stay_id"])
    stays = sorted(set(frame["stay_id"].dropna().astype(str)), key=_stable_key)
    if len(stays) <= 1:
        raise ValueError("clean cohort must contain at least two stay_id values")
    return stays


def _select_splits(
    all_stays: list[str],
    modifications: list[dict[str, str]],
    *,
    train_count: int,
    validation_count: int,
    seed: int,
) -> dict[str, set[str]]:
    if train_count + validation_count >= len(all_stays):
        raise ValueError("train_count + validation_count must leave at least one Test stay")
    by_stay: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in modifications:
        by_stay[row["canonical_stay_id"]].append(row)
    train = _coverage_select(by_stay, all_stays, train_count, seed=seed)
    remaining = [stay for stay in all_stays if stay not in train]
    validation = _coverage_select(by_stay, remaining, validation_count, seed=seed + 1)
    return {"train": train, "validation": validation, "test": set(all_stays) - train - validation}


def _coverage_select(
    by_stay: dict[str, list[dict[str, str]]],
    candidates: list[str],
    count: int,
    *,
    seed: int,
) -> set[str]:
    available = set(candidates)
    selected: list[str] = []
    classes: set[str] = set()
    subtypes: set[str] = set()
    while available and len(selected) < count:
        ranked = sorted(
            available,
            key=lambda stay: (
                -len({row.get("error_subtype", "") for row in by_stay.get(stay, [])} - subtypes),
                -len({row.get("error_class", "") for row in by_stay.get(stay, [])} - classes),
                -len(by_stay.get(stay, [])),
                _stable_key(f"{seed}:{stay}"),
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        available.remove(chosen)
        classes.update(row.get("error_class", "") for row in by_stay.get(chosen, []))
        subtypes.update(row.get("error_subtype", "") for row in by_stay.get(chosen, []))
    return set(selected)


def _materialize_clean_reference(clean: Path, destination: Path, stays: set[str]) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    for relative in REFERENCE_RELATIVES:
        source_relative = relative
        if relative == "csv/labels.csv":
            source_relative = "labels.csv"
        source = clean / source_relative
        if not source.is_file():
            source = source.with_suffix(source.suffix + ".gz")
        if not source.is_file():
            raise ValueError(f"clean reference source is missing: {source_relative}")
        target = destination / relative
        header = _read_header(source)
        if "stay_id" in header:
            count = _filter_csv(source, target, column="stay_id", allowed=stays)
        else:
            _copy_as_plain_csv(source, target)
            count = _count_rows(target)
        copied.append({"relative": relative, "row_count": count})
    return {"status": "SUCCESS", "file_count": len(copied), "files": copied}


def _target_for_subtype(subtype: str) -> tuple[str, str]:
    mapping = {
        "schema_range_violation": ("hosp/patients.csv", "cell_update"),
        "enum_violation": ("hosp/patients.csv", "cell_update"),
        "unexpected_missingness": ("hosp/admissions.csv", "cell_update"),
        "temporal_order_violation": ("icu/icustays.csv", "cell_update"),
        "med_end_before_start": ("icu/inputevents.csv", "cell_update"),
        "sentinel_or_negative_med_amount": ("icu/inputevents.csv", "cell_update"),
        "sentinel_chart_value": ("icu/chartevents.csv", "cell_update"),
        "unit_scale_error": ("icu/chartevents.csv", "cell_update"),
        "duplicate_row": ("icu/chartevents.csv", "row_insert"),
        "implausible_negative_event_time": ("icu/outputevents.csv", "cell_update"),
        "chart_itemid_alias_split": ("icu/chartevents.csv", "cell_update"),
        "medication_itemid_alias_split": ("icu/inputevents.csv", "cell_update"),
        "diagnosis_code_alias_split": ("hosp/diagnoses_icd.csv", "cell_update"),
        "orphan_feature_stay_id": ("icu/chartevents.csv", "cell_update"),
        "stay_hadm_mismatch": ("icu/inputevents.csv", "cell_update"),
        "label_conflict_with_cohort": ("hosp/patients.csv", "cell_update"),
        "future_window_leakage": ("icu/chartevents.csv", "cell_update"),
        "label_leakage_feature": ("icu/chartevents.csv", "row_insert"),
    }
    if subtype not in mapping:
        raise ValueError(f"unsupported error subtype: {subtype}")
    return mapping[subtype]


def _inject_raw_errors(raw: Path, stays: set[str], modifications: list[dict[str, str]]) -> list[dict[str, str]]:
    """Inject errors by streaming each raw table, including multi-GB event tables."""
    icu_path = raw / "icu/icustays.csv"
    icu_context = _read_icu_context(icu_path, stays)
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in modifications:
        target_file, _operation = _target_for_subtype(row["error_subtype"])
        by_file[target_file].append(row)
    records: list[dict[str, str]] = []
    for relative, source_rows in by_file.items():
        records.extend(_stream_inject_table(raw / relative, relative, source_rows, icu_context))
    if len(records) != len(modifications):
        raise RuntimeError(f"injection log count mismatch: {len(records)} != {len(modifications)}")
    return records


def _read_icu_context(path: Path, stays: set[str]) -> dict[str, dict[str, str]]:
    context: dict[str, dict[str, str]] = {}
    with _open_csv(path) as handle:
        for row in csv.DictReader(handle):
            stay = str(row.get("stay_id") or "")
            if stay in stays:
                context[stay] = dict(row)
    if set(context) != stays:
        raise ValueError(f"raw icustays does not match selected stays: {sorted(set(context) ^ stays)[:5]}")
    return context


def _stream_inject_table(
    path: Path,
    relative: str,
    source_rows: list[dict[str, str]],
    icu_context: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"raw subset is missing required table: {path}")
    pending: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(source_rows, key=lambda item: int(item["source_error_id"])):
        pending[row["canonical_stay_id"]].append(row)
    subject_to_stay = {str(row["subject_id"]): stay for stay, row in icu_context.items()}
    hadm_to_stay = {str(row["hadm_id"]): stay for stay, row in icu_context.items()}
    all_hadm = sorted(hadm_to_stay)
    records: list[dict[str, str]] = []
    target = path.with_name(f".{path.name}.injecting")
    with _open_csv(path) as reader_handle, target.open("w", encoding="utf-8", newline="") as writer_handle:
        reader = csv.DictReader(reader_handle)
        fields = list(reader.fieldnames or [])
        writer = csv.DictWriter(writer_handle, fieldnames=fields)
        writer.writeheader()
        output_row_index = 0
        for raw_row in reader:
            row = dict(raw_row)
            stay = _row_stay_for_table(row, subject_to_stay, hadm_to_stay)
            applicable = pending.pop(stay, []) if stay else []
            if applicable:
                insertions: list[dict[str, str]] = []
                for source_row in applicable:
                    record, inserted = _apply_stream_corruption(
                        row,
                        output_row_index,
                        relative,
                        source_row,
                        icu_context,
                        all_hadm,
                    )
                    if inserted is not None:
                        record["raw_row_index"] = str(output_row_index + 1 + len(insertions))
                        insertions.append(inserted)
                    records.append(record)
                writer.writerow(row)
                writer.writerows(insertions)
                output_row_index += 1 + len(insertions)
                continue
            writer.writerow(row)
            output_row_index += 1
    unresolved = [row for rows in pending.values() for row in rows]
    if unresolved:
        target.unlink(missing_ok=True)
        missing = ", ".join(f"{row['error_subtype']}:{row['canonical_stay_id']}" for row in unresolved[:5])
        raise ValueError(f"no raw target row in {relative} for {missing}")
    os.replace(target, path)
    return records


def _row_stay_for_table(
    row: dict[str, str],
    subject_to_stay: dict[str, str],
    hadm_to_stay: dict[str, str],
) -> str:
    if row.get("stay_id"):
        return str(row["stay_id"])
    if row.get("hadm_id"):
        return hadm_to_stay.get(str(row["hadm_id"]), "")
    return subject_to_stay.get(str(row.get("subject_id") or ""), "")


def _apply_stream_corruption(
    row: dict[str, str],
    row_index: int,
    target_file: str,
    source_row: dict[str, str],
    icu_context: dict[str, dict[str, str]],
    all_hadm: list[str],
) -> tuple[dict[str, str], dict[str, str] | None]:
    subtype = source_row["error_subtype"]
    column = ""
    new_value = ""
    effect = ""
    inserted: dict[str, str] | None = None
    if subtype == "schema_range_violation":
        column, new_value, effect = "anchor_age", "999", "invalid demographic range"
    elif subtype == "enum_violation":
        column, new_value, effect = "gender", "UNKNOWN_GENDER_CODE", "invalid demographic enum"
    elif subtype == "unexpected_missingness":
        column, new_value, effect = "insurance", "", "required admission attribute missing"
    elif subtype == "temporal_order_violation":
        column, new_value, effect = "outtime", _shift_time(row["intime"], hours=-1), "ICU outtime before intime"
    elif subtype == "med_end_before_start":
        column, new_value, effect = "endtime", row["starttime"], "medication endtime not after starttime"
    elif subtype == "sentinel_or_negative_med_amount":
        column, new_value, effect = "amount", "-999", "invalid negative medication amount"
    elif subtype == "sentinel_chart_value":
        column, new_value, effect = "valuenum", "-999", "sentinel chart value"
    elif subtype == "unit_scale_error":
        column, new_value, effect = "valuenum", _scaled_value(row.get("valuenum")), "chart value wrong scale"
    elif subtype == "duplicate_row":
        inserted = dict(row)
        return _record(source_row, target_file, row_index, "row_insert", "", "", json.dumps(inserted, sort_keys=True), "duplicate raw event"), inserted
    elif subtype == "implausible_negative_event_time":
        intime = icu_context[source_row["canonical_stay_id"]]["intime"]
        column, new_value, effect = "charttime", _shift_time(intime, hours=-1), "output event before ICU admission"
    elif subtype == "chart_itemid_alias_split":
        column, new_value, effect = "itemid", "990045", "noncanonical chart item alias"
    elif subtype == "medication_itemid_alias_split":
        column, new_value, effect = "itemid", "995158", "noncanonical medication item alias"
    elif subtype == "diagnosis_code_alias_split":
        column, new_value, effect = "icd_code", f"ALIAS_{row['icd_code']}", "noncanonical diagnosis alias"
    elif subtype == "orphan_feature_stay_id":
        column, new_value, effect = "stay_id", f"9{row['stay_id']}", "event references noncohort stay"
    elif subtype == "stay_hadm_mismatch":
        alternatives = [hadm for hadm in all_hadm if hadm != str(row["hadm_id"])]
        if not alternatives:
            raise ValueError("stay_hadm_mismatch requires at least two admissions in split")
        column, new_value, effect = "hadm_id", alternatives[0], "event hadm does not match stay"
    elif subtype == "label_conflict_with_cohort":
        column, new_value, effect = "dod", "2100-01-01", "patient death date conflicts with clean mortality label"
    elif subtype == "future_window_leakage":
        outtime = icu_context[source_row["canonical_stay_id"]]["outtime"]
        column, new_value, effect = "charttime", _shift_time(outtime, hours=1), "event occurs after ICU discharge"
    elif subtype == "label_leakage_feature":
        inserted = dict(row)
        inserted["itemid"] = "999001"
        inserted["value"] = source_row.get("dirty_value") or "label-derived"
        inserted["valuenum"] = source_row.get("dirty_value") or "1"
        return _record(source_row, target_file, row_index, "row_insert", "", "", json.dumps(inserted, sort_keys=True), "raw event encodes downstream label"), inserted
    else:  # defensive: _target_for_subtype has already validated this.
        raise ValueError(f"unsupported error subtype: {subtype}")
    old_value = str(row.get(column) or "")
    row[column] = str(new_value)
    return _record(source_row, target_file, row_index, "cell_update", column, old_value, str(new_value), effect), None



def _record(
    source_row: dict[str, str],
    raw_file: str,
    raw_row_index: int,
    operation: str,
    column: str,
    clean_value: str,
    dirty_value: str,
    effect: str,
) -> dict[str, str]:
    return {
        "source_error_id": source_row["source_error_id"],
        "canonical_stay_id": source_row["canonical_stay_id"],
        "error_class": source_row.get("error_class", ""),
        "error_subtype": source_row.get("error_subtype", ""),
        "target_effect_contract": effect,
        "operation": operation,
        "raw_file": raw_file,
        "raw_row_index": str(raw_row_index),
        "column": column,
        "clean_value": clean_value,
        "dirty_value": dirty_value,
    }


def _validate_prepared_split(
    prepared: Path,
    *,
    assignments: dict[str, set[str]],
    injection_records: dict[str, list[dict[str, str]]],
    source_fingerprint_before: dict[str, Any],
    source_fingerprint_after: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed, "expected": expected})

    key_sets = {name: _read_keys(prepared / name / "keys.csv") for name in SPLITS}
    add("key_counts", {name: len(key_sets[name]) for name in SPLITS} == {name: len(assignments[name]) for name in SPLITS}, {name: len(key_sets[name]) for name in SPLITS}, {name: len(assignments[name]) for name in SPLITS})
    overlap = sum(len(key_sets[left] & key_sets[right]) for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")))
    add("key_disjointness", overlap == 0, overlap, 0)
    add("key_full_coverage", set.union(*key_sets.values()) == set.union(*assignments.values()), len(set.union(*key_sets.values())), len(set.union(*assignments.values())))
    for split in SPLITS:
        icu = pd.read_csv(prepared / split / "raw/icu/icustays.csv", dtype=object)
        raw_stays = set(icu["stay_id"].dropna().astype(str))
        add(f"{split}_raw_icustay_coverage", raw_stays == key_sets[split], len(raw_stays), len(key_sets[split]))
        reference_name = "reference" if split == "train" else "reference_private"
        reference_files = [path for path in (prepared / split / reference_name).rglob("*") if path.is_file()]
        add(f"{split}_standard_reference_file_count", len(reference_files) == len(REFERENCE_RELATIVES), len(reference_files), len(REFERENCE_RELATIVES))
        add(f"{split}_raw_structure", (prepared / split / "raw/hosp").is_dir() and (prepared / split / "raw/icu").is_dir(), [str(path.relative_to(prepared)) for path in (prepared / split / "raw").iterdir()], ["hosp", "icu"])
        actual_raw_files = {
            path.relative_to(prepared / split / "raw").as_posix()
            for path in (prepared / split / "raw").rglob("*")
            if path.is_file()
        }
        add(
            f"{split}_raw_template",
            actual_raw_files == RAW_TEMPLATE_OUTPUT_RELATIVES,
            sorted(actual_raw_files),
            sorted(RAW_TEMPLATE_OUTPUT_RELATIVES),
        )
        add(f"{split}_injection_count", len(injection_records[split]) > 0, len(injection_records[split]), "> 0")
    links = [path.relative_to(prepared).as_posix() for path in prepared.rglob("*") if path.is_symlink()]
    add("physical_copy", not links, links, [])
    add("source_raw_unchanged", source_fingerprint_before == source_fingerprint_after, source_fingerprint_after, source_fingerprint_before)
    host_files = {path.name for path in (prepared / "host_private").iterdir()}
    add("private_injection_logs", {f"{split}_raw_modification_log.csv" for split in SPLITS}.issubset(host_files), sorted(host_files), "all split raw modification logs")
    return {"schema_version": 1, "status": "SUCCESS" if all(item["passed"] for item in checks) else "FAILED", "passed": all(item["passed"] for item in checks), "checks": checks}


def _rows_at_indices(path: Path, indices: set[int]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with _open_csv(path) as text:
        for index, row in enumerate(csv.DictReader(text)):
            if index in indices:
                result[index] = dict(row)
                if len(result) == len(indices):
                    break
    missing = indices - set(result)
    if missing:
        raise ValueError(f"clean package rows missing from {path}: {sorted(missing)[:5]}")
    return result


@contextmanager
def _open_csv(path: Path) -> Iterator[TextIO]:
    handle: TextIO = gzip.open(path, "rt", encoding="utf-8-sig", newline="") if path.name.endswith(".gz") else path.open(encoding="utf-8-sig", newline="")
    try:
        yield handle
    finally:
        handle.close()


def _read_header(path: Path) -> list[str]:
    with _open_csv(path) as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _filter_csv(source: Path, destination: Path, *, column: str, allowed: set[str]) -> int:
    with _open_csv(source) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        selected = [row for row in reader if str(row.get(column) or "") in allowed]
    _write_csv(destination, fields, selected)
    return len(selected)


def _copy_as_plain_csv(source: Path, destination: Path) -> None:
    with _open_csv(source) as handle:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(handle.read(), encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_keys(path: Path, stays: set[str]) -> None:
    _write_csv(path, ["stay_id"], [{"stay_id": stay} for stay in sorted(stays, key=_stable_key)])


def _read_keys(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {str(row["stay_id"]) for row in csv.DictReader(handle)}


def _shift_time(value: Any, *, hours: int) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"cannot corrupt invalid datetime value: {value}")
    return (parsed + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def _scaled_value(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "999999"
    return str(float(number) * 100.0)


def _optional_index(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(str(value)))


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _stable_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def directory_metadata_fingerprint(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    digest = hashlib.sha256()
    for path in sorted(files):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return {"path": str(root), "file_count": len(files), "total_size": sum(path.stat().st_size for path in files), "digest": digest.hexdigest()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a physical raw-MIMIC error-injection split.")
    parser.add_argument("--source-raw-3-1", required=True)
    parser.add_argument("--clean-package", required=True)
    parser.add_argument("--source-modification-log", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=20260804)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_raw_level_error_split(
        args.source_raw_3_1,
        args.clean_package,
        args.source_modification_log,
        args.output_root,
        train_count=args.train_count,
        validation_count=args.validation_count,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
