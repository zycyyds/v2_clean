from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from reproduction.mimic_icu_mortality import (
    FEATURE_FILES,
    GOLD_RELATIVE_PATHS,
    SUMMARY_FILES,
    ReproductionConfig,
    RunLayout,
    config_fingerprint,
    directory_metadata_fingerprint,
    file_fingerprint,
)


COHORT_COLUMNS = (
    "subject_id",
    "stay_id",
    "intime",
    "outtime",
    "Age",
    "gender",
    "ethnicity",
    "insurance",
    "label",
    "dod",
    "hadm_id",
)
DIAGNOSIS_COLUMNS = ("subject_id", "hadm_id", "stay_id", "new_icd_code")


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any,
    expected: Any,
    details: Any = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )


def _csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _csv_row_count(path: Path, columns: list[str], chunksize: int = 500_000) -> int:
    return sum(len(chunk) for chunk in pd.read_csv(path, usecols=columns[:1], chunksize=chunksize))


def _csv_unique_values(path: Path, column: str, chunksize: int = 500_000) -> set[Any]:
    values: set[Any] = set()
    for chunk in pd.read_csv(path, usecols=[column], chunksize=chunksize):
        values.update(chunk[column].dropna().tolist())
    return values


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    columns = _csv_columns(path)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "row_count": _csv_row_count(path, columns),
        "columns": columns,
    }


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.is_file() and path.stat().st_size > 0]


def validate_reproduction(
    config: ReproductionConfig,
    layout: RunLayout,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    cohort_exists = layout.cohort_path.is_file() and layout.cohort_path.stat().st_size > 0
    _check(checks, "cohort_exists", cohort_exists, observed=cohort_exists, expected=True)

    cohort = pd.DataFrame()
    if cohort_exists:
        cohort = pd.read_csv(layout.cohort_path)
        cohort_columns = list(cohort.columns)
        _check(
            checks,
            "cohort_schema",
            cohort_columns == list(COHORT_COLUMNS),
            observed=cohort_columns,
            expected=list(COHORT_COLUMNS),
        )
        adult_ok = "Age" in cohort and bool((pd.to_numeric(cohort["Age"], errors="coerce") >= 18).all())
        _check(checks, "cohort_adults", adult_ok, observed=int((~(pd.to_numeric(cohort.get("Age"), errors="coerce") >= 18)).sum()) if "Age" in cohort else None, expected=0)
        labels = set(pd.to_numeric(cohort.get("label"), errors="coerce").dropna().unique()) if "label" in cohort else set()
        _check(checks, "cohort_label_domain", labels <= {0, 1}, observed=sorted(labels), expected=[0, 1])
        duplicate_stays = int(cohort["stay_id"].duplicated().sum()) if "stay_id" in cohort else len(cohort)
        missing_stays = int(cohort["stay_id"].isna().sum()) if "stay_id" in cohort else len(cohort)
        _check(
            checks,
            "cohort_stay_keys",
            duplicate_stays == 0 and missing_stays == 0,
            observed={"duplicates": duplicate_stays, "missing": missing_stays},
            expected={"duplicates": 0, "missing": 0},
        )
        mortality_mismatches = None
        if {"dod", "intime", "outtime", "label"}.issubset(cohort.columns):
            dod = pd.to_datetime(cohort["dod"], errors="coerce")
            intime = pd.to_datetime(cohort["intime"], errors="coerce")
            outtime = pd.to_datetime(cohort["outtime"], errors="coerce")
            expected_label = (dod.notna() & (dod >= intime) & (dod <= outtime)).astype(int)
            mortality_mismatches = int((expected_label != pd.to_numeric(cohort["label"], errors="coerce")).sum())
        _check(
            checks,
            "cohort_mortality_rule",
            mortality_mismatches == 0,
            observed=mortality_mismatches,
            expected=0,
        )

    feature_paths = [layout.features_dir / name for name in FEATURE_FILES]
    existing_features = _existing(feature_paths)
    _check(
        checks,
        "five_feature_files",
        len(existing_features) == len(feature_paths),
        observed=[path.name for path in existing_features],
        expected=list(FEATURE_FILES),
    )

    diagnosis_path = layout.features_dir / "preproc_diag_icu.csv.gz"
    diagnosis_columns = _csv_columns(diagnosis_path) if diagnosis_path in existing_features else []
    _check(
        checks,
        "diagnosis_schema",
        diagnosis_columns == list(DIAGNOSIS_COLUMNS),
        observed=diagnosis_columns,
        expected=list(DIAGNOSIS_COLUMNS),
    )

    cohort_stays = set(cohort["stay_id"].dropna().tolist()) if "stay_id" in cohort else set()
    feature_extras: dict[str, int | None] = {}
    for path in existing_features:
        columns = _csv_columns(path)
        if "stay_id" not in columns:
            feature_extras[path.name] = None
            continue
        feature_extras[path.name] = len(_csv_unique_values(path, "stay_id") - cohort_stays)
    _check(
        checks,
        "feature_stays_in_cohort",
        len(feature_extras) == 5 and all(value == 0 for value in feature_extras.values()),
        observed=feature_extras,
        expected={name: 0 for name in FEATURE_FILES},
    )

    labels_exists = layout.labels_path.is_file() and layout.labels_path.stat().st_size > 0
    _check(checks, "labels_exists", labels_exists, observed=labels_exists, expected=True)
    labels_frame = pd.read_csv(layout.labels_path) if labels_exists else pd.DataFrame(columns=["stay_id", "label"])
    label_columns = list(labels_frame.columns)
    _check(
        checks,
        "labels_schema",
        label_columns == ["stay_id", "label"],
        observed=label_columns,
        expected=["stay_id", "label"],
    )
    label_stays = set(labels_frame["stay_id"].dropna().tolist()) if "stay_id" in labels_frame else set()
    label_extras = len(label_stays - cohort_stays)
    label_values = set(pd.to_numeric(labels_frame.get("label"), errors="coerce").dropna().unique())
    _check(
        checks,
        "labels_keys_and_domain",
        label_extras == 0 and label_values <= {0, 1},
        observed={"outside_cohort": label_extras, "values": sorted(label_values)},
        expected={"outside_cohort": 0, "values": [0, 1]},
    )

    short_stays: list[Any] = []
    if label_stays and {"stay_id", "intime", "outtime"}.issubset(cohort.columns):
        selected = cohort[cohort["stay_id"].isin(label_stays)].copy()
        duration = pd.to_datetime(selected["outtime"], errors="coerce") - pd.to_datetime(selected["intime"], errors="coerce")
        integer_hours = duration.dt.days * 24 + duration.dt.components["hours"]
        invalid_duration = duration.isna()
        short_stays = selected.loc[
            invalid_duration | (integer_hours < config.include_hours + config.prediction_hours), "stay_id"
        ].tolist()
    _check(
        checks,
        "labels_observation_window",
        len(short_stays) == 0 and len(label_stays) == len(labels_frame),
        observed={"short_stay_count": len(short_stays), "duplicate_label_rows": len(labels_frame) - len(label_stays)},
        expected={"short_stay_count": 0, "duplicate_label_rows": 0},
        details=short_stays[:20],
    )

    actual_stay_dirs = {path.name for path in layout.csv_dir.iterdir() if path.is_dir()} if layout.csv_dir.is_dir() else set()
    expected_stay_dirs = {str(int(value)) if isinstance(value, float) and value.is_integer() else str(value) for value in label_stays}
    missing_files: list[str] = []
    empty_files: list[str] = []
    for stay in sorted(expected_stay_dirs):
        for name in ("demo.csv", "static.csv", "dynamic.csv"):
            path = layout.csv_dir / stay / name
            if not path.is_file():
                missing_files.append(f"{stay}/{name}")
            elif path.stat().st_size == 0:
                empty_files.append(f"{stay}/{name}")
    extra_dirs = sorted(actual_stay_dirs - expected_stay_dirs)
    missing_dirs = sorted(expected_stay_dirs - actual_stay_dirs)
    _check(
        checks,
        "stay_csv_completeness",
        not missing_files and not empty_files and not extra_dirs and not missing_dirs,
        observed={
            "expected_stays": len(expected_stay_dirs),
            "actual_stays": len(actual_stay_dirs),
            "missing_files": len(missing_files),
            "empty_files": len(empty_files),
            "extra_dirs": len(extra_dirs),
            "missing_dirs": len(missing_dirs),
        },
        expected={"missing_files": 0, "empty_files": 0, "extra_dirs": 0, "missing_dirs": 0},
        details={
            "missing_files_sample": missing_files[:20],
            "empty_files_sample": empty_files[:20],
            "extra_dirs_sample": extra_dirs[:20],
            "missing_dirs_sample": missing_dirs[:20],
        },
    )

    dict_paths = [layout.dict_dir / name for name in layout.expected_dict_names()]
    existing_dicts = _existing(dict_paths)
    _check(
        checks,
        "dictionary_inventory",
        len(existing_dicts) == len(dict_paths),
        observed=[path.name for path in existing_dicts],
        expected=list(layout.expected_dict_names()),
    )

    summary_paths = [layout.summary_dir / name for name in SUMMARY_FILES]
    existing_summaries = _existing(summary_paths)
    actual_summary_csvs = sorted(path.name for path in layout.summary_dir.glob("*.csv")) if layout.summary_dir.is_dir() else []
    _check(
        checks,
        "summary_inventory",
        len(existing_summaries) == 10 and actual_summary_csvs == sorted(SUMMARY_FILES),
        observed=actual_summary_csvs,
        expected=sorted(SUMMARY_FILES),
    )

    forbidden = [
        path.relative_to(layout.root).as_posix()
        for path in layout.root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".onnx"}
    ]
    _check(checks, "no_model_artifacts", not forbidden, observed=forbidden, expected=[])

    source_snapshot_matches = False
    source_snapshot_observed: dict[str, Any] | None = None
    if layout.source_snapshot_path.is_file():
        source_snapshot_expected = json.loads(layout.source_snapshot_path.read_text(encoding="utf-8"))
        source_snapshot_observed = directory_metadata_fingerprint(config.source_root / "data")
        source_snapshot_matches = source_snapshot_expected == source_snapshot_observed
    else:
        source_snapshot_expected = None
    _check(
        checks,
        "original_data_untouched",
        source_snapshot_matches,
        observed=source_snapshot_observed,
        expected=source_snapshot_expected,
    )

    gold_paths = [layout.root / relative for relative in GOLD_RELATIVE_PATHS]
    existing_gold = _existing(gold_paths)
    _check(
        checks,
        "gold_candidate_inventory",
        len(existing_gold) == 17,
        observed=[path.relative_to(layout.root).as_posix() for path in existing_gold],
        expected=list(GOLD_RELATIVE_PATHS),
    )

    passed = all(item["passed"] for item in checks)
    report = {
        "schema_version": 1,
        "status": "success" if passed else "failed_validation",
        "passed": passed,
        "config_fingerprint": config_fingerprint(config),
        "checks": checks,
    }
    record_cache: dict[Path, dict[str, Any]] = {}

    def record(path: Path) -> dict[str, Any]:
        if path not in record_cache:
            record_cache[path] = _file_record(layout.root, path)
        return record_cache[path]

    state = {}
    if layout.state_path.is_file():
        state = json.loads(layout.state_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "status": report["status"],
        "config": config.as_serializable(),
        "source_inputs": {
            "raw_files": [file_fingerprint(config.raw_root / relative) for relative in config.required_raw_files()],
            "mapping": file_fingerprint(config.mapping_path),
            "original_data_snapshot": source_snapshot_observed,
        },
        "stage_state": state.get("stages", {}),
        "gold_candidates": [record(path) for path in existing_gold],
        "feature_files": [record(path) for path in existing_features],
        "summary_files": [record(path) for path in existing_summaries],
        "stay_directory_count": len(actual_stay_dirs),
        "dictionary_files": [
            {
                "relative_path": path.relative_to(layout.root).as_posix(),
                "size_bytes": path.stat().st_size,
            }
            for path in existing_dicts
        ],
    }
    return report, manifest
