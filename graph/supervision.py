from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .builder import build_graph
from .labels import build_labels


LOG_FIELDS = (
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
OUTPUT_LOG_FIELDS = LOG_FIELDS + (
    "source_error_ids",
    "source_error_count",
    "injection_seed",
)
ERROR_CLASSES = (
    "class1_intra_table",
    "class2_entity_alignment",
    "class3_cross_table",
    "class4_task_oriented",
)


class SupervisionBuildError(ValueError):
    pass


@dataclass(frozen=True)
class ErrorSpec:
    subtype: str
    error_class: str
    raw_file: str


CELL_ERROR_SPECS = (
    ErrorSpec("schema_range_violation", "class1_intra_table", "hosp/patients.csv"),
    ErrorSpec("enum_violation", "class1_intra_table", "hosp/patients.csv"),
    ErrorSpec("unexpected_missingness", "class1_intra_table", "hosp/admissions.csv"),
    ErrorSpec("temporal_order_violation", "class1_intra_table", "icu/icustays.csv"),
    ErrorSpec("med_end_before_start", "class1_intra_table", "icu/inputevents.csv"),
    ErrorSpec("sentinel_or_negative_med_amount", "class1_intra_table", "icu/inputevents.csv"),
    ErrorSpec("sentinel_chart_value", "class1_intra_table", "icu/chartevents.csv"),
    ErrorSpec("unit_scale_error", "class1_intra_table", "icu/chartevents.csv"),
    ErrorSpec("implausible_negative_event_time", "class1_intra_table", "icu/outputevents.csv"),
    ErrorSpec("chart_itemid_alias_split", "class2_entity_alignment", "icu/chartevents.csv"),
    ErrorSpec("medication_itemid_alias_split", "class2_entity_alignment", "icu/inputevents.csv"),
    ErrorSpec("diagnosis_code_alias_split", "class2_entity_alignment", "hosp/diagnoses_icd.csv"),
    ErrorSpec("orphan_feature_stay_id", "class3_cross_table", "icu/chartevents.csv"),
    ErrorSpec("stay_hadm_mismatch", "class3_cross_table", "icu/inputevents.csv"),
    ErrorSpec("label_conflict_with_cohort", "class3_cross_table", "hosp/patients.csv"),
    ErrorSpec("future_window_leakage", "class4_task_oriented", "icu/chartevents.csv"),
)


def _csv_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _csv_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _ensure_new_or_empty(path: Path, label: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SupervisionBuildError(f"{label} must be new or empty: {path}")


def _read_log(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SupervisionBuildError(f"missing base injection log: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SupervisionBuildError("base injection log is empty")
    missing = sorted(set(LOG_FIELDS) - set(rows[0]))
    if missing:
        raise SupervisionBuildError(f"base injection log is missing fields: {missing}")
    return [{field: str(row.get(field) or "") for field in LOG_FIELDS} for row in rows]


def _read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _row_count(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _normalize_base_cell_records(
    base_raw: Path,
    clean_raw: Path,
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], set[tuple[str, int]], set[tuple[str, int, str]], dict[str, int], dict[str, int]]:
    """Map original dirty-row coordinates back onto the reconstructed clean raw.

    The source benchmark includes two inserted-row errors. They are excluded
    from Cell supervision, so every later source row in the same table shifts
    left by the number of preceding insertions.
    """
    inserted_by_file: dict[str, list[int]] = defaultdict(list)
    for record in rows:
        error_class = record["error_class"]
        if error_class not in ERROR_CLASSES:
            raise SupervisionBuildError(f"unsupported error class in base log: {error_class}")
        try:
            row_index = int(record["raw_row_index"])
        except ValueError as exc:
            raise SupervisionBuildError(f"invalid raw_row_index in base log: {record['raw_row_index']}") from exc
        if row_index < 0:
            raise SupervisionBuildError(f"negative raw_row_index in base log: {row_index}")
        if record["operation"] == "row_insert":
            inserted_by_file[record["raw_file"]].append(row_index)
        elif record["operation"] != "cell_update":
            raise SupervisionBuildError(f"unsupported base operation: {record['operation']}")

    for raw_file, indices in inserted_by_file.items():
        if len(indices) != len(set(indices)):
            raise SupervisionBuildError(f"duplicate row_insert coordinate in base log: {raw_file}")
        indices.sort()
        dirty_path = base_raw / raw_file
        clean_path = clean_raw / raw_file
        if not dirty_path.is_file() or not clean_path.is_file():
            raise SupervisionBuildError(f"row_insert table is missing from dirty or clean raw: {raw_file}")
        if _row_count(dirty_path) - _row_count(clean_path) != len(indices):
            raise SupervisionBuildError(f"row_insert count does not match reconstructed clean table: {raw_file}")

    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for record in rows:
        if record["operation"] == "row_insert":
            continue
        raw_file = record["raw_file"]
        source_index = int(record["raw_row_index"])
        insertions = inserted_by_file.get(raw_file, [])
        if source_index in insertions:
            raise SupervisionBuildError(f"cell update targets an inserted row: {raw_file}:{source_index}")
        clean_index = source_index - sum(index < source_index for index in insertions)
        column = record["column"]
        coordinate = (raw_file, clean_index, column)
        mapped = dict(record)
        mapped["raw_row_index"] = str(clean_index)
        mapped["operation"] = "cell_update"
        grouped[coordinate].append(mapped)

    by_file: dict[str, dict[int, list[tuple[str, list[dict[str, str]]]]]] = defaultdict(lambda: defaultdict(list))
    for (raw_file, row_index, column), records in grouped.items():
        by_file[raw_file][row_index].append((column, records))
    normalized: list[dict[str, str]] = []
    for raw_file, expected_rows in by_file.items():
        path = clean_raw / raw_file
        if not path.is_file():
            raise SupervisionBuildError(f"base Cell table is missing from clean raw: {raw_file}")
        seen_rows: set[int] = set()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                cell_chains = expected_rows.get(row_index, [])
                for column, records in cell_chains:
                    if column not in row:
                        raise SupervisionBuildError(f"base Cell column is missing: {raw_file}:{column}")
                    original = str(row.get(column) or "")
                    current = original
                    effective = records[0]
                    source_ids: list[str] = []
                    for record in records:
                        if record["clean_value"] != current:
                            raise SupervisionBuildError(
                                f"base injection chain mismatch at {raw_file}:{row_index}:{column}: "
                                f"{record['clean_value']!r} != {current!r}"
                            )
                        source_ids.append(record["source_error_id"])
                        if record["dirty_value"] != current:
                            effective = record
                        current = record["dirty_value"]
                    if current == original:
                        raise SupervisionBuildError(f"base injection chain has no final change: {raw_file}:{row_index}:{column}")
                    collapsed = dict(effective)
                    collapsed["clean_value"] = original
                    collapsed["dirty_value"] = current
                    collapsed["source_error_ids"] = ",".join(source_ids)
                    collapsed["source_error_count"] = str(len(source_ids))
                    normalized.append(collapsed)
                if cell_chains:
                    seen_rows.add(row_index)
        missing = set(expected_rows) - seen_rows
        if missing:
            raise SupervisionBuildError(f"base Cell rows are missing from clean raw {raw_file}: {sorted(missing)[:5]}")

    dirty_cells = {
        (record["raw_file"], int(record["raw_row_index"]), record["column"])
        for record in normalized
    }
    dirty_rows = {coordinate[:2] for coordinate in dirty_cells}
    counts = Counter(record["error_class"] for record in normalized)
    excluded_row_inserts = sum(len(indices) for indices in inserted_by_file.values())
    source_cell_records = sum(record["operation"] == "cell_update" for record in rows)
    stats = {
        "source_cell_record_count": source_cell_records,
        "unique_base_cell_count": len(normalized),
        "collapsed_cell_record_count": source_cell_records - len(normalized),
        "excluded_row_insert_count": excluded_row_inserts,
    }
    return normalized, dirty_rows, dirty_cells, dict(counts), stats


def _read_context(clean_raw: Path) -> dict[str, Any]:
    path = clean_raw / "icu/icustays.csv"
    if not path.is_file():
        raise SupervisionBuildError(f"missing clean ICU context: {path}")
    stays: dict[str, dict[str, str]] = {}
    hadm_to_stay: dict[str, str] = {}
    subject_to_stays: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stay = str(row.get("stay_id") or "").strip()
            subject = str(row.get("subject_id") or "").strip()
            hadm = str(row.get("hadm_id") or "").strip()
            if not stay or not subject or not hadm:
                raise SupervisionBuildError("clean icustays rows require subject_id, hadm_id and stay_id")
            stays[stay] = {str(key): str(value or "") for key, value in row.items()}
            hadm_to_stay[hadm] = stay
            subject_to_stays[subject].append(stay)
    return {
        "stays": stays,
        "hadm_to_stay": hadm_to_stay,
        "subject_to_stays": {subject: sorted(values) for subject, values in subject_to_stays.items()},
        "all_hadm": sorted(hadm_to_stay),
    }


def _resolve_stay(row: dict[str, str], context: dict[str, Any]) -> str:
    stay = str(row.get("stay_id") or "").strip()
    if stay in context["stays"]:
        return stay
    hadm = str(row.get("hadm_id") or "").strip()
    if hadm in context["hadm_to_stay"]:
        return str(context["hadm_to_stay"][hadm])
    subject = str(row.get("subject_id") or "").strip()
    matches = context["subject_to_stays"].get(subject, [])
    return str(matches[0]) if len(matches) == 1 else ""


def _shift_time(value: str, hours: int) -> str | None:
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None
    return (parsed + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def _scaled_value(value: str) -> str | None:
    try:
        return str(float(value) * 100.0)
    except (TypeError, ValueError):
        return None


def _make_candidate(
    spec: ErrorSpec,
    row: dict[str, str],
    row_index: int,
    context: dict[str, Any],
) -> dict[str, str] | None:
    subtype = spec.subtype
    stay = _resolve_stay(row, context)
    column = ""
    new_value: str | None = None
    effect = ""
    if subtype == "schema_range_violation":
        column, new_value, effect = "anchor_age", "999", "invalid demographic range"
    elif subtype == "enum_violation":
        column, new_value, effect = "gender", "UNKNOWN_GENDER_CODE", "invalid demographic enum"
    elif subtype == "unexpected_missingness":
        column, new_value, effect = "insurance", "", "required admission attribute missing"
    elif subtype == "temporal_order_violation" and stay:
        column = "outtime"
        new_value = _shift_time(context["stays"][stay].get("intime", ""), -1)
        effect = "ICU outtime before intime"
    elif subtype == "med_end_before_start":
        column, new_value, effect = "endtime", str(row.get("starttime") or ""), "medication endtime not after starttime"
    elif subtype == "sentinel_or_negative_med_amount":
        column, new_value, effect = "amount", "-999", "invalid negative medication amount"
    elif subtype == "sentinel_chart_value":
        column, new_value, effect = "valuenum", "-999", "sentinel chart value"
    elif subtype == "unit_scale_error":
        column = "valuenum"
        new_value = _scaled_value(str(row.get(column) or ""))
        effect = "chart value wrong scale"
    elif subtype == "implausible_negative_event_time" and stay:
        column = "charttime"
        new_value = _shift_time(context["stays"][stay].get("intime", ""), -1)
        effect = "output event before ICU admission"
    elif subtype == "chart_itemid_alias_split":
        column, new_value, effect = "itemid", "990045", "noncanonical chart item alias"
    elif subtype == "medication_itemid_alias_split":
        column, new_value, effect = "itemid", "995158", "noncanonical medication item alias"
    elif subtype == "diagnosis_code_alias_split":
        column = "icd_code"
        value = str(row.get(column) or "")
        new_value = f"ALIAS_{value}" if value else None
        effect = "noncanonical diagnosis alias"
    elif subtype == "orphan_feature_stay_id":
        column = "stay_id"
        value = str(row.get(column) or "")
        new_value = f"9{value}" if value else None
        effect = "event references noncohort stay"
    elif subtype == "stay_hadm_mismatch":
        column = "hadm_id"
        current = str(row.get(column) or "")
        alternatives = [value for value in context["all_hadm"] if value != current]
        new_value = alternatives[0] if alternatives else None
        effect = "event hadm does not match stay"
    elif subtype == "label_conflict_with_cohort":
        column, new_value, effect = "dod", "2100-01-01", "patient death date conflicts with mortality label"
    elif subtype == "future_window_leakage" and stay:
        column = "charttime"
        new_value = _shift_time(context["stays"][stay].get("outtime", ""), 1)
        effect = "event occurs after ICU discharge"
    if not column or column not in row or new_value is None:
        return None
    old_value = str(row.get(column) or "")
    if old_value == str(new_value):
        return None
    if subtype == "unexpected_missingness" and not old_value:
        return None
    if subtype in {"sentinel_or_negative_med_amount", "sentinel_chart_value"} and not old_value:
        return None
    if not stay:
        return None
    return {
        "canonical_stay_id": stay,
        "error_class": spec.error_class,
        "error_subtype": subtype,
        "target_effect_contract": effect,
        "operation": "cell_update",
        "raw_file": spec.raw_file,
        "raw_row_index": str(row_index),
        "column": column,
        "clean_value": old_value,
        "dirty_value": str(new_value),
    }


def _priority(seed: int, candidate: dict[str, str]) -> int:
    text = "|".join((
        str(seed),
        candidate["error_subtype"],
        candidate["raw_file"],
        candidate["raw_row_index"],
        candidate["column"],
        candidate["clean_value"],
        candidate["dirty_value"],
    ))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _push_candidate(heap: list[tuple[int, str, dict[str, str]]], limit: int, priority: int, candidate: dict[str, str]) -> None:
    tie = f"{candidate['raw_file']}:{int(candidate['raw_row_index']):012d}:{candidate['column']}"
    item = (-priority, tie, candidate)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _collect_candidates(
    base_raw: Path,
    context: dict[str, Any],
    base_dirty_rows: set[tuple[str, int]],
    seeds: tuple[int, ...],
    pool_limit: int,
) -> dict[tuple[int, str], list[dict[str, str]]]:
    specs_by_file: dict[str, list[ErrorSpec]] = defaultdict(list)
    for spec in CELL_ERROR_SPECS:
        specs_by_file[spec.raw_file].append(spec)
    heaps: dict[tuple[int, str], list[tuple[int, str, dict[str, str]]]] = defaultdict(list)
    for relative, specs in specs_by_file.items():
        path = base_raw / relative
        if not path.is_file():
            raise SupervisionBuildError(f"missing injection target table: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_index, raw_row in enumerate(csv.DictReader(handle)):
                if (relative, row_index) in base_dirty_rows:
                    continue
                row = {str(key): str(value or "") for key, value in raw_row.items()}
                for spec in specs:
                    candidate = _make_candidate(spec, row, row_index, context)
                    if candidate is None:
                        continue
                    for seed in seeds:
                        _push_candidate(heaps[(seed, spec.subtype)], pool_limit, _priority(seed, candidate), candidate)
    result: dict[tuple[int, str], list[dict[str, str]]] = {}
    for key, heap in heaps.items():
        ordered = sorted(heap, key=lambda item: (-item[0], item[1]))
        result[key] = [candidate for _, _, candidate in ordered]
    return result


def _class_targets(dirty_target: int) -> dict[str, int]:
    base, remainder = divmod(dirty_target, len(ERROR_CLASSES))
    return {name: base + (1 if index < remainder else 0) for index, name in enumerate(ERROR_CLASSES)}


def _select_synthetic(
    pools: dict[tuple[int, str], list[dict[str, str]]],
    base_dirty_rows: set[tuple[str, int]],
    base_class_counts: dict[str, int],
    dirty_target: int,
    seeds: tuple[int, ...],
) -> list[dict[str, str]]:
    targets = _class_targets(dirty_target)
    needs = {name: max(0, targets[name] - int(base_class_counts.get(name, 0))) for name in ERROR_CLASSES}
    expected = dirty_target - sum(base_class_counts.values())
    shortfall = expected - sum(needs.values())
    for name in ERROR_CLASSES:
        if shortfall <= 0:
            break
        needs[name] += shortfall
        shortfall = 0
    used_rows = set(base_dirty_rows)
    selected: list[dict[str, str]] = []
    subtype_by_class = {
        error_class: sorted(spec.subtype for spec in CELL_ERROR_SPECS if spec.error_class == error_class)
        for error_class in ERROR_CLASSES
    }
    for error_class in ERROR_CLASSES:
        cursors = defaultdict(int)
        remaining = needs[error_class]
        while remaining:
            progressed = False
            for seed in seeds:
                for subtype in subtype_by_class[error_class]:
                    candidates = pools.get((seed, subtype), [])
                    cursor_key = (seed, subtype)
                    while cursors[cursor_key] < len(candidates):
                        candidate = candidates[cursors[cursor_key]]
                        cursors[cursor_key] += 1
                        row_key = (candidate["raw_file"], int(candidate["raw_row_index"]))
                        if row_key in used_rows:
                            continue
                        record = dict(candidate)
                        record["source_error_id"] = f"synthetic:{seed}:{len(selected)}"
                        record["injection_seed"] = str(seed)
                        selected.append(record)
                        used_rows.add(row_key)
                        remaining -= 1
                        progressed = True
                        break
                    if remaining == 0:
                        break
                if remaining == 0:
                    break
            if not progressed:
                capacities = {subtype: sum(len(pools.get((seed, subtype), [])) for seed in seeds) for subtype in subtype_by_class[error_class]}
                raise SupervisionBuildError(
                    f"insufficient unique candidates for {error_class}: remaining={remaining} capacities={capacities}"
                )
    if len(selected) != expected:
        raise SupervisionBuildError(f"synthetic selection count mismatch: {len(selected)} != {expected}")
    return selected


def _write_augmented_raw(
    base_raw: Path,
    output_raw: Path,
    records: list[dict[str, str]],
) -> None:
    by_file: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    coordinates: set[tuple[str, int, str]] = set()
    for record in records:
        row_index = int(record["raw_row_index"])
        coordinate = (record["raw_file"], row_index, record["column"])
        if coordinate in coordinates:
            raise SupervisionBuildError(f"duplicate augmented Cell coordinate: {coordinate}")
        coordinates.add(coordinate)
        by_file[record["raw_file"]][row_index].append(record)
    output_raw.mkdir(parents=True, exist_ok=True)
    for source in _csv_files(base_raw):
        relative = source.relative_to(base_raw).as_posix()
        destination = output_raw / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        modifications = by_file.get(relative)
        if not modifications:
            shutil.copy2(source, destination)
            continue
        with source.open(encoding="utf-8-sig", newline="") as source_handle, destination.open("w", encoding="utf-8", newline="") as destination_handle:
            reader = csv.DictReader(source_handle)
            fields = list(reader.fieldnames or [])
            writer = csv.DictWriter(destination_handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            seen: set[int] = set()
            for row_index, row in enumerate(reader):
                row_records = modifications.get(row_index, [])
                for record in row_records:
                    column = record["column"]
                    observed = str(row.get(column) or "")
                    if observed != record["clean_value"]:
                        raise SupervisionBuildError(
                            f"candidate clean value changed at {relative}:{row_index}:{column}"
                        )
                    row[column] = record["dirty_value"]
                if row_records:
                    seen.add(row_index)
                writer.writerow(row)
            missing = set(modifications) - seen
            if missing:
                raise SupervisionBuildError(f"selected rows disappeared from {relative}: {sorted(missing)[:5]}")


def _write_merged_log(path: Path, base_rows: list[dict[str, str]], synthetic: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_LOG_FIELDS), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(base_rows)
        writer.writerows(synthetic)


def _subject_folds(context: dict[str, Any], seed: int) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    subjects = sorted(
        context["subject_to_stays"],
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    subject_fold = {subject: index % 5 for index, subject in enumerate(subjects)}
    stay_subject: dict[str, str] = {}
    hadm_subject: dict[str, str] = {}
    for subject, stays in context["subject_to_stays"].items():
        for stay in stays:
            stay_subject[stay] = subject
            hadm_subject[context["stays"][stay]["hadm_id"]] = subject
    return subject_fold, stay_subject, hadm_subject


def _observation_subject(values: dict[str, str], stay_subject: dict[str, str], hadm_subject: dict[str, str]) -> str:
    subject = values.get("subject_id", "").strip()
    if subject:
        return subject
    hadm = values.get("hadm_id", "").strip()
    if hadm in hadm_subject:
        return hadm_subject[hadm]
    stay = values.get("stay_id", "").strip()
    return stay_subject.get(stay, "")


def _heap_observation(
    heap: list[tuple[int, int, int]],
    limit: int,
    priority: int,
    observation_index: int,
    fold: int,
) -> None:
    item = (-priority, observation_index, fold)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _build_supervision_masks(
    graph_dir: Path,
    labels_path: Path,
    output_path: Path,
    context: dict[str, Any],
    base_rows: set[tuple[str, int]],
    base_cells: set[tuple[str, int, str]],
    clean_target: int,
    dirty_target: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.load(labels_path)["cell_dirty"]
    dirty_indices_expected = set(np.flatnonzero(labels).tolist())
    if len(dirty_indices_expected) != dirty_target:
        raise SupervisionBuildError(f"dirty observation count mismatch: {len(dirty_indices_expected)} != {dirty_target}")
    subject_fold, stay_subject, hadm_subject = _subject_folds(context, seed)
    per_group_limit = max(64, min(512, math.ceil(clean_target / 20)))
    group_heaps: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    global_heap: list[tuple[int, int, int]] = []
    dirty_records: list[tuple[int, int, int]] = []

    current_row_id: int | None = None
    current: list[tuple[int, dict[str, Any]]] = []

    def flush() -> None:
        if not current:
            return
        values = {str(item["column"]): str(item.get("raw_value") or "") for _, item in current}
        subject = _observation_subject(values, stay_subject, hadm_subject)
        if not subject or subject not in subject_fold:
            if any(index in dirty_indices_expected for index, _ in current):
                raise SupervisionBuildError(f"dirty row cannot be assigned to a subject: row_id={current[0][1]['row_id']}")
            return
        fold = subject_fold[subject]
        for observation_index, item in current:
            if labels[observation_index]:
                coordinate = (str(item["table"]) + ".csv", int(item["row_number"]) - 1, str(item["column"]))
                row_key = coordinate[:2]
                source = 1 if coordinate in base_cells or row_key in base_rows else 2
                dirty_records.append((observation_index, fold, source))
                continue
            relation = str(item["relation"])
            priority = int.from_bytes(hashlib.sha256(f"{seed}:{observation_index}".encode("utf-8")).digest()[:8], "big")
            _heap_observation(group_heaps[relation], per_group_limit, priority, observation_index, fold)
            _heap_observation(global_heap, clean_target * 3, priority, observation_index, fold)

    observations_path = graph_dir / "cell_observations.jsonl"
    with observations_path.open(encoding="utf-8") as handle:
        for observation_index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = int(item["row_id"])
            if current_row_id is not None and row_id != current_row_id:
                flush()
                current = []
            current_row_id = row_id
            current.append((observation_index, item))
    flush()
    if len(dirty_records) != dirty_target:
        raise SupervisionBuildError(f"subject-linked dirty count mismatch: {len(dirty_records)} != {dirty_target}")

    grouped = {
        relation: sorted([(-neg_priority, index, fold) for neg_priority, index, fold in heap])
        for relation, heap in group_heaps.items()
    }
    selected_clean: list[tuple[int, int]] = []
    used_clean: set[int] = set()
    offset = 0
    relations = sorted(grouped)
    while len(selected_clean) < clean_target:
        progressed = False
        for relation in relations:
            candidates = grouped[relation]
            if offset < len(candidates):
                _, index, fold = candidates[offset]
                if index not in used_clean:
                    selected_clean.append((index, fold))
                    used_clean.add(index)
                    progressed = True
                    if len(selected_clean) == clean_target:
                        break
        if not progressed:
            break
        offset += 1
    if len(selected_clean) < clean_target:
        for _, index, fold in sorted((-neg_priority, index, fold) for neg_priority, index, fold in global_heap):
            if index in used_clean:
                continue
            selected_clean.append((index, fold))
            used_clean.add(index)
            if len(selected_clean) == clean_target:
                break
    if len(selected_clean) != clean_target:
        raise SupervisionBuildError(f"insufficient subject-linked clean observations: {len(selected_clean)} != {clean_target}")

    rows = [(index, 1, fold, source) for index, fold, source in dirty_records]
    rows.extend((index, 0, fold, 0) for index, fold in selected_clean)
    rows.sort(key=lambda item: hashlib.sha256(f"{seed}:mask:{item[0]}".encode("utf-8")).digest())
    np.savez_compressed(
        output_path,
        cell_indices=np.asarray([item[0] for item in rows], dtype=np.int64),
        cell_labels=np.asarray([item[1] for item in rows], dtype=np.int8),
        cell_folds=np.asarray([item[2] for item in rows], dtype=np.int8),
        cell_source=np.asarray([item[3] for item in rows], dtype=np.int8),
    )
    return {
        "clean_training_count": clean_target,
        "dirty_training_count": dirty_target,
        "fold_counts": dict(Counter(str(item[2]) for item in rows)),
        "clean_group_count": len(grouped),
    }


def build_supervised_graph(
    *,
    base_dirty_raw: str | Path,
    clean_raw: str | Path,
    clean_graph: str | Path,
    base_injection_log: str | Path,
    output_raw: str | Path,
    output_graph: str | Path,
    private_output: str | Path,
    dirty_target: int = 20_000,
    clean_target: int = 10_000,
    seeds: tuple[int, ...] = (666, 667),
) -> dict[str, Any]:
    base_dirty_raw = Path(base_dirty_raw).expanduser().resolve()
    clean_raw = Path(clean_raw).expanduser().resolve()
    clean_graph = Path(clean_graph).expanduser().resolve()
    base_injection_log = Path(base_injection_log).expanduser().resolve()
    output_raw = Path(output_raw).expanduser().resolve()
    output_graph = Path(output_graph).expanduser().resolve()
    private_output = Path(private_output).expanduser().resolve()
    if not base_dirty_raw.is_dir() or not clean_raw.is_dir() or not clean_graph.is_dir():
        raise SupervisionBuildError("base_dirty_raw, clean_raw and clean_graph must exist")
    if dirty_target <= 0 or clean_target <= 0:
        raise SupervisionBuildError("dirty_target and clean_target must be positive")
    if not seeds or len(set(seeds)) != len(seeds):
        raise SupervisionBuildError("seeds must be nonempty and unique")
    for output, label in ((output_raw, "output_raw"), (output_graph, "output_graph"), (private_output, "private_output")):
        _ensure_new_or_empty(output, label)
    relation_path = clean_graph / "relation_ids.json"
    graph_manifest_path = clean_graph / "graph_manifest.json"
    if not relation_path.is_file() or not graph_manifest_path.is_file():
        raise SupervisionBuildError("clean_graph is missing relation_ids.json or graph_manifest.json")

    source_base_rows = _read_log(base_injection_log)
    base_rows, base_dirty_rows, base_dirty_cells, base_class_counts, normalization_stats = _normalize_base_cell_records(
        base_dirty_raw,
        clean_raw,
        source_base_rows,
    )
    base_dirty_count = len(base_dirty_cells)
    if base_dirty_count >= dirty_target:
        raise SupervisionBuildError(f"dirty_target must exceed base dirty Cells: {base_dirty_count}")
    before_dirty_hash = _directory_sha256(base_dirty_raw)
    before_clean_hash = _directory_sha256(clean_raw)
    context = _read_context(clean_raw)
    pool_limit = max(_class_targets(dirty_target).values())
    pools = _collect_candidates(clean_raw, context, base_dirty_rows, tuple(seeds), pool_limit)
    synthetic = _select_synthetic(pools, base_dirty_rows, base_class_counts, dirty_target, tuple(seeds))

    coordinate_lines = [
        "|".join((record["raw_file"], record["raw_row_index"], record["column"], record["error_subtype"], record["injection_seed"]))
        for record in sorted(synthetic, key=lambda item: (item["raw_file"], int(item["raw_row_index"]), item["column"]))
    ]
    synthetic_coordinate_sha256 = hashlib.sha256("\n".join(coordinate_lines).encode("utf-8")).hexdigest()
    merged_log = private_output / "merged_injection_log.csv"
    try:
        private_output.mkdir(parents=True, exist_ok=True)
        _write_augmented_raw(clean_raw, output_raw, base_rows + synthetic)
        _write_merged_log(merged_log, base_rows, synthetic)
        raw_manifest = {
            "schema_version": 2,
            "status": "SUCCESS",
            "workflow": "graph_cell_supervised_error_injection",
            "source_dirty_raw_name": base_dirty_raw.name,
            "clean_raw_name": clean_raw.name,
            "source_base_log_record_count": len(source_base_rows),
            "base_log_record_count": len(base_rows),
            **normalization_stats,
            "synthetic_log_record_count": len(synthetic),
            "dirty_observation_target": dirty_target,
            "allowed_operation": "cell_update",
            "seeds": list(seeds),
            "synthetic_coordinate_sha256": synthetic_coordinate_sha256,
        }
        (output_raw / "raw_supervision_manifest.json").write_text(
            json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        relation_ids = json.loads(relation_path.read_text(encoding="utf-8"))
        build_graph(output_raw, output_graph, relation_ids=relation_ids)
        label_manifest = build_labels(output_graph, merged_log, private_output)
        masks = _build_supervision_masks(
            output_graph,
            private_output / "edge_labels.npz",
            private_output / "supervision_masks.npz",
            context,
            base_dirty_rows,
            base_dirty_cells,
            clean_target,
            dirty_target,
            int(seeds[0]),
        )
        if _directory_sha256(base_dirty_raw) != before_dirty_hash or _directory_sha256(clean_raw) != before_clean_hash:
            raise SupervisionBuildError("source raw directories changed during supervision build")
        result = {
            "schema_version": 2,
            "status": "SUCCESS",
            "workflow": "build_cell_supervised_graph",
            "source_base_log_record_count": len(source_base_rows),
            "base_log_record_count": len(base_rows),
            **normalization_stats,
            "synthetic_log_record_count": len(synthetic),
            "dirty_observation_count": int(label_manifest["dirty_cell_count"]),
            "clean_training_count": clean_target,
            "training_observation_count": dirty_target + clean_target,
            "seeds": list(seeds),
            "synthetic_coordinate_sha256": synthetic_coordinate_sha256,
            "base_raw_sha256": before_dirty_hash,
            "clean_raw_sha256": before_clean_hash,
            "clean_graph_manifest_sha256": _file_sha256(graph_manifest_path),
            "relation_ids_sha256": _file_sha256(relation_path),
            "by_error_class_records": dict(Counter(record["error_class"] for record in base_rows + synthetic)),
            "by_error_subtype_records": dict(Counter(record["error_subtype"] for record in base_rows + synthetic)),
            "by_operation_records": dict(Counter(record["operation"] for record in base_rows + synthetic)),
            **masks,
        }
        (private_output / "supervision_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    except Exception:
        shutil.rmtree(output_graph, ignore_errors=True)
        shutil.rmtree(output_raw, ignore_errors=True)
        shutil.rmtree(private_output, ignore_errors=True)
        raise
