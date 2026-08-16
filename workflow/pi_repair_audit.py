from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DELETE_ROW = "__DELETE_ROW__"
MAX_REPAIR_CANDIDATES = 5
PUBLIC_VIEW_SCHEMA_VERSION = 1
REPAIR_SUBMISSION_SCHEMA_VERSION = 1
REPAIR_REPORT_SCHEMA_VERSION = 1
REPAIR_SCORER_VERSION = "raw_repair_candidates_v2"

_CANDIDATE_KEYS = {"value", "rule_id", "evidence"}
_RECORD_KEYS = {
    "table",
    "row_index",
    "column",
    "original_value",
    "candidates",
    "selected_value",
}
_OUTPUT_KEYS = {"corrected_raw", "repair_candidates"}
_SHELL_LAUNCHERS = {"bash", "dash", "env", "fish", "ksh", "sh", "zsh"}
_SEMANTIC_ID_COLUMNS = {
    "itemid",
    "spec_itemid",
    "test_itemid",
    "org_itemid",
}
_EXPLICIT_LOCATOR_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "caregiver_id",
    "provider_id",
    "orderid",
    "linkorderid",
    "pharmacy_id",
    "poe_id",
    "emar_id",
    "transfer_id",
    "labevent_id",
    "microevent_id",
    "row_id",
    "row_number",
    "raw_row_index",
    "observation_id",
    "observation_index",
}


class PiRepairAuditError(ValueError):
    pass


@dataclass(frozen=True)
class RepairSubmission:
    workdir: Path
    replay_argv: tuple[str, ...]
    corrected_raw: Path
    repair_candidates: Path


@dataclass(frozen=True)
class RepairCandidate:
    table: str
    row_index: int
    column: str
    original_value: str
    candidates: tuple[dict[str, str], ...]
    selected_value: str

    @property
    def coordinate(self) -> tuple[str, int, str]:
        return self.table, self.row_index, self.column


@dataclass(frozen=True)
class GoldRepair:
    table: str
    row_index: int
    column: str
    dirty_value: str
    clean_value: str
    error_class: str
    operation: str

    @property
    def coordinate(self) -> tuple[str, int, str]:
        return self.table, self.row_index, self.column

    @property
    def expected_value(self) -> str:
        return DELETE_ROW if self.operation == "row_insert" else self.clean_value


def build_public_train_view(
    *,
    dirty_raw: str | Path,
    clean_raw: str | Path,
    graph_dir: str | Path,
    supervision_dir: str | Path,
    paired_log: str | Path,
    output_dir: str | Path,
    expected_dirty: int = 20_000,
    expected_clean: int = 10_000,
    expected_table_count: int = 29,
) -> dict[str, Any]:
    dirty = _require_directory(dirty_raw, "dirty raw")
    clean = _require_directory(clean_raw, "clean raw")
    graph = _require_directory(graph_dir, "graph")
    supervision = _require_directory(supervision_dir, "supervision")
    paired = _require_file(paired_log, "paired log")
    output = _new_directory(output_dir)

    dirty_files = _csv_files(dirty)
    clean_files = _csv_files(clean)
    if set(dirty_files) != set(clean_files):
        raise PiRepairAuditError("dirty and clean raw CSV file sets differ")
    if len(dirty_files) != expected_table_count:
        raise PiRepairAuditError(
            f"raw table count mismatch: expected={expected_table_count} actual={len(dirty_files)}"
        )
    table_reports: dict[str, Any] = {}
    for relative in sorted(dirty_files):
        dirty_header = _csv_header(dirty_files[relative])
        clean_header = _csv_header(clean_files[relative])
        if dirty_header != clean_header:
            raise PiRepairAuditError(f"dirty/clean columns differ: {relative}")
        dirty_report = _write_public_csv(
            dirty_files[relative], output / "dirty_raw" / relative
        )
        clean_report = _write_public_csv(
            clean_files[relative], output / "clean_raw" / relative
        )
        table_reports[relative] = {
            "columns": dirty_header,
            "dirty_rows": dirty_report["rows"],
            "clean_rows": clean_report["rows"],
            "dirty_sha256": dirty_report["sha256"],
            "clean_sha256": clean_report["sha256"],
        }

    evidence = _build_public_supervision_evidence(
        graph_dir=graph,
        supervision_dir=supervision,
        paired_log=paired,
        output_dir=output / "evidence",
        expected_dirty=expected_dirty,
        expected_clean=expected_clean,
    )
    manifest = {
        "schema_version": PUBLIC_VIEW_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "pi_agent_public_paired_train_view",
        "table_count": len(table_reports),
        "supervision": {
            "dirty_count": evidence["dirty_count"],
            "clean_count": evidence["clean_count"],
            "field_count": evidence["field_count"],
        },
        "tables": table_reports,
        "privacy": {
            "locator_values_pseudonymized": True,
            "fold_exported": False,
            "label_exported": False,
            "error_taxonomy_exported": False,
            "private_paths_exported": False,
        },
        "inputs": {
            "dirty_raw_sha256": _directory_sha256(dirty),
            "clean_raw_sha256": _directory_sha256(clean),
            "graph_manifest_sha256": _optional_sha256(graph / "graph_manifest.json"),
            "supervision_masks_sha256": _sha256(supervision / "supervision_masks.npz"),
            "paired_log_sha256": _sha256(paired),
        },
    }
    _write_json(output / "public_train_manifest.json", manifest)
    return manifest


def load_repair_submission(workdir: str | Path) -> RepairSubmission:
    root = _require_directory(workdir, "Agent workdir")
    path = root / "repair_submission.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PiRepairAuditError("repair_submission.json is missing") from exc
    except json.JSONDecodeError as exc:
        raise PiRepairAuditError("repair_submission.json is invalid JSON") from exc
    if payload.get("schema_version") != REPAIR_SUBMISSION_SCHEMA_VERSION:
        raise PiRepairAuditError("repair submission schema_version must be 1")
    replay = payload.get("replay")
    argv = replay.get("argv") if isinstance(replay, dict) else None
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise PiRepairAuditError("repair submission replay.argv must be a string array")
    if Path(argv[0]).name.lower() in _SHELL_LAUNCHERS:
        raise PiRepairAuditError("repair replay cannot execute shell command strings")
    for item in argv:
        static = item
        for placeholder in ("raw_root", "output_dir", "workdir"):
            static = static.replace("{" + placeholder + "}", "placeholder")
        if Path(static).is_absolute() and "{" not in item:
            raise PiRepairAuditError("repair replay cannot contain static absolute paths")
    joined = "\0".join(argv)
    if "{raw_root}" not in joined or "{output_dir}" not in joined:
        raise PiRepairAuditError("repair replay requires raw_root and output_dir placeholders")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != _OUTPUT_KEYS:
        raise PiRepairAuditError(
            "repair submission outputs must define corrected_raw and repair_candidates"
        )
    output_root = (root / str(payload.get("current_output_root") or "outputs/train")).resolve()
    if not _is_within(output_root, root):
        raise PiRepairAuditError("current_output_root escapes Agent workdir")
    resolved: dict[str, Path] = {}
    for key in sorted(_OUTPUT_KEYS):
        value = outputs[key]
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise PiRepairAuditError(f"repair output path is invalid: {key}")
        destination = (output_root / value).resolve()
        if not _is_within(destination, output_root):
            raise PiRepairAuditError(f"repair output path escapes output root: {key}")
        resolved[key] = destination
    return RepairSubmission(
        workdir=root,
        replay_argv=tuple(argv),
        corrected_raw=resolved["corrected_raw"],
        repair_candidates=resolved["repair_candidates"],
    )


def render_repair_replay_argv(
    submission: RepairSubmission,
    *,
    raw_root: str | Path,
    output_dir: str | Path,
    workdir: str | Path,
) -> list[str]:
    values = {
        "raw_root": str(Path(raw_root).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "workdir": str(Path(workdir).expanduser().resolve()),
    }
    rendered: list[str] = []
    for item in submission.replay_argv:
        try:
            rendered.append(item.format_map(values))
        except KeyError as exc:
            raise PiRepairAuditError(
                f"unsupported repair replay placeholder: {exc.args[0]}"
            ) from exc
    return rendered


def load_repair_candidates(path: str | Path) -> list[RepairCandidate]:
    source = _require_file(path, "repair candidate report")
    records: list[RepairCandidate] = []
    previous: tuple[str, int, str] | None = None
    seen: set[tuple[str, int, str]] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PiRepairAuditError(
                    f"invalid repair candidate JSONL at line {line_number}"
                ) from exc
            if not isinstance(payload, dict) or set(payload) != _RECORD_KEYS:
                raise PiRepairAuditError(
                    f"repair candidate record has invalid keys at line {line_number}"
                )
            table = _normalize_csv_path(str(payload["table"]))
            row_index = payload["row_index"]
            if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
                raise PiRepairAuditError(f"invalid row_index at line {line_number}")
            column = str(payload["column"])
            original = str(payload["original_value"])
            raw_candidates = payload["candidates"]
            if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 5:
                raise PiRepairAuditError(f"candidate count must be 1..5 at line {line_number}")
            candidates: list[dict[str, str]] = []
            values: list[str] = []
            for item in raw_candidates:
                if not isinstance(item, dict) or set(item) != _CANDIDATE_KEYS:
                    raise PiRepairAuditError(f"invalid candidate fields at line {line_number}")
                candidate = {key: str(item[key]) for key in sorted(_CANDIDATE_KEYS)}
                if not candidate["rule_id"] or not candidate["evidence"]:
                    raise PiRepairAuditError(
                        f"candidate rule/evidence is empty at line {line_number}"
                    )
                candidates.append(candidate)
                values.append(candidate["value"])
            if len(set(values)) != len(values):
                raise PiRepairAuditError(f"candidate values are not unique at line {line_number}")
            selected = str(payload["selected_value"])
            if selected not in values:
                raise PiRepairAuditError(f"selected_value is not a candidate at line {line_number}")
            if column == "" and any(value != DELETE_ROW for value in values):
                raise PiRepairAuditError(f"row action must use {DELETE_ROW} at line {line_number}")
            if column and any(value == DELETE_ROW for value in values):
                raise PiRepairAuditError(
                    f"Cell action cannot use {DELETE_ROW} at line {line_number}"
                )
            record = RepairCandidate(
                table=table,
                row_index=row_index,
                column=column,
                original_value=original,
                candidates=tuple(candidates),
                selected_value=selected,
            )
            if record.coordinate in seen:
                raise PiRepairAuditError(f"duplicate repair coordinate at line {line_number}")
            if previous is not None and record.coordinate < previous:
                raise PiRepairAuditError(
                    "repair candidate report must use deterministic coordinate order"
                )
            previous = record.coordinate
            seen.add(record.coordinate)
            records.append(record)
    return records


def validate_declared_repair_output(
    *,
    raw_root: str | Path,
    corrected_raw: str | Path,
    candidate_report: str | Path,
) -> dict[str, Any]:
    raw = _require_directory(raw_root, "raw root")
    corrected = _require_directory(corrected_raw, "corrected raw")
    candidates = load_repair_candidates(candidate_report)
    raw_files = _csv_files(raw)
    corrected_files = _csv_files(corrected)
    if set(raw_files) != set(corrected_files):
        raise PiRepairAuditError("corrected raw file set differs from input raw")
    by_table: dict[str, list[RepairCandidate]] = defaultdict(list)
    for item in candidates:
        if item.table not in raw_files:
            raise PiRepairAuditError(f"repair candidate references unknown table: {item.table}")
        by_table[item.table].append(item)

    verified_rows = 0
    for table in sorted(raw_files):
        if not by_table.get(table):
            if _sha256(raw_files[table]) != _sha256(corrected_files[table]):
                raise PiRepairAuditError(f"undeclared file modification: {table}")
            continue
        verified_rows += _validate_table_repairs(
            raw_files[table], corrected_files[table], by_table[table]
        )
    return {
        "status": "SUCCESS",
        "declared_repair_count": len(candidates),
        "modified_table_count": len(by_table),
        "verified_input_row_count": verified_rows,
    }


def score_repair_candidates(
    *,
    candidate_report: str | Path,
    gold_log: str | Path,
    raw_root: str | Path,
    row_gold_log: str | Path | None = None,
    pseudonymize_locator_values: bool = False,
    max_failure_cases: int = 200,
) -> dict[str, Any]:
    predictions = load_repair_candidates(candidate_report)
    gold, gold_event_summary = _load_gold_repairs(
        gold_log,
        row_gold_log=row_gold_log,
        pseudonymize_locator_values=pseudonymize_locator_values,
    )
    predicted = {item.coordinate: item for item in predictions}
    gold_by_coordinate = {item.coordinate: item for item in gold}
    predicted_coordinates = set(predicted)
    gold_coordinates = set(gold_by_coordinate)
    detected = predicted_coordinates & gold_coordinates
    false_positives = predicted_coordinates - gold_coordinates
    missed = gold_coordinates - predicted_coordinates

    exact_coordinates = {
        coordinate
        for coordinate in detected
        if predicted[coordinate].selected_value == gold_by_coordinate[coordinate].expected_value
    }
    wrong_coordinates = detected - exact_coordinates
    recall_at: dict[int, float] = {}
    for k in (1, 3, 5):
        hits = {
            coordinate
            for coordinate in detected
            if gold_by_coordinate[coordinate].expected_value
            in [item["value"] for item in predicted[coordinate].candidates[:k]]
        }
        recall_at[k] = _ratio(len(hits), len(gold_coordinates))

    detection_precision = _ratio(len(detected), len(predicted_coordinates))
    detection_recall = _ratio(len(detected), len(gold_coordinates))
    repair_precision = _ratio(len(exact_coordinates), len(predicted_coordinates))
    repair_recall = _ratio(len(exact_coordinates), len(gold_coordinates))
    opportunities = _count_repair_opportunities(raw_root)
    clean_opportunities = max(opportunities - len(gold_coordinates), 0)
    clean_preservation = (
        1.0
        if clean_opportunities == 0 and not false_positives
        else _ratio(clean_opportunities - len(false_positives), clean_opportunities)
    )
    by_class: dict[str, Any] = {}
    for error_class in sorted({item.error_class for item in gold}):
        class_coordinates = {
            item.coordinate for item in gold if item.error_class == error_class
        }
        class_detected = class_coordinates & predicted_coordinates
        class_exact = class_coordinates & exact_coordinates
        by_class[error_class] = {
            "gold_count": len(class_coordinates),
            "detected_count": len(class_detected),
            "exact_repair_count": len(class_exact),
            "detection_recall": _rounded(_ratio(len(class_detected), len(class_coordinates))),
            "exact_repair_recall": _rounded(_ratio(len(class_exact), len(class_coordinates))),
        }

    failures: list[dict[str, Any]] = []
    for coordinate in sorted(missed | wrong_coordinates | false_positives):
        if len(failures) >= max_failure_cases:
            break
        truth = gold_by_coordinate.get(coordinate)
        prediction = predicted.get(coordinate)
        locator = bool(coordinate[2] and _is_locator_column(coordinate[2]))
        dirty_value = (
            truth.dirty_value
            if truth
            else (prediction.original_value if prediction else "")
        )
        expected_value = truth.expected_value if truth else ""
        selected_value = prediction.selected_value if prediction else ""
        candidate_values = (
            [item["value"] for item in prediction.candidates] if prediction else []
        )
        if coordinate[2] == "":
            dirty_value = (
                "row_sha256:" + hashlib.sha256(dirty_value.encode()).hexdigest()[:16]
                if dirty_value
                else ""
            )
            expected_value = DELETE_ROW if truth else ""
            selected_value = DELETE_ROW if selected_value == DELETE_ROW else ""
            candidate_values = [
                DELETE_ROW for value in candidate_values if value == DELETE_ROW
            ]
        elif locator:
            dirty_value = _public_value(coordinate[2], dirty_value)
            expected_value = _public_value(coordinate[2], expected_value)
            selected_value = _public_value(coordinate[2], selected_value)
            candidate_values = [
                _public_value(coordinate[2], value) for value in candidate_values
            ]
        failures.append(
            {
                "case_id": hashlib.sha256(
                    ("\0".join((coordinate[0], str(coordinate[1]), coordinate[2]))).encode()
                ).hexdigest()[:16],
                "table": coordinate[0],
                "column": coordinate[2],
                "failure": (
                    "missed"
                    if prediction is None
                    else "false_positive"
                    if truth is None
                    else "wrong_value"
                ),
                "error_class": truth.error_class if truth else "",
                "dirty_value": dirty_value,
                "expected_value": expected_value,
                "selected_value": selected_value,
                "candidate_values": candidate_values,
            }
        )

    return {
        "schema_version": REPAIR_REPORT_SCHEMA_VERSION,
        "scorer_version": REPAIR_SCORER_VERSION,
        "status": "SUCCESS",
        "mode": "host_private_raw_repair_audit",
        "counts": {
            "gold_log_events": gold_event_summary["event_count"],
            "overlapping_gold_events": gold_event_summary["overlapping_event_count"],
            "gold_repairs": len(gold_coordinates),
            "predicted_repairs": len(predicted_coordinates),
            "detected_gold": len(detected),
            "exact_repairs": len(exact_coordinates),
            "wrong_value_repairs": len(wrong_coordinates),
            "false_positive_repairs": len(false_positives),
            "missed_repairs": len(missed),
            "clean_action_opportunities": clean_opportunities,
        },
        "metrics": {
            "detection_precision": _rounded(detection_precision),
            "detection_recall": _rounded(detection_recall),
            "detection_f1": _rounded(_f1(detection_precision, detection_recall)),
            "candidate_recall_at_1": _rounded(recall_at[1]),
            "candidate_recall_at_3": _rounded(recall_at[3]),
            "candidate_recall_at_5": _rounded(recall_at[5]),
            "exact_repair_precision": _rounded(repair_precision),
            "exact_repair_recall": _rounded(repair_recall),
            "exact_repair_f1": _rounded(_f1(repair_precision, repair_recall)),
            "clean_preservation": _rounded(clean_preservation),
        },
        "by_error_class": by_class,
        "failure_cases": failures,
        "failure_case_count": len(missed | wrong_coordinates | false_positives),
        "inputs": {
            "candidate_report_sha256": _sha256(Path(candidate_report)),
            "gold_log_sha256": _sha256(Path(gold_log)),
            "row_gold_log_sha256": (
                _sha256(Path(row_gold_log)) if row_gold_log is not None else ""
            ),
        },
    }


def score_repair_output(
    *,
    output_root: str | Path,
    raw_root: str | Path,
    gold_log: str | Path,
    row_gold_log: str | Path | None = None,
    pseudonymize_locator_values: bool = False,
) -> dict[str, Any]:
    root = _require_directory(output_root, "repair output")
    corrected = root / "corrected_raw"
    candidates = root / "repair_candidates.jsonl"
    integrity = validate_declared_repair_output(
        raw_root=raw_root,
        corrected_raw=corrected,
        candidate_report=candidates,
    )
    repair = score_repair_candidates(
        candidate_report=candidates,
        gold_log=gold_log,
        row_gold_log=row_gold_log,
        raw_root=raw_root,
        pseudonymize_locator_values=pseudonymize_locator_values,
    )
    return {
        "schema_version": 1,
        "status": "SUCCESS",
        "workflow": "pi_agent_train_only_raw_repair_audit",
        "integrity": integrity,
        "repair": repair,
        "metrics": dict(repair["metrics"]),
    }


def write_sanitized_failure_cases(report: Mapping[str, Any], path: str | Path) -> None:
    failures = ((report.get("repair") or report).get("failure_cases") or [])
    _write_json(Path(path), {"schema_version": 1, "failure_cases": failures})


def _build_public_supervision_evidence(
    *,
    graph_dir: Path,
    supervision_dir: Path,
    paired_log: Path,
    output_dir: Path,
    expected_dirty: int,
    expected_clean: int,
) -> dict[str, Any]:
    masks = supervision_dir / "supervision_masks.npz"
    observations = graph_dir / "cell_observations.jsonl"
    _require_file(masks, "supervision masks")
    _require_file(observations, "Cell observations")
    with np.load(masks) as arrays:
        required = {"cell_indices", "cell_labels"}
        if not required.issubset(arrays.files):
            raise PiRepairAuditError("supervision masks have an invalid schema")
        indices = np.asarray(arrays["cell_indices"], dtype=np.int64)
        labels = np.asarray(arrays["cell_labels"], dtype=np.int8)
    if len(indices) != len(labels) or len(set(indices.tolist())) != len(indices):
        raise PiRepairAuditError("supervision masks are inconsistent")
    selected = {int(index): int(label) for index, label in zip(indices, labels)}
    dirty_count = sum(label == 1 for label in selected.values())
    clean_count = sum(label == 0 for label in selected.values())
    if dirty_count != expected_dirty or clean_count != expected_clean:
        raise PiRepairAuditError(
            "supervision counts differ from the Train-only contract: "
            f"dirty={dirty_count} clean={clean_count}"
        )

    observed: dict[int, dict[str, Any]] = {}
    requested_rows: dict[str, set[int]] = defaultdict(set)
    with observations.open(encoding="utf-8") as handle:
        for observation_index, line in enumerate(handle):
            if observation_index not in selected:
                continue
            payload = json.loads(line)
            table = str(payload["table"])
            row_number = int(payload["row_number"])
            column = str(payload["column"])
            observed[observation_index] = {
                "table": table,
                "row_number": row_number,
                "column": column,
                "current": str(payload.get("raw_value") or ""),
            }
            requested_rows[table].add(row_number)
    if set(observed) != set(selected):
        raise PiRepairAuditError("supervision references missing Cell observations")

    row_context: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    with observations.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            table = str(payload["table"])
            row_number = int(payload["row_number"])
            if row_number not in requested_rows.get(table, set()):
                continue
            column = str(payload["column"])
            row_context[(table, row_number)][column] = str(payload.get("raw_value") or "")

    injection = _read_gold_cell_map(paired_log)
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"dirty_clean_pairs": [], "clean_examples": []}
    )
    dirty_coordinates: set[tuple[str, int, str]] = set()
    for observation_index in sorted(selected):
        label = selected[observation_index]
        item = observed[observation_index]
        table = item["table"]
        row_number = item["row_number"]
        column = item["column"]
        current = item["current"]
        context = {
            name: _public_value(name, value)
            for name, value in sorted(row_context[(table, row_number)].items())
            if name != column
        }
        key = (table, row_number, column)
        if label == 1:
            truth = injection.get(key)
            if truth is None or current != truth["dirty_value"]:
                raise PiRepairAuditError(f"dirty supervision mismatch: {table}.{column}")
            dirty_coordinates.add(key)
            groups[(table, column)]["dirty_clean_pairs"].append(
                {
                    "dirty": _public_value(column, current),
                    "clean": _public_value(column, truth["clean_value"]),
                    "row_context": context,
                }
            )
        else:
            if key in injection:
                raise PiRepairAuditError("sampled clean Cell appears in injection log")
            groups[(table, column)]["clean_examples"].append(
                {"value": _public_value(column, current), "row_context": context}
            )
    if dirty_coordinates != set(injection):
        raise PiRepairAuditError("dirty supervision and paired log coordinates differ")

    fields_dir = output_dir / "fields"
    fields_dir.mkdir(parents=True)
    fields: list[dict[str, Any]] = []
    for (table, column), values in sorted(groups.items()):
        field_id = hashlib.sha256(f"{table}\0{column}".encode()).hexdigest()[:16]
        relative = Path("fields") / f"{field_id}.json"
        payload = {
            "table": table,
            "column": column,
            "dirty_clean_pairs": values["dirty_clean_pairs"],
            "clean_examples": values["clean_examples"],
        }
        _write_json(output_dir / relative, payload)
        fields.append(
            {
                "field_id": field_id,
                "table": table,
                "column": column,
                "path": relative.as_posix(),
                "dirty_pair_count": len(values["dirty_clean_pairs"]),
                "clean_example_count": len(values["clean_examples"]),
                "sha256": _sha256(output_dir / relative),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "SUCCESS",
        "dirty_count": dirty_count,
        "clean_count": clean_count,
        "field_count": len(fields),
        "fields": fields,
        "privacy": {
            "locator_values_pseudonymized": True,
            "row_coordinates_exported": False,
            "fold_exported": False,
            "error_taxonomy_exported": False,
        },
    }
    _write_json(output_dir / "evidence_manifest.json", manifest)
    return manifest


def _read_gold_cell_map(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    result: dict[tuple[str, int, str], dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"raw_file", "raw_row_index", "column", "clean_value", "dirty_value"}
        if not required.issubset(reader.fieldnames or []):
            raise PiRepairAuditError("paired log has an invalid schema")
        for row in reader:
            table = _normalize_table_name(row["raw_file"])
            key = (table, int(row["raw_row_index"]) + 1, row["column"])
            if key in result:
                raise PiRepairAuditError("paired log has duplicate Cell coordinates")
            result[key] = {name: str(value or "") for name, value in row.items()}
    return result


def _load_gold_repairs(
    gold_log: str | Path,
    *,
    row_gold_log: str | Path | None,
    pseudonymize_locator_values: bool,
) -> tuple[list[GoldRepair], dict[str, int]]:
    rows: list[dict[str, str]] = []
    with _require_file(gold_log, "Gold repair log").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows.extend(csv.DictReader(handle))
    if row_gold_log is not None:
        with _require_file(row_gold_log, "row Gold repair log").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            rows.extend(
                row
                for row in csv.DictReader(handle)
                if row.get("operation") == "row_insert"
            )
    grouped: dict[tuple[str, int, str], list[GoldRepair]] = defaultdict(list)
    for row in rows:
        operation = str(row.get("operation") or "")
        if operation not in {"cell_update", "row_insert"}:
            raise PiRepairAuditError(f"unsupported Gold repair operation: {operation}")
        table = _normalize_csv_path(str(row.get("raw_file") or ""))
        row_index = int(row.get("raw_row_index") or 0)
        column = "" if operation == "row_insert" else str(row.get("column") or "")
        dirty_value = str(row.get("dirty_value") or "")
        clean_value = str(row.get("clean_value") or "")
        if pseudonymize_locator_values and column:
            dirty_value = _public_value(column, dirty_value)
            clean_value = _public_value(column, clean_value)
        item = GoldRepair(
            table=table,
            row_index=row_index,
            column=column,
            dirty_value=dirty_value,
            clean_value=clean_value,
            error_class=str(row.get("error_class") or "unknown"),
            operation=operation,
        )
        grouped[item.coordinate].append(item)
    result = [_collapse_gold_repair_chain(grouped[key]) for key in sorted(grouped)]
    event_count = sum(len(items) for items in grouped.values())
    return result, {
        "event_count": event_count,
        "overlapping_event_count": event_count - len(result),
    }


def _collapse_gold_repair_chain(items: Sequence[GoldRepair]) -> GoldRepair:
    """Collapse sequential injections on one Cell into its final repair target."""

    if not items:
        raise PiRepairAuditError("Gold repair chain is empty")
    operations = {item.operation for item in items}
    if len(operations) != 1:
        raise PiRepairAuditError(f"mixed Gold operations at coordinate: {items[0].coordinate}")
    classes = sorted({item.error_class for item in items})
    error_class = classes[0] if len(classes) == 1 else "multiple:" + "+".join(classes)
    if items[0].operation == "row_insert":
        first = items[0]
        return GoldRepair(
            table=first.table,
            row_index=first.row_index,
            column=first.column,
            dirty_value=first.dirty_value,
            clean_value=first.clean_value,
            error_class=error_class,
            operation=first.operation,
        )

    unique = set(items)
    if len(unique) == 1:
        return items[0]
    dirty_values = {item.dirty_value for item in items}
    clean_values = {item.clean_value for item in items}
    current_candidates = dirty_values - clean_values
    if len(current_candidates) == 1:
        current = next(iter(current_candidates))
    elif len(dirty_values) == 1:
        current = next(iter(dirty_values))
    else:
        raise PiRepairAuditError(
            f"ambiguous final dirty value in Gold repair chain: {items[0].coordinate}"
        )
    original_candidates = clean_values - dirty_values
    if len(original_candidates) == 1:
        original = next(iter(original_candidates))
    elif len(clean_values) == 1:
        original = next(iter(clean_values))
    else:
        raise PiRepairAuditError(
            f"ambiguous original clean value in Gold repair chain: {items[0].coordinate}"
        )

    edges: dict[str, set[str]] = defaultdict(set)
    for item in items:
        edges[item.dirty_value].add(item.clean_value)
    frontier = [current]
    visited: set[str] = set()
    while frontier:
        value = frontier.pop()
        if value == original:
            break
        if value in visited:
            continue
        visited.add(value)
        frontier.extend(sorted(edges.get(value, ())))
    else:
        raise PiRepairAuditError(
            f"disconnected Gold repair chain: {items[0].coordinate}"
        )

    first = items[0]
    return GoldRepair(
        table=first.table,
        row_index=first.row_index,
        column=first.column,
        dirty_value=current,
        clean_value=original,
        error_class=error_class,
        operation=first.operation,
    )


def _validate_table_repairs(
    raw_path: Path,
    corrected_path: Path,
    candidates: Sequence[RepairCandidate],
) -> int:
    by_row: dict[int, list[RepairCandidate]] = defaultdict(list)
    for item in candidates:
        by_row[item.row_index].append(item)
    with raw_path.open(encoding="utf-8-sig", newline="") as raw_handle, corrected_path.open(
        encoding="utf-8-sig", newline=""
    ) as corrected_handle:
        raw_reader = csv.reader(raw_handle)
        corrected_reader = csv.reader(corrected_handle)
        raw_header = next(raw_reader, None)
        corrected_header = next(corrected_reader, None)
        if raw_header != corrected_header or raw_header is None:
            raise PiRepairAuditError(f"corrected raw header mismatch: {raw_path.name}")
        columns = {name: index for index, name in enumerate(raw_header)}
        seen_rows: set[int] = set()
        count = 0
        for row_index, raw_row in enumerate(raw_reader):
            count += 1
            repairs = by_row.get(row_index, [])
            if repairs:
                seen_rows.add(row_index)
            row_deletes = [item for item in repairs if item.column == ""]
            if row_deletes:
                if len(row_deletes) != 1 or len(repairs) != 1:
                    raise PiRepairAuditError("row deletion cannot be combined with Cell repairs")
                expected_hash = "sha256:" + hashlib.sha256(
                    json.dumps(raw_row, ensure_ascii=True, separators=(",", ":")).encode()
                ).hexdigest()
                if row_deletes[0].original_value not in {"", expected_hash}:
                    raise PiRepairAuditError("row deletion original_value hash mismatch")
                continue
            expected = list(raw_row)
            for item in repairs:
                if item.column not in columns:
                    raise PiRepairAuditError(f"repair references unknown column: {item.column}")
                index = columns[item.column]
                if expected[index] != item.original_value:
                    raise PiRepairAuditError("repair original_value does not match raw input")
                expected[index] = item.selected_value
            actual = next(corrected_reader, None)
            if actual != expected:
                raise PiRepairAuditError(
                    "corrected raw contains undeclared or incorrect changes: "
                    f"{raw_path.name} row {row_index}"
                )
        if next(corrected_reader, None) is not None:
            raise PiRepairAuditError(f"corrected raw contains extra rows: {raw_path.name}")
        missing_rows = sorted(set(by_row) - seen_rows)
        if missing_rows:
            raise PiRepairAuditError(
                f"repair references missing rows: {raw_path.name} {missing_rows[:3]}"
            )
        return count


def _write_public_csv(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    rows = 0
    with source.open(encoding="utf-8-sig", newline="") as input_handle, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if not reader.fieldnames:
            raise PiRepairAuditError(f"CSV has no header: {source.name}")
        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            writer.writerow(
                {column: _public_value(column, str(value or "")) for column, value in row.items()}
            )
            rows += 1
    os.replace(temporary, destination)
    return {"rows": rows, "sha256": _sha256(destination), "bytes": destination.stat().st_size}


def _public_value(column: str, value: str) -> str:
    if not value or not _is_locator_column(column):
        return value
    namespace = "order" if column.lower() in {"orderid", "linkorderid"} else column.lower()
    digest = hashlib.sha256(f"pi-public-v1\0{namespace}\0{value}".encode()).hexdigest()
    if value.lstrip("-").isdigit():
        return str(int(digest[:15], 16))
    return f"token_{digest[:16]}"


def _is_locator_column(column: str) -> bool:
    lowered = column.lower()
    if lowered in _SEMANTIC_ID_COLUMNS:
        return False
    return lowered in _EXPLICIT_LOCATOR_COLUMNS or lowered.endswith("_id")


def _count_repair_opportunities(raw_root: str | Path) -> int:
    total = 0
    for path in _csv_files(_require_directory(raw_root, "raw root")).values():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                continue
            for _ in reader:
                total += len(header) + 1
    return total


def _csv_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.csv")):
        if path.is_symlink() or not path.is_file():
            raise PiRepairAuditError(f"CSV input is not a regular file: {path.name}")
        relative = path.relative_to(root).as_posix()
        result[relative] = path
    return result


def _csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(next(csv.reader(handle), []))


def _normalize_table_name(raw_file: str) -> str:
    value = raw_file.replace("\\", "/")
    if value.endswith(".csv.gz"):
        return value[:-7]
    if value.endswith(".csv"):
        return value[:-4]
    return value


def _normalize_csv_path(raw_file: str) -> str:
    value = raw_file.replace("\\", "/")
    if value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/")):
        raise PiRepairAuditError(f"unsafe table path: {raw_file}")
    if value.endswith(".csv.gz"):
        value = value[:-3]
    elif not value.endswith(".csv"):
        value += ".csv"
    return value


def _new_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise PiRepairAuditError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _require_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise PiRepairAuditError(f"{label} directory does not exist: {resolved}")
    return resolved


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PiRepairAuditError(f"{label} does not exist: {resolved}")
    return resolved


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: Path) -> str:
    return _sha256(path) if path.is_file() else ""


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return max(0.0, float(numerator) / float(denominator))


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, 6)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
