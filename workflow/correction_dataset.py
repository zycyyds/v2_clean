from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


IGNORED_ARCHIVE_NAMES = {".DS_Store"}


def build_correction_dataset(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    train_count: int = 10,
) -> dict[str, Any]:
    archive = Path(archive_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not archive.is_file():
        raise ValueError(f"correction source archive does not exist: {archive}")
    if train_count < 1:
        raise ValueError("train_count must be positive")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    existing_manifest = output / "split_manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if (
            existing.get("source_archive_sha256") == archive_hash
            and int(existing.get("counts", {}).get("train") or 0) == train_count
        ):
            return {"status": "SUCCESS", "dataset_root": str(output), "manifest": existing}
        raise ValueError(f"existing correction dataset manifest does not match: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"correction dataset output is not empty: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp_text:
        prepared = Path(temp_text) / output.name
        prepared.mkdir()
        with zipfile.ZipFile(archive) as source:
            clean_files, dirty_files = _package_files(source)
            if clean_files != dirty_files:
                raise ValueError("clean and dirty package file lists differ")
            modifications, modification_fields = _load_modifications(source)
            canonical_stays = _canonical_modification_stays(source, modifications)
            for row, stay_id in zip(modifications, canonical_stays, strict=True):
                row["canonical_stay_id"] = stay_id

            cohort_relative = _cohort_relative_path(clean_files)
            cohort_rows = list(_iter_archive_table(source, f"clean/{cohort_relative}"))
            all_stays = sorted(
                {str(row.get("stay_id") or "").strip() for row in cohort_rows if row.get("stay_id")},
                key=_stable_key,
            )
            if train_count >= len(all_stays):
                raise ValueError(
                    f"train_count must be smaller than the number of stays: {len(all_stays)}"
                )
            train_stays = _select_training_stays(modifications, all_stays, train_count)
            correction_stays = set(all_stays) - train_stays

            _write_keys(prepared / "train/keys.csv", train_stays)
            _write_keys(prepared / "correction/keys.csv", correction_stays)
            dirty_overrides = _dirty_row_split_overrides(modifications)
            for relative in clean_files:
                _partition_archive_file(
                    source,
                    member=f"clean/{relative}",
                    relative=relative,
                    train_target=prepared / "train/reference" / relative,
                    correction_target=prepared / "correction/reference_private" / relative,
                    train_stays=train_stays,
                    row_stay_overrides={},
                )
                _partition_archive_file(
                    source,
                    member=f"dirty/{relative}",
                    relative=relative,
                    train_target=prepared / "train/raw" / relative,
                    correction_target=prepared / "correction/raw" / relative,
                    train_stays=train_stays,
                    row_stay_overrides=dirty_overrides.get(relative, {}),
                )

            host = prepared / "host_private"
            _write_modification_partition(
                host / "train_modification_log.csv",
                modifications,
                [
                    *modification_fields,
                    "canonical_stay_id",
                    "clean_row_json",
                    "dirty_row_json",
                ],
                train_stays,
                include=True,
            )
            _write_modification_partition(
                host / "correction_modification_log.csv",
                modifications,
                [
                    *modification_fields,
                    "canonical_stay_id",
                    "clean_row_json",
                    "dirty_row_json",
                ],
                train_stays,
                include=False,
            )
            summary_member = "ground_truth/summary.json"
            if summary_member in source.namelist():
                (host / "source_summary.json").parent.mkdir(parents=True, exist_ok=True)
                (host / "source_summary.json").write_bytes(source.read(summary_member))

        manifest = {
            "schema_version": 1,
            "status": "SUCCESS",
            "workflow": "reference-guided-correction",
            "source_archive": str(archive),
            "source_archive_sha256": archive_hash,
            "split_key": "stay_id",
            "selection": "deterministic_greedy_difference_coverage",
            "counts": {"train": len(train_stays), "correction": len(correction_stays)},
            "file_count": len(clean_files),
            "paths": {
                "train_raw": str(output / "train/raw"),
                "train_reference": str(output / "train/reference"),
                "train_keys": str(output / "train/keys.csv"),
                "correction_raw": str(output / "correction/raw"),
                "correction_reference_private": str(output / "correction/reference_private"),
                "correction_keys": str(output / "correction/keys.csv"),
            },
        }
        _write_json(prepared / "split_manifest.json", manifest)
        if output.exists():
            output.rmdir()
        os.replace(prepared, output)
    return {"status": "SUCCESS", "dataset_root": str(output), "manifest": manifest}


def _package_files(source: zipfile.ZipFile) -> tuple[list[str], list[str]]:
    names = [name for name in source.namelist() if not name.endswith("/")]

    def collect(prefix: str) -> list[str]:
        return sorted(
            name[len(prefix) :]
            for name in names
            if name.startswith(prefix)
            and not name.startswith("__MACOSX/")
            and Path(name).name not in IGNORED_ARCHIVE_NAMES
        )

    return collect("clean/"), collect("dirty/")


def _load_modifications(source: zipfile.ZipFile) -> tuple[list[dict[str, str]], list[str]]:
    with source.open("ground_truth/modification_log.csv") as binary:
        with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            reader = csv.DictReader(text)
            rows = [dict(row) for row in reader]
            return rows, list(reader.fieldnames or [])


def _canonical_modification_stays(
    source: zipfile.ZipFile,
    modifications: list[dict[str, str]],
) -> list[str]:
    clean_targets: dict[str, set[int]] = defaultdict(set)
    dirty_targets: dict[str, set[int]] = defaultdict(set)
    for row in modifications:
        dirty_targets[row["file"]].add(_required_index(row.get("dirty_row_index")))
        clean_index = _optional_index(row.get("clean_row_index"))
        if clean_index is not None:
            clean_targets[row["file"]].add(clean_index)
    clean_rows = {
        relative: _rows_at_indices(source, f"clean/{relative}", indices)
        for relative, indices in clean_targets.items()
    }
    dirty_rows = {
        relative: _rows_at_indices(source, f"dirty/{relative}", indices)
        for relative, indices in dirty_targets.items()
    }
    canonical: list[str] = []
    for row in modifications:
        clean_index = _optional_index(row.get("clean_row_index"))
        dirty_index = _required_index(row.get("dirty_row_index"))
        dirty_row = dirty_rows[row["file"]][dirty_index]
        if clean_index is not None:
            clean_row = clean_rows[row["file"]][clean_index]
            source_row = clean_row
            row["clean_row_json"] = _row_json(clean_row)
        else:
            source_row = dirty_row
            row["clean_row_json"] = ""
        row["dirty_row_json"] = _row_json(dirty_row)
        stay_id = str(source_row.get("stay_id") or row.get("stay_id") or "").strip()
        if not stay_id:
            raise ValueError(
                f"could not resolve canonical stay_id for {row['file']} row {row.get('dirty_row_index')}"
            )
        canonical.append(stay_id)
    return canonical


def _row_json(row: dict[str, str]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _select_training_stays(
    modifications: list[dict[str, str]],
    all_stays: list[str],
    train_count: int,
) -> set[str]:
    by_stay: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in modifications:
        by_stay[row["canonical_stay_id"]].append(row)
    candidates = set(by_stay) & set(all_stays)
    selected: list[str] = []
    covered_subtypes: set[str] = set()
    covered_classes: set[str] = set()
    while candidates and len(selected) < train_count:
        ranked = sorted(
            candidates,
            key=lambda stay: (
                -len(
                    {row.get("error_subtype", "") for row in by_stay[stay]}
                    - covered_subtypes
                ),
                -len(
                    {row.get("error_class", "") for row in by_stay[stay]}
                    - covered_classes
                ),
                -len(by_stay[stay]),
                _stable_key(stay),
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        candidates.remove(chosen)
        covered_subtypes.update(row.get("error_subtype", "") for row in by_stay[chosen])
        covered_classes.update(row.get("error_class", "") for row in by_stay[chosen])
    for stay in all_stays:
        if len(selected) >= train_count:
            break
        if stay not in selected:
            selected.append(stay)
    if len(selected) != train_count:
        raise ValueError(f"could not select {train_count} train stays")
    return set(selected)


def _dirty_row_split_overrides(
    modifications: list[dict[str, str]],
) -> dict[str, dict[int, str]]:
    overrides: dict[str, dict[int, str]] = defaultdict(dict)
    for row in modifications:
        index = _required_index(row.get("dirty_row_index"))
        canonical = row["canonical_stay_id"]
        existing = overrides[row["file"]].get(index)
        if existing is not None and existing != canonical:
            raise ValueError(f"conflicting canonical stay assignment for {row['file']} row {index}")
        overrides[row["file"]][index] = canonical
    return overrides


def _partition_archive_file(
    source: zipfile.ZipFile,
    *,
    member: str,
    relative: str,
    train_target: Path,
    correction_target: Path,
    train_stays: set[str],
    row_stay_overrides: dict[int, str],
) -> None:
    if Path(relative).suffix.casefold() not in {".csv", ".gz"}:
        payload = source.read(member)
        for target in (train_target, correction_target):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return
    with _archive_text(source, member) as text:
        reader = csv.DictReader(text)
        fields = list(reader.fieldnames or [])
        if "stay_id" not in fields:
            payload = source.read(member)
            for target in (train_target, correction_target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            return
        with _csv_output(train_target, fields) as train_writer, _csv_output(
            correction_target, fields
        ) as correction_writer:
            for index, row in enumerate(reader):
                canonical = row_stay_overrides.get(index) or str(row.get("stay_id") or "").strip()
                (train_writer if canonical in train_stays else correction_writer).writerow(row)


def _write_modification_partition(
    path: Path,
    modifications: list[dict[str, str]],
    fieldnames: list[str],
    train_stays: set[str],
    *,
    include: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in modifications:
            if (row["canonical_stay_id"] in train_stays) is include:
                writer.writerow(row)


def _rows_at_indices(
    source: zipfile.ZipFile,
    member: str,
    indices: set[int],
) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with _archive_text(source, member) as text:
        for index, row in enumerate(csv.DictReader(text)):
            if index in indices:
                rows[index] = dict(row)
                if len(rows) == len(indices):
                    break
    missing = indices - set(rows)
    if missing:
        raise ValueError(f"archive rows are missing from {member}: {sorted(missing)[:10]}")
    return rows


def _iter_archive_table(source: zipfile.ZipFile, member: str) -> Iterator[dict[str, str]]:
    with _archive_text(source, member) as text:
        for row in csv.DictReader(text):
            yield dict(row)


@contextmanager
def _archive_text(source: zipfile.ZipFile, member: str) -> Iterator[TextIO]:
    binary = source.open(member)
    decompressed = gzip.GzipFile(fileobj=binary) if member.endswith(".gz") else binary
    text = io.TextIOWrapper(decompressed, encoding="utf-8-sig", newline="")
    try:
        yield text
    finally:
        text.close()


@contextmanager
def _csv_output(path: Path, fields: list[str]) -> Iterator[csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    binary = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) if path.suffix == ".gz" else raw
    text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    try:
        yield writer
    finally:
        text.flush()
        text.close()


def _write_keys(path: Path, values: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stay_id"])
        writer.writerows([[value] for value in sorted(values, key=_stable_key)])


def _cohort_relative_path(files: list[str]) -> str:
    candidates = [name for name in files if name.startswith("cohort/") and name.endswith(".csv.gz")]
    if not candidates:
        raise ValueError("clean package does not contain a cohort CSV")
    return candidates[0]


def _optional_index(value: object) -> int | None:
    text = str(value or "").strip()
    return None if not text else int(float(text))


def _required_index(value: object) -> int:
    index = _optional_index(value)
    if index is None:
        raise ValueError("modification row is missing dirty_row_index")
    return index


def _stable_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
