from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DIRTY_URL = (
    "https://raw.githubusercontent.com/BigDaMa/raha/master/"
    "datasets/hospital/dirty.csv"
)
DEFAULT_CLEAN_URL = (
    "https://raw.githubusercontent.com/BigDaMa/raha/master/"
    "datasets/hospital/clean.csv"
)
SELECTION = "curated_greedy_changed_column_coverage_v1"
KEY_COLUMN = "index"
TABLE_NAME = "hospital.csv"
EXPECTED_DIRTY_FIELDS = [
    "index",
    "provider_number",
    "name",
    "address_1",
    "address_2",
    "address_3",
    "city",
    "state",
    "zip",
    "county",
    "phone",
    "type",
    "owner",
    "emergency_service",
    "condition",
    "measure_code",
    "measure_name",
    "score",
    "sample",
    "state_average",
]
EXPECTED_CLEAN_FIELDS = [
    "index",
    "ProviderNumber",
    "HospitalName",
    "Address1",
    "Address2",
    "Address3",
    "City",
    "State",
    "ZipCode",
    "CountyName",
    "PhoneNumber",
    "HospitalType",
    "HospitalOwner",
    "EmergencyService",
    "Condition",
    "MeasureCode",
    "MeasureName",
    "Score",
    "Sample",
    "Stateavg",
]


def build_hospital_dataset(
    *,
    dirty_source: str | Path,
    clean_source: str | Path,
    output_dir: str | Path,
    train_count: int = 10,
    validation_count: int = 0,
    validation_seed: int = 666,
    expected_rows: int = 1_000,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if train_count < 1 or train_count >= expected_rows:
        raise ValueError("train_count must be positive and smaller than expected_rows")
    if validation_count < 0 or train_count + validation_count >= expected_rows:
        raise ValueError("validation_count must leave at least one correction row")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"dataset output must be new and empty: {output}")

    dirty_bytes, dirty_identity = _read_source(dirty_source)
    clean_bytes, clean_identity = _read_source(clean_source)
    dirty_fields, dirty_rows = _parse_csv(dirty_bytes)
    clean_fields, clean_rows = _parse_csv(clean_bytes)
    clean_rows, clean_schema_mode = _canonicalize_clean_schema(
        dirty_fields,
        clean_fields,
        clean_rows,
    )
    if KEY_COLUMN not in dirty_fields:
        raise ValueError(f"missing key column: {KEY_COLUMN}")
    if len(dirty_rows) != expected_rows or len(clean_rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} rows, got dirty={len(dirty_rows)} "
            f"clean={len(clean_rows)}"
        )

    dirty_by_key = _rows_by_key(dirty_rows)
    clean_by_key = _rows_by_key(clean_rows)
    if list(dirty_by_key) != list(clean_by_key):
        raise ValueError("dirty and clean keys or row order differ")

    differences = _cell_differences(dirty_fields, dirty_rows, clean_rows)
    changed_columns_by_key: dict[str, set[str]] = {}
    for item in differences:
        changed_columns_by_key.setdefault(item[KEY_COLUMN], set()).add(item["column"])
    if len(changed_columns_by_key) < train_count:
        raise ValueError("not enough dirty rows to construct curated examples")
    train_keys = _select_train_keys(changed_columns_by_key, train_count)
    train_key_set = set(train_keys)
    remaining_keys = [
        row[KEY_COLUMN] for row in dirty_rows if row[KEY_COLUMN] not in train_key_set
    ]
    validation_keys = _select_validation_keys(
        remaining_keys,
        validation_count,
        validation_seed,
    )
    validation_key_set = set(validation_keys)
    train_dirty = [row for row in dirty_rows if row[KEY_COLUMN] in train_key_set]
    train_clean = [row for row in clean_rows if row[KEY_COLUMN] in train_key_set]
    validation_dirty = [row for row in dirty_rows if row[KEY_COLUMN] in validation_key_set]
    validation_clean = [row for row in clean_rows if row[KEY_COLUMN] in validation_key_set]
    held_out_keys = train_key_set | validation_key_set
    correction_dirty = [row for row in dirty_rows if row[KEY_COLUMN] not in held_out_keys]
    correction_clean = [row for row in clean_rows if row[KEY_COLUMN] not in held_out_keys]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp_text:
        prepared = Path(temp_text) / output.name
        prepared.mkdir()
        _write_csv(prepared / "train/raw" / TABLE_NAME, dirty_fields, train_dirty)
        _write_csv(prepared / "train/reference" / TABLE_NAME, dirty_fields, train_clean)
        _write_csv(prepared / "train_only_replay/raw" / TABLE_NAME, dirty_fields, train_dirty)
        _write_csv(prepared / "train_only_replay/gold" / TABLE_NAME, dirty_fields, train_clean)
        if validation_count:
            _write_csv(
                prepared / "validation/raw" / TABLE_NAME,
                dirty_fields,
                validation_dirty,
            )
            _write_csv(
                prepared / "validation/gold" / TABLE_NAME,
                dirty_fields,
                validation_clean,
            )
        _write_csv(prepared / "correction/raw" / TABLE_NAME, dirty_fields, correction_dirty)
        _write_csv(
            prepared / "correction/reference_private" / TABLE_NAME,
            dirty_fields,
            correction_clean,
        )
        _write_keys(prepared / "train/keys.csv", train_dirty)
        _write_keys(prepared / "train_only_replay/keys.csv", train_dirty)
        if validation_count:
            _write_keys(prepared / "validation/keys.csv", validation_dirty)
        _write_keys(prepared / "correction/keys.csv", correction_dirty)
        _write_differences(
            prepared / "host_private/cell_diff.jsonl",
            differences,
            train_key_set,
            validation_key_set,
        )

        train_difference_count = sum(item[KEY_COLUMN] in train_key_set for item in differences)
        validation_difference_count = sum(
            item[KEY_COLUMN] in validation_key_set for item in differences
        )
        manifest = {
            "schema_version": 1,
            "status": "SUCCESS",
            "workflow": (
                "raha_hospital_curated_10_20_holdout"
                if validation_count
                else "raha_hospital_curated_10shot"
            ),
            "benchmark_label": (
                "curated 10-shot with independent 20-row validation"
                if validation_count
                else "curated 10-shot internal audit"
            ),
            "table": TABLE_NAME,
            "key_column": KEY_COLUMN,
            "clean_schema_mode": clean_schema_mode,
            "selection": SELECTION,
            "validation_selection": (
                "seeded_stable_hash_over_remaining_rows_v1"
                if validation_count
                else "none"
            ),
            "validation_seed": validation_seed if validation_count else None,
            "source": {
                "dirty": dirty_identity,
                "clean": clean_identity,
                "dirty_sha256": _sha256(dirty_bytes),
                "clean_sha256": _sha256(clean_bytes),
            },
            "counts": {
                "all": expected_rows,
                "train": len(train_dirty),
                "validation": len(validation_dirty),
                "correction": len(correction_dirty),
            },
            "difference_counts": {
                "cells": len(differences),
                "rows": len(changed_columns_by_key),
                "train_cells": train_difference_count,
                "validation_cells": validation_difference_count,
                "correction_cells": (
                    len(differences)
                    - train_difference_count
                    - validation_difference_count
                ),
            },
            "train_indices": [row[KEY_COLUMN] for row in train_dirty],
            "validation_indices": [row[KEY_COLUMN] for row in validation_dirty],
            "paths": {
                "train_raw": "train/raw",
                "train_reference": "train/reference",
                "train_only_replay_raw": "train_only_replay/raw",
                "train_only_replay_gold": "train_only_replay/gold",
                "validation_raw": "validation/raw" if validation_count else None,
                "validation_gold": "validation/gold" if validation_count else None,
                "correction_raw": "correction/raw",
                "correction_reference_private": "correction/reference_private",
            },
        }
        if not validation_count:
            manifest.pop("validation_selection")
            manifest.pop("validation_seed")
            manifest.pop("validation_indices")
            manifest["counts"].pop("validation")
            manifest["difference_counts"].pop("validation_cells")
            manifest["paths"].pop("validation_raw")
            manifest["paths"].pop("validation_gold")
        manifest["file_sha256"] = {
            path.relative_to(prepared).as_posix(): _sha256(path.read_bytes())
            for path in sorted(prepared.rglob("*"))
            if path.is_file()
        }
        _write_json(prepared / "split_manifest.json", manifest)
        if output.exists():
            output.rmdir()
        os.replace(prepared, output)
    return manifest


def _read_source(source: str | Path) -> tuple[bytes, str]:
    text = str(source)
    if text.startswith(("https://", "http://")):
        request = urllib.request.Request(text, headers={"User-Agent": "v2-clean-benchmark/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read(), text
    path = Path(source).expanduser().resolve()
    return path.read_bytes(), str(path)


def _parse_csv(payload: bytes) -> tuple[list[str], list[dict[str, str]]]:
    with io.StringIO(payload.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError("CSV has no header")
        rows = [{field: str(row.get(field) or "") for field in fields} for row in reader]
    return fields, rows


def _rows_by_key(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row[KEY_COLUMN]
        if not key or key in result:
            raise ValueError(f"missing or duplicate {KEY_COLUMN}: {key!r}")
        result[key] = row
    return result


def _canonicalize_clean_schema(
    dirty_fields: list[str],
    clean_fields: list[str],
    clean_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    if dirty_fields == clean_fields:
        return clean_rows, "identical"
    if dirty_fields != EXPECTED_DIRTY_FIELDS or clean_fields != EXPECTED_CLEAN_FIELDS:
        raise ValueError("dirty and clean schemas differ outside the approved Hospital mapping")
    remapped = [
        {
            dirty_column: row[clean_column]
            for dirty_column, clean_column in zip(dirty_fields, clean_fields, strict=True)
        }
        for row in clean_rows
    ]
    return remapped, "raha_official_positional_mapping_v1"


def _cell_differences(
    fields: list[str],
    dirty_rows: list[dict[str, str]],
    clean_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    differences: list[dict[str, str]] = []
    for dirty, clean in zip(dirty_rows, clean_rows, strict=True):
        if dirty[KEY_COLUMN] != clean[KEY_COLUMN]:
            raise ValueError("dirty and clean row keys differ")
        for column in fields:
            if column == KEY_COLUMN or dirty[column] == clean[column]:
                continue
            differences.append(
                {
                    KEY_COLUMN: dirty[KEY_COLUMN],
                    "column": column,
                    "dirty_value": dirty[column],
                    "clean_value": clean[column],
                }
            )
    return differences


def _select_train_keys(changed_columns_by_key: dict[str, set[str]], count: int) -> list[str]:
    remaining = set(changed_columns_by_key)
    covered: set[str] = set()
    selected: list[str] = []
    while len(selected) < count:
        ranked = sorted(
            remaining,
            key=lambda key: (
                -len(changed_columns_by_key[key] - covered),
                -len(changed_columns_by_key[key]),
                _stable_key(key),
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        remaining.remove(chosen)
        covered.update(changed_columns_by_key[chosen])
    return selected


def _select_validation_keys(keys: list[str], count: int, seed: int) -> list[str]:
    if count == 0:
        return []
    ranked = sorted(
        keys,
        key=lambda key: (
            hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest(),
            _stable_key(key),
        ),
    )
    return ranked[:count]


def _stable_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_keys(path: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(path, [KEY_COLUMN], [{KEY_COLUMN: row[KEY_COLUMN]} for row in rows])


def _write_differences(
    path: Path,
    differences: list[dict[str, str]],
    train_keys: set[str],
    validation_keys: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in differences:
            key = item[KEY_COLUMN]
            split = (
                "train"
                if key in train_keys
                else "validation"
                if key in validation_keys
                else "correction"
            )
            payload = {**item, "split": split}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a curated-train Raha Hospital benchmark split."
    )
    parser.add_argument("--dirty-source", default=DEFAULT_DIRTY_URL)
    parser.add_argument("--clean-source", default=DEFAULT_CLEAN_URL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--validation-count", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=666)
    parser.add_argument("--expected-rows", type=int, default=1_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_hospital_dataset(
        dirty_source=args.dirty_source,
        clean_source=args.clean_source,
        output_dir=args.output_dir,
        train_count=args.train_count,
        validation_count=args.validation_count,
        validation_seed=args.validation_seed,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
