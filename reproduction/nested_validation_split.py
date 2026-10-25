"""Build a smaller hidden-validation split from an existing physical split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from reproduction.mimic_icu_mortality import directory_metadata_fingerprint
from workflow.reference_splits import (
    copy_mimic_raw_subset,
    copy_reference_table_for_keys,
)


IGNORED_NAMES = {".DS_Store", "__pycache__", ".pytest_cache"}
REFERENCE_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz")


def build_nested_validation_split(
    parent_split: str | Path,
    output_root: str | Path,
    *,
    validation_count: int = 20,
    parent_seed: int | None = None,
) -> dict[str, Any]:
    parent = Path(parent_split).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not parent.is_dir():
        raise ValueError(f"parent split does not exist: {parent}")
    if validation_count < 1:
        raise ValueError("validation_count must be positive")
    if output == parent or parent in output.parents:
        raise ValueError("output root cannot be inside the parent split")

    key_frames = {name: _read_keys(parent / name / "keys.csv") for name in ("train", "validation", "test")}
    _validate_parent_keys(key_frames)
    if validation_count > len(key_frames["validation"]):
        raise ValueError(
            f"validation_count exceeds parent validation size: {validation_count} > {len(key_frames['validation'])}"
        )
    identity = {
        "parent_split": str(parent),
        "validation_count": validation_count,
        "parent_seed": parent_seed,
        "selection_method": "ordered_prefix",
    }
    existing_manifest = output / "split_manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("identity") == identity and existing.get("status") == "SUCCESS":
            return {"status": "SUCCESS", "dataset_root": str(output), "manifest": existing}
        raise ValueError(f"existing nested split manifest does not match: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"nested split output is not empty: {output}")

    before = directory_metadata_fingerprint(parent)
    selected_validation = key_frames["validation"].iloc[:validation_count].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        prepared = Path(temporary) / output.name
        prepared.mkdir()
        _copy_split(parent / "train", prepared / "train")
        _copy_split(parent / "test", prepared / "test")
        validation = prepared / "validation"
        validation.mkdir()
        selected_validation.to_csv(validation / "keys.csv", index=False)

        raw_record = copy_mimic_raw_subset(
            parent / "validation/raw",
            validation / "raw",
            validation / "keys.csv",
            split_key="stay_id",
        )
        reference_record = _copy_reference_package(
            parent / "validation/reference_private",
            validation / "reference_private",
            selected_validation,
        )

        counts = {
            "train": len(key_frames["train"]),
            "validation": len(selected_validation),
            "test": len(key_frames["test"]),
        }
        for split in ("train", "validation", "test"):
            record = {
                "schema_version": 1,
                "split": split,
                "key_count": counts[split],
                "keys_sha256": _sha256(prepared / split / "keys.csv"),
                "raw_root": str(output / split / "raw"),
                "reference_root": str(
                    output / split / ("reference" if split == "train" else "reference_private")
                ),
                "copy_mode": "filtered_physical_copy" if split == "validation" else "physical_copy",
            }
            if split == "validation":
                record["raw_record"] = raw_record
                record["reference_record"] = reference_record
            _write_json(prepared / split / "split_record.json", record)

        after = directory_metadata_fingerprint(parent)
        report = _validate_prepared_split(
            prepared,
            parent,
            key_frames,
            selected_validation,
            before=before,
            after=after,
        )
        _write_json(prepared / "split_validation_report.json", report)
        if not report["passed"]:
            failures = [item["name"] for item in report["checks"] if not item["passed"]]
            raise RuntimeError("nested split validation failed: " + ", ".join(failures))

        manifest = {
            "schema_version": 1,
            "status": "SUCCESS",
            "workflow": "nested_validation_split",
            "identity": identity,
            "parent_split": str(parent),
            "counts": counts,
            "split_key": "stay_id",
            "selection": {
                "method": "ordered_prefix",
                "parent_seed": parent_seed,
                "parent_validation_count": len(key_frames["validation"]),
                "selected_validation_count": validation_count,
            },
            "paths": {
                split: {
                    "raw": str(output / split / "raw"),
                    "reference": str(
                        output / split / ("reference" if split == "train" else "reference_private")
                    ),
                    "keys": str(output / split / "keys.csv"),
                }
                for split in ("train", "validation", "test")
            },
            "parent_metadata_before": before,
            "parent_metadata_after": after,
        }
        _write_json(prepared / "split_manifest.json", manifest)
        if output.exists():
            output.rmdir()
        os.replace(prepared, output)
    return {"status": "SUCCESS", "dataset_root": str(output), "manifest": manifest}


def _read_keys(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"split keys file is missing: {path}")
    frame = pd.read_csv(path, dtype={"stay_id": "string"})
    if list(frame.columns) != ["stay_id"]:
        raise ValueError(f"keys file must contain only stay_id: {path}")
    if frame["stay_id"].isna().any() or frame["stay_id"].duplicated().any():
        raise ValueError(f"keys file contains missing or duplicate stay_id values: {path}")
    return frame


def _validate_parent_keys(frames: dict[str, pd.DataFrame]) -> None:
    sets = {name: set(frame["stay_id"].astype(str)) for name, frame in frames.items()}
    overlaps = {
        f"{left}_{right}": sorted(sets[left] & sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        if sets[left] & sets[right]
    }
    if overlaps:
        raise ValueError(f"parent split keys overlap: {overlaps}")


def _copy_split(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"parent split directory is missing: {source}")
    shutil.copytree(source, destination, symlinks=False, ignore=_ignore_entries)


def _ignore_entries(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORED_NAMES or name.endswith(".pyc")}


def _copy_reference_package(source: Path, destination: Path, keys: pd.DataFrame) -> dict[str, Any]:
    if not source.is_dir():
        raise ValueError(f"parent validation reference is missing: {source}")
    allowed = set(keys["stay_id"].astype(str))
    files: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name in IGNORED_NAMES or not path.name.lower().endswith(REFERENCE_SUFFIXES):
            continue
        relative = path.relative_to(source)
        destination_path = destination / relative
        rows = copy_reference_table_for_keys(
            path,
            destination_path,
            split_key="stay_id",
            allowed=allowed,
        )
        files.append({"relative_path": relative.as_posix(), "row_count": rows})
    return {"file_count": len(files), "files": files}


def _validate_prepared_split(
    prepared: Path,
    parent: Path,
    parent_keys: dict[str, pd.DataFrame],
    selected_validation: pd.DataFrame,
    *,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed, "expected": expected})

    expected_counts = {
        "train": len(parent_keys["train"]),
        "validation": len(selected_validation),
        "test": len(parent_keys["test"]),
    }
    actual_frames = {split: _read_keys(prepared / split / "keys.csv") for split in expected_counts}
    actual_counts = {split: len(frame) for split, frame in actual_frames.items()}
    add("key_counts", actual_counts == expected_counts, actual_counts, expected_counts)
    sets = {split: set(frame["stay_id"].astype(str)) for split, frame in actual_frames.items()}
    overlap_count = sum(
        len(sets[left] & sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    add("key_disjointness", overlap_count == 0, overlap_count, 0)

    icustays = _find_table(prepared / "validation/raw/icu", "icustays.csv")
    raw_stays = set(pd.read_csv(icustays, dtype={"stay_id": "string"})["stay_id"].dropna().astype(str))
    add("raw_key_coverage", raw_stays == sets["validation"], sorted(raw_stays), sorted(sets["validation"]))
    reference_count = sum(
        1
        for path in (prepared / "validation/reference_private").rglob("*")
        if path.is_file() and path.name not in IGNORED_NAMES
    )
    add("reference_file_count", reference_count == 17, reference_count, 17)
    links = [path.relative_to(prepared).as_posix() for path in prepared.rglob("*") if path.is_symlink()]
    add("physical_copy", not links, links, [])
    add("parent_unchanged", before == after, after, before)
    return {
        "schema_version": 1,
        "status": "SUCCESS" if all(item["passed"] for item in checks) else "FAILED",
        "passed": all(item["passed"] for item in checks),
        "parent_split": str(parent),
        "checks": checks,
    }


def _find_table(root: Path, name: str) -> Path:
    plain = root / name
    compressed = root / f"{name}.gz"
    if plain.is_file():
        return plain
    if compressed.is_file():
        return compressed
    raise ValueError(f"required table is missing: {plain}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a physical nested validation split.")
    parser.add_argument("--parent-split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--parent-seed", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_nested_validation_split(
        args.parent_split,
        args.output_root,
        validation_count=args.validation_count,
        parent_seed=args.parent_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
