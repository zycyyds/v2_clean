from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from reproduction.mimic_icu_mortality import GOLD_RELATIVE_PATHS


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _tree_digest(root: Path, *, ignore_metadata: bool = False) -> str:
    digest = hashlib.sha256()
    ignored = {".DS_Store", "split_record.json"} if ignore_metadata else set()
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.name not in ignored
    ):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_parent_split(root: Path) -> Path:
    for split, keys in (
        ("train", [1, 2]),
        ("validation", [101, 102, 103, 104]),
        ("test", [201, 202, 203]),
    ):
        _write_csv(root / split / "keys.csv", [{"stay_id": key} for key in keys])
        reference_name = "reference" if split == "train" else "reference_private"
        reference = root / split / reference_name
        for relative in GOLD_RELATIVE_PATHS:
            path = reference / relative.removeprefix("data/")
            if "summary/" in relative:
                _write_csv(path, [{"itemid": 1, "total_count": len(keys)}])
            elif relative.endswith("labels.csv"):
                _write_csv(path, [{"stay_id": key, "label": key % 2} for key in keys])
            else:
                _write_csv(path, [{"stay_id": key, "value": key * 10} for key in keys])

        raw = root / split / "raw"
        _write_csv(
            raw / "icu/icustays.csv",
            [
                {"subject_id": key + 1000, "hadm_id": key + 2000, "stay_id": key}
                for key in keys
            ],
        )
        _write_csv(
            raw / "hosp/admissions.csv",
            [{"subject_id": key + 1000, "hadm_id": key + 2000} for key in keys],
        )
        _write_csv(
            raw / "icu/chartevents.csv",
            [{"stay_id": key, "itemid": 220045, "valuenum": key} for key in keys],
        )
        _write_csv(
            raw / "hosp/labevents.csv",
            [{"subject_id": key + 1000, "hadm_id": key + 2000, "value": key} for key in keys],
        )
        _write_csv(raw / "icu/d_items.csv", [{"itemid": 220045, "label": "Heart Rate"}])
        (raw / ".DS_Store").write_bytes(b"ignored")
    return root


def test_nested_split_physically_copies_train_test_and_filters_validation(tmp_path: Path) -> None:
    from reproduction.nested_validation_split import build_nested_validation_split

    parent = _build_parent_split(tmp_path / "parent")
    before = _tree_digest(parent)
    output = tmp_path / "nested"

    result = build_nested_validation_split(
        parent,
        output,
        validation_count=2,
        parent_seed=1759733077,
    )

    assert result["status"] == "SUCCESS"
    assert _tree_digest(parent) == before
    assert pd.read_csv(output / "validation/keys.csv")["stay_id"].tolist() == [101, 102]
    assert pd.read_csv(output / "validation/raw/icu/icustays.csv")["stay_id"].tolist() == [101, 102]
    assert pd.read_csv(output / "validation/raw/hosp/labevents.csv")["hadm_id"].tolist() == [2101, 2102]
    assert pd.read_csv(output / "validation/reference_private/csv/labels.csv")["stay_id"].tolist() == [101, 102]
    assert pd.read_csv(output / "validation/reference_private/summary/chart_summary.csv").shape[0] == 1
    assert (output / "validation/raw/icu/d_items.csv").is_file()
    assert not any(path.name == ".DS_Store" for path in output.rglob("*"))
    assert not any(path.is_symlink() for path in output.rglob("*"))

    assert _tree_digest(output / "train", ignore_metadata=True) == _tree_digest(
        parent / "train", ignore_metadata=True
    )
    assert _tree_digest(output / "test", ignore_metadata=True) == _tree_digest(
        parent / "test", ignore_metadata=True
    )
    reference_files = [
        path
        for path in (output / "validation/reference_private").rglob("*")
        if path.is_file()
    ]
    assert len(reference_files) == 17

    manifest = json.loads((output / "split_manifest.json").read_text())
    report = json.loads((output / "split_validation_report.json").read_text())
    assert manifest["counts"] == {"train": 2, "validation": 2, "test": 3}
    assert manifest["selection"]["method"] == "ordered_prefix"
    assert manifest["selection"]["parent_seed"] == 1759733077
    assert report["passed"] is True
    assert {item["name"] for item in report["checks"] if item["passed"]} >= {
        "key_counts",
        "key_disjointness",
        "raw_key_coverage",
        "reference_file_count",
        "physical_copy",
        "parent_unchanged",
    }
    for split in ("train", "validation", "test"):
        assert (output / split / "split_record.json").is_file()


def test_nested_split_reuses_matching_manifest_and_rejects_different_identity(tmp_path: Path) -> None:
    from reproduction.nested_validation_split import build_nested_validation_split

    parent = _build_parent_split(tmp_path / "parent")
    output = tmp_path / "nested"
    first = build_nested_validation_split(parent, output, validation_count=2, parent_seed=7)
    second = build_nested_validation_split(parent, output, validation_count=2, parent_seed=7)

    assert second["manifest"] == first["manifest"]
    with pytest.raises(ValueError, match="existing nested split manifest"):
        build_nested_validation_split(parent, output, validation_count=3, parent_seed=7)


def test_nested_split_rejects_overlapping_parent_keys(tmp_path: Path) -> None:
    from reproduction.nested_validation_split import build_nested_validation_split

    parent = _build_parent_split(tmp_path / "parent")
    _write_csv(parent / "test/keys.csv", [{"stay_id": 101}, {"stay_id": 202}])

    with pytest.raises(ValueError, match="overlap"):
        build_nested_validation_split(parent, tmp_path / "nested", validation_count=2)
