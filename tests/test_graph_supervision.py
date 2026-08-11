from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graph.builder import build_graph
from graph.supervision import build_supervised_graph


LOG_FIELDS = [
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
]


def _csv(root: Path, relative: str, rows: list[dict[str, object]]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    clean = tmp_path / "clean_raw"
    dirty = tmp_path / "dirty_raw"
    patients = []
    admissions = []
    stays = []
    diagnoses = []
    chart = []
    inputs = []
    outputs = []
    for offset in range(4):
        subject, hadm, stay = 100 + offset, 200 + offset, 300 + offset
        patients.append({"subject_id": subject, "gender": "M", "anchor_age": 50 + offset, "dod": ""})
        admissions.append({"subject_id": subject, "hadm_id": hadm, "insurance": "Medicare"})
        stays.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "intime": "2020-01-01 00:00:00", "outtime": "2020-01-02 00:00:00"})
        diagnoses.append({"subject_id": subject, "hadm_id": hadm, "icd_code": f"I50{offset}", "icd_version": 10})
        for event in range(12):
            chart.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "charttime": f"2020-01-01 {event:02d}:00:00", "itemid": 220045, "valuenum": 70 + event})
        for event in range(8):
            inputs.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "starttime": "2020-01-01 01:00:00", "endtime": "2020-01-01 02:00:00", "itemid": 225158, "amount": 1 + event})
        for event in range(4):
            outputs.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay, "charttime": "2020-01-01 03:00:00", "itemid": 226559, "value": 10 + event})
    for relative, rows in {
        "hosp/patients.csv": patients,
        "hosp/admissions.csv": admissions,
        "hosp/diagnoses_icd.csv": diagnoses,
        "icu/icustays.csv": stays,
        "icu/chartevents.csv": chart,
        "icu/inputevents.csv": inputs,
        "icu/outputevents.csv": outputs,
        "icu/d_items.csv": [{"itemid": 220045, "label": "Heart Rate"}],
    }.items():
        _csv(clean, relative, rows)
        _csv(dirty, relative, rows)

    dirty_chart = pd.read_csv(dirty / "icu/chartevents.csv", dtype=str)
    clean_value = dirty_chart.loc[0, "valuenum"]
    dirty_chart.loc[0, "valuenum"] = "7000"
    duplicate = dirty_chart.iloc[[1]].copy()
    dirty_chart = pd.concat([dirty_chart.iloc[:2], duplicate, dirty_chart.iloc[2:]], ignore_index=True)
    dirty_chart.to_csv(dirty / "icu/chartevents.csv", index=False)

    log = tmp_path / "host_private/base.csv"
    log.parent.mkdir()
    with log.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerow({
            "source_error_id": "base-1", "canonical_stay_id": "300", "error_class": "class1_intra_table",
            "error_subtype": "unit_scale_error", "target_effect_contract": "scale", "operation": "cell_update",
            "raw_file": "icu/chartevents.csv", "raw_row_index": "0", "column": "valuenum",
            "clean_value": clean_value, "dirty_value": "700",
        })
        writer.writerow({
            "source_error_id": "base-2", "canonical_stay_id": "300", "error_class": "class1_intra_table",
            "error_subtype": "duplicate_row", "target_effect_contract": "duplicate", "operation": "row_insert",
            "raw_file": "icu/chartevents.csv", "raw_row_index": "2", "column": "",
            "clean_value": "", "dirty_value": duplicate.iloc[0].to_json(),
        })

    clean_graph = tmp_path / "clean_graph"
    build_graph(clean, clean_graph)
    return dirty, clean, clean_graph, log


def test_builds_deterministic_supervised_graph_without_mutating_sources(tmp_path: Path) -> None:
    dirty, clean, clean_graph, base_log = _fixture(tmp_path)
    before_dirty = _tree_hash(dirty)
    before_clean = _tree_hash(clean)

    result = build_supervised_graph(
        base_dirty_raw=dirty,
        clean_raw=clean,
        clean_graph=clean_graph,
        base_injection_log=base_log,
        output_raw=tmp_path / "supervised_raw",
        output_graph=tmp_path / "supervised_graph",
        private_output=tmp_path / "private/supervision",
        dirty_target=40,
        clean_target=12,
        seeds=(666, 667),
    )

    assert result["status"] == "SUCCESS"
    assert result["dirty_observation_count"] == 40
    assert result["clean_training_count"] == 12
    assert result["source_base_log_record_count"] == 2
    assert result["base_log_record_count"] == 1
    assert result["excluded_row_insert_count"] == 1
    assert result["synthetic_log_record_count"] > 0
    assert result["by_operation_records"] == {"cell_update": 40}
    assert _tree_hash(dirty) == before_dirty
    assert _tree_hash(clean) == before_clean
    assert json.loads((tmp_path / "supervised_graph/relation_ids.json").read_text()) == json.loads((clean_graph / "relation_ids.json").read_text())

    labels = np.load(tmp_path / "private/supervision/edge_labels.npz")
    masks = np.load(tmp_path / "private/supervision/supervision_masks.npz")
    assert int(labels["cell_dirty"].sum()) == 40
    assert len(masks["cell_indices"]) == 52
    assert int(masks["cell_labels"].sum()) == 40
    assert set(masks["cell_folds"].tolist()) <= {0, 1, 2, 3, 4}

    with (tmp_path / "private/supervision/merged_injection_log.csv").open(encoding="utf-8", newline="") as handle:
        merged = list(csv.DictReader(handle))
    assert merged[0]["source_error_id"] == "base-1"
    assert all(row["operation"] == "cell_update" for row in merged)
    assert not any(row["error_subtype"] == "duplicate_row" for row in merged)
    synthetic_cells = [row for row in merged if row["source_error_id"].startswith("synthetic:")]
    assert len({(row["raw_file"], row["raw_row_index"], row["column"]) for row in synthetic_cells}) == len(synthetic_cells)


def test_same_seeds_produce_same_synthetic_coordinates(tmp_path: Path) -> None:
    dirty, clean, clean_graph, base_log = _fixture(tmp_path)
    manifests = []
    for suffix in ("a", "b"):
        manifests.append(build_supervised_graph(
            base_dirty_raw=dirty,
            clean_raw=clean,
            clean_graph=clean_graph,
            base_injection_log=base_log,
            output_raw=tmp_path / f"raw_{suffix}",
            output_graph=tmp_path / f"graph_{suffix}",
            private_output=tmp_path / f"private_{suffix}",
            dirty_target=40,
            clean_target=12,
            seeds=(666, 667),
        ))

    assert manifests[0]["synthetic_coordinate_sha256"] == manifests[1]["synthetic_coordinate_sha256"]
