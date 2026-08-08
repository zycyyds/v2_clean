from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graph.builder import GraphBuildError, build_graph


def _csv(root: Path, relative: str, rows: list[dict[str, object]]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _nodes(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "nodes.jsonl").read_text().splitlines()]


def test_typed_shared_values_deduplicate_and_dictionary_links(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _csv(raw, "icu/chartevents.csv", [{"subject_id": 10, "hadm_id": 20, "stay_id": 30, "itemid": 225664, "valuenum": 96.0}])
    _csv(raw, "icu/d_items.csv", [{"itemid": 225664, "label": "Glucose finger stick"}])
    _csv(raw, "hosp/admissions.csv", [{"subject_id": 10, "hadm_id": 20}])
    output = tmp_path / "graph"

    manifest = build_graph(raw, output)
    nodes = _nodes(output)
    values = [node for node in nodes if node["node_type"] == "Value"]
    subject_values = [node for node in values if node["domain"] == "entity.subject_id"]
    item_values = [node for node in values if node["domain"] == "dictionary.icu_item"]

    assert len(subject_values) == 1
    assert len(item_values) == 1
    assert len(values) == 4
    assert manifest["triple_count"] == 7
    assert manifest["edge_count"] == 14
    edges = np.load(output / "edges.npz")
    assert edges["edge_index"].shape == (2, 14)
    assert (output / "cell_observations.jsonl").exists()


def test_missing_shared_values_are_not_interned(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _csv(raw, "icu/icustays.csv", [{"subject_id": "", "hadm_id": "null", "stay_id": 30}])
    output = tmp_path / "graph"
    build_graph(raw, output)
    values = [node for node in _nodes(output) if node["node_type"] == "Value"]
    assert all(node.get("canonical_value") not in {None, "", "null"} for node in values)
    assert not any(node.get("domain") == "entity.subject_id" for node in values)
    assert not any(node.get("domain") == "entity.hadm_id" for node in values)


def test_icd_version_is_part_of_dictionary_identity(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _csv(raw, "hosp/diagnoses_icd.csv", [{"icd_code": "25000", "icd_version": "9"}, {"icd_code": "25000", "icd_version": "10"}])
    output = tmp_path / "graph"
    build_graph(raw, output)
    values = [node for node in _nodes(output) if node["node_type"] == "Value" and node.get("domain") == "dictionary.diagnosis_icd"]
    assert {node["canonical_value"] for node in values} == {"9:25000", "10:25000"}


def test_forbidden_raw_root_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "reference_private"
    raw.mkdir()
    with pytest.raises(GraphBuildError, match="forbidden/private"):
        build_graph(raw, tmp_path / "graph")
