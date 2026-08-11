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
    assert len(values) == 6
    assert manifest["triple_count"] == 9
    assert manifest["edge_count"] == 18
    assert np.load(output / "edge_index.npy", mmap_mode="r").shape == (2, 18)
    assert np.load(output / "edge_type.npy", mmap_mode="r").shape == (18,)
    assert (output / "cell_observations.jsonl").exists()
    assert all(node.get("embedding_text") for node in nodes)


def test_missing_shared_values_are_column_scoped_instead_of_globally_shared(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _csv(raw, "icu/icustays.csv", [{"subject_id": "", "hadm_id": "null", "stay_id": 30}])
    output = tmp_path / "graph"
    build_graph(raw, output)
    values = [node for node in _nodes(output) if node["node_type"] == "Value"]
    assert not any(node.get("domain") == "entity.subject_id" for node in values)
    assert not any(node.get("domain") == "entity.hadm_id" for node in values)
    assert any(node.get("domain") == "icu/icustays.subject_id" and node.get("canonical_value") == "<MISSING>" for node in values)
    assert any(node.get("domain") == "icu/icustays.hadm_id" and node.get("canonical_value") == "<MISSING>" for node in values)


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


def test_relation_vocabulary_is_reused_and_unknown_relations_are_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _csv(raw, "icu/icustays.csv", [{"subject_id": 10, "hadm_id": 20, "stay_id": 30}])
    relation_ids = {
        "icu/icustays.subject_id": 0,
        "icu/icustays.subject_id__rev": 1,
        "icu/icustays.hadm_id": 2,
        "icu/icustays.hadm_id__rev": 3,
        "icu/icustays.stay_id": 4,
        "icu/icustays.stay_id__rev": 5,
    }

    manifest = build_graph(raw, tmp_path / "graph", relation_ids=relation_ids)

    assert json.loads((tmp_path / "graph/relation_ids.json").read_text()) == relation_ids
    assert manifest["relation_count"] == len(relation_ids)

    incomplete = dict(relation_ids)
    incomplete.pop("icu/icustays.stay_id__rev")
    with pytest.raises(GraphBuildError, match="unknown relation"):
        build_graph(raw, tmp_path / "bad_graph", relation_ids=incomplete)


def test_ordinary_values_share_within_a_column_but_not_across_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _csv(raw, "icu/example.csv", [
        {"left": "same", "right": "same"},
        {"left": "same", "right": "different"},
    ])

    output = tmp_path / "graph"
    manifest = build_graph(raw, output)
    values = [node for node in _nodes(output) if node["node_type"] == "Value"]

    assert manifest["triple_count"] == 4
    assert manifest["relation_count"] == 4
    assert len([node for node in values if node["domain"] == "icu/example.left"]) == 1
    assert len([node for node in values if node["domain"] == "icu/example.right"]) == 2
    observations = [json.loads(line) for line in (output / "cell_observations.jsonl").read_text().splitlines()]
    assert all(item["value_node_id"] is not None for item in observations)
    assert [item["forward_edge_id"] for item in observations] == [0, 2, 4, 6]
