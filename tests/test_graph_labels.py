from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from graph.builder import build_graph
from graph.labels import build_labels


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_injection_log_labels_cell_and_inserted_row_without_reference(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write(raw / "icu/chartevents.csv", "subject_id,stay_id,itemid,valuenum\n10,30,1,10\n10,30,2,20\n10,30,2,20\n10,30,3,30\n")
    graph_dir = tmp_path / "graph"
    build_graph(raw, graph_dir)
    log = tmp_path / "host_private" / "train_raw_modification_log.csv"
    log.parent.mkdir()
    fields = ["source_error_id", "error_class", "error_subtype", "operation", "raw_file", "raw_row_index", "column"]
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"source_error_id": "1", "error_class": "class1", "error_subtype": "unit_scale_error", "operation": "cell_update", "raw_file": "icu/chartevents.csv", "raw_row_index": "1", "column": "valuenum"})
        writer.writerow({"source_error_id": "2", "error_class": "class2", "error_subtype": "duplicate_row", "operation": "row_insert", "raw_file": "icu/chartevents.csv", "raw_row_index": "2", "column": ""})
    labels = build_labels(graph_dir, log)
    assert labels["dirty_cell_count"] == 5
    assert labels["dirty_row_count"] == 2
    arrays = np.load(graph_dir / "edge_labels.npz")
    assert int(arrays["cell_dirty"].sum()) == 5
    assert int(arrays["row_dirty_node"].sum()) == 2


def test_label_builder_rejects_unknown_operation(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write(raw / "icu/icustays.csv", "stay_id\n30\n")
    graph_dir = tmp_path / "graph"
    build_graph(raw, graph_dir)
    log = tmp_path / "log.csv"
    _write(log, "operation,raw_file,raw_row_index,column,error_class,error_subtype\nweird,icu/icustays.csv,0,stay_id,c,x\n")
    try:
        build_labels(graph_dir, log)
    except ValueError as exc:
        assert "unsupported injection operation" in str(exc)
    else:
        raise AssertionError("unknown operation was accepted")
