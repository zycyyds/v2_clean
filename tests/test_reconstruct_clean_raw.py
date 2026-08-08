from __future__ import annotations

import csv
import json
from pathlib import Path

from graph.reconstruct_clean_raw import reconstruct_clean_raw


def test_reconstructs_cell_updates_and_removes_inserted_rows(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty"
    source = dirty / "icu/chartevents.csv"
    source.parent.mkdir(parents=True)
    source.write_text("stay_id,value\n1,clean\n1,dirty\n1,inserted\n", encoding="utf-8")
    log = tmp_path / "host_private/log.csv"
    log.parent.mkdir()
    fields = ["source_error_id", "operation", "raw_file", "raw_row_index", "column", "clean_value", "dirty_value"]
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"source_error_id": "1", "operation": "cell_update", "raw_file": "icu/chartevents.csv", "raw_row_index": "1", "column": "value", "clean_value": "clean2", "dirty_value": "dirty"})
        writer.writerow({"source_error_id": "2", "operation": "row_insert", "raw_file": "icu/chartevents.csv", "raw_row_index": "2", "column": "", "clean_value": "", "dirty_value": "{}"})
    output = tmp_path / "clean"
    manifest = reconstruct_clean_raw(dirty, log, output)
    assert manifest["restored_cells"] == 1
    assert manifest["removed_rows"] == 1
    assert (output / "icu/chartevents.csv").read_text(encoding="utf-8") == "stay_id,value\n1,clean\n1,clean2\n"


def test_reconstructs_chained_updates_in_reverse_log_order(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty"
    source = dirty / "hosp/patients.csv"
    source.parent.mkdir(parents=True)
    source.write_text("subject_id,value\n1,final\n", encoding="utf-8")
    log = tmp_path / "log.csv"
    fields = ["source_error_id", "operation", "raw_file", "raw_row_index", "column", "clean_value", "dirty_value"]
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"source_error_id": "1", "operation": "cell_update", "raw_file": "hosp/patients.csv", "raw_row_index": "0", "column": "value", "clean_value": "original", "dirty_value": "first"})
        writer.writerow({"source_error_id": "2", "operation": "cell_update", "raw_file": "hosp/patients.csv", "raw_row_index": "0", "column": "value", "clean_value": "first", "dirty_value": "final"})
    output = tmp_path / "clean"
    reconstruct_clean_raw(dirty, log, output)
    assert (output / "hosp/patients.csv").read_text(encoding="utf-8") == "subject_id,value\n1,original\n"
