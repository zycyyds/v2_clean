from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from main import parse_args


def _csv_text(fieldnames: list[str], rows: list[dict[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _gzip_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def _build_synthetic_correction_archive(path: Path, *, stay_count: int = 12) -> Path:
    cohort_clean = [
        {
            "subject_id": 1000 + index,
            "hadm_id": 2000 + index,
            "stay_id": 3000 + index,
            "Age": 40 + index,
            "label": index % 2,
        }
        for index in range(stay_count)
    ]
    cohort_dirty = [dict(row) for row in cohort_clean]
    cohort_dirty[0]["Age"] = 999

    chart_clean = [
        {"stay_id": 3000 + index, "itemid": 220045, "valuenum": 70 + index}
        for index in range(stay_count)
    ]
    chart_dirty = [dict(row) for row in chart_clean]
    chart_dirty[1]["stay_id"] = 999999
    chart_dirty.insert(
        3,
        {"stay_id": 3002, "itemid": 999001, "valuenum": cohort_clean[2]["label"]},
    )

    labels = [
        {"stay_id": row["stay_id"], "label": row["label"]}
        for row in cohort_clean
    ]
    modifications = [
        {
            "operation": "cell_update",
            "file": "cohort/cohort_icu_mortality_0__.csv.gz",
            "clean_row_index": 0,
            "dirty_row_index": 0,
            "column": "Age",
            "clean_value": 40,
            "dirty_value": 999,
            "error_class": "class1_intra_table",
            "error_subtype": "schema_range_violation",
            "repair_hint": "hidden",
            "stay_id": "",
        },
        {
            "operation": "cell_update",
            "file": "features/preproc_chart_icu.csv.gz",
            "clean_row_index": 1,
            "dirty_row_index": 1,
            "column": "stay_id",
            "clean_value": 3001,
            "dirty_value": 999999,
            "error_class": "class3_cross_table",
            "error_subtype": "orphan_feature_stay_id",
            "repair_hint": "hidden",
            "stay_id": "",
        },
        {
            "operation": "row_insert",
            "file": "features/preproc_chart_icu.csv.gz",
            "clean_row_index": "",
            "dirty_row_index": 3,
            "column": "",
            "clean_value": "",
            "dirty_value": "inserted",
            "error_class": "class4_task_oriented",
            "error_subtype": "label_leakage_feature",
            "repair_hint": "hidden",
            "stay_id": 3002,
        },
    ]
    mod_fields = list(modifications[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "clean/cohort/cohort_icu_mortality_0__.csv.gz",
            _gzip_bytes(_csv_text(list(cohort_clean[0]), cohort_clean)),
        )
        archive.writestr(
            "dirty/cohort/cohort_icu_mortality_0__.csv.gz",
            _gzip_bytes(_csv_text(list(cohort_dirty[0]), cohort_dirty)),
        )
        archive.writestr(
            "clean/features/preproc_chart_icu.csv.gz",
            _gzip_bytes(_csv_text(list(chart_clean[0]), chart_clean)),
        )
        archive.writestr(
            "dirty/features/preproc_chart_icu.csv.gz",
            _gzip_bytes(_csv_text(list(chart_dirty[0]), chart_dirty)),
        )
        labels_text = _csv_text(list(labels[0]), labels)
        archive.writestr("clean/labels.csv", labels_text)
        archive.writestr("dirty/labels.csv", labels_text)
        archive.writestr("clean/summary/chart_summary.csv", "itemid,total_count\n220045,12\n")
        archive.writestr("dirty/summary/chart_summary.csv", "itemid,total_count\n220045,12\n")
        archive.writestr(
            "ground_truth/modification_log.csv",
            _csv_text(mod_fields, modifications),
        )
        archive.writestr(
            "ground_truth/summary.json",
            json.dumps({"modifications_total": len(modifications)}),
        )
    return path


def _read_keys(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["stay_id"] for row in csv.DictReader(handle)}


def _read_gzip_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_correction_dataset_partitions_canonical_stays_and_private_truth(
    tmp_path: Path,
) -> None:
    from workflow.correction_dataset import build_correction_dataset

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    output = tmp_path / "dataset"

    report = build_correction_dataset(archive, output, train_count=2)

    train_keys = _read_keys(output / "train/keys.csv")
    correction_keys = _read_keys(output / "correction/keys.csv")
    assert report["status"] == "SUCCESS"
    assert len(train_keys) == 2
    assert len(correction_keys) == 10
    assert train_keys.isdisjoint(correction_keys)
    assert train_keys | correction_keys == {str(3000 + index) for index in range(12)}
    assert {"3000", "3001"}.issubset(train_keys)

    dirty_chart = _read_gzip_rows(output / "train/raw/features/preproc_chart_icu.csv.gz")
    assert any(row["stay_id"] == "999999" for row in dirty_chart)
    assert (output / "train/reference/features/preproc_chart_icu.csv.gz").is_file()
    assert (output / "correction/reference_private/features/preproc_chart_icu.csv.gz").is_file()
    assert (output / "train/raw/summary/chart_summary.csv").read_bytes() == (
        output / "correction/raw/summary/chart_summary.csv"
    ).read_bytes()

    with (output / "host_private/train_modification_log.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        train_modifications = list(csv.DictReader(handle))
    with (output / "host_private/correction_modification_log.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        correction_modifications = list(csv.DictReader(handle))
    assert len(train_modifications) + len(correction_modifications) == 3
    assert all(row["canonical_stay_id"] in train_keys for row in train_modifications)
    assert all(row["canonical_stay_id"] in correction_keys for row in correction_modifications)
    assert all(row["dirty_row_json"] for row in train_modifications + correction_modifications)
    assert all(
        row["clean_row_json"]
        for row in train_modifications + correction_modifications
        if row["operation"] == "cell_update"
    )

    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    serialized_manifest = json.dumps(manifest)
    assert manifest["counts"] == {"train": 2, "correction": 10}
    assert "error_subtype" not in serialized_manifest
    assert "repair_hint" not in serialized_manifest


def test_build_correction_dataset_refuses_mismatched_existing_manifest(tmp_path: Path) -> None:
    from workflow.correction_dataset import build_correction_dataset

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "split_manifest.json").write_text(
        json.dumps({"source_archive_sha256": "wrong"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="existing correction dataset manifest"):
        build_correction_dataset(archive, output, train_count=2)


def _write_plain_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_csv_text(list(rows[0]), rows), encoding="utf-8")


def _write_private_log(path: Path) -> None:
    clean_changed = {"stay_id": "1", "value": "1"}
    dirty_changed = {"stay_id": "1", "value": "999"}
    inserted = {"stay_id": "3", "value": "1"}
    rows = [
        {
            "operation": "cell_update",
            "file": "features/table.csv",
            "error_class": "class1_intra_table",
            "error_subtype": "schema_range_violation",
            "clean_row_json": json.dumps(clean_changed, sort_keys=True),
            "dirty_row_json": json.dumps(dirty_changed, sort_keys=True),
        },
        {
            "operation": "row_insert",
            "file": "features/table.csv",
            "error_class": "class4_task_oriented",
            "error_subtype": "label_leakage_feature",
            "clean_row_json": "",
            "dirty_row_json": json.dumps(inserted, sort_keys=True),
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_csv_text(list(rows[0]), rows), encoding="utf-8")


def test_correction_evaluator_scores_exact_repairs_and_collateral_changes(
    tmp_path: Path,
) -> None:
    from workflow.correction_evaluation import evaluate_correction_package

    clean = tmp_path / "clean"
    dirty = tmp_path / "dirty"
    result = tmp_path / "result"
    clean_rows = [{"stay_id": "1", "value": "1"}, {"stay_id": "2", "value": "2"}]
    dirty_rows = [
        {"stay_id": "1", "value": "999"},
        {"stay_id": "2", "value": "2"},
        {"stay_id": "3", "value": "1"},
    ]
    result_rows = [
        {"stay_id": "1", "value": "1"},
        {"stay_id": "2", "value": "777"},
    ]
    _write_plain_table(clean / "features/table.csv", clean_rows)
    _write_plain_table(dirty / "features/table.csv", dirty_rows)
    _write_plain_table(result / "features/table.csv", result_rows)
    log = tmp_path / "modification_log.csv"
    _write_private_log(log)

    report = evaluate_correction_package(
        dirty_root=dirty,
        result_root=result,
        clean_root=clean,
        modification_log=log,
        output_dir=tmp_path / "evaluation",
    )

    assert report["metrics"]["ground_truth_errors"] == 2
    assert report["metrics"]["exact_repairs"] == 2
    assert report["metrics"]["exact_repair_recall"] == 1.0
    assert report["metrics"]["inserted_row_deletion_rate"] == 1.0
    assert report["metrics"]["collateral_row_changes"] == 2
    assert report["metrics"]["clean_preservation"] == 0.0
    assert report["by_error_class"]["class1_intra_table"]["exact_repairs"] == 1
    assert Path(report["evaluation_report"]).is_file()


def test_correction_evaluator_distinguishes_missed_and_incorrect_repairs(
    tmp_path: Path,
) -> None:
    from workflow.correction_evaluation import evaluate_correction_package

    clean = tmp_path / "clean"
    dirty = tmp_path / "dirty"
    result = tmp_path / "result"
    clean_rows = [{"stay_id": "1", "value": "1"}, {"stay_id": "2", "value": "2"}]
    dirty_rows = [
        {"stay_id": "1", "value": "999"},
        {"stay_id": "2", "value": "2"},
        {"stay_id": "3", "value": "1"},
    ]
    _write_plain_table(clean / "features/table.csv", clean_rows)
    _write_plain_table(dirty / "features/table.csv", dirty_rows)
    shutil.copytree(dirty, result)
    log = tmp_path / "modification_log.csv"
    _write_private_log(log)

    missed = evaluate_correction_package(
        dirty_root=dirty,
        result_root=result,
        clean_root=clean,
        modification_log=log,
        output_dir=tmp_path / "missed_evaluation",
    )
    assert missed["metrics"]["exact_repairs"] == 0
    assert missed["metrics"]["missed_repairs"] == 2
    assert missed["metrics"]["incorrect_repairs"] == 0

    _write_plain_table(
        result / "features/table.csv",
        [
            {"stay_id": "1", "value": "555"},
            {"stay_id": "2", "value": "2"},
            {"stay_id": "3", "value": "1"},
        ],
    )
    incorrect = evaluate_correction_package(
        dirty_root=dirty,
        result_root=result,
        clean_root=clean,
        modification_log=log,
        output_dir=tmp_path / "incorrect_evaluation",
    )
    assert incorrect["metrics"]["incorrect_repairs"] == 1
    assert incorrect["metrics"]["missed_repairs"] == 1


def test_correction_cli_exposes_prepare_and_single_run_workflows() -> None:
    prepare = parse_args(
        [
            "--workflow",
            "prepare-correction-split",
            "--source-archive",
            "/tmp/source.zip",
            "--split-output",
            "/tmp/output",
            "--train-count",
            "10",
        ]
    )
    run = parse_args(
        [
            "--workflow",
            "reference-guided-correct",
            "--dataset-split",
            "/tmp/dataset",
            "--experiment-dir",
            "/tmp/experiment",
            "task",
        ]
    )

    assert prepare.workflow == "prepare-correction-split"
    assert prepare.train_count == 10
    assert run.workflow == "reference-guided-correct"
    assert run.resume is False

    resumed = parse_args(
        [
            "--workflow",
            "reference-guided-correct",
            "--dataset-split",
            "/tmp/dataset",
            "--experiment-dir",
            "/tmp/experiment",
            "--resume",
            "task",
        ]
    )
    assert resumed.resume is True


def test_correction_agent_contract_and_prompt_hide_private_truth_and_taxonomy(
    tmp_path: Path,
) -> None:
    from workflow.correction_dataset import build_correction_dataset
    from workflow.reference_correction import ReferenceCorrectionConfig, ReferenceCorrectionWorkflow

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    dataset = tmp_path / "dataset"
    build_correction_dataset(archive, dataset, train_count=2)
    workflow = ReferenceCorrectionWorkflow(
        ReferenceCorrectionConfig(
            dataset_split=dataset,
            experiment_dir=tmp_path / "experiment",
            task_text="discover and correct",
            max_iters=20,
        )
    )

    serialized = json.dumps(workflow.sanitized_contract)
    prompt = workflow._task_text(tmp_path / "phase")
    roots = {str(Path(path)) for path in workflow._agent_read_roots()}

    assert "host_private" not in serialized
    assert "reference_private" not in serialized
    assert "modification_log" not in serialized
    assert "error_subtype" not in prompt
    assert "schema_range_violation" not in prompt
    assert "repair_hint" not in prompt
    assert "根据2个成对标准示例" in prompt
    assert "根据10个成对标准示例" not in prompt
    assert 'key_column="stay_id"' in prompt
    assert "不得按 `hadm_id` 去重" in prompt
    assert workflow.sanitized_contract["key_column"] == "stay_id"
    assert str(dataset / "train/raw") in roots
    assert str(dataset / "train/reference") in roots
    assert str(dataset / "correction/raw") in roots
    assert str(dataset / "train/keys.csv") not in roots
    assert str(dataset / "correction/keys.csv") not in roots
    assert str(dataset / "host_private") not in roots


def test_correction_result_gate_requires_complete_matching_package(tmp_path: Path) -> None:
    from workflow.correction_dataset import build_correction_dataset
    from workflow.reference_correction import validate_correction_result_package

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    dataset = tmp_path / "dataset"
    build_correction_dataset(archive, dataset, train_count=2)
    result = tmp_path / "result"
    shutil.copytree(dataset / "correction/raw", result)

    valid = validate_correction_result_package(
        result_package=result,
        input_package=dataset / "correction/raw",
        split_keys=dataset / "correction/keys.csv",
        key_column="stay_id",
    )
    assert valid["valid"] is True

    (result / "features/preproc_chart_icu.csv.gz").unlink()
    invalid = validate_correction_result_package(
        result_package=result,
        input_package=dataset / "correction/raw",
        split_keys=dataset / "correction/keys.csv",
        key_column="stay_id",
    )
    assert invalid["valid"] is False
    assert any("missing" in issue for issue in invalid["issues"])


def test_correction_result_gate_uses_stay_id_not_hadm_id(tmp_path: Path) -> None:
    from workflow.correction_dataset import build_correction_dataset
    from workflow.reference_correction import validate_correction_result_package

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    dataset = tmp_path / "dataset"
    build_correction_dataset(archive, dataset, train_count=2)
    result = tmp_path / "result"
    shutil.copytree(dataset / "correction/reference_private", result)

    cohort_path = result / "cohort/cohort_icu_mortality_0__.csv.gz"
    with gzip.open(cohort_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    rows[1]["hadm_id"] = rows[0]["hadm_id"]
    with gzip.open(cohort_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    valid = validate_correction_result_package(
        result_package=result,
        input_package=dataset / "correction/raw",
        split_keys=dataset / "correction/keys.csv",
        key_column="stay_id",
    )
    assert valid["valid"] is True

    rows.pop()
    with gzip.open(cohort_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    invalid = validate_correction_result_package(
        result_package=result,
        input_package=dataset / "correction/raw",
        split_keys=dataset / "correction/keys.csv",
        key_column="stay_id",
    )
    assert invalid["valid"] is False
    assert any("correction stay_id" in issue for issue in invalid["issues"])


def test_correction_workflow_runs_one_agent_session_then_private_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workflow.correction_dataset import build_correction_dataset
    from workflow.reference_correction import ReferenceCorrectionConfig, ReferenceCorrectionWorkflow

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    dataset = tmp_path / "dataset"
    build_correction_dataset(archive, dataset, train_count=2)
    experiment = tmp_path / "experiment"
    workflow = ReferenceCorrectionWorkflow(
        ReferenceCorrectionConfig(
            dataset_split=dataset,
            experiment_dir=experiment,
            task_text="discover and correct",
            max_iters=20,
        )
    )
    calls = []

    def fake_agent_session() -> Path:
        calls.append("agent")
        package = experiment / "agent_output/result_package"
        shutil.copytree(dataset / "correction/reference_private", package)
        return package

    monkeypatch.setattr(workflow, "_run_agent_session", fake_agent_session)

    report = workflow.run_sync()

    assert calls == ["agent"]
    assert report["status"] == "SUCCESS"
    assert report["metrics"]["exact_repair_recall"] == 1.0
    assert Path(report["result_package"]).is_dir()
    assert Path(report["evaluation_report"]).is_file()


def test_correction_workflow_explicitly_resumes_failed_same_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workflow.correction_dataset import build_correction_dataset
    from workflow.reference_correction import ReferenceCorrectionConfig, ReferenceCorrectionWorkflow

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    dataset = tmp_path / "dataset"
    build_correction_dataset(archive, dataset, train_count=2)
    experiment = tmp_path / "experiment"

    first = ReferenceCorrectionWorkflow(
        ReferenceCorrectionConfig(
            dataset_split=dataset,
            experiment_dir=experiment,
            task_text="discover and correct",
            max_iters=20,
        )
    )

    def fail_agent_session() -> Path:
        phase = experiment / "agent_runs/data_cleaning_agent"
        phase.mkdir(parents=True, exist_ok=True)
        (phase / "rules.json").write_text('{"rules": {}}', encoding="utf-8")
        raise ConnectionError("network unavailable")

    monkeypatch.setattr(first, "_run_agent_session", fail_agent_session)
    failed = first.run_sync()
    assert failed["status"] == "correction_failed"

    resumed = ReferenceCorrectionWorkflow(
        ReferenceCorrectionConfig(
            dataset_split=dataset,
            experiment_dir=experiment,
            task_text="discover and correct",
            max_iters=20,
            resume=True,
        )
    )

    def successful_agent_session() -> Path:
        assert resumed.resume_context is not None
        assert resumed.resume_context["previous_error"] == "ConnectionError: network unavailable"
        package = experiment / "agent_output/result_package"
        shutil.copytree(dataset / "correction/reference_private", package)
        return package

    monkeypatch.setattr(resumed, "_run_agent_session", successful_agent_session)
    report = resumed.run_sync()

    assert report["status"] == "SUCCESS"
    history = sorted((experiment / "resume_history").glob("resume_*_previous_report.json"))
    assert len(history) == 1
    previous = json.loads(history[0].read_text(encoding="utf-8"))
    assert previous["status"] == "correction_failed"
    manifest = json.loads((experiment / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["resume_count"] == 1


def test_correction_resume_rejects_changed_experiment_identity(tmp_path: Path) -> None:
    from workflow.correction_dataset import build_correction_dataset
    from workflow.reference_correction import ReferenceCorrectionConfig, ReferenceCorrectionWorkflow

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    dataset = tmp_path / "dataset"
    build_correction_dataset(archive, dataset, train_count=2)
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "reference-guided-correct",
                "dataset_split": str(dataset.resolve()),
                "source_archive_sha256": "wrong",
                "task_text_sha256": "wrong",
                "max_iters": 20,
            }
        ),
        encoding="utf-8",
    )
    (experiment / "correction_run_report.json").write_text(
        json.dumps({"status": "correction_failed", "error": "ConnectionError: offline"}),
        encoding="utf-8",
    )
    workflow = ReferenceCorrectionWorkflow(
        ReferenceCorrectionConfig(
            dataset_split=dataset,
            experiment_dir=experiment,
            task_text="discover and correct",
            max_iters=20,
            resume=True,
        )
    )

    with pytest.raises(ValueError, match="resume manifest does not match"):
        workflow.run_sync()


def test_correction_resume_context_includes_host_gate_issues(tmp_path: Path) -> None:
    from workflow.correction_dataset import build_correction_dataset
    from workflow.reference_correction import ReferenceCorrectionConfig, ReferenceCorrectionWorkflow

    archive = _build_synthetic_correction_archive(tmp_path / "source.zip")
    dataset = tmp_path / "dataset"
    build_correction_dataset(archive, dataset, train_count=2)
    experiment = tmp_path / "experiment"
    workflow = ReferenceCorrectionWorkflow(
        ReferenceCorrectionConfig(
            dataset_split=dataset,
            experiment_dir=experiment,
            task_text="discover and correct",
            max_iters=20,
            resume=True,
        )
    )
    experiment.mkdir()
    (experiment / "run_manifest.json").write_text(
        json.dumps(
            {
                **workflow._manifest_identity(),
                "created_at": "2026-01-01T00:00:00",
                "resume_count": 0,
                "resumes": [],
            }
        ),
        encoding="utf-8",
    )
    (experiment / "correction_run_report.json").write_text(
        json.dumps(
            {
                "status": "correction_failed",
                "gate": {
                    "valid": False,
                    "issues": [
                        "features/preproc_chart_icu.csv.gz: contains 2 unknown stay_id values"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    context = workflow._prepare_experiment(experiment / "correction_run_report.json")
    assert context is not None
    assert context["previous_error"] == (
        "Host result gate failed: features/preproc_chart_icu.csv.gz: "
        "contains 2 unknown stay_id values"
    )
