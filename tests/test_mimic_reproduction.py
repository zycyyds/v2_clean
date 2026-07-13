from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from reproduction.mimic_icu_mortality import (
    GOLD_RELATIVE_PATHS,
    STAGES,
    ReproductionConfig,
    RunLayout,
    checkpoint_is_reusable,
    config_fingerprint,
    initialize_workspace,
    stage_outputs,
    stage_worker_arguments,
    write_checkpoint,
)
from reproduction.mimic_validation import validate_reproduction


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _config(tmp_path: Path) -> ReproductionConfig:
    source_root = tmp_path / "source"
    raw_root = source_root / "mimiciv" / "3.1"
    mapping = source_root / "utils" / "mappings" / "ICD9_to_ICD10_mapping.txt"
    raw_root.mkdir(parents=True)
    mapping.parent.mkdir(parents=True)
    mapping.write_text("icd9\ticd10\n", encoding="utf-8")
    for relative in ReproductionConfig.required_raw_files():
        path = raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("header\n", encoding="utf-8")
    for relative in ReproductionConfig.required_source_files():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture\n", encoding="utf-8")
    return ReproductionConfig(source_root=source_root, run_root=tmp_path / "run")


def test_default_config_matches_approved_icu_mortality_profile(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.mimic_version == "3.1"
    assert config.care_setting == "ICU"
    assert config.task == "Mortality"
    assert config.disease_filter == "No Disease Filter"
    assert config.feature_flags == {
        "diagnosis": True,
        "output": True,
        "chart": True,
        "procedures": True,
        "medications": True,
    }
    assert config.diagnosis_mode == "Convert ICD-9 to ICD-10 and group ICD-10 codes"
    assert (config.include_hours, config.prediction_hours, config.bucket_hours) == (72, 2, 1)
    assert config.imputation is False
    assert config.cohort_output == "cohort_icu_mortality_0__"


def test_layout_never_targets_original_data_directory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = RunLayout(config)

    assert layout.data_root == config.run_root / "data"
    assert layout.data_root != config.source_root / "data"
    with pytest.raises(ValueError, match="original project data"):
        ReproductionConfig(source_root=config.source_root, run_root=config.source_root / "data")


def test_initialize_workspace_creates_only_output_dirs_and_readonly_links(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = initialize_workspace(config)

    for directory in (
        layout.cohort_dir,
        layout.features_dir,
        layout.summary_dir,
        layout.csv_dir,
        layout.dict_dir,
        layout.logs_dir,
        layout.checkpoints_dir,
    ):
        assert directory.is_dir()
    assert layout.raw_link.is_symlink()
    assert layout.raw_link.resolve() == config.raw_root.resolve()
    assert layout.mapping_link.is_symlink()
    assert layout.mapping_link.resolve() == config.mapping_path.resolve()
    assert json.loads(layout.metadata_path.read_text(encoding="utf-8"))["owner"] == "mimic-reproduction-runner"


def test_initialize_workspace_rejects_unmanaged_nonempty_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.run_root.mkdir()
    (config.run_root / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unmanaged"):
        initialize_workspace(config)


def test_initialize_workspace_rejects_wrong_existing_input_link(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.run_root.mkdir()
    (config.run_root / "reproduction_metadata.json").write_text(
        json.dumps({"owner": "mimic-reproduction-runner"}), encoding="utf-8"
    )
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (config.run_root / "mimiciv").mkdir()
    (config.run_root / "mimiciv" / "3.1").symlink_to(wrong, target_is_directory=True)

    with pytest.raises(RuntimeError, match="points to"):
        initialize_workspace(config)


def test_checkpoint_requires_matching_config_and_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = initialize_workspace(config)
    output = layout.data_root / "sample.csv"
    output.write_text("stay_id\n1\n", encoding="utf-8")
    write_checkpoint(layout, "cohort", config, [output], elapsed_seconds=1.25)

    assert checkpoint_is_reusable(layout, "cohort", config, [output]) is True
    output.write_text("stay_id\n2\n", encoding="utf-8")
    assert checkpoint_is_reusable(layout, "cohort", config, [output]) is False

    changed = ReproductionConfig(
        source_root=config.source_root,
        run_root=config.run_root,
        include_hours=48,
    )
    assert config_fingerprint(changed) != config_fingerprint(config)
    assert checkpoint_is_reusable(layout, "cohort", changed, [output]) is False


def test_checkpoint_is_invalidated_when_a_raw_input_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = initialize_workspace(config)
    output = layout.data_root / "sample.csv"
    output.write_text("stay_id\n1\n", encoding="utf-8")
    write_checkpoint(layout, "cohort", config, [output], elapsed_seconds=1.0)

    raw_input = config.raw_root / config.required_raw_files()[0]
    raw_input.write_text("changed\n", encoding="utf-8")

    assert checkpoint_is_reusable(layout, "cohort", config, [output]) is False


def test_diagnosis_grouping_does_not_invalidate_feature_extraction_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = initialize_workspace(config)
    outputs = stage_outputs(layout, "features")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stay_id\n1\n", encoding="utf-8")
    write_checkpoint(layout, "features", config, outputs, elapsed_seconds=2.0)

    layout.features_dir.joinpath("preproc_diag_icu.csv.gz").write_text(
        "subject_id,hadm_id,stay_id,new_icd_code\n10,20,1,I10\n",
        encoding="utf-8",
    )

    assert checkpoint_is_reusable(layout, "features", config, outputs) is True
    assert layout.intermediate_diag_path in outputs
    assert layout.features_dir / "preproc_diag_icu.csv.gz" not in outputs


def test_stage_order_and_worker_arguments_are_exact(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert STAGES == ("cohort", "features", "diagnosis", "summaries", "stay_generation", "validation")
    args = stage_worker_arguments(config, "stay_generation")
    assert args[-2:] == ["--stage", "stay_generation"]
    assert str(config.source_root) in args
    assert str(config.run_root) in args


def _build_valid_fixture(config: ReproductionConfig) -> RunLayout:
    layout = initialize_workspace(config)
    cohort_rows = [
        {
            "subject_id": 10,
            "stay_id": 100,
            "intime": "2100-01-01 00:00:00",
            "outtime": "2100-01-05 00:00:00",
            "Age": 65,
            "gender": "F",
            "ethnicity": "WHITE",
            "insurance": "Medicare",
            "label": 1,
            "dod": "2100-01-03 00:00:00",
            "hadm_id": 1000,
        }
    ]
    _write_csv(layout.cohort_path, cohort_rows)
    _write_csv(layout.labels_path, [{"stay_id": 100, "label": 1}])
    _write_csv(
        layout.features_dir / "preproc_diag_icu.csv.gz",
        [{"subject_id": 10, "hadm_id": 1000, "stay_id": 100, "new_icd_code": "I10"}],
    )
    _write_csv(
        layout.features_dir / "preproc_chart_icu.csv.gz",
        [{"stay_id": 100, "itemid": 220045, "event_time_from_admit": "0 days 01:00:00", "valuenum": 80}],
    )
    for name in ("preproc_med_icu.csv.gz", "preproc_out_icu.csv.gz", "preproc_proc_icu.csv.gz"):
        _write_csv(layout.features_dir / name, [{"stay_id": 100, "itemid": 1}])
    for name in (
        "diag_summary.csv",
        "diag_features.csv",
        "med_summary.csv",
        "med_features.csv",
        "proc_summary.csv",
        "proc_features.csv",
        "out_summary.csv",
        "out_features.csv",
        "chart_summary.csv",
        "chart_features.csv",
    ):
        _write_csv(layout.summary_dir / name, [{"value": 1}])
    stay_dir = layout.csv_dir / "100"
    _write_csv(stay_dir / "demo.csv", [{"Age": 65}])
    _write_csv(stay_dir / "static.csv", [{"COND": 1}])
    _write_csv(stay_dir / "dynamic.csv", [{"CHART": 80}])
    for name in RunLayout.expected_dict_names():
        path = layout.dict_dir / name
        path.write_bytes(b"pickle")
    return layout


def test_validation_accepts_complete_fixture_and_lists_17_gold_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = _build_valid_fixture(config)

    report, manifest = validate_reproduction(config, layout)

    assert report["passed"] is True
    assert report["status"] == "success"
    assert len(manifest["gold_candidates"]) == 17
    assert tuple(item["relative_path"] for item in manifest["gold_candidates"]) == GOLD_RELATIVE_PATHS


def test_validation_rejects_six_column_diagnosis_and_missing_stay_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = _build_valid_fixture(config)
    _write_csv(
        layout.features_dir / "preproc_diag_icu.csv.gz",
        [
            {
                "subject_id": 10,
                "hadm_id": 1000,
                "stay_id": 100,
                "icd_code": "4019",
                "root_icd10_convert": "I10",
                "root": "I10",
            }
        ],
    )
    (layout.csv_dir / "100" / "dynamic.csv").unlink()

    report, _ = validate_reproduction(config, layout)

    assert report["passed"] is False
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert "diagnosis_schema" in failed
    assert "stay_csv_completeness" in failed


def test_validation_rejects_label_stay_shorter_than_74_integer_hours(tmp_path: Path) -> None:
    config = _config(tmp_path)
    layout = _build_valid_fixture(config)
    cohort = pd.read_csv(layout.cohort_path)
    cohort.loc[0, "outtime"] = "2100-01-04 01:59:59"
    cohort.to_csv(layout.cohort_path, index=False)

    report, _ = validate_reproduction(config, layout)

    check = next(item for item in report["checks"] if item["name"] == "labels_observation_window")
    assert check["passed"] is False
