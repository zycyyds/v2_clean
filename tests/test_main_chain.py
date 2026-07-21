from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import workflow.reference_guided as reference_guided

from agent.reference_runtime import DATA_CLEANING_AGENT_PIPELINE_SKILLS, create_reference_toolkit
from agent_tools.context import EngineerToolContext, EngineerToolPermissionError
from main import parse_args
from workflow.mimic_pipeline import split_teacher_reference
from workflow.reference_guided import (
    DataCleaningAgentRuntime,
    ReferenceGuidedConfig,
    ReferenceGuidedWorkflow,
    _finish_reference_attempt_state,
    _ensure_run_manifest,
    _archive_promoted_round,
    _promote_active_bundle_atomically,
    _recover_interrupted_promotion,
    _write_json,
    _migrate_reference_experiment_state,
    _reference_candidate_quality_gate,
    _reference_stop_reason,
    _complete_reference_package_paths,
    _inherit_pipeline_to_workspace,
    _audit_reference_feedback_response,
    _merge_test_rule_feedback_into_repair_targets,
    _stage_workspace_pipeline,
    evaluate_reference_directory_package,
    infer_reference_contract,
)
from workflow.reference_test_stage import (
    ReferenceCheckpointTestConfig,
    ReferenceCheckpointTestRuntime,
    ReferenceTestStageConfig,
    _build_sanitized_test_contract,
    _runner_spec_quality_gate,
    _test_package_quality_gate,
    run_reference_test_stage,
)
from workflow.reference_package_gate import validate_business_result_package
from workflow.reference_pipeline import (
    PIPELINE_MODULES,
    _build_train_regression_report,
    business_file_hashes,
    pipeline_directory_sha256,
    run_pipeline_isolated,
    validate_and_replay_candidate_pipeline,
    validate_pipeline_structure,
)


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_reference_package(root: Path) -> None:
    _write_csv(root / "cohort" / "cohort_icu_mortality.csv", "stay_id,label\n1,0\n")
    _write_csv(root / "csv" / "labels.csv", "stay_id,label\n1,0\n")
    _write_csv(
        root / "features" / "preproc_chart_icu.csv",
        "stay_id,itemid,event_time_from_admit\n1,220045,1.0\n",
    )
    _write_csv(root / "features" / "preproc_diag_icu.csv", "stay_id,new_icd_code\n1,A01\n")
    _write_csv(
        root / "features" / "preproc_med_icu.csv",
        "subject_id,hadm_id,stay_id,itemid,starttime,endtime,orderid\n10,20,1,30,1,2,40\n",
    )
    _write_csv(
        root / "features" / "preproc_out_icu.csv",
        "subject_id,hadm_id,stay_id,itemid,charttime\n10,20,1,30,1\n",
    )
    _write_csv(
        root / "features" / "preproc_proc_icu.csv",
        "subject_id,hadm_id,stay_id,itemid,starttime\n10,20,1,30,1\n",
    )
    for source in ("chart", "med", "out", "proc"):
        _write_csv(root / "summary" / f"{source}_features.csv", "itemid\n30\n")
        _write_csv(root / "summary" / f"{source}_summary.csv", "itemid,total_count\n30,1\n")
    _write_csv(root / "summary" / "diag_features.csv", "new_icd_code\nA01\n")
    _write_csv(root / "summary" / "diag_summary.csv", "new_icd_code,total_count\nA01,1\n")
    _write_csv(root / "reference.csv", "stay_id,label\n1,0\n")


def _build_split(root: Path) -> Path:
    for split_name in ("train", "validation", "test"):
        split_root = root / split_name
        (split_root / "raw" / "hosp").mkdir(parents=True)
        (split_root / "raw" / "icu").mkdir(parents=True)
        _write_csv(split_root / "keys.csv", "stay_id\n1\n")
        reference_root = split_root / ("reference" if split_name == "train" else "reference_private")
        _write_reference_package(reference_root)
    (root / "split_manifest.json").write_text(
        json.dumps({"dataset_profile": "mimic_iv_3_1", "split_key": "stay_id"}),
        encoding="utf-8",
    )
    return root


def _write_synthetic_pipeline(
    pipeline: Path,
    *,
    expected_files: list[str],
    parent_pipeline_sha256: str = "",
) -> None:
    pipeline.mkdir(parents=True, exist_ok=True)
    run_source = """\
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--raw-root", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--split-mode", choices=("train", "validation", "test"), required=True)
args = parser.parse_args()
source = Path(args.raw_root) / "source"
target = Path(args.output_dir)
if target.exists():
    shutil.rmtree(target)
shutil.copytree(source, target)
"""
    _write_csv(pipeline / "run.py", run_source)
    for module in PIPELINE_MODULES:
        _write_csv(pipeline / module, "# synthetic pipeline module\n")
    _write_csv(pipeline / "config.yaml", "schema_version: 1\n")
    (pipeline / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "SUCCESS",
                "entrypoint": "run.py",
                "modules": list(PIPELINE_MODULES),
                "expected_files": expected_files,
                "parent_pipeline_sha256": parent_pipeline_sha256,
            }
        ),
        encoding="utf-8",
    )


def _write_host_replay_success(candidate: Path) -> None:
    _write_json(
        candidate / "host_replay" / "pipeline_replay_report.json",
        {"schema_version": 1, "status": "SUCCESS", "valid": True, "issues": []},
    )


def _expected_business_files(reference: Path) -> list[str]:
    return sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )


def _build_runtime_with_formal_round(
    tmp_path: Path,
) -> tuple[DataCleaningAgentRuntime, dict, Path]:
    split = _build_split(tmp_path / "split")
    experiment = tmp_path / "experiment"
    contract = infer_reference_contract(split)
    contract_path = experiment / "reference_contract.json"
    _write_json(contract_path, contract)
    runtime = DataCleaningAgentRuntime(
        config=ReferenceGuidedConfig(split, experiment, "task", max_iters=10).normalized(),
        contract=contract,
        contract_path=contract_path,
    )
    candidate = experiment / "candidates/attempt_0001"
    shutil.copytree(split / "train/reference", candidate / "results/result_package")
    _write_synthetic_pipeline(
        candidate / "script_bundle/pipeline",
        expected_files=_expected_business_files(split / "train/reference"),
    )
    _write_host_replay_success(candidate)
    evaluation_report = experiment / "evaluations/attempt_0001/evaluation_report.json"
    _write_json(evaluation_report, {"schema_version": 2, "metrics": {"composite_score": 0.8}})
    round_dir = _archive_promoted_round(
        experiment_dir=experiment,
        candidate=candidate,
        evaluation={"evaluation_report": str(evaluation_report)},
        round_index=1,
        attempt_index=1,
        score=0.8,
    )
    shutil.copytree(round_dir / "script_bundle", experiment / "active_bundle/script_bundle")
    shutil.copytree(round_dir / "validation_result", experiment / "active_bundle/results/result_package")
    state = {
        "schema_version": 2,
        "status": "active",
        "best_score": 0.8,
        "best_round": 1,
        "best_attempt": 1,
        "best_feedback": "",
        "rounds": [{"round": 1, "attempt": 1, "score": 0.8}],
        "attempts": [],
        "checkpoints": [],
        "sessions": [],
        "abandoned_checkpoints": [],
        "consecutive_valid_no_improvement": 0,
        "invalid_attempt_count": 0,
    }
    return runtime, state, round_dir


def test_cli_exposes_only_three_main_workflows() -> None:
    args = parse_args(["--workflow", "reference-test-evaluate"])
    assert args.workflow == "reference-test-evaluate"
    with pytest.raises(SystemExit):
        parse_args(["--workflow", "train-validate"])


def test_cli_exposes_attempt_and_promotion_limits() -> None:
    args = parse_args(
        [
            "--workflow",
            "reference-guided-train-validate",
            "--patience",
            "3",
            "--max-attempts",
            "25",
            "--target-score",
            "0.96",
        ]
    )
    assert args.patience == 3
    assert args.max_attempts == 25
    assert args.target_score == 0.96


def test_reference_contract_accepts_directory_package(tmp_path: Path) -> None:
    contract = infer_reference_contract(_build_split(tmp_path / "split"))
    assert contract["status"] == "supported"
    assert contract["reference_type"] == "reference_directory"
    assert contract["key_column"] == "stay_id"


def test_run_manifest_allows_same_run_resume_and_rejects_legacy_or_changed_prompt(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    config = ReferenceGuidedConfig(split, experiment, "original prompt").normalized()

    first = _ensure_run_manifest(config, contract, experiment)
    resumed = _ensure_run_manifest(config, contract, experiment)

    assert first["run_id"] == resumed["run_id"]
    assert first["scorer_version"] == "balanced_row_aligned_v2"
    with pytest.raises(ValueError, match="run manifest does not match"):
        _ensure_run_manifest(
            ReferenceGuidedConfig(split, experiment, "changed prompt").normalized(),
            contract,
            experiment,
        )

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "experiment_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="historical experiment"):
        _ensure_run_manifest(
            ReferenceGuidedConfig(split, legacy, "prompt").normalized(),
            contract,
            legacy,
        )


def test_directory_evaluation_replays_identical_package(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    result = tmp_path / "result"
    _write_reference_package(reference)
    shutil.copytree(reference, result)
    paths = evaluate_reference_directory_package(
        result_package=result,
        validation_reference_root=reference,
        output_dir=tmp_path / "evaluation",
        key_column="stay_id",
    )
    report = json.loads(Path(paths["evaluation_report"]).read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["scorer_version"] == "balanced_row_aligned_v2"
    assert report["metrics"]["composite_score"] == 1.0
    assert report["metrics"]["file_coverage"] == 1.0


def test_reference_round_requires_a_published_workspace_package(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    phase_root = tmp_path / "phase"
    artifact_package = phase_root / "artifacts" / "step01_ExecutePython" / "result_package"
    _write_reference_package(artifact_package)

    assert _complete_reference_package_paths(phase_root, contract) is None

    shutil.copytree(artifact_package, phase_root / "workspace" / "result_package")
    package = _complete_reference_package_paths(phase_root, contract)
    assert package is not None
    assert package["result_package"] == phase_root / "workspace" / "result_package"


def test_reference_toolkit_registers_only_retained_skills(tmp_path: Path) -> None:
    context = EngineerToolContext.from_task(task_text="MIMIC ICU mortality", engineer_phase_root=tmp_path)
    _, manifest = create_reference_toolkit(context)
    assert {item["name"] for item in manifest} == DATA_CLEANING_AGENT_PIPELINE_SKILLS


def test_reference_test_stage_runs_adapter_against_hidden_test_reference(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import os\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "source = Path(os.environ['TRAIN_REF_ROOT'])\n"
        "target = Path(os.environ['OUTPUT_DIR']) / 'result_package'\n"
        "shutil.copytree(source, target)\n",
        encoding="utf-8",
    )
    report = run_reference_test_stage(
        ReferenceTestStageConfig(
            dataset_split=split,
            experiment_dir=tmp_path / "experiment",
            adapter_script=adapter,
        )
    )
    assert report["status"] == "SUCCESS"
    assert report["metrics"]["composite_score"] == 1.0


def test_checkpoint_test_contract_contains_no_private_reference_paths(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)

    sanitized = _build_sanitized_test_contract(contract)
    serialized = json.dumps(sanitized)

    assert sanitized["split_mode"] == "test"
    assert sanitized["paths"]["test_raw"].endswith("test/raw")
    assert sanitized["paths"]["test_keys"].endswith("test/keys.csv")
    assert "reference_private" not in serialized
    assert "validation_reference" not in serialized
    assert "test_reference" not in serialized


def test_checkpoint_agent_read_roots_block_private_reference_and_other_experiments(
    tmp_path: Path,
) -> None:
    split = _build_split(tmp_path / "split")
    checkpoint = tmp_path / "experiment/test_checkpoints/checkpoint_0001_best_round_0001"
    bundle = checkpoint / "frozen_script_bundle"
    _write_csv(bundle / "runner.py", "print('runner')\n")
    runtime = ReferenceCheckpointTestRuntime(
        ReferenceCheckpointTestConfig(
            dataset_split=split,
            experiment_dir=tmp_path / "experiment",
            checkpoint_dir=checkpoint,
            frozen_script_bundle=bundle,
            best_round=1,
            best_attempt=1,
            best_score=0.9,
            max_iters=10,
        )
    )
    phase_root = checkpoint / "agent_runs/data_cleaning_agent/phase"
    sanitized_contract = phase_root / "sanitized_test_contract.json"
    _write_json(sanitized_contract, runtime.sanitized_contract)
    context = EngineerToolContext(
        engineer_phase_root=phase_root,
        read_roots=runtime._read_roots(sanitized_contract),
        split_mode="test",
    )
    other_experiment = tmp_path / "other_experiment"
    other_experiment.mkdir()

    assert context.resolve_read_path(bundle / "runner.py").is_file()
    with pytest.raises(EngineerToolPermissionError):
        context.resolve_read_path(split / "test/reference_private")
    with pytest.raises(EngineerToolPermissionError):
        context.resolve_read_path(other_experiment)


def test_checkpoint_prompt_only_allows_runner_spec_not_business_scripts(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    checkpoint = tmp_path / "experiment/test_checkpoints/checkpoint_0001_best_round_0001"
    bundle = checkpoint / "frozen_script_bundle"
    _write_csv(bundle / "pipeline/run.py", "print('runner')\n")
    runtime = ReferenceCheckpointTestRuntime(
        ReferenceCheckpointTestConfig(
            dataset_split=split,
            experiment_dir=tmp_path / "experiment",
            checkpoint_dir=checkpoint,
            frozen_script_bundle=bundle,
            best_round=1,
            best_attempt=1,
            best_score=0.9,
            max_iters=10,
        )
    )
    phase_root = checkpoint / "agent_runs/data_cleaning_agent/phase"
    prompt = runtime._task_text(phase_root, phase_root / "sanitized_test_contract.json")

    assert "runner_spec.json" in prompt
    assert "不得创建 build_test.py" in prompt
    assert "自己阅读冻结脚本，决定" not in prompt
    assert "发布完整目录" not in prompt


def test_runner_spec_gate_rejects_test_business_script(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    reference = split / "train/reference"
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    bundle = tmp_path / "frozen_script_bundle"
    _write_synthetic_pipeline(bundle / "pipeline", expected_files=expected)
    workspace = tmp_path / "workspace"
    output = tmp_path / "test_run/result_package"
    runner_spec = workspace / "runner_spec.json"
    _write_json(
        runner_spec,
        {
            "schema_version": 1,
            "entrypoint": str((bundle / "pipeline/run.py").resolve()),
            "pipeline_sha256": pipeline_directory_sha256(bundle / "pipeline"),
            "raw_root": str((split / "test/raw").resolve()),
            "output_dir": str(output.resolve()),
            "split_mode": "test",
        },
    )

    valid = _runner_spec_quality_gate(
        runner_spec=runner_spec,
        workspace=workspace,
        frozen_script_bundle=bundle,
        test_raw=split / "test/raw",
        output_dir=output,
        train_reference_root=reference,
    )
    assert valid["valid"] is True

    _write_csv(workspace / "build_test.py", "print('rewritten rules')\n")
    invalid = _runner_spec_quality_gate(
        runner_spec=runner_spec,
        workspace=workspace,
        frozen_script_bundle=bundle,
        test_raw=split / "test/raw",
        output_dir=output,
        train_reference_root=reference,
    )
    assert invalid["valid"] is False
    assert any("forbidden" in issue for issue in invalid["issues"])


def test_checkpoint_test_gate_failure_never_calls_private_evaluator(tmp_path: Path, monkeypatch) -> None:
    split = _build_split(tmp_path / "split")
    checkpoint = tmp_path / "experiment/test_checkpoints/checkpoint_0001_best_round_0001"
    bundle = checkpoint / "frozen_script_bundle"
    reference = split / "train/reference"
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    _write_synthetic_pipeline(bundle / "pipeline", expected_files=expected)
    _write_csv(split / "test/raw/source/csv/labels.csv", "stay_id,label\n1,0\n")
    runtime = ReferenceCheckpointTestRuntime(
        ReferenceCheckpointTestConfig(
            dataset_split=split,
            experiment_dir=tmp_path / "experiment",
            checkpoint_dir=checkpoint,
            frozen_script_bundle=bundle,
            best_round=1,
            best_attempt=1,
            best_score=0.9,
            max_iters=10,
        )
    )

    def fake_agent_session() -> Path:
        runner_spec = checkpoint / "agent_runs/runner_spec.json"
        _write_json(runner_spec, {"schema_version": 1})
        runtime._last_gate = {"valid": True, "status": "SUCCESS", "issues": []}
        return runner_spec

    monkeypatch.setattr(runtime, "_run_agent_session", fake_agent_session)
    monkeypatch.setattr(
        reference_guided,
        "evaluate_reference_directory_package",
        lambda **kwargs: pytest.fail("private evaluator must not run after gate failure"),
    )

    report = runtime.run_sync()

    assert report["test_status"] == "pipeline_rule_failure"
    assert report["evaluation"] == {}
    assert report["gate"]["valid"] is False
    assert not (checkpoint / "test_evaluation").exists()


def test_checkpoint_test_evaluates_only_after_gate_and_preserves_frozen_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    split = _build_split(tmp_path / "split")
    checkpoint = tmp_path / "experiment/test_checkpoints/checkpoint_0001_best_round_0001"
    bundle = checkpoint / "frozen_script_bundle"
    reference = split / "train/reference"
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    _write_synthetic_pipeline(bundle / "pipeline", expected_files=expected)
    shutil.copytree(reference, split / "test/raw/source")
    runtime = ReferenceCheckpointTestRuntime(
        ReferenceCheckpointTestConfig(
            dataset_split=split,
            experiment_dir=tmp_path / "experiment",
            checkpoint_dir=checkpoint,
            frozen_script_bundle=bundle,
            best_round=1,
            best_attempt=1,
            best_score=0.9,
            max_iters=10,
        )
    )

    def fake_agent_session() -> Path:
        runner_spec = checkpoint / "agent_runs/runner_spec.json"
        _write_json(runner_spec, {"schema_version": 1})
        runtime._last_gate = {"valid": True, "status": "SUCCESS", "issues": []}
        return runner_spec

    monkeypatch.setattr(runtime, "_run_agent_session", fake_agent_session)

    report = runtime.run_sync()

    assert report["test_status"] == "success"
    assert report["metrics"]["composite_score"] == 1.0
    assert report["frozen_script_bundle_sha256_before"] == report["frozen_script_bundle_sha256_after"]
    assert (checkpoint / "test_run/result_package/csv/labels.csv").is_file()
    assert (checkpoint / "test_evaluation/private_report.json").is_file()


def test_checkpoint_gate_rejects_unknown_test_key_in_feature_file(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    package = tmp_path / "result_package"
    shutil.copytree(split / "train/reference", package)
    diag = package / "features/preproc_diag_icu.csv"
    diag.write_text("stay_id,new_icd_code\n999,A01\n", encoding="utf-8")
    bundle = tmp_path / "frozen_script_bundle"
    _write_csv(bundle / "runner.py", "print('runner')\n")

    gate = _test_package_quality_gate(
        result_package=package,
        train_reference_root=split / "train/reference",
        test_keys=split / "test/keys.csv",
        key_column="stay_id",
        frozen_script_bundle=bundle,
        frozen_hash_before=reference_guided._directory_sha256(bundle),
    )

    assert gate["valid"] is False
    assert any("unknown test keys" in issue for issue in gate["issues"])


def test_checkpoint_gate_allows_labels_to_cover_a_test_key_subset(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    _write_csv(split / "test/keys.csv", "stay_id\n1\n2\n")
    package = tmp_path / "result_package"
    shutil.copytree(split / "train/reference", package)
    _write_csv(
        package / "cohort/cohort_icu_mortality.csv",
        "stay_id,label\n1,0\n2,1\n",
    )
    bundle = tmp_path / "frozen_script_bundle"
    _write_csv(bundle / "runner.py", "print('runner')\n")

    gate = _test_package_quality_gate(
        result_package=package,
        train_reference_root=split / "train/reference",
        test_keys=split / "test/keys.csv",
        key_column="stay_id",
        frozen_script_bundle=bundle,
        frozen_hash_before=reference_guided._directory_sha256(bundle),
    )

    assert gate["valid"] is True
    assert not any("csv/labels.csv: missing" in issue for issue in gate["issues"])


def test_checkpoint_gate_rejects_unknown_test_key_in_labels_subset(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    package = tmp_path / "result_package"
    shutil.copytree(split / "train/reference", package)
    _write_csv(package / "csv/labels.csv", "stay_id,label\n999,0\n")
    bundle = tmp_path / "frozen_script_bundle"
    _write_csv(bundle / "runner.py", "print('runner')\n")

    gate = _test_package_quality_gate(
        result_package=package,
        train_reference_root=split / "train/reference",
        test_keys=split / "test/keys.csv",
        key_column="stay_id",
        frozen_script_bundle=bundle,
        frozen_hash_before=reference_guided._directory_sha256(bundle),
    )

    assert gate["valid"] is False
    assert any("csv/labels.csv: contains 1 unknown test keys" in issue for issue in gate["issues"])


def test_checkpoint_gate_still_requires_complete_cohort_key_coverage(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    _write_csv(split / "test/keys.csv", "stay_id\n1\n2\n")
    package = tmp_path / "result_package"
    shutil.copytree(split / "train/reference", package)
    bundle = tmp_path / "frozen_script_bundle"
    _write_csv(bundle / "runner.py", "print('runner')\n")

    gate = _test_package_quality_gate(
        result_package=package,
        train_reference_root=split / "train/reference",
        test_keys=split / "test/keys.csv",
        key_column="stay_id",
        frozen_script_bundle=bundle,
        frozen_hash_before=reference_guided._directory_sha256(bundle),
    )

    assert gate["valid"] is False
    assert any(
        "cohort/cohort_icu_mortality.csv: missing 1 test keys" in issue
        for issue in gate["issues"]
    )


def test_checkpoint_gate_reports_feature_business_key_duplicates_without_rejecting(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    package = tmp_path / "result_package"
    shutil.copytree(split / "train/reference", package)
    chart = package / "features/preproc_chart_icu.csv"
    chart.write_text(
        "stay_id,itemid,event_time_from_admit\n"
        "1,220045,1.0\n"
        "1,220045,1.0\n",
        encoding="utf-8",
    )
    bundle = tmp_path / "frozen_script_bundle"
    _write_csv(bundle / "runner.py", "print('runner')\n")

    gate = _test_package_quality_gate(
        result_package=package,
        train_reference_root=split / "train/reference",
        test_keys=split / "test/keys.csv",
        key_column="stay_id",
        frozen_script_bundle=bundle,
        frozen_hash_before=reference_guided._directory_sha256(bundle),
    )

    assert gate["valid"] is True
    chart_report = next(
        item for item in gate["files"]
        if item["relative_path"] == "features/preproc_chart_icu.csv"
    )
    assert chart_report["duplicate_business_key_count"] == 1
    assert any("duplicate business keys" in item for item in chart_report["diagnostics"])


def test_validation_and_test_share_the_same_business_package_gate(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    package = tmp_path / "result_package"
    shutil.copytree(split / "train/reference", package)
    _write_csv(
        package / "features/preproc_chart_icu.csv",
        "stay_id,itemid,event_time_from_admit\n"
        "1,220045,1.0\n"
        "1,220045,1.0\n",
    )

    validation_gate = validate_business_result_package(
        result_package=package,
        train_reference_root=split / "train/reference",
        split_keys=split / "validation/keys.csv",
        key_column="stay_id",
        split_mode="validation",
    )
    test_gate = validate_business_result_package(
        result_package=package,
        train_reference_root=split / "train/reference",
        split_keys=split / "test/keys.csv",
        key_column="stay_id",
        split_mode="test",
    )

    assert validation_gate["valid"] is True
    assert test_gate["valid"] is True
    assert validation_gate["business_gate_fingerprint"] == test_gate["business_gate_fingerprint"]
    assert validation_gate["files"] == test_gate["files"]


def test_shared_business_gate_rejects_extra_and_empty_files(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    package = tmp_path / "package"
    shutil.copytree(split / "train/reference", package)
    _write_csv(package / "features/unexpected.csv", "stay_id,value\n1,1\n")
    _write_csv(package / "summary/chart_features.csv", "itemid\n")

    gate = validate_business_result_package(
        result_package=package,
        train_reference_root=split / "train/reference",
        split_keys=split / "validation/keys.csv",
        key_column="stay_id",
        split_mode="validation",
        required_file_count=17,
    )

    assert gate["valid"] is False
    assert any("unexpected business files" in issue for issue in gate["issues"])
    assert any("empty output file" in issue for issue in gate["issues"])


def test_shared_business_gate_rejects_rows_with_extra_csv_cells(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    package = tmp_path / "package"
    shutil.copytree(split / "train/reference", package)
    cohort = next((package / "cohort").glob("*.csv"))
    lines = cohort.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1] + ",unexpected-extra-cell"
    cohort.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gate = validate_business_result_package(
        result_package=package,
        train_reference_root=split / "train/reference",
        split_keys=split / "train/keys.csv",
        key_column="stay_id",
        split_mode="train",
        required_file_count=17,
    )

    assert gate["valid"] is False
    assert any("extra CSV cells" in issue for issue in gate["issues"])


def test_shared_business_gate_checks_placeholder_values_beyond_first_thousand_rows(
    tmp_path: Path,
) -> None:
    split = _build_split(tmp_path / "split")
    package = tmp_path / "package"
    shutil.copytree(split / "train/reference", package)
    values = "itemid\n" + "placeholder\n" * 1000 + "30\n"
    _write_csv(package / "summary/chart_features.csv", values)

    gate = validate_business_result_package(
        result_package=package,
        train_reference_root=split / "train/reference",
        split_keys=split / "validation/keys.csv",
        key_column="stay_id",
        split_mode="validation",
        required_file_count=17,
    )

    chart_report = next(
        item for item in gate["files"] if item["relative_path"] == "summary/chart_features.csv"
    )
    assert chart_report["passed"] is True


def test_pipeline_structure_rejects_missing_entrypoint_and_wrong_parent(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    _write_reference_package(reference)
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=expected)
    (pipeline / "run.py").unlink()

    missing_entrypoint = validate_pipeline_structure(
        pipeline,
        expected_files=expected,
        expected_parent_sha256="",
    )
    assert missing_entrypoint["valid"] is False
    assert any("run.py" in issue for issue in missing_entrypoint["issues"])

    _write_synthetic_pipeline(pipeline, expected_files=expected, parent_pipeline_sha256="wrong")
    wrong_parent = validate_pipeline_structure(
        pipeline,
        expected_files=expected,
        expected_parent_sha256="expected-parent",
    )
    assert wrong_parent["valid"] is False
    assert any("parent_pipeline_sha256" in issue for issue in wrong_parent["issues"])


def test_pipeline_structure_rejects_undeclared_entrypoints_and_embedded_data(
    tmp_path: Path,
) -> None:
    expected = ["cohort/cohort.csv"]
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=expected)
    (pipeline / "build_test.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    embedded = pipeline / "_static_summary" / "summary.csv"
    embedded.parent.mkdir(parents=True)
    embedded.write_text("itemid,total_count\n1,10\n", encoding="utf-8")

    report = validate_pipeline_structure(
        pipeline,
        expected_files=expected,
        expected_parent_sha256="",
    )

    assert report["valid"] is False
    assert any("undeclared files" in issue for issue in report["issues"])
    assert any("embedded result data" in issue for issue in report["issues"])


def test_pipeline_structure_allows_declared_train_derived_summary_assets(
    tmp_path: Path,
) -> None:
    split = _build_split(tmp_path / "split")
    reference = split / "train/reference"
    expected = _expected_business_files(reference)
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=expected)
    source_relative = "summary/chart_summary.csv"
    asset_relative = "summary_assets/chart_summary.csv"
    (pipeline / "summary_assets").mkdir()
    shutil.copy2(reference / source_relative, pipeline / asset_relative)
    manifest_path = pipeline / "pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["learned_summary_assets"] = [
        {
            "source": source_relative,
            "path": asset_relative,
            "sha256": hashlib.sha256((reference / source_relative).read_bytes()).hexdigest(),
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_pipeline_structure(
        pipeline,
        expected_files=expected,
        expected_parent_sha256="",
        train_reference_root=reference,
    )

    assert report["valid"] is True
    assert report["learned_summary_assets"] == [asset_relative]


def test_pipeline_structure_rejects_summary_assets_with_wrong_source_or_patient_columns(
    tmp_path: Path,
) -> None:
    split = _build_split(tmp_path / "split")
    reference = split / "train/reference"
    expected = _expected_business_files(reference)
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=expected)
    asset_relative = "summary_assets/chart_summary.csv"
    _write_csv(pipeline / asset_relative, "stay_id,total_count\n1,10\n")
    manifest_path = pipeline / "pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["learned_summary_assets"] = [
        {
            "source": "summary/chart_summary.csv",
            "path": asset_relative,
            "sha256": hashlib.sha256((pipeline / asset_relative).read_bytes()).hexdigest(),
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_pipeline_structure(
        pipeline,
        expected_files=expected,
        expected_parent_sha256="",
        train_reference_root=reference,
    )

    assert report["valid"] is False
    assert any("train/reference summary source" in issue for issue in report["issues"])
    assert any("patient-level columns" in issue for issue in report["issues"])


def test_pipeline_structure_rejects_forbidden_result_source_reference(tmp_path: Path) -> None:
    expected = ["cohort/cohort.csv"]
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=expected)
    (pipeline / "run.py").write_text(
        'SOURCE = "/tmp/experiment/active_bundle/results/result_package"\n',
        encoding="utf-8",
    )

    report = validate_pipeline_structure(
        pipeline,
        expected_files=expected,
        expected_parent_sha256="",
    )

    assert report["valid"] is False
    assert any("forbidden result/private" in issue for issue in report["issues"])


def test_pipeline_structure_rejects_invalid_manifest_types_without_raising(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=["cohort/cohort.csv"])
    (pipeline / "pipeline_manifest.json").write_text("[]\n", encoding="utf-8")

    list_manifest = validate_pipeline_structure(
        pipeline,
        expected_files=["cohort/cohort.csv"],
        expected_parent_sha256="",
    )
    assert list_manifest["valid"] is False
    assert any("JSON object" in issue for issue in list_manifest["issues"])

    (pipeline / "pipeline_manifest.json").write_text(
        json.dumps({"schema_version": "one"}),
        encoding="utf-8",
    )
    bad_version = validate_pipeline_structure(
        pipeline,
        expected_files=["cohort/cohort.csv"],
        expected_parent_sha256="",
    )
    assert bad_version["valid"] is False
    assert any("schema_version" in issue for issue in bad_version["issues"])

    (pipeline / "pipeline_manifest.json").write_text(
        json.dumps({"schema_version": 1, "modules": 1, "expected_files": 2}),
        encoding="utf-8",
    )
    bad_fields = validate_pipeline_structure(
        pipeline,
        expected_files=["cohort/cohort.csv"],
        expected_parent_sha256="",
    )
    assert bad_fields["valid"] is False
    assert any("modules must be a list" in issue for issue in bad_fields["issues"])
    assert any("expected_files must be a list" in issue for issue in bad_fields["issues"])


def test_pipeline_structure_rejects_large_embedded_literal_payload(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=["cohort/cohort.csv"])
    payload = ",".join(str(30_000_000 + index) for index in range(600))
    (pipeline / "cohort.py").write_text(f'STAY_IDS = "{payload}"\n', encoding="utf-8")

    report = validate_pipeline_structure(
        pipeline,
        expected_files=["cohort/cohort.csv"],
        expected_parent_sha256="",
    )

    assert report["valid"] is False
    assert any("embedded literal" in issue for issue in report["issues"])


def test_pipeline_structure_rejects_split_embedded_literal_payload(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=["cohort/cohort.csv"])
    chunks = ["x" * 3000 for _ in range(100)]
    (pipeline / "cohort.py").write_text(
        "PAYLOAD = " + repr(chunks) + "\n",
        encoding="utf-8",
    )

    report = validate_pipeline_structure(
        pipeline,
        expected_files=["cohort/cohort.csv"],
        expected_parent_sha256="",
    )

    assert report["valid"] is False
    assert any("total literal payload" in issue for issue in report["issues"])


def test_pipeline_structure_rejects_unrelated_module_change(tmp_path: Path) -> None:
    expected = ["features/preproc_diag_icu.csv"]
    previous = tmp_path / "previous"
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(previous, expected_files=expected)
    parent_hash = pipeline_directory_sha256(previous)
    shutil.copytree(previous, pipeline)
    manifest = json.loads((pipeline / "pipeline_manifest.json").read_text(encoding="utf-8"))
    manifest["parent_pipeline_sha256"] = parent_hash
    (pipeline / "pipeline_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pipeline / "chart.py").write_text("CHANGED = True\n", encoding="utf-8")

    report = validate_pipeline_structure(
        pipeline,
        expected_files=expected,
        expected_parent_sha256=parent_hash,
        previous_pipeline_dir=previous,
        allowed_changed_modules={"diag.py"},
    )

    assert report["valid"] is False
    assert any("unrelated pipeline modules" in issue for issue in report["issues"])

    (pipeline / "chart.py").write_text(
        (previous / "chart.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (pipeline / "run.py").write_text("CHANGED = True\n", encoding="utf-8")
    (pipeline / "config.yaml").write_text("changed: true\n", encoding="utf-8")
    support_change = validate_pipeline_structure(
        pipeline,
        expected_files=expected,
        expected_parent_sha256=parent_hash,
        previous_pipeline_dir=previous,
        allowed_changed_modules={"diag.py"},
    )
    assert support_change["valid"] is False
    assert any("unrelated pipeline rule files" in issue for issue in support_change["issues"])


def test_pipeline_isolated_replay_is_deterministic(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    reference = split / "train/reference"
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=expected)
    for name in ("train", "validation", "test"):
        shutil.copytree(reference, split / name / "raw/source")

    first = tmp_path / "replay_first"
    second = tmp_path / "replay_second"
    first_run = run_pipeline_isolated(
        pipeline_dir=pipeline,
        raw_root=split / "validation/raw",
        output_dir=first,
        split_mode="validation",
        log_path=tmp_path / "first.log",
    )
    second_run = run_pipeline_isolated(
        pipeline_dir=pipeline,
        raw_root=split / "validation/raw",
        output_dir=second,
        split_mode="validation",
        log_path=tmp_path / "second.log",
    )

    assert first_run["returncode"] == 0
    assert second_run["returncode"] == 0
    assert business_file_hashes(first) == business_file_hashes(second)


def test_pipeline_isolation_blocks_process_escape(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(
        pipeline,
        expected_files=_expected_business_files(split / "train/reference"),
    )
    (pipeline / "run.py").write_text(
        'import os\nos.execv("/bin/sh", ["/bin/sh", "-c", "exit 0"])\n',
        encoding="utf-8",
    )

    result = run_pipeline_isolated(
        pipeline_dir=pipeline,
        raw_root=split / "validation/raw",
        output_dir=tmp_path / "output",
        split_mode="validation",
        log_path=tmp_path / "escape.log",
    )

    assert result["returncode"] != 0


def test_pipeline_isolation_blocks_native_read_of_private_reference(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    private_file = split / "validation/reference_private/csv/labels.csv"
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(
        pipeline,
        expected_files=_expected_business_files(split / "train/reference"),
    )
    (pipeline / "run.py").write_text(
        "import ctypes\n"
        'libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")\n'
        f'fd = libc.open({str(private_file).encode()!r}, 0)\n'
        "raise SystemExit(0 if fd >= 0 else 7)\n",
        encoding="utf-8",
    )

    result = run_pipeline_isolated(
        pipeline_dir=pipeline,
        raw_root=split / "validation/raw",
        output_dir=tmp_path / "output",
        split_mode="validation",
        log_path=tmp_path / "native_escape.log",
    )

    assert result["returncode"] == 7


def test_pipeline_isolation_blocks_native_read_and_write_outside_allowlist(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    outside_read = tmp_path / "outside-secret.txt"
    outside_read.write_text("secret", encoding="utf-8")
    outside_write = tmp_path / "outside-write.txt"
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(
        pipeline,
        expected_files=_expected_business_files(split / "train/reference"),
    )
    (pipeline / "run.py").write_text(
        "import ctypes\n"
        'libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")\n'
        f'fd_read = libc.open({str(outside_read).encode()!r}, 0)\n'
        f'fd_write = libc.open({str(outside_write).encode()!r}, 0x601, 0o600)\n'
        "raise SystemExit(0 if fd_read >= 0 or fd_write >= 0 else 9)\n",
        encoding="utf-8",
    )

    result = run_pipeline_isolated(
        pipeline_dir=pipeline,
        raw_root=split / "validation/raw",
        output_dir=tmp_path / "output",
        split_mode="validation",
        log_path=tmp_path / "native_allowlist.log",
    )

    assert result["returncode"] == 9
    assert outside_write.exists() is False


def test_pipeline_isolation_times_out_infinite_pipeline(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(
        pipeline,
        expected_files=_expected_business_files(split / "train/reference"),
    )
    (pipeline / "run.py").write_text("while True:\n    pass\n", encoding="utf-8")

    result = run_pipeline_isolated(
        pipeline_dir=pipeline,
        raw_root=split / "validation/raw",
        output_dir=tmp_path / "output",
        split_mode="validation",
        log_path=tmp_path / "timeout.log",
        timeout_seconds=1,
    )

    assert result["status"] == "TIMEOUT"
    assert result["timed_out"] is True


def test_pipeline_isolation_stops_excessive_output_file_count(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(
        pipeline,
        expected_files=_expected_business_files(split / "train/reference"),
    )
    (pipeline / "run.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--raw-root'); p.add_argument('--output-dir'); p.add_argument('--split-mode'); a=p.parse_args()\n"
        "out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)\n"
        "[(out / f'{i}.txt').write_text('x') for i in range(20)]\n",
        encoding="utf-8",
    )

    result = run_pipeline_isolated(
        pipeline_dir=pipeline,
        raw_root=split / "validation/raw",
        output_dir=tmp_path / "output",
        split_mode="validation",
        log_path=tmp_path / "resource.log",
        max_output_files=5,
    )

    assert result["status"] == "RESOURCE_LIMIT"
    assert "file count" in result["resource_limit_reason"]


def test_pipeline_isolation_allows_normal_pandas_import(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    reference = split / "train/reference"
    shutil.copytree(reference, split / "validation/raw/source")
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(
        pipeline,
        expected_files=_expected_business_files(reference),
    )
    run_path = pipeline / "run.py"
    run_path.write_text("import pandas as pd\n" + run_path.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_pipeline_isolated(
        pipeline_dir=pipeline,
        raw_root=split / "validation/raw",
        output_dir=tmp_path / "output",
        split_mode="validation",
        log_path=tmp_path / "pandas.log",
    )

    assert result["returncode"] == 0


def test_pipeline_replay_rejects_manual_candidate_csv_change(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    reference = split / "train/reference"
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    for name in ("train", "validation"):
        shutil.copytree(reference, split / name / "raw/source")
    pipeline = tmp_path / "pipeline"
    _write_synthetic_pipeline(pipeline, expected_files=expected)
    candidate = tmp_path / "candidate_result"
    shutil.copytree(reference, candidate)
    _write_csv(candidate / "csv/labels.csv", "stay_id,label\n1,1\n")

    gate = validate_and_replay_candidate_pipeline(
        pipeline_dir=pipeline,
        previous_pipeline_dir=None,
        candidate_result_package=candidate,
        train_raw=split / "train/raw",
        train_reference_root=reference,
        train_keys=split / "train/keys.csv",
        validation_raw=split / "validation/raw",
        validation_keys=split / "validation/keys.csv",
        key_column="stay_id",
        report_root=tmp_path / "host_replay",
    )

    assert gate["valid"] is False
    assert gate["pipeline_sha256"] == pipeline_directory_sha256(pipeline)
    assert gate["hash_consistency"]["matched"] is False
    assert "csv/labels.csv" in gate["hash_consistency"]["mismatched_files"]


def test_train_regression_uses_exact_counts_not_rounded_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_score = {
        "file_reports": [
            {
                "relative_path": "features/preproc_chart_icu.csv",
                "status": "compared",
                "reference_rows": 2_000_000,
                "result_rows": 2_000_000,
                "matched_key_rows": 2_000_000,
                "exact_row_matches": 1_999_999,
                "reference_columns": ["stay_id", "valuenum"],
                "result_columns": ["stay_id", "valuenum"],
                "per_column": {
                    "stay_id": {"matched_cells": 2_000_000},
                    "valuenum": {"matched_cells": 1_999_999},
                },
                "metrics": {
                    "schema_f1": 1.0,
                    "key_structure_f1": 1.0,
                    "row_aligned_cell_f1": 1.0,
                    "exact_row_f1": 1.0,
                },
            }
        ]
    }
    monkeypatch.setattr(
        "workflow.reference_pipeline.score_reference_directory",
        lambda *args, **kwargs: fake_score,
    )

    report = _build_train_regression_report(
        result_package=tmp_path / "result",
        train_reference_root=tmp_path / "reference",
        key_column="stay_id",
    )

    assert report["status"] == "NEEDS_REPAIR"
    assert report["files"][0]["passed"] is False


def test_formal_pipeline_is_inherited_and_staged_as_one_canonical_directory(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    _write_reference_package(reference)
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    active = tmp_path / "active"
    _write_synthetic_pipeline(
        active / "script_bundle/pipeline",
        expected_files=expected,
    )
    workspace = tmp_path / "workspace"
    parent_hash = _inherit_pipeline_to_workspace(active, workspace)

    assert parent_hash == pipeline_directory_sha256(active / "script_bundle/pipeline")
    assert (workspace / "pipeline/run.py").is_file()
    _write_csv(workspace / "pipeline/cohort.py", "UPDATED = True\n")

    candidate = tmp_path / "candidate"
    staged = _stage_workspace_pipeline(workspace, candidate)
    assert staged == candidate / "script_bundle/pipeline"
    assert (staged / "cohort.py").read_text(encoding="utf-8") == "UPDATED = True\n"
    assert (staged / "run.py").is_file()


def test_reference_split_preserves_requested_counts(tmp_path: Path) -> None:
    source = tmp_path / "reference.csv"
    _write_csv(
        source,
        "stay_id,label\n" + "".join(f"{index},{index % 2}\n" for index in range(10)),
    )
    result = split_teacher_reference(source, tmp_path / "split", counts=(2, 3, 5), seed=7)
    assert result["manifest"]["row_counts"] == {"source": 10, "train": 2, "validation": 3, "test": 5}


def test_v1_experiment_state_migrates_rounds_into_attempts_and_promotions() -> None:
    state = _migrate_reference_experiment_state(
        {
            "schema_version": 1,
            "status": "frozen",
            "best_score": 0.9342,
            "best_round": 1,
            "consecutive_no_improvement": 2,
            "rounds": [
                {
                    "round": 1,
                    "score": 0.9342,
                    "improved": True,
                    "evaluation": {"public_feedback": "/tmp/best-feedback.json"},
                },
                {"round": 2, "score": 0.8906, "improved": False, "evaluation": {}},
                {"round": 3, "score": 0.9342, "improved": False, "evaluation": {}},
            ],
        }
    )

    assert state["schema_version"] == 2
    assert [item["attempt"] for item in state["attempts"]] == [1, 2, 3]
    assert len(state["rounds"]) == 1
    assert state["rounds"][0]["attempt"] == 1
    assert state["best_round"] == 1
    assert state["best_attempt"] == 1
    assert state["best_feedback"] == "/tmp/best-feedback.json"
    assert state["consecutive_valid_no_improvement"] == 2


def test_v1_migration_does_not_count_needs_repair_candidate_as_valid_failure(tmp_path: Path) -> None:
    round_one = tmp_path / "round_0001"
    round_two = tmp_path / "round_0002"
    round_three = tmp_path / "round_0003"
    for path, status in ((round_one, "SUCCESS"), (round_two, "SUCCESS"), (round_three, "NEEDS_REPAIR")):
        path.mkdir()
        (path / "candidate_delta.json").write_text(
            json.dumps({"status": status, "has_feedback_linked_material_delta": status == "SUCCESS"}),
            encoding="utf-8",
        )
    state = _migrate_reference_experiment_state(
        {
            "schema_version": 1,
            "status": "frozen",
            "best_score": 0.9342,
            "best_round": 1,
            "consecutive_no_improvement": 2,
            "rounds": [
                {"round": 1, "score": 0.9342, "improved": True, "candidate_dir": str(round_one)},
                {"round": 2, "score": 0.8906, "improved": False, "candidate_dir": str(round_two)},
                {"round": 3, "score": 0.9342, "improved": False, "candidate_dir": str(round_three)},
            ],
        }
    )

    assert state["attempts"][-1]["status"] == "invalid"
    assert state["invalid_attempt_count"] == 1
    assert state["consecutive_valid_no_improvement"] == 1


def test_only_improved_attempt_creates_a_formal_round() -> None:
    state = _migrate_reference_experiment_state(
        {
            "schema_version": 2,
            "status": "active",
            "best_score": 0.8,
            "best_round": 1,
            "best_attempt": 1,
            "best_feedback": "/tmp/feedback-1.json",
            "rounds": [{"round": 1, "attempt": 1, "score": 0.8}],
            "attempts": [{"attempt": 1, "score": 0.8, "improved": True}],
            "consecutive_valid_no_improvement": 0,
            "invalid_attempt_count": 0,
        }
    )

    state, improved = _finish_reference_attempt_state(
        state,
        attempt_index=2,
        score=0.7,
        candidate_dir="/tmp/candidate-2",
        result={},
        evaluation={"public_feedback": "/tmp/feedback-2.json"},
    )
    assert improved is False
    assert len(state["rounds"]) == 1
    assert len(state["attempts"]) == 2
    assert state["best_score"] == 0.8
    assert state["best_feedback"] == "/tmp/feedback-1.json"
    assert state["consecutive_valid_no_improvement"] == 1

    state, improved = _finish_reference_attempt_state(
        state,
        attempt_index=3,
        score=0.9,
        candidate_dir="/tmp/candidate-3",
        result={},
        evaluation={"public_feedback": "/tmp/feedback-3.json"},
    )
    assert improved is True
    assert len(state["rounds"]) == 2
    assert state["best_round"] == 2
    assert state["best_attempt"] == 3
    assert state["best_feedback"] == "/tmp/feedback-3.json"
    assert state["consecutive_valid_no_improvement"] == 0


def test_promoted_round_archives_cumulative_scripts_results_and_evaluation(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    candidate = experiment / "candidates" / "attempt_0002"
    _write_csv(candidate / "script_bundle" / "workspace" / "previous.py", "PREVIOUS = True\n")
    _write_csv(
        candidate / "agent_runs" / "data_cleaning_agent" / "phase" / "workspace" / "build_package.py",
        "CURRENT = True\n",
    )
    _write_csv(candidate / "results" / "result_package" / "csv" / "labels.csv", "stay_id,label\n1,0\n")
    _write_synthetic_pipeline(
        candidate / "script_bundle/pipeline",
        expected_files=["csv/labels.csv"],
    )
    _write_json(candidate / "host_replay/pipeline_replay_report.json", {"valid": True})
    evaluation_dir = experiment / "evaluations" / "attempt_0002"
    evaluation_report = evaluation_dir / "evaluation_report.json"
    _write_csv(evaluation_report, '{"schema_version": 2, "metrics": {"composite_score": 0.9}}\n')

    round_dir = _archive_promoted_round(
        experiment_dir=experiment,
        candidate=candidate,
        evaluation={"evaluation_report": str(evaluation_report)},
        round_index=2,
        attempt_index=2,
        score=0.9,
    )

    assert (round_dir / "script_bundle/workspace/previous.py").is_file()
    assert (round_dir / "script_bundle/workspace/build_package.py").is_file()
    assert (round_dir / "validation_result/csv/labels.csv").is_file()
    assert (round_dir / "evaluation/evaluation_report.json").is_file()
    provenance = json.loads((round_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["round"] == 2
    assert provenance["attempt"] == 2
    assert provenance["score"] == 0.9
    assert provenance["script_bundle_sha256"]
    assert provenance["pipeline_sha256"] == pipeline_directory_sha256(
        round_dir / "script_bundle/pipeline"
    )
    assert provenance["entrypoint_sha256"]
    assert (round_dir / "host_replay/pipeline_replay_report.json").is_file()
    assert (candidate / "script_bundle/manifest.json").is_file()


def test_checkpoint_rejects_pipeline_tampered_after_formal_promotion(tmp_path: Path) -> None:
    runtime, state, round_dir = _build_runtime_with_formal_round(tmp_path)
    run_path = round_dir / "script_bundle/pipeline/run.py"
    run_path.write_text(run_path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    updated, canonical, _ = runtime._ensure_canonical_pipeline(state)

    assert canonical is None
    assert updated["reproducible"] is False
    assert any(
        "promotion provenance" in issue
        for issue in updated["checkpoint_pipeline_validation"]["issues"]
    )


def test_test_pipeline_rule_failure_returns_to_validation_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime, state, _ = _build_runtime_with_formal_round(tmp_path)

    def fail_test(self):
        report_path = self.checkpoint_dir / "checkpoint_report.json"
        report = {
            "test_status": "pipeline_rule_failure",
            "gate": {"issues": ["missing test business file"], "diagnostics": []},
            "metrics": {},
            "evaluation": {},
            "report_path": str(report_path),
        }
        _write_json(report_path, report)
        return report

    monkeypatch.setattr(ReferenceCheckpointTestRuntime, "run_sync", fail_test)

    result = runtime._checkpoint_and_test("patience", state)

    assert result["status"] == "validation_resume_required"
    assert result["force_validation_attempt"] is True
    feedback = Path(result["pending_test_rule_feedback"])
    assert feedback.is_file()
    assert "missing test business file" in feedback.read_text(encoding="utf-8")


def test_test_pipeline_rule_feedback_is_merged_into_machine_repair_targets(
    tmp_path: Path,
) -> None:
    feedback = tmp_path / "test_feedback.json"
    targets = tmp_path / "repair_targets.json"
    _write_json(
        feedback,
        {
            "issues": [
                "features/preproc_diag_icu.csv: missing output file",
                "summary/diag_summary.csv: empty output file",
            ]
        },
    )

    merged = _merge_test_rule_feedback_into_repair_targets(feedback, targets)

    assert merged["target_count"] == 2
    assert {
        item["relative_path"] for item in merged["targets"]
    } == {"features/preproc_diag_icu.csv", "summary/diag_summary.csv"}


def test_generic_test_rule_target_links_to_any_changed_business_file(tmp_path: Path) -> None:
    targets = tmp_path / "repair_targets.json"
    response = tmp_path / "feedback_response.json"
    _write_json(
        targets,
        {
            "targets": [
                {
                    "repair_target_id": "generic-test-rule",
                    "type": "test_pipeline_rule_failure",
                    "relative_path": "",
                }
            ]
        },
    )
    _write_json(
        response,
        {
            "responses": [
                {
                    "repair_target_id": "generic-test-rule",
                    "changed_files": ["results/result_package"],
                }
            ]
        },
    )

    audit = _audit_reference_feedback_response(
        changed_files=["results/result_package/features/preproc_diag_icu.csv"],
        repair_targets=targets,
        feedback_response=response,
    )

    assert audit["is_complete"] is True
    assert audit["has_linked_delta"] is True


def test_legacy_best_consolidates_pipeline_before_checkpoint_without_mutating_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    split = _build_split(tmp_path / "split")
    reference = split / "train/reference"
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    for name in ("train", "validation", "test"):
        shutil.copytree(reference, split / name / "raw/source")
    experiment = tmp_path / "experiment"
    contract = infer_reference_contract(split)
    contract_path = experiment / "reference_contract.json"
    _write_json(contract_path, contract)
    runtime = DataCleaningAgentRuntime(
        config=ReferenceGuidedConfig(
            split,
            experiment,
            "task",
            max_iters=10,
        ).normalized(),
        contract=contract,
        contract_path=contract_path,
    )
    for index in range(1, 4):
        round_dir = experiment / f"rounds/round_{index:04d}"
        _write_csv(round_dir / "script_bundle/workspace/legacy.py", f"ROUND = {index}\n")
        _write_json(round_dir / "provenance.json", {"round": index, "score": index / 10})
    shutil.copytree(reference, experiment / "rounds/round_0003/validation_result")
    shutil.copytree(experiment / "rounds/round_0003/script_bundle", experiment / "active_bundle/script_bundle")
    shutil.copytree(reference, experiment / "active_bundle/results/result_package")
    provenance_before = (
        experiment / "rounds/round_0003/provenance.json"
    ).read_bytes()

    def fake_consolidation_agent(consolidation_root: Path, best_round: int) -> Path:
        pipeline = consolidation_root / "agent_pipeline"
        _write_synthetic_pipeline(pipeline, expected_files=expected)
        return pipeline

    monkeypatch.setattr(runtime, "_run_pipeline_consolidation_agent", fake_consolidation_agent)

    def fake_checkpoint_test(self):
        report_path = self.checkpoint_dir / "checkpoint_report.json"
        _write_json(report_path, {"test_status": "success", "metrics": {}, "evaluation": {}})
        return {
            "test_status": "success",
            "metrics": {},
            "evaluation": {},
            "report_path": str(report_path),
        }

    monkeypatch.setattr(ReferenceCheckpointTestRuntime, "run_sync", fake_checkpoint_test)
    state = {
        "schema_version": 2,
        "status": "active",
        "best_score": 0.821243,
        "best_round": 3,
        "best_attempt": 6,
        "best_feedback": "",
        "rounds": [
            {"round": index, "attempt": index * 2, "score": score}
            for index, score in ((1, 0.637221), (2, 0.811414), (3, 0.821243))
        ],
        "attempts": [],
        "checkpoints": [],
        "sessions": [],
        "abandoned_checkpoints": [],
        "consecutive_valid_no_improvement": 0,
        "invalid_attempt_count": 0,
    }

    result = runtime._checkpoint_and_test("manual", state)

    consolidation = experiment / "rounds/round_0003/pipeline_consolidation"
    assert result["best_score"] == 0.821243
    assert result["reproducible"] is True
    assert Path(result["canonical_pipeline"]).is_dir()
    assert (consolidation / "consolidation_report.json").is_file()
    assert (
        experiment
        / "test_checkpoints/checkpoint_0001_best_round_0003/frozen_script_bundle/pipeline/run.py"
    ).is_file()
    assert (experiment / "rounds/round_0003/provenance.json").read_bytes() == provenance_before


def test_failed_legacy_consolidation_keeps_best_and_skips_test(tmp_path: Path, monkeypatch) -> None:
    split = _build_split(tmp_path / "split")
    experiment = tmp_path / "experiment"
    contract = infer_reference_contract(split)
    contract_path = experiment / "reference_contract.json"
    _write_json(contract_path, contract)
    runtime = DataCleaningAgentRuntime(
        config=ReferenceGuidedConfig(split, experiment, "task", max_iters=10).normalized(),
        contract=contract,
        contract_path=contract_path,
    )
    _write_csv(experiment / "rounds/round_0003/script_bundle/legacy.py", "BROKEN = True\n")
    shutil.copytree(split / "train/reference", experiment / "rounds/round_0003/validation_result")
    shutil.copytree(experiment / "rounds/round_0003/script_bundle", experiment / "active_bundle/script_bundle")
    shutil.copytree(split / "train/reference", experiment / "active_bundle/results/result_package")

    monkeypatch.setattr(
        runtime,
        "_run_pipeline_consolidation_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cannot consolidate")),
    )
    monkeypatch.setattr(
        ReferenceCheckpointTestRuntime,
        "run_sync",
        lambda self: pytest.fail("test must not run after consolidation failure"),
    )
    state = {
        "schema_version": 2,
        "status": "active",
        "best_score": 0.821243,
        "best_round": 3,
        "best_attempt": 6,
        "rounds": [{"round": 3, "attempt": 6, "score": 0.821243}],
        "attempts": [],
        "checkpoints": [],
        "sessions": [],
        "abandoned_checkpoints": [],
        "consecutive_valid_no_improvement": 0,
        "invalid_attempt_count": 0,
    }

    result = runtime._checkpoint_and_test("manual", state)

    assert result["status"] == "pipeline_consolidation_failed"
    assert result["best_score"] == 0.821243
    assert result["checkpoints"] == []


def test_atomic_json_failure_preserves_previous_state(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"version": 1}\n', encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(reference_guided.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        _write_json(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_atomic_promotion_commits_active_bundle_and_state_together(tmp_path: Path) -> None:
    active = tmp_path / "active_bundle"
    candidate = tmp_path / "candidate"
    state_path = tmp_path / "experiment_state.json"
    journal = tmp_path / "promotion_journal.json"
    _write_csv(active / "value.txt", "old\n")
    _write_csv(candidate / "value.txt", "new\n")
    _write_json(state_path, {"best_attempt": 1})

    _promote_active_bundle_atomically(
        active_dir=active,
        candidate=candidate,
        state_path=state_path,
        state={"best_attempt": 2, "best_round": 2},
        journal_path=journal,
    )

    assert (active / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert (candidate / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert json.loads(state_path.read_text(encoding="utf-8"))["best_attempt"] == 2
    assert not journal.exists()


def test_promotion_recovery_rolls_back_uncommitted_directory_swap(tmp_path: Path) -> None:
    active = tmp_path / "active_bundle"
    backup = tmp_path / ".active_bundle.previous"
    prepared = tmp_path / ".active_bundle.next"
    state_path = tmp_path / "experiment_state.json"
    journal = tmp_path / "promotion_journal.json"
    _write_csv(backup / "value.txt", "old\n")
    _write_csv(prepared / "value.txt", "new\n")
    _write_json(state_path, {"best_attempt": 1})
    _write_json(
        journal,
        {
            "schema_version": 1,
            "stage": "old_moved",
            "active_dir": str(active),
            "backup_dir": str(backup),
            "prepared_dir": str(prepared),
            "candidate_sha256": "unused",
            "best_attempt": 2,
        },
    )

    _recover_interrupted_promotion(journal, state_path)

    assert (active / "value.txt").read_text(encoding="utf-8") == "old\n"
    assert not backup.exists()
    assert not prepared.exists()
    assert not journal.exists()


def test_invalid_attempt_is_audited_without_consuming_patience() -> None:
    state = _migrate_reference_experiment_state(
        {
            "schema_version": 2,
            "status": "active",
            "best_score": 0.8,
            "best_round": 1,
            "best_attempt": 1,
            "rounds": [{"round": 1, "attempt": 1, "score": 0.8}],
            "attempts": [],
            "consecutive_valid_no_improvement": 1,
            "invalid_attempt_count": 0,
        }
    )

    state, improved = _finish_reference_attempt_state(
        state,
        attempt_index=2,
        score=None,
        candidate_dir="/tmp/candidate-2",
        result={},
        evaluation={},
        gate={"valid": False, "issues": ["no changed business file"]},
    )

    assert improved is False
    assert len(state["rounds"]) == 1
    assert state["attempts"][-1]["status"] == "invalid"
    assert state["invalid_attempt_count"] == 1
    assert state["consecutive_valid_no_improvement"] == 1


def test_first_zero_score_candidate_does_not_create_formal_round() -> None:
    state, improved = _finish_reference_attempt_state(
        {
            "schema_version": 2,
            "status": "active",
            "best_score": 0.0,
            "best_round": 0,
            "best_attempt": 0,
            "rounds": [],
            "attempts": [],
            "consecutive_valid_no_improvement": 0,
            "invalid_attempt_count": 0,
        },
        attempt_index=1,
        score=0.0,
        candidate_dir="/tmp/candidate",
        result={"producer": "test"},
        evaluation={},
        gate={"valid": True, "issues": []},
    )

    assert improved is False
    assert state["rounds"] == []
    assert state["best_round"] == 0
    assert state["consecutive_valid_no_improvement"] == 1


def test_invalid_attempt_between_valid_failures_does_not_change_patience() -> None:
    state = _migrate_reference_experiment_state(
        {
            "schema_version": 2,
            "status": "active",
            "best_score": 0.8,
            "best_round": 1,
            "best_attempt": 1,
            "rounds": [{"round": 1, "attempt": 1, "score": 0.8}],
            "attempts": [],
            "consecutive_valid_no_improvement": 0,
            "invalid_attempt_count": 0,
        }
    )
    state, _ = _finish_reference_attempt_state(
        state,
        attempt_index=2,
        score=0.7,
        candidate_dir="/tmp/a2",
        result={},
        evaluation={},
    )
    state, _ = _finish_reference_attempt_state(
        state,
        attempt_index=3,
        score=None,
        candidate_dir="/tmp/a3",
        result={},
        evaluation={},
        gate={"valid": False, "issues": ["invalid"]},
    )
    state, _ = _finish_reference_attempt_state(
        state,
        attempt_index=4,
        score=0.8,
        candidate_dir="/tmp/a4",
        result={},
        evaluation={},
    )
    assert state["consecutive_valid_no_improvement"] == 2


def test_candidate_gate_rejects_manifest_only_and_unrelated_business_changes(tmp_path: Path) -> None:
    active = tmp_path / "active"
    candidate = tmp_path / "candidate"
    for root in (active, candidate):
        _write_csv(root / "results/result_package/features/diag.csv", "stay_id,value\n1,A\n")
        _write_csv(root / "results/result_package/features/chart.csv", "stay_id,value\n1,10\n")
        (root / "adapter_bundle").mkdir(parents=True)
        (root / "adapter_bundle/manifest.json").write_text("{}", encoding="utf-8")
    (candidate / "adapter_bundle/manifest.json").write_text('{"changed": true}', encoding="utf-8")

    repair_targets = candidate / "repair_targets.json"
    repair_targets.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "repair_target_id": "diag",
                        "relative_path": "features/diag.csv",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = candidate / "train_regression_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "SUCCESS",
                "files": [
                    {
                        "relative_path": "features/diag.csv",
                        "train_rows": 1,
                        "reference_rows": 1,
                        "column_coverage": 1.0,
                        "key_coverage": 1.0,
                        "value_recall": 1.0,
                        "passed": True,
                        "failure_reason": "",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    gate = _reference_candidate_quality_gate(
        active=active,
        candidate=candidate,
        contract={"paths": {}, "key_column": "stay_id"},
        repair_targets=repair_targets,
    )
    assert gate["valid"] is False
    assert "no changed business file" in gate["issues"]

    _write_csv(candidate / "results/result_package/features/diag.csv", "stay_id,value\n1,B\n")
    _write_csv(candidate / "results/result_package/features/chart.csv", "stay_id,value\n1,11\n")
    gate = _reference_candidate_quality_gate(
        active=active,
        candidate=candidate,
        contract={"paths": {}, "key_column": "stay_id"},
        repair_targets=repair_targets,
    )
    assert gate["valid"] is False
    assert any("unrelated business files changed" in issue for issue in gate["issues"])


def test_candidate_gate_accepts_feedback_linked_business_change_with_passing_regression(tmp_path: Path) -> None:
    active = tmp_path / "active"
    candidate = tmp_path / "candidate"
    _write_csv(active / "results/result_package/features/diag.csv", "stay_id,value\n1,A\n")
    _write_csv(candidate / "results/result_package/features/diag.csv", "stay_id,value\n1,B\n")
    (candidate / "repair_targets.json").write_text(
        json.dumps({"targets": [{"repair_target_id": "diag", "relative_path": "features/diag.csv"}]}),
        encoding="utf-8",
    )
    (candidate / "train_regression_report.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "SUCCESS",
                "files": [
                    {
                        "relative_path": "features/diag.csv",
                        "train_rows": 1,
                        "reference_rows": 1,
                        "column_coverage": 1.0,
                        "key_coverage": 1.0,
                        "value_recall": 1.0,
                        "passed": True,
                        "failure_reason": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    gate = _reference_candidate_quality_gate(
        active=active,
        candidate=candidate,
        contract={"paths": {}, "key_column": "stay_id"},
        repair_targets=candidate / "repair_targets.json",
    )
    assert gate["valid"] is True


def test_candidate_gate_rejects_complete_package_without_canonical_pipeline(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    active = tmp_path / "active"
    candidate = tmp_path / "candidate"
    shutil.copytree(split / "train/reference", candidate / "results/result_package")

    gate = _reference_candidate_quality_gate(
        active=active,
        candidate=candidate,
        contract=contract,
        repair_targets=None,
    )

    assert gate["valid"] is False
    assert any("pipeline" in issue.casefold() for issue in gate["issues"])


def test_reference_directory_candidate_gate_fails_closed_when_host_paths_are_missing(
    tmp_path: Path,
) -> None:
    active = tmp_path / "active"
    candidate = tmp_path / "candidate"
    _write_csv(candidate / "results/result_package/cohort/cohort.csv", "stay_id,label\n1,0\n")
    contract = {
        "reference_type": "reference_directory",
        "key_column": "stay_id",
        "paths": {
            "train_raw": str(tmp_path / "missing_train_raw"),
            "train_reference_root": str(tmp_path / "missing_reference"),
            "validation_raw": str(tmp_path / "missing_validation_raw"),
        },
    }

    gate = _reference_candidate_quality_gate(
        active=active,
        candidate=candidate,
        contract=contract,
        repair_targets=None,
    )

    assert gate["valid"] is False
    assert any("required host" in issue for issue in gate["issues"])
    assert gate["pipeline_replay_gate"] == {}


def test_candidate_gate_accepts_host_replayable_canonical_pipeline(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    reference = split / "train/reference"
    expected = sorted(
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*.csv")
        if path.name != "reference.csv"
    )
    for name in ("train", "validation"):
        shutil.copytree(reference, split / name / "raw/source")
    active = tmp_path / "active"
    candidate = tmp_path / "candidate"
    shutil.copytree(reference, candidate / "results/result_package")
    _write_synthetic_pipeline(
        candidate / "script_bundle/pipeline",
        expected_files=expected,
    )

    gate = _reference_candidate_quality_gate(
        active=active,
        candidate=candidate,
        contract=contract,
        repair_targets=None,
    )

    assert gate["valid"] is True
    assert gate["pipeline_replay_gate"]["valid"] is True
    assert gate["pipeline_replay_gate"]["hash_consistency"]["matched"] is True
    assert (candidate / "host_replay/train_regression_report.json").is_file()


def test_candidate_gate_rejects_incomplete_train_regression_schema(tmp_path: Path) -> None:
    active = tmp_path / "active"
    candidate = tmp_path / "candidate"
    _write_csv(active / "results/result_package/features/diag.csv", "stay_id,value\n1,A\n")
    _write_csv(candidate / "results/result_package/features/diag.csv", "stay_id,value\n1,B\n")
    (candidate / "train_regression_report.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "SUCCESS",
                "files": [{"relative_path": "features/diag.csv", "passed": True}],
            }
        ),
        encoding="utf-8",
    )

    gate = _reference_candidate_quality_gate(
        active=active,
        candidate=candidate,
        contract={"paths": {}, "key_column": "stay_id"},
        repair_targets=None,
    )
    assert gate["valid"] is False
    assert any("train regression fields missing" in issue for issue in gate["issues"])


def test_stop_reason_counts_promotions_patience_attempts_and_target_independently() -> None:
    base = {
        "best_score": 0.9,
        "consecutive_valid_no_improvement": 0,
    }
    assert _reference_stop_reason(
        base,
        promotions_this_run=2,
        attempts_this_run=5,
        round_limit=2,
        patience=2,
        max_attempts=100,
        target_score=None,
    ) == "round_limit"
    assert _reference_stop_reason(
        {**base, "consecutive_valid_no_improvement": 2},
        promotions_this_run=0,
        attempts_this_run=3,
        round_limit=20,
        patience=2,
        max_attempts=100,
        target_score=None,
    ) == "patience"
    assert _reference_stop_reason(
        base,
        promotions_this_run=0,
        attempts_this_run=100,
        round_limit=20,
        patience=2,
        max_attempts=100,
        target_score=None,
    ) == "max_attempts"
    assert _reference_stop_reason(
        {**base, "best_score": 0.9603},
        promotions_this_run=0,
        attempts_this_run=1,
        round_limit=20,
        patience=2,
        max_attempts=100,
        target_score=0.9603,
    ) == "target_score"


def test_reference_config_validates_attempt_controls(tmp_path: Path) -> None:
    split = _build_split(tmp_path / "split")
    with pytest.raises(ValueError, match="patience"):
        ReferenceGuidedConfig(split, tmp_path / "experiment", "task", patience=0).normalized()
    with pytest.raises(ValueError, match="max_attempts"):
        ReferenceGuidedConfig(split, tmp_path / "experiment", "task", max_attempts=0).normalized()
    with pytest.raises(ValueError, match="target_score"):
        ReferenceGuidedConfig(split, tmp_path / "experiment", "task", target_score=1.1).normalized()


def test_runtime_counts_only_promotions_and_reuses_best_feedback(tmp_path: Path, monkeypatch) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    contract_path = tmp_path / "reference_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    runtime = DataCleaningAgentRuntime(
        config=ReferenceGuidedConfig(
            split,
            tmp_path / "experiment",
            "task",
            round_limit=2,
            patience=3,
            max_attempts=5,
        ).normalized(),
        contract=contract,
        contract_path=contract_path,
    )
    seen_feedback: list[str] = []

    async def fake_run_attempt(
        *,
        candidate,
        attempt_index,
        public_feedback,
        previous_outcome=None,
        test_rule_feedback=None,
    ):
        seen_feedback.append(str(public_feedback) if public_feedback else "")
        package = candidate / "results/result_package"
        _write_csv(package / "cohort/cohort_icu_mortality.csv", f"stay_id,label\n1,{attempt_index}\n")
        _write_synthetic_pipeline(
            candidate / "script_bundle/pipeline",
            expected_files=_expected_business_files(split / "train/reference"),
        )
        _write_host_replay_success(candidate)
        return {"producer": "test", "result_package": str(package)}

    scores = iter((0.8, 0.7, 0.9))
    checkpoint_calls: list[Path] = []

    def fake_evaluate(*, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True)
        score = next(scores)
        evaluation = output / "evaluation_report.json"
        feedback = output / "public_feedback.json"
        private = output / "private_report.json"
        evaluation.write_text(json.dumps({"metrics": {"composite_score": score}}), encoding="utf-8")
        feedback.write_text(json.dumps({"repair_targets": []}), encoding="utf-8")
        private.write_text("{}", encoding="utf-8")
        return {
            "evaluation_report": str(evaluation),
            "public_feedback": str(feedback),
            "private_report": str(private),
        }

    monkeypatch.setattr(runtime, "_run_attempt", fake_run_attempt)
    monkeypatch.setattr(
        reference_guided,
        "_reference_candidate_quality_gate",
        lambda **kwargs: {"valid": True, "status": "SUCCESS", "issues": []},
    )
    monkeypatch.setattr(reference_guided, "evaluate_reference_directory_package", fake_evaluate)

    def fake_checkpoint_test(self):
        checkpoint_calls.append(self.checkpoint_dir)
        report_path = self.checkpoint_dir / "checkpoint_report.json"
        _write_json(
            report_path,
            {
                "test_status": "success",
                "metrics": {"composite_score": 0.91},
                "evaluation": {},
            },
        )
        return {
            "test_status": "success",
            "metrics": {"composite_score": 0.91},
            "evaluation": {},
            "report_path": str(report_path),
        }

    monkeypatch.setattr(ReferenceCheckpointTestRuntime, "run_sync", fake_checkpoint_test)

    result = runtime.run_sync()
    state = json.loads((tmp_path / "experiment/experiment_state.json").read_text(encoding="utf-8"))

    assert result["termination_reason"] == "round_limit"
    assert result["round_count"] == 2
    assert result["attempt_count"] == 3
    assert [item["score"] for item in state["rounds"]] == [0.8, 0.9]
    assert seen_feedback[0] == ""
    assert seen_feedback[1].endswith("evaluations/attempt_0001/public_feedback.json")
    assert seen_feedback[2] == seen_feedback[1]
    active_result = tmp_path / "experiment/active_bundle/results/result_package/cohort/cohort_icu_mortality.csv"
    frozen_result = tmp_path / "experiment/frozen_bundle/results/result_package/cohort/cohort_icu_mortality.csv"
    assert active_result.read_bytes() == frozen_result.read_bytes()
    formal_rounds = sorted((tmp_path / "experiment/rounds").glob("round_*"))
    assert [path.name for path in formal_rounds] == ["round_0001", "round_0002"]
    assert [
        json.loads((path / "provenance.json").read_text(encoding="utf-8"))["attempt"]
        for path in formal_rounds
    ] == [1, 3]
    assert (tmp_path / "experiment/active_bundle/script_bundle/manifest.json").is_file()
    assert result["status"] == "checkpointed"
    assert len(checkpoint_calls) == 1
    assert checkpoint_calls[0].name == "checkpoint_0001_best_round_0002"
    assert (checkpoint_calls[0] / "frozen_script_bundle/manifest.json").is_file()
    assert len(state["checkpoints"]) == 1
    assert state["checkpoints"][0]["test_status"] == "success"

    runtime._checkpoint_and_test("manual_checkpoint", state)
    reused_state = json.loads(
        (tmp_path / "experiment/experiment_state.json").read_text(encoding="utf-8")
    )
    assert len(checkpoint_calls) == 1
    assert len(reused_state["checkpoints"]) == 2
    assert reused_state["checkpoints"][-1]["test_status"] == "reused"
    assert reused_state["checkpoints"][-1]["reused_from"] == str(checkpoint_calls[0])


def test_fresh_synthetic_workflow_runs_promotion_checkpoint_and_test_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    split = _build_split(tmp_path / "split")
    experiment = tmp_path / "fresh_experiment"
    workflow = ReferenceGuidedWorkflow(
        ReferenceGuidedConfig(
            dataset_split=split,
            experiment_dir=experiment,
            task_text="synthetic task",
            round_limit=1,
            max_attempts=3,
            max_iters=10,
        )
    )

    async def fake_run_attempt(self, *, candidate, **kwargs):
        package = candidate / "results/result_package"
        shutil.copytree(split / "train/reference", package)
        reference = split / "train/reference"
        _write_synthetic_pipeline(
            candidate / "script_bundle/pipeline",
            expected_files=sorted(
                path.relative_to(reference).as_posix()
                for path in reference.rglob("*.csv")
                if path.name != "reference.csv"
            ),
        )
        _write_host_replay_success(candidate)
        return {"producer": "mock-agent", "result_package": str(package)}

    monkeypatch.setattr(DataCleaningAgentRuntime, "_run_attempt", fake_run_attempt)
    monkeypatch.setattr(
        reference_guided,
        "_reference_candidate_quality_gate",
        lambda **kwargs: {"valid": True, "status": "SUCCESS", "issues": []},
    )

    def fake_checkpoint_test(self):
        report_path = self.checkpoint_dir / "checkpoint_report.json"
        _write_json(report_path, {"test_status": "success", "metrics": {}, "evaluation": {}})
        return {
            "test_status": "success",
            "metrics": {},
            "evaluation": {},
            "report_path": str(report_path),
        }

    monkeypatch.setattr(ReferenceCheckpointTestRuntime, "run_sync", fake_checkpoint_test)

    result = workflow.run_sync()

    assert result["status"] == "checkpointed"
    assert result["best_score"] == 1.0
    assert result["round_count"] == 1
    assert result["checkpoint_count"] == 1
    assert (experiment / "run_manifest.json").is_file()
    assert (experiment / "rounds/round_0001/script_bundle/manifest.json").is_file()
    assert (
        experiment
        / "test_checkpoints/checkpoint_0001_best_round_0001/frozen_script_bundle/manifest.json"
    ).is_file()


def test_runtime_invalid_attempts_stop_at_safety_limit_without_hidden_evaluation(tmp_path: Path, monkeypatch) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    contract_path = tmp_path / "reference_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    runtime = DataCleaningAgentRuntime(
        config=ReferenceGuidedConfig(
            split,
            tmp_path / "experiment",
            "task",
            round_limit=2,
            patience=2,
            max_attempts=2,
        ).normalized(),
        contract=contract,
        contract_path=contract_path,
    )

    async def fail_attempt(**kwargs):
        raise RuntimeError("no valid package")

    monkeypatch.setattr(runtime, "_run_attempt", fail_attempt)
    monkeypatch.setattr(
        reference_guided,
        "evaluate_reference_directory_package",
        lambda **kwargs: pytest.fail("invalid attempt must not reach hidden evaluation"),
    )

    result = runtime.run_sync()
    assert result["termination_reason"] == "max_attempts"
    assert result["round_count"] == 0
    assert result["attempt_count"] == 2
    assert result["invalid_attempt_count"] == 2
    assert result["status"] == "checkpointed_without_best"
    assert not (tmp_path / "experiment/test_checkpoints").exists()
    assert result["consecutive_valid_no_improvement"] == 0


def test_first_ctrl_c_cancels_attempt_and_checkpoints_committed_best(
    tmp_path: Path,
    monkeypatch,
) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    contract_path = tmp_path / "reference_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    runtime = DataCleaningAgentRuntime(
        config=ReferenceGuidedConfig(
            split,
            tmp_path / "experiment",
            "task",
            round_limit=5,
            max_attempts=5,
        ).normalized(),
        contract=contract,
        contract_path=contract_path,
    )

    async def interrupt_second_attempt(*, candidate, attempt_index, **kwargs):
        if attempt_index == 2:
            raise KeyboardInterrupt
        package = candidate / "results/result_package"
        _write_csv(package / "cohort/cohort_icu_mortality.csv", "stay_id,label\n1,1\n")
        _write_synthetic_pipeline(
            candidate / "script_bundle/pipeline",
            expected_files=_expected_business_files(split / "train/reference"),
        )
        _write_host_replay_success(candidate)
        return {"producer": "test", "result_package": str(package)}

    def fake_evaluate(*, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True)
        evaluation = output / "evaluation_report.json"
        feedback = output / "public_feedback.json"
        private = output / "private_report.json"
        evaluation.write_text(json.dumps({"metrics": {"composite_score": 0.8}}), encoding="utf-8")
        feedback.write_text("{}", encoding="utf-8")
        private.write_text("{}", encoding="utf-8")
        return {
            "evaluation_report": str(evaluation),
            "public_feedback": str(feedback),
            "private_report": str(private),
        }

    monkeypatch.setattr(runtime, "_run_attempt", interrupt_second_attempt)
    monkeypatch.setattr(
        reference_guided,
        "_reference_candidate_quality_gate",
        lambda **kwargs: {"valid": True, "status": "SUCCESS", "issues": []},
    )
    monkeypatch.setattr(reference_guided, "evaluate_reference_directory_package", fake_evaluate)
    monkeypatch.setattr(
        ReferenceCheckpointTestRuntime,
        "run_sync",
        lambda self: {
            "test_status": "success",
            "metrics": {"composite_score": 0.75},
            "evaluation": {},
            "report_path": str(self.checkpoint_dir / "checkpoint_report.json"),
        },
    )

    result = runtime.run_sync()
    state = json.loads((tmp_path / "experiment/experiment_state.json").read_text(encoding="utf-8"))

    assert result["status"] == "checkpointed"
    assert result["termination_reason"] == "cancelled_by_user"
    assert state["best_score"] == 0.8
    assert [item["status"] for item in state["attempts"]] == ["evaluated", "cancelled_by_user"]
    assert state["attempts"][-1]["valid"] is False
    assert state["invalid_attempt_count"] == 0
    assert state["consecutive_valid_no_improvement"] == 0
    assert (tmp_path / "experiment/candidates/attempt_0002/attempt_outcome.json").is_file()


def test_ctrl_c_while_copying_first_candidate_checkpoints_without_best(
    tmp_path: Path,
    monkeypatch,
) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    contract_path = tmp_path / "reference_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    runtime = DataCleaningAgentRuntime(
        config=ReferenceGuidedConfig(
            split,
            tmp_path / "experiment",
            "task",
            round_limit=5,
            max_attempts=5,
        ).normalized(),
        contract=contract,
        contract_path=contract_path,
    )
    monkeypatch.setattr(runtime, "_begin_attempt", lambda attempt_index: (_ for _ in ()).throw(KeyboardInterrupt()))

    result = runtime.run_sync()
    state = json.loads((tmp_path / "experiment/experiment_state.json").read_text(encoding="utf-8"))

    assert result["status"] == "checkpointed_without_best"
    assert result["termination_reason"] == "cancelled_by_user"
    assert state["attempts"][-1]["status"] == "cancelled_by_user"
    assert state["invalid_attempt_count"] == 0


def test_second_ctrl_c_interrupts_test_without_changing_validation_best(
    tmp_path: Path,
    monkeypatch,
) -> None:
    split = _build_split(tmp_path / "split")
    contract = infer_reference_contract(split)
    contract_path = tmp_path / "reference_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    runtime = DataCleaningAgentRuntime(
        config=ReferenceGuidedConfig(
            split,
            tmp_path / "experiment",
            "task",
            round_limit=5,
            max_attempts=5,
        ).normalized(),
        contract=contract,
        contract_path=contract_path,
    )

    async def interrupt_second_attempt(*, candidate, attempt_index, **kwargs):
        if attempt_index == 2:
            raise KeyboardInterrupt
        package = candidate / "results/result_package"
        _write_csv(package / "cohort/cohort_icu_mortality.csv", "stay_id,label\n1,1\n")
        _write_synthetic_pipeline(
            candidate / "script_bundle/pipeline",
            expected_files=_expected_business_files(split / "train/reference"),
        )
        _write_host_replay_success(candidate)
        return {"producer": "test", "result_package": str(package)}

    def fake_evaluate(*, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True)
        evaluation = output / "evaluation_report.json"
        feedback = output / "public_feedback.json"
        private = output / "private_report.json"
        evaluation.write_text(json.dumps({"metrics": {"composite_score": 0.8}}), encoding="utf-8")
        feedback.write_text("{}", encoding="utf-8")
        private.write_text("{}", encoding="utf-8")
        return {
            "evaluation_report": str(evaluation),
            "public_feedback": str(feedback),
            "private_report": str(private),
        }

    monkeypatch.setattr(runtime, "_run_attempt", interrupt_second_attempt)
    monkeypatch.setattr(
        reference_guided,
        "_reference_candidate_quality_gate",
        lambda **kwargs: {"valid": True, "status": "SUCCESS", "issues": []},
    )
    monkeypatch.setattr(reference_guided, "evaluate_reference_directory_package", fake_evaluate)

    def interrupt_test(self):
        private_evaluation = self.checkpoint_dir / "test_evaluation"
        private_evaluation.mkdir(parents=True)
        (private_evaluation / "private_report.json").write_text("{}", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(ReferenceCheckpointTestRuntime, "run_sync", interrupt_test)

    result = runtime.run_sync()
    state = json.loads((tmp_path / "experiment/experiment_state.json").read_text(encoding="utf-8"))

    assert result["status"] == "interrupted"
    assert result["termination_reason"] == "second_interrupt"
    assert state["best_score"] == 0.8
    assert state["best_round"] == 1
    assert (tmp_path / "experiment/active_bundle/results/result_package").is_dir()
    assert not any((tmp_path / "experiment/test_checkpoints").rglob("test_evaluation"))

    monkeypatch.setattr(
        ReferenceCheckpointTestRuntime,
        "run_sync",
        lambda self: {
            "test_status": "success",
            "metrics": {},
            "evaluation": {},
            "report_path": str(self.checkpoint_dir / "checkpoint_report.json"),
        },
    )
    resumed = runtime._checkpoint_and_test("resume_after_interrupt", state)
    assert Path(resumed["latest_checkpoint"]).name == "checkpoint_0002_best_round_0001"
