from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import workflow.reference_guided as reference_guided

from agent.reference_runtime import REFERENCE_CODE_AGENT_PIPELINE_SKILLS, create_reference_toolkit
from agent_tools.context import EngineerToolContext, EngineerToolPermissionError
from main import parse_args
from workflow.mimic_pipeline import split_teacher_reference
from workflow.reference_guided import (
    ReferenceCodeAgentRuntime,
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
    evaluate_reference_directory_package,
    infer_reference_contract,
)
from workflow.reference_test_stage import (
    ReferenceCheckpointTestConfig,
    ReferenceCheckpointTestRuntime,
    ReferenceTestStageConfig,
    _build_sanitized_test_contract,
    _test_package_quality_gate,
    run_reference_test_stage,
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
    assert {item["name"] for item in manifest} == REFERENCE_CODE_AGENT_PIPELINE_SKILLS


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
    phase_root = checkpoint / "agent_runs/reference_code_agent/phase"
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


def test_checkpoint_test_gate_failure_never_calls_private_evaluator(tmp_path: Path, monkeypatch) -> None:
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

    def fake_agent_session() -> Path:
        package = checkpoint / "agent_runs/incomplete/result_package"
        _write_csv(package / "csv/labels.csv", "stay_id,label\n1,0\n")
        return package

    monkeypatch.setattr(runtime, "_run_agent_session", fake_agent_session)
    monkeypatch.setattr(
        reference_guided,
        "evaluate_reference_directory_package",
        lambda **kwargs: pytest.fail("private evaluator must not run after gate failure"),
    )

    report = runtime.run_sync()

    assert report["test_status"] == "test_failed"
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

    def fake_agent_session() -> Path:
        package = checkpoint / "agent_runs/complete/result_package"
        shutil.copytree(split / "train/reference", package)
        return package

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


def test_checkpoint_gate_enforces_train_business_key_grain(tmp_path: Path) -> None:
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

    assert gate["valid"] is False
    assert any("duplicate business keys" in issue for issue in gate["issues"])


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
        candidate / "agent_runs" / "reference_code_agent" / "phase" / "workspace" / "build_package.py",
        "CURRENT = True\n",
    )
    _write_csv(candidate / "results" / "result_package" / "csv" / "labels.csv", "stay_id,label\n1,0\n")
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
    assert (candidate / "script_bundle/manifest.json").is_file()


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
    runtime = ReferenceCodeAgentRuntime(
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

    async def fake_run_attempt(*, candidate, attempt_index, public_feedback, previous_outcome=None):
        seen_feedback.append(str(public_feedback) if public_feedback else "")
        package = candidate / "results/result_package"
        _write_csv(package / "cohort/cohort_icu_mortality.csv", f"stay_id,label\n1,{attempt_index}\n")
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
        return {"producer": "mock-agent", "result_package": str(package)}

    monkeypatch.setattr(ReferenceCodeAgentRuntime, "_run_attempt", fake_run_attempt)
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
    runtime = ReferenceCodeAgentRuntime(
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
    runtime = ReferenceCodeAgentRuntime(
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
    runtime = ReferenceCodeAgentRuntime(
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
    runtime = ReferenceCodeAgentRuntime(
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
