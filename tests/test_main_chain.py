from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import workflow.reference_guided as reference_guided

from agent.reference_runtime import REFERENCE_CODE_AGENT_PIPELINE_SKILLS, create_reference_toolkit
from agent_tools.context import EngineerToolContext
from main import parse_args
from workflow.mimic_pipeline import split_teacher_reference
from workflow.reference_guided import (
    ReferenceCodeAgentRuntime,
    ReferenceGuidedConfig,
    _finish_reference_attempt_state,
    _migrate_reference_experiment_state,
    _reference_candidate_quality_gate,
    _reference_stop_reason,
    _complete_reference_package_paths,
    evaluate_reference_directory_package,
    infer_reference_contract,
)
from workflow.reference_test_stage import ReferenceTestStageConfig, run_reference_test_stage


def _write_csv(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_reference_package(root: Path) -> None:
    _write_csv(root / "cohort" / "cohort_icu_mortality.csv", "stay_id,label\n1,0\n")
    _write_csv(
        root / "features" / "preproc_chart_icu.csv",
        "stay_id,itemid,event_time_from_admit\n1,220045,1.0\n",
    )
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
    assert result["consecutive_valid_no_improvement"] == 0
