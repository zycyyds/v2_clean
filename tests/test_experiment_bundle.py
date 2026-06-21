from __future__ import annotations

import json
from pathlib import Path

from workflow.bundle import ExperimentBundleManager


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_candidate_promote_reject_freeze_and_resume(tmp_path: Path) -> None:
    explorer = tmp_path / "explorer"
    rules = explorer / "field_extraction_rules.json"
    report = explorer / "data_analysis_report.json"
    record = explorer / "explorer_run_record.json"
    _write_json(rules, {"rules": [{"target_field_id": "field_a"}]})
    _write_json(report, {"field_count": 1})
    _write_json(record, {"status": "SUCCESS"})

    manager = ExperimentBundleManager(tmp_path / "experiment")
    active = manager.initialize(
        {
            "field_extraction_rules": str(rules),
            "data_analysis_report": str(report),
            "explorer_run_record": str(record),
        }
    )
    assert active.name == "active_bundle"
    assert (active / "field_extraction_rules.json").is_file()
    assert (active / "capabilities").is_dir()

    first = manager.create_candidate(1)
    (first / "capabilities" / "entity_adapter.py").write_text("VERSION = 1\n", encoding="utf-8")
    promoted = manager.finish_round(1, score=0.4, candidate_dir=first)
    assert promoted["improved"] is True
    assert promoted["stop"] is False
    assert (manager.active_dir / "capabilities" / "entity_adapter.py").is_file()

    second = manager.create_candidate(2)
    (second / "capabilities" / "entity_adapter.py").write_text("VERSION = 2\n", encoding="utf-8")
    rejected = manager.finish_round(2, score=0.4005, candidate_dir=second)
    assert rejected["improved"] is False
    assert rejected["consecutive_no_improvement"] == 1
    assert "VERSION = 1" in (manager.active_dir / "capabilities" / "entity_adapter.py").read_text()

    third = manager.create_candidate(3)
    stopped = manager.finish_round(3, score=0.2, candidate_dir=third)
    assert stopped["stop"] is True
    assert stopped["consecutive_no_improvement"] == 2

    frozen = manager.freeze()
    assert frozen.name == "frozen_bundle"
    assert json.loads((frozen / "manifest.json").read_text())["status"] == "frozen"

    resumed = ExperimentBundleManager(tmp_path / "experiment")
    state = resumed.load_state()
    assert state["best_score"] == 0.4
    assert state["status"] == "frozen"
    assert len(state["rounds"]) == 3


def test_candidate_is_idempotent_and_active_hash_is_recorded(tmp_path: Path) -> None:
    explorer = tmp_path / "explorer"
    rules = explorer / "field_extraction_rules.json"
    report = explorer / "data_analysis_report.json"
    record = explorer / "explorer_run_record.json"
    _write_json(rules, {"rules": []})
    _write_json(report, {})
    _write_json(record, {"status": "SUCCESS"})
    manager = ExperimentBundleManager(tmp_path / "experiment")
    manager.initialize(
        {
            "field_extraction_rules": str(rules),
            "data_analysis_report": str(report),
            "explorer_run_record": str(record),
        }
    )

    first = manager.create_candidate(1)
    second = manager.create_candidate(1)

    assert first == second
    state = manager.load_state()
    assert state["active_bundle_hash"]


def test_first_round_is_retained_even_when_score_is_zero(tmp_path: Path) -> None:
    explorer = tmp_path / "explorer"
    rules = explorer / "field_extraction_rules.json"
    report = explorer / "data_analysis_report.json"
    record = explorer / "explorer_run_record.json"
    _write_json(rules, {"rules": []})
    _write_json(report, {})
    _write_json(record, {"status": "SUCCESS"})
    manager = ExperimentBundleManager(tmp_path / "experiment")
    manager.initialize(
        {
            "field_extraction_rules": str(rules),
            "data_analysis_report": str(report),
            "explorer_run_record": str(record),
        }
    )
    candidate = manager.create_candidate(1)
    (candidate / "results").mkdir()

    result = manager.finish_round(1, score=0.0, candidate_dir=candidate)

    assert result["improved"] is True
    assert manager.load_state()["best_round"] == 1
    assert (manager.active_dir / "results").is_dir()


def test_failed_candidate_round_is_recorded_and_can_be_retried(tmp_path: Path) -> None:
    explorer = tmp_path / "explorer"
    artifacts = {
        "field_extraction_rules": explorer / "field_extraction_rules.json",
        "data_analysis_report": explorer / "data_analysis_report.json",
        "explorer_run_record": explorer / "explorer_run_record.json",
    }
    for path in artifacts.values():
        _write_json(path, {})
    manager = ExperimentBundleManager(tmp_path / "experiment")
    manager.initialize({name: str(path) for name, path in artifacts.items()})

    candidate = manager.begin_round(1)
    failed = manager.fail_round(1, candidate, "invalid target mapping")

    assert failed["status"] == "failed"
    assert manager.load_state()["current_round"] is None
    assert manager.load_state()["round_attempts"][-1]["error"] == "invalid target mapping"
    retried = manager.begin_round(1)
    assert retried == candidate
    assert manager.load_state()["current_round"]["status"] == "running"


def test_evaluator_invalid_candidate_is_never_promoted(tmp_path: Path) -> None:
    explorer = tmp_path / "explorer"
    artifacts = {
        "field_extraction_rules": explorer / "field_extraction_rules.json",
        "data_analysis_report": explorer / "data_analysis_report.json",
        "explorer_run_record": explorer / "explorer_run_record.json",
    }
    for path in artifacts.values():
        _write_json(path, {})
    manager = ExperimentBundleManager(tmp_path / "experiment")
    manager.initialize({name: str(path) for name, path in artifacts.items()})
    candidate = manager.begin_round(1)
    (candidate / "invalid.txt").write_text("invalid", encoding="utf-8")

    result = manager.finish_round(
        1,
        score=0.9,
        candidate_dir=candidate,
        eligible_for_promotion=False,
    )

    assert result["improved"] is False
    assert result["eligible_for_promotion"] is False
    assert manager.load_state()["best_round"] == 0
    assert not (manager.active_dir / "invalid.txt").exists()


def test_stale_running_candidate_is_rebuilt_from_active_bundle(tmp_path: Path) -> None:
    explorer = tmp_path / "explorer"
    artifacts = {
        "field_extraction_rules": explorer / "field_extraction_rules.json",
        "data_analysis_report": explorer / "data_analysis_report.json",
        "explorer_run_record": explorer / "explorer_run_record.json",
    }
    for path in artifacts.values():
        _write_json(path, {})
    manager = ExperimentBundleManager(tmp_path / "experiment")
    manager.initialize({name: str(path) for name, path in artifacts.items()})
    candidate = manager.begin_round(1)
    (candidate / "partial.txt").write_text("partial", encoding="utf-8")

    recovered = ExperimentBundleManager(tmp_path / "experiment").begin_round(1)

    assert recovered == candidate
    assert not (recovered / "partial.txt").exists()
    state = manager.load_state()
    assert state["round_attempts"][-1]["status"] == "recovered"
