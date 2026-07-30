from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent.pi_full_experiment import (
    PiFullExperiment,
    PiFullExperimentConfig,
)
from agent.pi_harness import PiHarnessResult, PiValidationHarnessConfig
from agent.pi_test_harness import PiTestHarnessResult


def _full_config(tmp_path: Path) -> PiFullExperimentConfig:
    validation = PiValidationHarnessConfig(
        project_root=Path(__file__).parents[1],
        experiment_dir=tmp_path / "validation_experiment",
        train_raw=tmp_path / "train/raw",
        train_reference=tmp_path / "train/reference",
        validation_raw=tmp_path / "validation/raw",
        validation_gold=tmp_path / "private/validation_gold",
        evaluation_manifest=tmp_path / "evaluation_manifest.json",
        dataset_manifest=tmp_path / "dataset_manifest.json",
    )
    return PiFullExperimentConfig(
        validation=validation,
        test_experiment=tmp_path / "test_experiment",
        test_raw=tmp_path / "test/raw",
        test_gold=tmp_path / "private/test_gold",
        evaluation_manifest=tmp_path / "evaluation_manifest.json",
    )


def _validation_result(
    tmp_path: Path,
    *,
    status: str = "SUCCESS_REPRODUCIBLE",
) -> PiHarnessResult:
    experiment = tmp_path / "validation_experiment"
    return PiHarnessResult(
        status=status,
        stop_reason="target_score" if status == "SUCCESS_REPRODUCIBLE" else "failed",
        rounds=3,
        repair_rounds=1,
        best_score=0.91,
        reproducible_score=0.9,
        best_snapshot=experiment / "host/best_snapshot",
        reproducible_snapshot=experiment / "host/reproducible_snapshot",
    )


def _test_result(
    tmp_path: Path,
    *,
    status: str = "SUCCESS",
    phase: str = "complete",
    score: float | None = 0.88,
) -> PiTestHarnessResult:
    experiment = tmp_path / "test_experiment"
    return PiTestHarnessResult(
        status=status,
        phase=phase,
        scored=score is not None,
        score=score,
        test_execution_count=1,
        replay_status="SUCCESS" if phase != "test_replay" else status,
        frozen_snapshot_sha256="a" * 64,
        frozen_snapshot=experiment / "host/frozen_snapshot",
        result_package=experiment / "host/test_result_package",
        preflight_status="NOT_RUN",
        replay_duration_seconds=1.25,
        scoring_duration_seconds=2.5 if phase in {"scoring", "complete"} else 0.0,
    )


def test_validation_success_automatically_runs_test_once(tmp_path: Path) -> None:
    calls: list[str] = []
    validation_result = _validation_result(tmp_path)
    test_result = _test_result(tmp_path)

    async def run_validation(_prompt: str) -> PiHarnessResult:
        calls.append("validation")
        return validation_result

    async def run_test() -> PiTestHarnessResult:
        calls.append("test")
        return test_result

    result = asyncio.run(
        PiFullExperiment(
            _full_config(tmp_path),
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    assert calls == ["validation", "test"]
    assert result.status == "SUCCESS"
    assert result.phase == "complete"
    assert result.validation_result is validation_result
    assert result.test_result is test_result


@pytest.mark.parametrize("status", ["INTERRUPTED", "FAILED", "REPLAY_FAILED"])
def test_validation_non_success_never_starts_test(
    tmp_path: Path,
    status: str,
) -> None:
    test_calls = 0
    config = _full_config(tmp_path)

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return _validation_result(tmp_path, status=status)

    async def run_test() -> PiTestHarnessResult:
        nonlocal test_calls
        test_calls += 1
        raise AssertionError("Test must not start")

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    assert result.status == "VALIDATION_FAILED"
    assert result.phase == "validation"
    assert result.test_result is None
    assert test_calls == 0
    assert not config.test_experiment.exists()
    assert result.report_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "VALIDATION_FAILED"
    assert report["test"] is None
    assert report["test_execution_count"] == 0
    assert report["scoring_execution_count"] == 0


@pytest.mark.parametrize(
    ("status", "phase", "expected_scoring_count"),
    [
        ("REPLAY_FAILED", "test_replay", 0),
        ("SCORING_FAILED", "scoring", 1),
        ("INTERRUPTED", "test_replay", 0),
        ("INTERRUPTED", "scoring", 1),
    ],
)
def test_test_failure_status_is_terminal_without_retry(
    tmp_path: Path,
    status: str,
    phase: str,
    expected_scoring_count: int,
) -> None:
    test_calls = 0

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return _validation_result(tmp_path)

    async def run_test() -> PiTestHarnessResult:
        nonlocal test_calls
        test_calls += 1
        return _test_result(tmp_path, status=status, phase=phase, score=None)

    result = asyncio.run(
        PiFullExperiment(
            _full_config(tmp_path),
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    assert result.status == status
    assert result.phase == phase
    assert result.test_result is not None
    assert test_calls == 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["test_execution_count"] == 1
    assert report["scoring_execution_count"] == expected_scoring_count


def test_combined_report_is_host_only_complete_and_gold_free(tmp_path: Path) -> None:
    config = _full_config(tmp_path)

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return _validation_result(tmp_path)

    async def run_test() -> PiTestHarnessResult:
        return _test_result(tmp_path)

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    expected_path = config.validation.experiment_dir / "host/full_experiment_report.json"
    assert result.report_path == expected_path.resolve()
    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["status"] == "SUCCESS"
    assert report["phase"] == "complete"
    assert report["error_code"] == ""
    assert report["validation"]["status"] == "SUCCESS_REPRODUCIBLE"
    assert report["validation"]["rounds"] == 3
    assert report["validation"]["best_score"] == 0.91
    assert report["validation"]["reproducible_score"] == 0.9
    assert report["test"]["status"] == "SUCCESS"
    assert report["test"]["phase"] == "complete"
    assert report["test"]["replay_status"] == "SUCCESS"
    assert report["test"]["score"] == 0.88
    assert report["test_execution_count"] == 1
    assert report["scoring_execution_count"] == 1
    assert report["test"]["frozen_snapshot_sha256"] == "a" * 64
    assert report["validation"]["reproducible_snapshot"] == str(
        result.validation_result.reproducible_snapshot,
    )
    assert report["test"]["frozen_snapshot"] == str(result.test_result.frozen_snapshot)
    assert report["test"]["result_package"] == str(result.test_result.result_package)
    assert report["validation_duration_seconds"] >= 0.0
    assert report["test_duration_seconds"] >= 0.0
    assert report["duration_seconds"] >= 0.0
    assert report["test"]["replay_duration_seconds"] == 1.25
    assert report["test"]["scoring_duration_seconds"] == 2.5
    assert str(config.validation.validation_gold.resolve()) not in report_text
    assert str(config.test_gold.resolve()) not in report_text
    assert "GOLD_VALUE_MUST_NOT_LEAK" not in report_text


def test_combined_report_rejects_injected_gold_paths_and_redacts_free_text(
    tmp_path: Path,
) -> None:
    config = _full_config(tmp_path)
    sentinel = "GOLD_VALUE_MUST_NOT_LEAK"
    validation_gold = config.validation.validation_gold.resolve()
    test_gold = config.test_gold.resolve()
    validation_result = replace(
        _validation_result(tmp_path),
        stop_reason=f"blocked by {validation_gold}",
        best_snapshot=validation_gold / sentinel / "best_snapshot",
        reproducible_snapshot=validation_gold / sentinel / "reproducible_snapshot",
    )
    test_result = replace(
        _test_result(
            tmp_path,
            status="REPLAY_FAILED",
            phase="test_replay",
            score=None,
        ),
        replay_status=f"failed near {test_gold}",
        frozen_snapshot=test_gold / sentinel / "frozen_snapshot",
        result_package=test_gold / sentinel / "test_result_package",
    )

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return validation_result

    async def run_test() -> PiTestHarnessResult:
        return test_result

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert result.status == "REPLAY_FAILED"
    assert result.report_path.is_file()
    assert report["validation"]["reproducible_snapshot"] == ""
    assert report["test"]["frozen_snapshot"] == ""
    assert report["test"]["result_package"] == ""
    assert "stop_reason" not in report["validation"]
    assert report["test"]["replay_status"] == "UNKNOWN"
    assert str(validation_gold) not in report_text
    assert str(test_gold) not in report_text
    assert sentinel not in report_text


def test_combined_report_constrains_untrusted_test_text_fields(tmp_path: Path) -> None:
    config = _full_config(tmp_path)
    sentinel = "GOLD_VALUE_MUST_NOT_LEAK"
    validation_result = replace(
        _validation_result(tmp_path),
        stop_reason=sentinel,
    )
    test_result = replace(
        _test_result(tmp_path, score=None),
        status=f"FAILED_{sentinel}",
        phase=f"phase_{sentinel}",
        replay_status=sentinel,
        frozen_snapshot_sha256=sentinel,
    )

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return validation_result

    async def run_test() -> PiTestHarnessResult:
        return test_result

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert result.status == "UNKNOWN"
    assert result.phase == "unknown"
    assert "stop_reason" not in report["validation"]
    assert report["status"] == "UNKNOWN"
    assert report["phase"] == "unknown"
    assert report["test"]["status"] == "UNKNOWN"
    assert report["test"]["phase"] == "unknown"
    assert report["test"]["replay_status"] == "UNKNOWN"
    assert report["test"]["frozen_snapshot_sha256"] == ""
    assert sentinel not in report_text


def test_validation_exception_writes_fixed_failure_without_starting_test(
    tmp_path: Path,
) -> None:
    config = _full_config(tmp_path)
    sentinel = "GOLD_VALUE_MUST_NOT_LEAK"
    test_calls = 0

    async def run_validation(_prompt: str) -> PiHarnessResult:
        raise RuntimeError(
            f"{sentinel} {config.validation.validation_gold.resolve()} "
            f"{config.test_gold.resolve()}",
        )

    async def run_test() -> PiTestHarnessResult:
        nonlocal test_calls
        test_calls += 1
        raise AssertionError("Test must not start")

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert test_calls == 0
    assert result.status == report["status"] == "VALIDATION_FAILED"
    assert result.phase == report["phase"] == "validation"
    assert report["error_code"] == "VALIDATION_EXCEPTION"
    assert sentinel not in report_text
    assert str(config.validation.validation_gold.resolve()) not in report_text
    assert str(config.test_gold.resolve()) not in report_text


def test_validation_cancellation_writes_fixed_interrupted_report(
    tmp_path: Path,
) -> None:
    config = _full_config(tmp_path)
    sentinel = "GOLD_VALUE_MUST_NOT_LEAK"
    test_calls = 0

    async def run_validation(_prompt: str) -> PiHarnessResult:
        raise asyncio.CancelledError(
            f"{sentinel} {config.validation.validation_gold.resolve()}",
        )

    async def run_test() -> PiTestHarnessResult:
        nonlocal test_calls
        test_calls += 1
        raise AssertionError("Test must not start")

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert test_calls == 0
    assert result.status == report["status"] == "INTERRUPTED"
    assert result.phase == report["phase"] == "validation"
    assert report["error_code"] == "VALIDATION_CANCELLED"
    assert sentinel not in report_text
    assert str(config.validation.validation_gold.resolve()) not in report_text


@pytest.mark.parametrize("marker_exists", [False, True])
def test_test_exception_uses_started_marker_for_execution_count(
    tmp_path: Path,
    marker_exists: bool,
) -> None:
    config = _full_config(tmp_path)
    sentinel = "GOLD_VALUE_MUST_NOT_LEAK"
    test_calls = 0
    marker = config.test_experiment / "host/test_started.json"
    if marker_exists:
        marker.parent.mkdir(parents=True)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frozen_snapshot_sha256": "b" * 64,
                    "started_at_unix": 1.0,
                },
            ),
            encoding="utf-8",
        )

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return _validation_result(tmp_path)

    async def run_test() -> PiTestHarnessResult:
        nonlocal test_calls
        test_calls += 1
        raise RuntimeError(f"{sentinel} {config.test_gold.resolve()}")

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert test_calls == 1
    assert result.status == report["status"] == "REPLAY_FAILED"
    assert result.phase == report["phase"] == "test_replay"
    assert report["error_code"] == "TEST_EXCEPTION"
    assert report["test_execution_count"] == int(marker_exists)
    assert result.test_result is not None
    assert result.test_result.test_execution_count == int(marker_exists)
    assert report["test"]["replay_status"] == "EXECUTION_FAILED"
    assert report["test"]["frozen_snapshot_sha256"] == (
        "b" * 64 if marker_exists else ""
    )
    assert report["test"]["frozen_snapshot"] == ""
    assert report["test"]["result_package"] == ""
    assert sentinel not in report_text
    assert str(config.test_gold.resolve()) not in report_text


def test_test_cancellation_writes_fixed_interrupted_report_without_retry(
    tmp_path: Path,
) -> None:
    config = _full_config(tmp_path)
    sentinel = "GOLD_VALUE_MUST_NOT_LEAK"
    test_calls = 0

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return _validation_result(tmp_path)

    async def run_test() -> PiTestHarnessResult:
        nonlocal test_calls
        test_calls += 1
        raise asyncio.CancelledError(f"{sentinel} {config.test_gold.resolve()}")

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert test_calls == 1
    assert result.status == report["status"] == "INTERRUPTED"
    assert result.phase == report["phase"] == "test_replay"
    assert report["error_code"] == "TEST_CANCELLED"
    assert report["test_execution_count"] == 0
    assert report["test"]["replay_status"] == "INTERRUPTED"
    assert sentinel not in report_text
    assert str(config.test_gold.resolve()) not in report_text


@pytest.mark.parametrize("cancelled", [False, True])
def test_test_failure_after_scoring_marker_recovers_scoring_phase(
    tmp_path: Path,
    cancelled: bool,
) -> None:
    config = _full_config(tmp_path)
    frozen_hash = "c" * 64
    test_calls = 0

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return _validation_result(tmp_path)

    async def run_test() -> PiTestHarnessResult:
        nonlocal test_calls
        test_calls += 1
        host = config.test_experiment / "host"
        host.mkdir(parents=True)
        marker = {
            "schema_version": 1,
            "frozen_snapshot_sha256": frozen_hash,
            "started_at_unix": 1.0,
        }
        (host / "test_started.json").write_text(
            json.dumps(marker),
            encoding="utf-8",
        )
        (host / "scoring_started.json").write_text(
            json.dumps(marker),
            encoding="utf-8",
        )
        if cancelled:
            raise asyncio.CancelledError("GOLD_VALUE_MUST_NOT_LEAK")
        raise RuntimeError("GOLD_VALUE_MUST_NOT_LEAK")

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    expected_status = "INTERRUPTED" if cancelled else "SCORING_FAILED"
    expected_error = "TEST_CANCELLED" if cancelled else "TEST_EXCEPTION"
    assert test_calls == 1
    assert result.status == report["status"] == expected_status
    assert result.phase == report["phase"] == "scoring"
    assert report["error_code"] == expected_error
    assert report["test_execution_count"] == 1
    assert report["scoring_execution_count"] == 1
    assert report["test"]["frozen_snapshot_sha256"] == frozen_hash
    assert report["test"]["replay_status"] == "SUCCESS"
    assert report["test"]["frozen_snapshot"] == ""
    assert report["test"]["result_package"] == ""
    assert "GOLD_VALUE_MUST_NOT_LEAK" not in report_text


@pytest.mark.parametrize(
    "marker_text",
    [
        "{",
        json.dumps({"schema_version": True, "frozen_snapshot_sha256": "d" * 64}),
        json.dumps({"schema_version": 2, "frozen_snapshot_sha256": "d" * 64}),
        json.dumps({"schema_version": 1, "frozen_snapshot_sha256": "invalid"}),
    ],
)
def test_test_exception_ignores_invalid_start_markers(
    tmp_path: Path,
    marker_text: str,
) -> None:
    config = _full_config(tmp_path)

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return _validation_result(tmp_path)

    async def run_test() -> PiTestHarnessResult:
        host = config.test_experiment / "host"
        host.mkdir(parents=True)
        (host / "test_started.json").write_text(marker_text, encoding="utf-8")
        (host / "scoring_started.json").write_text(marker_text, encoding="utf-8")
        raise RuntimeError("failed")

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert result.status == "REPLAY_FAILED"
    assert result.phase == "test_replay"
    assert report["test_execution_count"] == 0
    assert report["scoring_execution_count"] == 0
    assert report["test"]["frozen_snapshot_sha256"] == ""


def test_public_paths_are_serialized_in_resolved_form(tmp_path: Path) -> None:
    config = _full_config(tmp_path)
    validation_result = replace(
        _validation_result(tmp_path),
        reproducible_snapshot=(
            config.validation.experiment_dir
            / "host/temporary/../reproducible_snapshot"
        ),
    )
    test_result = replace(
        _test_result(tmp_path),
        frozen_snapshot=config.test_experiment / "host/temporary/../frozen_snapshot",
        result_package=config.test_experiment / "host/temporary/../test_result_package",
    )

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return validation_result

    async def run_test() -> PiTestHarnessResult:
        return test_result

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["validation"]["reproducible_snapshot"] == str(
        validation_result.reproducible_snapshot.resolve(),
    )
    assert report["test"]["frozen_snapshot"] == str(
        test_result.frozen_snapshot.resolve(),
    )
    assert report["test"]["result_package"] == str(
        test_result.result_package.resolve(),
    )


def test_combined_report_constrains_unknown_validation_status(tmp_path: Path) -> None:
    config = _full_config(tmp_path)
    sentinel = "GOLD_VALUE_MUST_NOT_LEAK"

    async def run_validation(_prompt: str) -> PiHarnessResult:
        return replace(
            _validation_result(tmp_path),
            status=sentinel,
            stop_reason=sentinel,
        )

    async def run_test() -> PiTestHarnessResult:
        raise AssertionError("Test must not start")

    result = asyncio.run(
        PiFullExperiment(
            config,
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    report_text = result.report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["validation"]["status"] == "UNKNOWN"
    assert "stop_reason" not in report["validation"]
    assert sentinel not in report_text


def test_default_runners_reuse_existing_harnesses_without_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_full_experiment as module

    config = _full_config(tmp_path)
    validation_result = _validation_result(tmp_path)
    test_result = _test_result(tmp_path)
    seen: dict[str, object] = {}

    class FakeValidationHarness:
        def __init__(self, validation_config: PiValidationHarnessConfig) -> None:
            seen["validation_config"] = validation_config

        async def run(self, prompt: str) -> PiHarnessResult:
            seen["prompt"] = prompt
            return validation_result

    class FakeTestHarness:
        def __init__(self, test_config: object) -> None:
            seen["test_config"] = test_config

        async def run(self) -> PiTestHarnessResult:
            return test_result

    monkeypatch.setattr(module, "PiValidationHarness", FakeValidationHarness)
    monkeypatch.setattr(module, "PiTestHarness", FakeTestHarness)

    result = asyncio.run(PiFullExperiment(config).run("default prompt"))

    assert result.status == "SUCCESS"
    assert seen["validation_config"] is config.validation
    assert seen["prompt"] == "default prompt"
    test_config = seen["test_config"]
    assert test_config.validation_experiment == config.validation.experiment_dir
    assert test_config.preflight_attestation is None
    assert test_config.test_experiment == config.test_experiment
    assert test_config.test_raw == config.test_raw
    assert test_config.test_gold == config.test_gold
    assert test_config.evaluation_manifest == config.evaluation_manifest
    assert test_config.replay_timeout_seconds == config.test_replay_timeout_seconds
    assert test_config.scoring_timeout_seconds == config.test_scoring_timeout_seconds
