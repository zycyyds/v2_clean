from __future__ import annotations

import asyncio
import io
import json
import signal
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent.pi_harness import directory_sha256
from agent.pi_test_harness import (
    PiTestHarness,
    PiTestHarnessConfig,
    PiTestPreflight,
    PiTestPreflightConfig,
    ReplayRequest,
    ScoreExecution,
    TestReplayExecution,
    _communicate_with_escalation,
)


def _validation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    validation = tmp_path / "validation_experiment"
    snapshot = validation / "host/reproducible_snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "scripts/build.py").write_text("print('frozen')\n", encoding="utf-8")
    (snapshot / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result_root": "result_package",
                "replay": {
                    "argv": [
                        "python",
                        "scripts/build.py",
                        "--raw",
                        "{raw_root}",
                        "--out",
                        "{output_dir}",
                    ],
                },
            },
        ),
        encoding="utf-8",
    )
    train_reference = tmp_path / "dataset/train/reference"
    train_reference.mkdir(parents=True)
    (train_reference / "catalog.csv").write_text("id\n1\n", encoding="utf-8")
    (validation / "host/run_report.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS_REPRODUCIBLE",
                "reproducible_snapshot": str(snapshot.resolve()),
                "best_score": 0.9,
                "reproducible_score": 0.9,
            },
        ),
        encoding="utf-8",
    )
    (validation / "host/run_manifest.json").write_text(
        json.dumps({"train_reference": str(train_reference.resolve())}),
        encoding="utf-8",
    )
    return validation, train_reference


def _preflight_config(tmp_path: Path) -> PiTestPreflightConfig:
    validation, _ = _validation_fixture(tmp_path)
    raw = tmp_path / "dataset/validation_4000/raw"
    raw.mkdir(parents=True)
    (raw / "data.csv").write_text("id,value\n1,public\n", encoding="utf-8")
    return PiTestPreflightConfig(
        project_root=Path(__file__).parents[1],
        validation_experiment=validation,
        preflight_experiment=tmp_path / "preflight_experiment",
        preflight_raw=raw,
        replay_timeout_seconds=10.0,
    )


def _test_config(tmp_path: Path, attestation: Path | None) -> PiTestHarnessConfig:
    validation = tmp_path / "validation_experiment"
    test_raw = tmp_path / "dataset/test/raw"
    test_gold = tmp_path / "dataset/test/reference_private"
    test_raw.mkdir(parents=True, exist_ok=True)
    test_gold.mkdir(parents=True, exist_ok=True)
    (test_raw / "data.csv").write_text("id,value\n2,test\n", encoding="utf-8")
    (test_gold / "data.csv").write_text("id,value\n2,gold\n", encoding="utf-8")
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "files": {"data.csv": {"key_columns": ["id"]}}}),
        encoding="utf-8",
    )
    return PiTestHarnessConfig(
        project_root=Path(__file__).parents[1],
        validation_experiment=validation,
        test_experiment=tmp_path / "test_experiment",
        test_raw=test_raw,
        test_gold=test_gold,
        evaluation_manifest=manifest,
        preflight_attestation=attestation,
        replay_timeout_seconds=10.0,
        scoring_timeout_seconds=10.0,
    )


def _successful_replay(requests: list[ReplayRequest]):
    async def replay(request: ReplayRequest) -> TestReplayExecution:
        requests.append(request)
        result = request.runtime_root / "result"
        result.mkdir(parents=True, exist_ok=True)
        (result / "data.csv").write_text("id,value\n2,result\n", encoding="utf-8")
        return TestReplayExecution(
            status="SUCCESS",
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            result_root=result,
            output_file_count=1,
            output_bytes=(result / "data.csv").stat().st_size,
        )

    return replay


def _score_report(value: float) -> dict:
    return {
        "status": "SUCCESS",
        "metrics": {"composite_score": value},
        "file_reports": [],
    }


def test_preflight_creates_success_attestation_without_scoring(tmp_path: Path) -> None:
    config = _preflight_config(tmp_path)
    requests: list[ReplayRequest] = []

    result = asyncio.run(
        PiTestPreflight(config, replay_runner=_successful_replay(requests)).run(),
    )

    assert result.status == "SUCCESS"
    assert result.phase == "complete"
    assert result.scored is False
    assert result.score is None
    assert result.preflight_status == "SUCCESS"
    assert len(requests) == 1
    assert requests[0].phase == "preflight"
    assert requests[0].raw_root == config.preflight_raw.resolve()
    attestation = json.loads(result.attestation.read_text(encoding="utf-8"))
    assert attestation["status"] == "SUCCESS"
    assert attestation["frozen_snapshot_sha256"] == directory_sha256(
        config.validation_experiment / "host/reproducible_snapshot",
    )
    assert attestation["output_file_count"] == 1
    assert not (config.preflight_experiment / "host/preflight_result_package").exists()


def test_preflight_and_test_emit_low_frequency_phase_updates(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight_output = io.StringIO()
    preflight = asyncio.run(
        PiTestPreflight(
            preflight_config,
            replay_runner=_successful_replay([]),
            output=preflight_output,
        ).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)
    test_output = io.StringIO()

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        return ScoreExecution("SUCCESS", 0, "", 0.1, _score_report(0.2))

    asyncio.run(
        PiTestHarness(
            config,
            replay_runner=_successful_replay([]),
            score_runner=score_runner,
            output=test_output,
        ).run(),
    )

    assert "[Test Preflight] replay started" in preflight_output.getvalue()
    assert "[Test Preflight] status=SUCCESS" in preflight_output.getvalue()
    assert "[Test Harness] Test replay started" in test_output.getvalue()
    assert "[Test Harness] hidden scoring started" in test_output.getvalue()
    assert "[Test Harness] status=SUCCESS" in test_output.getvalue()


def test_test_replay_and_hidden_scoring_each_run_once(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)
    replay_requests: list[ReplayRequest] = []
    score_calls = 0

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        nonlocal score_calls
        score_calls += 1
        return ScoreExecution("SUCCESS", 0, "", 0.2, _score_report(0.42))

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=_successful_replay(replay_requests),
            score_runner=score_runner,
        ).run(),
    )

    assert result.status == "SUCCESS"
    assert result.phase == "complete"
    assert result.scored is True
    assert result.score == 0.42
    assert result.test_execution_count == 1
    assert result.preflight_status == "SUCCESS"
    assert len(replay_requests) == 1
    assert replay_requests[0].phase == "test_replay"
    assert replay_requests[0].raw_root == config.test_raw.resolve()
    assert score_calls == 1
    assert (config.test_experiment / "host/score_report.json").is_file()
    report = json.loads(
        (config.test_experiment / "host/run_report.json").read_text(encoding="utf-8"),
    )
    assert report["preflight_status"] == "SUCCESS"


def test_test_config_preserves_preflight_attestation_positional_field(
    tmp_path: Path,
) -> None:
    expected = _test_config(tmp_path, None)

    config = PiTestHarnessConfig(
        expected.project_root,
        expected.validation_experiment,
        None,
        expected.test_experiment,
        expected.test_raw,
        expected.test_gold,
        expected.evaluation_manifest,
        expected.replay_timeout_seconds,
        expected.scoring_timeout_seconds,
    )

    assert config.preflight_attestation is None
    assert config.test_experiment == expected.test_experiment
    assert config.test_raw == expected.test_raw
    assert config.test_gold == expected.test_gold
    assert config.evaluation_manifest == expected.evaluation_manifest


def test_direct_test_writes_started_marker_before_replay(tmp_path: Path) -> None:
    _validation_fixture(tmp_path)
    config = _test_config(tmp_path, None)
    replay_requests: list[ReplayRequest] = []

    async def replay(request: ReplayRequest) -> TestReplayExecution:
        marker = json.loads(
            (config.test_experiment / "host/test_started.json").read_text(
                encoding="utf-8",
            ),
        )
        assert marker["schema_version"] == 1
        assert marker["frozen_snapshot_sha256"] == directory_sha256(
            config.validation_experiment / "host/reproducible_snapshot",
        )
        assert isinstance(marker["started_at_unix"], float)
        return await _successful_replay(replay_requests)(request)

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        return ScoreExecution("SUCCESS", 0, "", 0.1, _score_report(0.31))

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=replay,
            score_runner=score_runner,
        ).run(),
    )

    assert result.status == "SUCCESS"
    assert result.preflight_status == "NOT_RUN"
    assert len(replay_requests) == 1
    manifest = json.loads(
        (config.test_experiment / "host/run_manifest.json").read_text(encoding="utf-8"),
    )
    assert manifest["preflight_attestation_sha256"] == ""
    assert manifest["preflight_raw_identity"] == {}
    assert str(config.test_gold.resolve()) not in json.dumps(manifest)
    report = json.loads(
        (config.test_experiment / "host/run_report.json").read_text(encoding="utf-8"),
    )
    assert report["preflight_status"] == "NOT_RUN"
    marker = json.loads(
        (config.test_experiment / "host/test_started.json").read_text(encoding="utf-8"),
    )
    assert str(config.test_gold.resolve()) not in json.dumps(marker)


def test_scoring_started_marker_exists_before_score_runner(tmp_path: Path) -> None:
    _validation_fixture(tmp_path)
    config = _test_config(tmp_path, None)
    expected_hash = directory_sha256(
        config.validation_experiment / "host/reproducible_snapshot",
    )

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        marker_path = config.test_experiment / "host/scoring_started.json"
        marker_text = marker_path.read_text(encoding="utf-8")
        marker = json.loads(marker_text)
        assert marker["schema_version"] == 1
        assert marker["frozen_snapshot_sha256"] == expected_hash
        assert isinstance(marker["started_at_unix"], float)
        assert str(config.test_gold.resolve()) not in marker_text
        return ScoreExecution("SUCCESS", 0, "", 0.1, _score_report(0.31))

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=_successful_replay([]),
            score_runner=score_runner,
        ).run(),
    )

    assert result.status == "SUCCESS"
    assert (config.test_experiment / "host/scoring_started.json").is_file()


def test_same_test_experiment_cannot_replay_twice(tmp_path: Path) -> None:
    _validation_fixture(tmp_path)
    config = _test_config(tmp_path, None)
    replay_requests: list[ReplayRequest] = []

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        return ScoreExecution("SUCCESS", 0, "", 0.1, _score_report(0.31))

    harness = PiTestHarness(
        config,
        replay_runner=_successful_replay(replay_requests),
        score_runner=score_runner,
    )
    first = asyncio.run(harness.run())

    assert first.status == "SUCCESS"
    with pytest.raises(ValueError, match="experiment directory must be new and empty"):
        asyncio.run(harness.run())
    assert len(replay_requests) == 1


def test_test_replay_failure_is_not_retried_or_scored(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)
    replay_calls = 0
    score_calls = 0

    async def replay(_request: ReplayRequest) -> TestReplayExecution:
        nonlocal replay_calls
        replay_calls += 1
        return TestReplayExecution("EXECUTION_FAILED", 2, "", "failed", 0.1, None, 0, 0)

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        nonlocal score_calls
        score_calls += 1
        raise AssertionError("failed Test replay must not open or score Gold")

    result = asyncio.run(
        PiTestHarness(config, replay_runner=replay, score_runner=score_runner).run(),
    )

    assert result.status == "REPLAY_FAILED"
    assert result.phase == "test_replay"
    assert result.scored is False
    assert result.score is None
    assert result.test_execution_count == 1
    assert replay_calls == 1
    assert score_calls == 0
    assert (config.test_experiment / "host/test_started.json").is_file()
    assert not (config.test_experiment / "host/scoring_started.json").exists()

    second_harness = PiTestHarness(
        config,
        replay_runner=replay,
        score_runner=score_runner,
    )
    with pytest.raises(ValueError, match="experiment directory must be new and empty"):
        asyncio.run(second_harness.run())
    assert replay_calls == 1
    assert score_calls == 0


def test_invalid_validation_status_stops_before_preflight(tmp_path: Path) -> None:
    config = _preflight_config(tmp_path)
    report = config.validation_experiment / "host/run_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["status"] = "INTERRUPTED"
    report.write_text(json.dumps(payload), encoding="utf-8")
    replay_calls = 0

    async def replay(_request: ReplayRequest) -> TestReplayExecution:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("invalid Validation must not start preflight")

    with pytest.raises(ValueError, match="SUCCESS_REPRODUCIBLE"):
        asyncio.run(PiTestPreflight(config, replay_runner=replay).run())
    assert replay_calls == 0


def test_attestation_hash_mismatch_stops_before_test_raw(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    payload = json.loads(preflight.attestation.read_text(encoding="utf-8"))
    payload["frozen_snapshot_sha256"] = "changed"
    preflight.attestation.write_text(json.dumps(payload), encoding="utf-8")
    config = _test_config(tmp_path, preflight.attestation)
    replay_calls = 0

    async def replay(_request: ReplayRequest) -> TestReplayExecution:
        nonlocal replay_calls
        replay_calls += 1
        raise AssertionError("invalid attestation must not start Test")

    with pytest.raises(ValueError, match="attestation.*hash"):
        asyncio.run(PiTestHarness(config, replay_runner=replay).run())
    assert replay_calls == 0


def test_test_config_has_no_agent_or_declaration_repair_controls(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)

    assert not hasattr(config, "max_declaration_repairs")
    assert not hasattr(config, "max_iters")


def test_preflight_failure_uses_null_score(tmp_path: Path) -> None:
    config = _preflight_config(tmp_path)

    async def replay(_request: ReplayRequest) -> TestReplayExecution:
        return TestReplayExecution("TIMEOUT", None, "", "timeout", 10.0, None, 0, 0)

    result = asyncio.run(PiTestPreflight(config, replay_runner=replay).run())

    assert result.status == "PREFLIGHT_FAILED"
    assert result.phase == "preflight"
    assert result.scored is False
    assert result.score is None
    assert result.attestation is None


def test_preflight_runner_exception_is_reported_without_attestation(tmp_path: Path) -> None:
    config = _preflight_config(tmp_path)

    async def replay(_request: ReplayRequest) -> TestReplayExecution:
        raise OSError("cannot start replay")

    result = asyncio.run(PiTestPreflight(config, replay_runner=replay).run())

    assert result.status == "PREFLIGHT_FAILED"
    assert result.replay_status == "EXECUTION_FAILED"
    assert result.attestation is None


def test_frozen_snapshot_change_before_scoring_stops_without_gold(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)
    score_calls = 0

    async def replay(request: ReplayRequest) -> TestReplayExecution:
        execution = await _successful_replay([])(request)
        (config.validation_experiment / "host/reproducible_snapshot/scripts/build.py").write_text(
            "changed\n",
            encoding="utf-8",
        )
        return execution

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        nonlocal score_calls
        score_calls += 1
        raise AssertionError("changed frozen source must not be scored")

    with pytest.raises(RuntimeError, match="frozen.*changed"):
        asyncio.run(
            PiTestHarness(config, replay_runner=replay, score_runner=score_runner).run(),
        )
    assert score_calls == 0


def test_validation_report_must_point_to_actual_reproducible_snapshot(tmp_path: Path) -> None:
    config = _preflight_config(tmp_path)
    report = config.validation_experiment / "host/run_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["reproducible_snapshot"] = str(tmp_path / "somewhere_else")
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="reproducible_snapshot"):
        asyncio.run(PiTestPreflight(config).run())


def test_preflight_rejects_submission_that_requires_train_raw(tmp_path: Path) -> None:
    config = _preflight_config(tmp_path)
    submission = config.validation_experiment / "host/reproducible_snapshot/submission.json"
    payload = json.loads(submission.read_text(encoding="utf-8"))
    payload["replay"]["argv"].extend(["--train-raw", "{train_raw}"])
    submission.write_text(json.dumps(payload), encoding="utf-8")

    result = asyncio.run(PiTestPreflight(config).run())

    assert result.status == "PREFLIGHT_FAILED"
    assert result.replay_status == "INVALID_SUBMISSION"
    assert result.score is None


def test_preflight_requires_at_least_one_structured_output_file(tmp_path: Path) -> None:
    config = _preflight_config(tmp_path)

    async def replay(request: ReplayRequest) -> TestReplayExecution:
        result = request.runtime_root / "result"
        result.mkdir(parents=True)
        (result / "notes.txt").write_text("not structured\n", encoding="utf-8")
        return TestReplayExecution("SUCCESS", 0, "", "", 0.1, result, 0, 15)

    result = asyncio.run(PiTestPreflight(config, replay_runner=replay).run())

    assert result.status == "PREFLIGHT_FAILED"
    assert result.replay_status == "INVALID_OUTPUT"
    assert result.attestation is None


def test_scoring_failure_uses_null_score_and_does_not_retry(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)
    score_calls = 0

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        nonlocal score_calls
        score_calls += 1
        return ScoreExecution("EXECUTION_FAILED", 3, "scorer failed", 0.1, None)

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=_successful_replay([]),
            score_runner=score_runner,
        ).run(),
    )

    assert result.status == "SCORING_FAILED"
    assert result.phase == "scoring"
    assert result.scored is False
    assert result.score is None
    assert score_calls == 1


def test_test_runner_exception_is_not_retried_or_scored(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)
    replay_calls = 0
    score_calls = 0

    async def replay(_request: ReplayRequest) -> TestReplayExecution:
        nonlocal replay_calls
        replay_calls += 1
        raise OSError("cannot start Test replay")

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        nonlocal score_calls
        score_calls += 1
        raise AssertionError("replay exception must not score")

    result = asyncio.run(
        PiTestHarness(config, replay_runner=replay, score_runner=score_runner).run(),
    )

    assert result.status == "REPLAY_FAILED"
    assert result.score is None
    assert result.test_execution_count == 1
    assert replay_calls == 1
    assert score_calls == 0


def test_score_runner_exception_is_reported_once(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)
    score_calls = 0

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        nonlocal score_calls
        score_calls += 1
        raise OSError("cannot start scorer")

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=_successful_replay([]),
            score_runner=score_runner,
        ).run(),
    )

    assert result.status == "SCORING_FAILED"
    assert result.score is None
    assert score_calls == 1


def test_preflight_cancellation_writes_interrupted_report(tmp_path: Path) -> None:
    config = _preflight_config(tmp_path)

    async def replay(_request: ReplayRequest) -> TestReplayExecution:
        raise asyncio.CancelledError

    result = asyncio.run(PiTestPreflight(config, replay_runner=replay).run())

    assert result.status == "INTERRUPTED"
    assert result.phase == "preflight"
    assert result.scored is False
    assert result.score is None
    report = json.loads(
        (config.preflight_experiment / "host/run_report.json").read_text(encoding="utf-8"),
    )
    assert report["status"] == "INTERRUPTED"


def test_test_cancellation_during_scoring_never_writes_score(tmp_path: Path) -> None:
    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        raise asyncio.CancelledError

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=_successful_replay([]),
            score_runner=score_runner,
        ).run(),
    )

    assert result.status == "INTERRUPTED"
    assert result.phase == "scoring"
    assert result.test_execution_count == 1
    assert result.score is None
    assert not (config.test_experiment / "host/score_report.json").exists()


def test_default_preflight_replay_profile_has_only_public_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.pi_test_harness as harness_module

    config = _preflight_config(tmp_path)
    forbidden_test = tmp_path / "dataset/test/raw"
    forbidden_gold = tmp_path / "dataset/validation/reference_private"
    forbidden_train_raw = tmp_path / "dataset/train/raw"
    for path in (forbidden_test, forbidden_gold, forbidden_train_raw):
        path.mkdir(parents=True)
        (path / "secret.csv").write_text("secret\n", encoding="utf-8")
    captured = {}

    class FakeProcess:
        pid = 98765
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["profile"] = Path(command[2]).read_text(encoding="utf-8")
        workdir = Path(kwargs["cwd"])
        output = workdir / "result_package"
        output.mkdir(parents=True)
        (output / "data.csv").write_text("id\n1\n", encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr(
        harness_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    result = asyncio.run(PiTestPreflight(config).run())

    assert result.status == "SUCCESS"
    profile = captured["profile"]
    assert str(config.preflight_raw.resolve()) in profile
    assert str(forbidden_test.resolve()) not in profile
    assert str(forbidden_gold.resolve()) not in profile
    assert str(forbidden_train_raw.resolve()) not in profile
    assert str((config.project_root / "workflow").resolve()) not in profile
    assert "(deny network*)" in profile
    assert not any("API_KEY" in key for key in captured["env"])


def test_default_score_process_has_no_model_environment_and_hides_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.pi_test_harness as harness_module

    preflight_config = _preflight_config(tmp_path)
    preflight = asyncio.run(
        PiTestPreflight(preflight_config, replay_runner=_successful_replay([])).run(),
    )
    config = _test_config(tmp_path, preflight.attestation)
    captured = {}
    report = _score_report(0.37)
    report["file_reports"] = [
        {
            "relative_path": "data.csv",
            "reference_file": str(config.test_gold / "data.csv"),
            "result_file": str(config.test_experiment / "host/test_result_package/data.csv"),
            "issue": f"cannot read {config.test_gold / 'data.csv'}",
        },
    ]

    class FakeProcess:
        pid = 98766
        returncode = 0

        async def communicate(self):
            return json.dumps(report).encode(), b""

    async def fake_create_subprocess_exec(*command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["profile"] = Path(command[2]).read_text(encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr(
        harness_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-propagate")

    result = asyncio.run(
        PiTestHarness(config, replay_runner=_successful_replay([])).run(),
    )

    assert result.status == "SUCCESS"
    assert not any("API_KEY" in key for key in captured["env"])
    profile = captured["profile"]
    assert "(deny network*)" in profile
    saved = (config.test_experiment / "host/score_report.json").read_text(encoding="utf-8")
    assert str(config.test_gold.resolve()) not in saved
    assert '"reference_file"' not in saved
    assert '"result_file"' not in saved


def test_timed_out_process_ignoring_sigint_and_sigterm_is_killed() -> None:
    async def exercise() -> int:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            (
                "import signal,time; "
                "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "time.sleep(60)"
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        await _communicate_with_escalation(
            process,
            timeout=0.05,
            shutdown_timeout=0.05,
        )
        return process.returncode

    returncode = asyncio.run(exercise())

    assert returncode == -signal.SIGKILL


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="requires macOS sandbox-exec",
)
def test_real_preflight_and_test_replay_deny_private_inputs_and_network(
    tmp_path: Path,
) -> None:
    preflight_config = _preflight_config(tmp_path)
    source = preflight_config.validation_experiment / "host/reproducible_snapshot"
    forbidden_gold = tmp_path / "dataset/test/reference_private"
    forbidden_train_raw = tmp_path / "dataset/train/raw"
    forbidden_host = tmp_path / "forbidden_host"
    for path in (forbidden_gold, forbidden_train_raw, forbidden_host):
        path.mkdir(parents=True)
        (path / "secret.csv").write_text("secret\n", encoding="utf-8")
    workflow_file = preflight_config.project_root / "workflow/pi_harness_evaluation.py"
    forbidden_paths = [
        str(forbidden_gold / "secret.csv"),
        str(forbidden_train_raw / "secret.csv"),
        str(forbidden_host / "secret.csv"),
        str(workflow_file),
    ]
    (source / "scripts/build.py").write_text(
        "import argparse,pathlib,socket\n"
        "p=argparse.ArgumentParser(); p.add_argument('--raw'); p.add_argument('--out'); a=p.parse_args()\n"
        "checks=[]\n"
        f"for value in {forbidden_paths!r}:\n"
        " try: pathlib.Path(value).read_bytes(); checks.append(False)\n"
        " except OSError: checks.append(True)\n"
        "try:\n"
        " s=socket.socket(); s.bind(('127.0.0.1',0)); checks.append(False)\n"
        "except OSError: checks.append(True)\n"
        "assert all(checks), checks\n"
        "out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)\n"
        "(out/'data.csv').write_bytes((pathlib.Path(a.raw)/'data.csv').read_bytes())\n",
        encoding="utf-8",
    )

    preflight = asyncio.run(PiTestPreflight(preflight_config).run())
    config = _test_config(tmp_path, preflight.attestation)

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        return ScoreExecution("SUCCESS", 0, "", 0.1, _score_report(0.0))

    result = asyncio.run(PiTestHarness(config, score_runner=score_runner).run())

    assert preflight.status == "SUCCESS"
    assert result.status == "SUCCESS"
    assert result.test_execution_count == 1
    assert result.score == 0.0
