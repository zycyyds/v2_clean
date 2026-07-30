from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.pi_full_experiment_cli as cli


def _argv(prompt_file: Path) -> list[str]:
    return [
        "--experiment-dir", "/runs/validation",
        "--train-raw", "/data/train/raw",
        "--train-reference", "/data/train/reference",
        "--validation-raw", "/data/validation/raw",
        "--validation-gold", "/private/validation/gold",
        "--evaluation-manifest", "/manifests/evaluation.json",
        "--dataset-manifest", "/manifests/dataset.json",
        "--prompt-file", str(prompt_file),
        "--skills-dir", "/skills/one",
        "--skills-dir", "/skills/two",
        "--test-experiment", "/runs/test",
        "--test-raw", "/data/test/raw",
        "--test-gold", "/private/test/gold",
    ]


def _combined_report(*, status: str = "SUCCESS") -> dict:
    test = (
        {
            "status": "SUCCESS",
            "score": 0.89,
            "frozen_snapshot_sha256": "a" * 64,
        }
        if status == "SUCCESS"
        else None
    )
    return {
        "status": status,
        "phase": "complete" if status == "SUCCESS" else "validation",
        "validation": {
            "rounds": 4,
            "best_score": 0.93,
            "reproducible_score": 0.91,
        },
        "test": test,
        "test_execution_count": 1 if test is not None else 0,
    }


def _result(report_path: Path, *, status: str = "SUCCESS") -> SimpleNamespace:
    validation = SimpleNamespace(
        rounds="GOLD_VALUE_MUST_NOT_LEAK /private/test/gold",
        best_score="GOLD_VALUE_MUST_NOT_LEAK /private/test/gold",
        reproducible_score="GOLD_VALUE_MUST_NOT_LEAK /private/test/gold",
    )
    test = SimpleNamespace(
        status="GOLD_VALUE_MUST_NOT_LEAK /private/test/gold",
        score="GOLD_VALUE_MUST_NOT_LEAK /private/test/gold",
        test_execution_count="GOLD_VALUE_MUST_NOT_LEAK /private/test/gold",
        frozen_snapshot_sha256="invalid GOLD_VALUE_MUST_NOT_LEAK /private/test/gold",
    )
    return SimpleNamespace(
        status=status,
        phase="complete" if status == "SUCCESS" else "test_replay",
        validation_result=validation,
        test_result=test,
        report_path=report_path,
    )


def test_parse_args_accepts_complete_experiment_inputs_and_defaults(
    tmp_path: Path,
) -> None:
    args = cli.parse_args(_argv(tmp_path / "prompt.txt"))

    assert args.experiment_dir == "/runs/validation"
    assert args.train_raw == "/data/train/raw"
    assert args.train_reference == "/data/train/reference"
    assert args.validation_raw == "/data/validation/raw"
    assert args.validation_gold == "/private/validation/gold"
    assert args.evaluation_manifest == "/manifests/evaluation.json"
    assert args.dataset_manifest == "/manifests/dataset.json"
    assert args.prompt_file == str(tmp_path / "prompt.txt")
    assert args.max_rounds == 20
    assert args.patience == 3
    assert args.target_score == 1.0
    assert args.max_iters == 10_000
    assert args.skills_dir == ["/skills/one", "/skills/two"]
    assert args.replay_timeout == 1800.0
    assert args.test_experiment == "/runs/test"
    assert args.test_raw == "/data/test/raw"
    assert args.test_gold == "/private/test/gold"
    assert args.test_replay_timeout == 1800.0
    assert args.test_scoring_timeout == 3600.0
    assert not hasattr(args, "preflight_attestation")
    assert not hasattr(args, "confirmation")


@pytest.mark.parametrize(
    "forbidden_args",
    [
        ["--preflight-attestation", "/runs/preflight/attestation.json"],
        ["--confirm-test"],
    ],
)
def test_parse_args_rejects_preflight_and_confirmation_controls(
    tmp_path: Path,
    forbidden_args: list[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([*_argv(tmp_path / "prompt.txt"), *forbidden_args])


def test_run_builds_full_config_reads_prompt_once_and_prints_public_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("run the complete experiment", encoding="utf-8")
    report_path = tmp_path / "full_experiment_report.json"
    report_path.write_text(json.dumps(_combined_report()), encoding="utf-8")
    captured: dict[str, object] = {"run_calls": 0}

    class FakeExperiment:
        def __init__(self, config) -> None:
            captured["config"] = config

        async def run(self, prompt: str):
            captured["run_calls"] = int(captured["run_calls"]) + 1
            captured["prompt"] = prompt
            return _result(report_path)

    async def fake_signal_wrapper(operation):
        captured["operation"] = operation
        return await operation, None

    monkeypatch.setattr(cli, "PiFullExperiment", FakeExperiment)
    monkeypatch.setattr(cli, "_run_with_terminal_signals", fake_signal_wrapper)

    exit_code = asyncio.run(cli._run(cli.parse_args(_argv(prompt_file))))

    assert exit_code == 0
    assert captured["run_calls"] == 1
    assert captured["prompt"] == "run the complete experiment"
    config = captured["config"]
    assert config.validation.project_root == Path(cli.__file__).parents[1]
    assert config.validation.experiment_dir == Path("/runs/validation")
    assert config.validation.train_raw == Path("/data/train/raw")
    assert config.validation.train_reference == Path("/data/train/reference")
    assert config.validation.validation_raw == Path("/data/validation/raw")
    assert config.validation.validation_gold == Path("/private/validation/gold")
    assert config.validation.evaluation_manifest == Path("/manifests/evaluation.json")
    assert config.validation.dataset_manifest == Path("/manifests/dataset.json")
    assert config.validation.max_rounds == 20
    assert config.validation.patience == 3
    assert config.validation.target_score == 1.0
    assert config.validation.max_iters == 10_000
    assert config.validation.skill_dirs == (Path("/skills/one"), Path("/skills/two"))
    assert config.validation.replay_timeout_seconds == 1800.0
    assert config.test_experiment == Path("/runs/test")
    assert config.test_raw == Path("/data/test/raw")
    assert config.test_gold == Path("/private/test/gold")
    assert config.evaluation_manifest == config.validation.evaluation_manifest
    assert config.test_replay_timeout_seconds == 1800.0
    assert config.test_scoring_timeout_seconds == 3600.0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "status": "SUCCESS",
        "phase": "complete",
        "validation": {
            "rounds": 4,
            "best_score": 0.93,
            "reproducible_score": 0.91,
        },
        "test": {
            "status": "SUCCESS",
            "score": 0.89,
            "test_execution_count": 1,
            "frozen_snapshot_sha256": "a" * 64,
        },
        "report_path": str(report_path),
    }
    assert "GOLD_VALUE_MUST_NOT_LEAK" not in output
    assert "/private/validation/gold" not in output
    assert "/private/test/gold" not in output


def test_run_projects_none_test_from_combined_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt", encoding="utf-8")
    report_path = tmp_path / "full_experiment_report.json"
    report_path.write_text(
        json.dumps(_combined_report(status="VALIDATION_FAILED")),
        encoding="utf-8",
    )

    class FakeExperiment:
        def __init__(self, _config) -> None:
            pass

        async def run(self, _prompt: str):
            return _result(report_path, status="VALIDATION_FAILED")

    async def fake_signal_wrapper(operation):
        return await operation, None

    monkeypatch.setattr(cli, "PiFullExperiment", FakeExperiment)
    monkeypatch.setattr(cli, "_run_with_terminal_signals", fake_signal_wrapper)

    assert asyncio.run(cli._run(cli.parse_args(_argv(prompt_file)))) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["test"] is None
    assert "test_execution_count" not in payload


@pytest.mark.parametrize(
    ("status", "received_signal", "expected"),
    [
        ("SUCCESS", None, 0),
        ("INTERRUPTED", None, 130),
        ("REPLAY_FAILED", None, 1),
        ("SUCCESS", signal.SIGTERM, 130),
    ],
)
def test_run_maps_terminal_outcomes_to_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    received_signal: signal.Signals | None,
    expected: int,
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt", encoding="utf-8")
    report_path = tmp_path / "full_experiment_report.json"
    report_path.write_text(
        json.dumps(_combined_report(status=status)),
        encoding="utf-8",
    )

    class FakeExperiment:
        def __init__(self, _config) -> None:
            pass

        async def run(self, _prompt: str):
            return _result(report_path, status=status)

    async def fake_signal_wrapper(operation):
        return await operation, received_signal

    monkeypatch.setattr(cli, "PiFullExperiment", FakeExperiment)
    monkeypatch.setattr(cli, "_run_with_terminal_signals", fake_signal_wrapper)

    assert asyncio.run(cli._run(cli.parse_args(_argv(prompt_file)))) == expected


def test_main_maps_keyboard_interrupt_to_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(_args) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "parse_args", lambda _argv: SimpleNamespace())
    monkeypatch.setattr(cli, "_run", interrupt)

    assert cli.main([]) == 130


def test_main_redacts_regular_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(_args) -> int:
        raise ValueError("GOLD_VALUE_MUST_NOT_LEAK /private/test/gold")

    monkeypatch.setattr(cli, "parse_args", lambda _argv: SimpleNamespace())
    monkeypatch.setattr(cli, "_run", fail)

    assert cli.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "status": "CLI_FAILED",
        "error_code": "FULL_EXPERIMENT_EXCEPTION",
    }
    assert "GOLD_VALUE_MUST_NOT_LEAK" not in captured.err
    assert "/private/test/gold" not in captured.err
