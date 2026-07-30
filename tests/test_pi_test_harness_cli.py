from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.pi_test_harness_cli as cli
from agent.pi_test_preflight_cli import parse_args as parse_preflight_args


def test_preflight_cli_parses_strict_public_inputs() -> None:
    args = parse_preflight_args(
        [
            "--validation-experiment", "/runs/validation",
            "--preflight-experiment", "/runs/preflight",
            "--preflight-raw", "/data/validation/raw",
        ],
    )

    assert args.replay_timeout == 1800.0
    assert not hasattr(args, "max_iters")


def _test_argv() -> list[str]:
    return [
        "--validation-experiment", "/runs/validation",
        "--test-experiment", "/runs/test",
        "--test-raw", "/data/test/raw",
        "--test-gold", "/data/test/reference_private",
        "--evaluation-manifest", "/manifests/evaluation.json",
    ]


def test_test_cli_defaults_to_no_attestation_and_has_no_agent_controls() -> None:
    args = cli.parse_args(_test_argv())

    assert args.preflight_attestation is None
    assert args.replay_timeout == 1800.0
    assert args.scoring_timeout == 3600.0
    assert not hasattr(args, "max_declaration_repairs")
    assert not hasattr(args, "max_iters")


def test_test_cli_still_accepts_preflight_attestation() -> None:
    args = cli.parse_args(
        [
            *_test_argv(),
            "--preflight-attestation", "/runs/preflight/host/preflight_attestation.json",
        ],
    )

    assert args.preflight_attestation == "/runs/preflight/host/preflight_attestation.json"


@pytest.mark.parametrize(
    ("attestation", "expected"),
    [
        (None, None),
        (
            "/runs/preflight/host/preflight_attestation.json",
            Path("/runs/preflight/host/preflight_attestation.json"),
        ),
    ],
)
def test_test_cli_explicitly_passes_optional_attestation_to_config(
    monkeypatch: pytest.MonkeyPatch,
    attestation: str | None,
    expected: Path | None,
) -> None:
    captured: dict[str, object] = {}

    class FakeHarness:
        def __init__(self, config) -> None:
            captured["config"] = config

        async def run(self):
            return SimpleNamespace(
                status="SUCCESS",
                phase="complete",
                scored=True,
                score=0.8,
                test_execution_count=1,
                replay_status="SUCCESS",
                frozen_snapshot_sha256="b" * 64,
                result_package=Path("/runs/test/host/test_result_package"),
            )

    async def fake_signal_wrapper(operation):
        return await operation, None

    monkeypatch.setattr(cli, "PiTestHarness", FakeHarness)
    monkeypatch.setattr(cli, "_run_with_terminal_signals", fake_signal_wrapper)
    argv = _test_argv()
    if attestation is not None:
        argv.extend(["--preflight-attestation", attestation])

    assert asyncio.run(cli._run(cli.parse_args(argv))) == 0
    assert captured["config"].preflight_attestation == expected


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
        "error_code": "TEST_HARNESS_EXCEPTION",
    }
    assert "GOLD_VALUE_MUST_NOT_LEAK" not in captured.err
    assert "/private/test/gold" not in captured.err
