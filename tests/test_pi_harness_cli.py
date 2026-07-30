from __future__ import annotations

import asyncio
import signal

import pytest

from agent.pi_harness_cli import _run_with_terminal_signals, parse_args


def test_harness_cli_parses_public_and_private_inputs() -> None:
    args = parse_args(
        [
            "--experiment-dir", "/tmp/experiment",
            "--train-raw", "/data/train/raw",
            "--train-reference", "/data/train/reference",
            "--validation-raw", "/data/validation/raw",
            "--validation-gold", "/private/validation/gold",
            "--evaluation-manifest", "/manifests/mimic.json",
            "--dataset-manifest", "/data/split_manifest.json",
            "--prompt-file", "/prompts/task.txt",
            "--skills-dir", "/skills/one",
            "--skills-dir", "/skills/two",
        ],
    )

    assert args.max_rounds == 20
    assert args.patience == 3
    assert args.target_score == 1.0
    assert args.max_iters == 10_000
    assert args.dataset_manifest == "/data/split_manifest.json"
    assert args.skills_dir == ["/skills/one", "/skills/two"]


@pytest.mark.parametrize("requested_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_terminal_signals_cancel_once_and_allow_cleanup(
    requested_signal: signal.Signals,
    monkeypatch,
) -> None:
    async def exercise() -> tuple[str, signal.Signals | None, list[signal.Signals]]:
        loop = asyncio.get_running_loop()
        callbacks = {}
        removed = []

        def add_signal_handler(signum, callback, *args):
            callbacks[signum] = (callback, args)

        def remove_signal_handler(signum):
            removed.append(signum)
            return True

        monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
        monkeypatch.setattr(loop, "remove_signal_handler", remove_signal_handler)

        async def operation() -> str:
            callback, args = callbacks[requested_signal]
            callback(*args)
            callback(*args)
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                return "cleaned"
            raise AssertionError("signal callback did not cancel the operation")

        result, received = await _run_with_terminal_signals(operation())
        return result, received, removed

    result, received, removed = asyncio.run(exercise())

    assert result == "cleaned"
    assert received == requested_signal
    assert removed == [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
