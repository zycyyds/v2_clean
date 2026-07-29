from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path

from agent.pi_harness import (
    PiValidationHarness,
    PiValidationHarnessConfig,
    ReplayExecution,
)


class FakeWorker:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.prompts: list[str] = []
        self.events: list[str] = []
        self.closed = False

    async def start(self) -> None:
        self.events.append("start")

    async def run_turn(self, prompt: str) -> dict:
        self.events.append("turn")
        self.prompts.append(prompt)
        round_index = len(self.prompts)
        (self.workdir / "result_package").mkdir(parents=True, exist_ok=True)
        (self.workdir / "result_package/data.csv").write_text(
            f"id,value\n1,{round_index}\n",
            encoding="utf-8",
        )
        (self.workdir / "scripts").mkdir(exist_ok=True)
        (self.workdir / "scripts/build.py").write_text(
            f"# round {round_index}\n",
            encoding="utf-8",
        )
        (self.workdir / "submission.json").write_text(
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
        return {"status": "SUCCESS"}

    async def clear_file_cache(self) -> None:
        self.events.append("clear_file_cache")

    async def close(self) -> None:
        self.closed = True


class InvalidThenValidWorker(FakeWorker):
    async def run_turn(self, prompt: str) -> dict:
        result = await super().run_turn(prompt)
        if len(self.prompts) == 1:
            submission = json.loads((self.workdir / "submission.json").read_text())
            submission["replay"]["argv"] = "python scripts/build.py"
            (self.workdir / "submission.json").write_text(
                json.dumps(submission),
                encoding="utf-8",
            )
        return result


class InterruptingWorker(FakeWorker):
    def __init__(self, workdir: Path) -> None:
        super().__init__(workdir)
        self.interrupted = False

    async def run_turn(self, prompt: str) -> dict:
        if self.prompts:
            self.events.append("turn_cancelled")
            raise asyncio.CancelledError
        return await super().run_turn(prompt)

    async def interrupt(self) -> None:
        self.interrupted = True
        self.events.append("interrupt")


def _report(score: float) -> dict:
    return {
        "status": "SUCCESS",
        "metrics": {"composite_score": score},
        "file_reports": [
            {
                "relative_path": "data.csv",
                "status": "SUCCESS",
                "reference_rows": 1,
                "result_rows": 1,
                "metrics": {"file_score": score},
            },
        ],
        "extra_result_files": [],
    }


def _config(tmp_path: Path) -> PiValidationHarnessConfig:
    train_raw = tmp_path / "data/train/raw"
    train_reference = tmp_path / "data/train/reference"
    validation_raw = tmp_path / "data/validation/raw"
    validation_gold = tmp_path / "data/validation/gold"
    for path in (train_raw, train_reference, validation_raw, validation_gold):
        path.mkdir(parents=True)
    (validation_gold / "data.csv").write_text("id,value\n1,gold\n", encoding="utf-8")
    manifest = tmp_path / "evaluation_manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "files": {"data.csv": {"key_columns": ["id"]}}},
        ),
        encoding="utf-8",
    )
    return PiValidationHarnessConfig(
        project_root=Path(__file__).parents[1],
        experiment_dir=tmp_path / "experiment",
        train_raw=train_raw,
        train_reference=train_reference,
        validation_raw=validation_raw,
        validation_gold=validation_gold,
        evaluation_manifest=manifest,
        max_rounds=2,
        patience=2,
        target_score=1.0,
        max_iters=10,
    )


def test_harness_reuses_worker_and_rolls_back_before_feedback(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_rounds=3, patience=3)
    worker = FakeWorker(config.experiment_dir / "agent_workdir")
    scores = iter([0.8, 0.4, 0.9])

    def scorer(*_args, **_kwargs):
        return _report(next(scores))

    async def replay(_source: Path) -> ReplayExecution:
        return ReplayExecution(
            status="SUCCESS",
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            result_root=_source / "result_package",
            score_report=_report(0.9),
        )

    harness = PiValidationHarness(
        config,
        worker=worker,
        scorer=scorer,
        replay_runner=replay,
        sandbox_probe=lambda *_args, **_kwargs: None,
    )
    result = asyncio.run(harness.run("initial task"))

    assert result.status == "SUCCESS_REPRODUCIBLE"
    assert result.best_score == 0.9
    assert len(worker.prompts) == 3
    assert '"restored_best": true' in worker.prompts[2]
    assert worker.events == ["start", "turn", "turn", "clear_file_cache", "turn"]
    assert worker.closed
    assert (config.experiment_dir / "agent_workdir/scripts/build.py").read_text() == "# round 3\n"


def test_harness_emits_low_frequency_phase_updates(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_rounds=1)
    worker = FakeWorker(config.experiment_dir / "agent_workdir")
    output = StringIO()

    async def replay(source: Path) -> ReplayExecution:
        return ReplayExecution(
            status="SUCCESS",
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            result_root=source / "result_package",
            score_report=_report(0.7),
        )

    harness = PiValidationHarness(
        config,
        worker=worker,
        scorer=lambda *_args, **_kwargs: _report(0.7),
        replay_runner=replay,
        sandbox_probe=lambda *_args, **_kwargs: None,
        output=output,
    )

    asyncio.run(harness.run("initial task"))

    updates = output.getvalue()
    assert "[Harness] round 1/1: Agent turn started" in updates
    assert "[Harness] round 1: hidden scoring started" in updates
    assert "[Harness] round 1: score=0.700000 promoted=true" in updates
    assert "[Harness] independent replay started" in updates
    assert "[Harness] independent replay: status=SUCCESS score=0.700000" in updates


def test_cancelled_turn_replays_best_and_closes_same_worker(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_rounds=3, patience=3)
    worker = InterruptingWorker(config.experiment_dir / "agent_workdir")

    async def replay(source: Path) -> ReplayExecution:
        return ReplayExecution(
            status="SUCCESS",
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            result_root=source / "result_package",
            score_report=_report(0.7),
        )

    harness = PiValidationHarness(
        config,
        worker=worker,
        scorer=lambda *_args, **_kwargs: _report(0.7),
        replay_runner=replay,
        sandbox_probe=lambda *_args, **_kwargs: None,
    )

    result = asyncio.run(harness.run("initial task"))

    assert result.status == "INTERRUPTED_REPLAYED"
    assert result.stop_reason == "keyboard_interrupt"
    assert result.rounds == 1
    assert result.best_score == 0.7
    assert result.reproducible_score == 0.7
    assert worker.interrupted
    assert worker.closed
    assert worker.events == ["start", "turn", "turn_cancelled", "interrupt"]
    report = json.loads((config.experiment_dir / "host/run_report.json").read_text())
    assert report["status"] == "INTERRUPTED_REPLAYED"


def test_sandbox_probe_failure_closes_worker_without_starting_attempt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    worker = FakeWorker(config.experiment_dir / "agent_workdir")

    def failing_probe(*_args, **_kwargs) -> None:
        raise RuntimeError("unsafe sandbox")

    harness = PiValidationHarness(
        config,
        worker=worker,
        scorer=lambda *_args, **_kwargs: _report(0.7),
        sandbox_probe=failing_probe,
    )

    try:
        asyncio.run(harness.run("initial task"))
    except RuntimeError as exc:
        assert str(exc) == "unsafe sandbox"
    else:
        raise AssertionError("sandbox probe failure should abort the run")

    assert worker.closed
    assert worker.events == []


def test_replay_failure_returns_to_same_agent_until_reproducible(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(config, max_rounds=1)
    worker = FakeWorker(config.experiment_dir / "agent_workdir")
    replay_scores = iter([None, 0.7])

    async def replay(source: Path) -> ReplayExecution:
        score = next(replay_scores)
        return ReplayExecution(
            status="EXECUTION_FAILED" if score is None else "SUCCESS",
            returncode=1 if score is None else 0,
            stdout="",
            stderr="failed safely" if score is None else "",
            duration_seconds=0.1,
            result_root=source / "result_package",
            score_report=None if score is None else _report(score),
        )

    harness = PiValidationHarness(
        config,
        worker=worker,
        scorer=lambda *_args, **_kwargs: _report(0.7),
        replay_runner=replay,
        sandbox_probe=lambda *_args, **_kwargs: None,
    )
    result = asyncio.run(harness.run("initial task"))

    assert result.status == "SUCCESS_REPRODUCIBLE"
    assert len(worker.prompts) == 2
    assert "independent replay failed" in worker.prompts[1].lower()
    assert worker.events.count("start") == 1


def test_invalid_submission_returns_to_same_worker_for_correction(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_rounds=2, patience=2)
    worker = InvalidThenValidWorker(config.experiment_dir / "agent_workdir")

    async def replay(source: Path) -> ReplayExecution:
        return ReplayExecution(
            status="SUCCESS",
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            result_root=source / "result_package",
            score_report=_report(0.7),
        )

    harness = PiValidationHarness(
        config,
        worker=worker,
        scorer=lambda *_args, **_kwargs: _report(0.7),
        replay_runner=replay,
        sandbox_probe=lambda *_args, **_kwargs: None,
    )

    result = asyncio.run(harness.run("initial task"))

    assert result.status == "SUCCESS_REPRODUCIBLE"
    assert result.best_score == 0.7
    assert worker.events.count("start") == 1
    assert len(worker.prompts) == 2
    assert "submission_error" in worker.prompts[1]
    assert "replay.argv must be a non-empty string array" in worker.prompts[1]


def test_replay_removes_temporary_directory_after_invalid_source(tmp_path: Path) -> None:
    config = _config(tmp_path)
    harness = PiValidationHarness(
        config,
        sandbox_probe=lambda *_args, **_kwargs: None,
    )
    harness._prepare()
    invalid_source = tmp_path / "invalid-source"
    invalid_source.mkdir()
    (invalid_source / "link").symlink_to(config.validation_raw)

    replay = asyncio.run(harness._run_replay(invalid_source))

    assert replay.status == "INVALID_SOURCE"
    assert not list((config.experiment_dir / "host/runtime").glob("replay-*"))


def test_real_independent_replay_denies_gold_network_and_host_write(tmp_path: Path) -> None:
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        return
    config = _config(tmp_path)
    (config.validation_raw / "data.csv").write_text(
        "id,value\n1,gold\n",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "result_package").mkdir()
    (source / "result_package/data.csv").write_text("stale", encoding="utf-8")
    forbidden_write = config.experiment_dir / "host" / "forbidden.txt"
    script = source / "build.py"
    script.write_text(
        "import argparse, pathlib, socket\n"
        "p=argparse.ArgumentParser(); p.add_argument('--raw'); p.add_argument('--out'); a=p.parse_args()\n"
        "checks=[]\n"
        f"\ntry: pathlib.Path({str(config.validation_gold / 'data.csv')!r}).read_text(); checks.append(False)\n"
        "except PermissionError: checks.append(True)\n"
        "try:\n"
        " s=socket.socket(); s.bind(('127.0.0.1', 0)); checks.append(False)\n"
        "except PermissionError: checks.append(True)\n"
        f"\ntry: pathlib.Path({str(forbidden_write)!r}).write_text('bad'); checks.append(False)\n"
        "except PermissionError: checks.append(True)\n"
        "assert all(checks), checks\n"
        "out=pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'data.csv').write_bytes((pathlib.Path(a.raw)/'data.csv').read_bytes())\n",
        encoding="utf-8",
    )
    (source / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result_root": "result_package",
                "replay": {
                    "argv": [
                        "python", "build.py", "--raw", "{raw_root}", "--out", "{output_dir}",
                    ],
                },
            },
        ),
        encoding="utf-8",
    )
    harness = PiValidationHarness(
        config,
        sandbox_probe=lambda *_args, **_kwargs: None,
    )
    harness._prepare()

    replay = asyncio.run(harness._run_replay(source))

    assert replay.status == "SUCCESS"
    assert replay.score == 1.0
    assert not forbidden_write.exists()
