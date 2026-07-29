from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path

import pytest

from agent.pi_harness import directory_sha256


class DeclarationWorker:
    def __init__(self, workdir: Path, frozen_hash: str, *, invalid_first: bool = False) -> None:
        self.workdir = workdir
        self.frozen_hash = frozen_hash
        self.invalid_first = invalid_first
        self.prompts: list[str] = []
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def run_turn(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        payload = {
            "schema_version": 1,
            "frozen_snapshot_sha256": self.frozen_hash,
            "replay_argv": [
                "python",
                "{frozen_root}/scripts/build.py",
                "--raw",
                "{raw_root}",
                "--out",
                "{output_dir}",
                "--train-reference",
                "{train_reference}",
            ],
        }
        if self.invalid_first and len(self.prompts) == 1:
            payload["frozen_snapshot_sha256"] = "wrong"
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "runner_spec.json").write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "SUCCESS", "model_calls": 1, "tool_calls": 1}

    async def close(self) -> None:
        self.closed = True


def _fixture(tmp_path: Path):
    from agent.pi_test_harness import PiTestHarnessConfig

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
            }
        ),
        encoding="utf-8",
    )
    train_reference = tmp_path / "dataset/train/reference"
    test_raw = tmp_path / "dataset/test/raw"
    test_gold = tmp_path / "dataset/test/reference_private"
    for path in (train_reference, test_raw, test_gold):
        path.mkdir(parents=True)
    (test_raw / "data.csv").write_text("id,value\n1,test\n", encoding="utf-8")
    (test_gold / "data.csv").write_text("id,value\n1,test\n", encoding="utf-8")
    evaluation_manifest = tmp_path / "evaluation.json"
    evaluation_manifest.write_text(
        json.dumps({"schema_version": 1, "files": {"data.csv": {"key_columns": ["id"]}}}),
        encoding="utf-8",
    )
    (validation / "host/run_manifest.json").write_text(
        json.dumps({"train_reference": str(train_reference)}),
        encoding="utf-8",
    )
    return PiTestHarnessConfig(
        project_root=Path(__file__).parents[1],
        validation_experiment=validation,
        test_experiment=tmp_path / "test_experiment",
        test_raw=test_raw,
        test_gold=test_gold,
        evaluation_manifest=evaluation_manifest,
        max_declaration_repairs=3,
    )


def _score(value: float = 0.8) -> dict:
    return {
        "status": "SUCCESS",
        "metrics": {"composite_score": value},
        "file_reports": [],
    }


def test_direct_replay_scores_once_without_creating_test_agent(tmp_path: Path) -> None:
    from agent.pi_test_harness import PiTestHarness, TestReplayExecution

    config = _fixture(tmp_path)
    worker_created = False
    score_calls = 0

    async def replay(_snapshot: Path, _runner_spec: Path | None) -> TestReplayExecution:
        result = config.test_experiment / "host/test_result_package"
        result.mkdir(parents=True)
        (result / "data.csv").write_text("id,value\n1,test\n", encoding="utf-8")
        return TestReplayExecution("SUCCESS", 0, "", "", 0.1, result)

    def worker_factory(*_args):
        nonlocal worker_created
        worker_created = True
        raise AssertionError("direct replay must not create a Test Agent")

    def scorer(*_args):
        nonlocal score_calls
        score_calls += 1
        return _score()

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=replay,
            scorer=scorer,
            worker_factory=worker_factory,
            sandbox_probe=lambda *_args: None,
        ).run()
    )

    assert result.status == "SUCCESS"
    assert result.declaration_agent_used is False
    assert result.score == 0.8
    assert score_calls == 1
    assert not worker_created
    manifest = json.loads((config.test_experiment / "host/run_manifest.json").read_text())
    assert manifest["frozen_snapshot_sha256"] == directory_sha256(
        config.test_experiment / "host/frozen_snapshot"
    )


def test_failed_direct_replay_uses_one_fresh_agent_and_closes_before_scoring(tmp_path: Path) -> None:
    from agent.pi_test_harness import PiTestHarness, TestReplayExecution

    config = _fixture(tmp_path)
    frozen_hash = directory_sha256(config.validation_experiment / "host/reproducible_snapshot")
    worker = DeclarationWorker(
        config.test_experiment / "test_agent_workdir",
        frozen_hash,
        invalid_first=True,
    )
    replay_calls: list[Path | None] = []

    async def replay(_snapshot: Path, runner_spec: Path | None) -> TestReplayExecution:
        replay_calls.append(runner_spec)
        if runner_spec is None:
            return TestReplayExecution("EXECUTION_FAILED", 2, "", "bad argv", 0.1, None)
        result = config.test_experiment / "host/test_result_package"
        result.mkdir(parents=True, exist_ok=True)
        (result / "data.csv").write_text("id,value\n1,test\n", encoding="utf-8")
        return TestReplayExecution("SUCCESS", 0, "", "", 0.1, result)

    def scorer(*_args):
        assert worker.closed, "Gold scoring must happen only after the Test Agent closes"
        return _score(0.9)

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=replay,
            scorer=scorer,
            worker_factory=lambda *_args: worker,
            sandbox_probe=lambda *_args: None,
        ).run()
    )

    assert result.status == "SUCCESS"
    assert result.declaration_agent_used is True
    assert result.declaration_rounds == 2
    assert worker.started and worker.closed
    assert len(replay_calls) == 2
    assert replay_calls[0] is None
    assert replay_calls[1].name == "runner_spec.json"
    assert "wrong" in worker.prompts[1]
    assert all("0.9" not in prompt for prompt in worker.prompts)
    assert directory_sha256(config.test_experiment / "host/frozen_snapshot") == frozen_hash


def test_declaration_failure_stops_after_three_repairs_without_scoring(tmp_path: Path) -> None:
    from agent.pi_test_harness import PiTestHarness, TestReplayExecution

    config = _fixture(tmp_path)
    frozen_hash = directory_sha256(config.validation_experiment / "host/reproducible_snapshot")
    worker = DeclarationWorker(config.test_experiment / "test_agent_workdir", frozen_hash)

    async def replay(_snapshot: Path, _runner_spec: Path | None) -> TestReplayExecution:
        return TestReplayExecution("EXECUTION_FAILED", 2, "", "still broken", 0.1, None)

    def scorer(*_args):
        raise AssertionError("failed Test replay must not read or score Gold")

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=replay,
            scorer=scorer,
            worker_factory=lambda *_args: worker,
            sandbox_probe=lambda *_args: None,
        ).run()
    )

    assert result.status == "REPLAY_FAILED"
    assert result.declaration_rounds == 3
    assert len(worker.prompts) == 3
    assert worker.closed


def test_frozen_snapshot_change_is_detected_before_gold_scoring(tmp_path: Path) -> None:
    from agent.pi_test_harness import PiTestHarness, TestReplayExecution

    config = _fixture(tmp_path)

    async def replay(snapshot: Path, _runner_spec: Path | None) -> TestReplayExecution:
        (snapshot / "scripts/build.py").write_text("changed\n", encoding="utf-8")
        result = config.test_experiment / "host/test_result_package"
        result.mkdir(parents=True)
        (result / "data.csv").write_text("id,value\n1,test\n", encoding="utf-8")
        return TestReplayExecution("SUCCESS", 0, "", "", 0.1, result)

    with pytest.raises(RuntimeError, match="frozen Test snapshot changed"):
        asyncio.run(
            PiTestHarness(
                config,
                replay_runner=replay,
                scorer=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("Gold must not be scored after a frozen snapshot change")
                ),
                worker_factory=lambda *_args: None,
                sandbox_probe=lambda *_args: None,
            ).run()
        )


def test_runner_spec_rejects_pipeline_changes_shell_and_unknown_placeholders(tmp_path: Path) -> None:
    from agent.pi_test_harness import load_runner_spec

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "pipeline.py").write_text("x", encoding="utf-8")
    workdir = tmp_path / "agent"
    workdir.mkdir()
    path = workdir / "runner_spec.json"
    base = {
        "schema_version": 1,
        "frozen_snapshot_sha256": directory_sha256(frozen),
        "replay_argv": ["python", "{frozen_root}/pipeline.py", "{raw_root}", "{output_dir}"],
    }
    path.write_text(json.dumps(base), encoding="utf-8")
    assert load_runner_spec(path, frozen).replay_argv[0] == "python"

    for replacement, message in (
        ({**base, "frozen_snapshot_sha256": "changed"}, "hash"),
        ({**base, "replay_argv": ["sh", "-c", "echo bad"]}, "shell"),
        (
            {
                **base,
                "replay_argv": [
                    "python", "-c", "print('new logic')", "{frozen_root}",
                    "{raw_root}", "{output_dir}",
                ],
            },
            "frozen snapshot",
        ),
        (
            {
                **base,
                "replay_argv": [
                    "python", "{frozen_root}/../adapt.py", "{raw_root}", "{output_dir}",
                ],
            },
            "escapes",
        ),
        ({**base, "replay_argv": ["python", "{gold_root}/x.py", "{output_dir}"]}, "placeholder"),
    ):
        path.write_text(json.dumps(replacement), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_runner_spec(path, frozen)


def test_cancelled_test_replay_kills_and_reaps_process_group(monkeypatch) -> None:
    from agent.pi_test_harness import _communicate_with_cleanup

    class Process:
        pid = 4321
        returncode = None

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.waited = False

        async def communicate(self):
            self.started.set()
            await asyncio.Event().wait()

        async def wait(self):
            self.waited = True
            self.returncode = -signal.SIGKILL
            return self.returncode

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("agent.pi_test_harness.os.killpg", lambda pid, sig: killed.append((pid, sig)))

    async def exercise() -> Process:
        process = Process()
        task = asyncio.create_task(_communicate_with_cleanup(process, timeout=60.0))
        await process.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return process

    process = asyncio.run(exercise())
    assert killed == [(4321, signal.SIGKILL)]
    assert process.waited is True
