from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.pi_train_repair import (
    PiTrainRepairConfig,
    PiTrainRepairHarness,
    RepairReplayExecution,
    _offline_environment,
    default_repair_skill_dirs,
    freeze_repair_source,
)
class _FakeWorker:
    def __init__(self, workdir: Path) -> None:
        self.prompts: list[str] = []
        self.started = False
        self.closed = False
        self.workdir = workdir

    async def start(self) -> None:
        self.started = True

    async def run_turn(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        pipeline = self.workdir / "pipeline"
        pipeline.mkdir(exist_ok=True)
        (pipeline / "run.py").write_text("print('ok')\n", encoding="utf-8")
        (self.workdir / "repair_submission.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "outputs": {
                        "corrected_raw": "corrected_raw",
                        "repair_candidates": "repair_candidates.jsonl",
                    },
                    "replay": {
                        "argv": [
                            "python",
                            "pipeline/run.py",
                            "--raw-root",
                            "{raw_root}",
                            "--output-dir",
                            "{output_dir}",
                        ]
                    },
                }
            ) + "\n",
            encoding="utf-8",
        )
        return {
            "model_calls": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "react_iterations": 1,
            "tool_calls": 2,
            "tool_errors": 0,
            "api_failovers": 0,
            "compactions": 0,
        }

    async def close(self) -> None:
        self.closed = True


def _config(tmp_path: Path, *, max_repair_turns: int = 2) -> PiTrainRepairConfig:
    public = tmp_path / "public"
    for name in ("dirty_raw", "clean_raw", "evidence"):
        (public / name).mkdir(parents=True)
    (public / "public_train_manifest.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "supervision": {"dirty_count": 20_000, "clean_count": 10_000},
            }
        ),
        encoding="utf-8",
    )
    skills: list[Path] = []
    for index in range(4):
        skill = tmp_path / f"skill_{index}"
        skill.mkdir()
        (skill / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")
        skills.append(skill)
    files = {}
    for name in ("gold.csv", "row_gold.csv"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        files[name] = path
    return PiTrainRepairConfig(
        project_root=Path(__file__).parents[1],
        experiment_dir=tmp_path / "experiment",
        public_train_root=public,
        train_gold_log=files["gold.csv"],
        train_row_gold_log=files["row_gold.csv"],
        max_iters=20,
        max_repair_turns=max_repair_turns,
        skill_dirs=tuple(skills),
        replay_timeout_seconds=10,
    )


def test_train_only_harness_freezes_successful_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    worker = _FakeWorker(config.experiment_dir / "agent_workdir")

    async def replay(source: Path, output: Path) -> RepairReplayExecution:
        output.mkdir(parents=True, exist_ok=True)
        return RepairReplayExecution(
            status="SUCCESS",
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            output_root=output,
            score_report={
                "status": "SUCCESS",
                "metrics": {
                    "exact_repair_f1": 0.5,
                    "candidate_recall_at_5": 0.8,
                    "clean_preservation": 1.0,
                },
            },
        )

    monkeypatch.setattr(
        "agent.pi_train_repair.build_worker_model_environment",
        lambda *_: {
            "OPENAI_API_KEYS_JSON": "[]",
            "OPENAI_API_BASE": "https://example.invalid",
            "MODEL_NAME": "MiniMax-M3",
            "AGENT_TEMPERATURE": "0",
            "AGENT_SEED": "666",
        },
    )
    harness = PiTrainRepairHarness(
        config,
        worker=worker,
        replay_runner=replay,
        sandbox_probe=lambda *_: None,
    )

    result = asyncio.run(harness.run("Learn repairs"))

    assert result.status == "SUCCESS_REPRODUCIBLE"
    assert result.frozen_snapshot.is_dir()
    assert sorted(
        path.relative_to(result.frozen_snapshot).as_posix()
        for path in result.frozen_snapshot.rglob("*")
        if path.is_file()
    ) == ["pipeline/run.py", "repair_submission.json"]
    assert result.train_report is not None and result.train_report.is_file()
    assert worker.started and worker.closed
    assert len(worker.prompts) == 1
    assert "There is no Validation loop" in worker.prompts[0]
    assert "20,000 dirty pairs" in worker.prompts[0]
    assert "Do not build the 17-file formatting pipeline" in worker.prompts[0]
    assert "train-reference" not in worker.prompts[0]
    run_report = json.loads((config.experiment_dir / "host/run_report.json").read_text())
    assert run_report["usage"]["model_calls"] == 1


def test_train_only_harness_uses_bounded_public_train_repair_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, max_repair_turns=1)
    worker = _FakeWorker(config.experiment_dir / "agent_workdir")
    calls = 0

    async def replay(source: Path, output: Path) -> RepairReplayExecution:
        nonlocal calls
        calls += 1
        return RepairReplayExecution(
            status="INVALID_OUTPUT",
            returncode=None,
            stdout="",
            stderr="missing candidate report",
            duration_seconds=0.1,
            output_root=output,
            score_report=None,
        )

    monkeypatch.setattr(
        "agent.pi_train_repair.build_worker_model_environment",
        lambda *_: {
            "OPENAI_API_KEYS_JSON": "[]",
            "MODEL_NAME": "MiniMax-M3",
        },
    )
    harness = PiTrainRepairHarness(
        config,
        worker=worker,
        replay_runner=replay,
        sandbox_probe=lambda *_: None,
    )

    result = asyncio.run(harness.run())

    assert result.status == "TRAIN_REPLAY_FAILED"
    assert calls == 2
    assert len(worker.prompts) == 2
    assert "missing candidate report" in worker.prompts[1]
    assert not result.frozen_snapshot.exists()


def test_default_skill_set_is_exactly_the_four_correction_views(tmp_path: Path) -> None:
    names = {path.name for path in default_repair_skill_dirs(tmp_path)}
    assert names == {
        "correction_intra_table_errors",
        "correction_entity_alignment_errors",
        "correction_cross_table_errors",
        "correction_task_oriented_errors",
    }


def test_frozen_replay_environment_contains_no_model_credentials(tmp_path: Path) -> None:
    environment = _offline_environment(tmp_path / "home", tmp_path / "tmp")
    assert "OPENAI_API_KEY" not in environment
    assert "OPENAI_API_KEYS_JSON" not in environment
    assert "OPENAI_API_BASE" not in environment
    assert "MODEL_NAME" not in environment


def test_freeze_repair_source_excludes_outputs_and_caches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pipeline/__pycache__").mkdir(parents=True)
    (source / "pipeline/run.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "pipeline/__pycache__/run.pyc").write_bytes(b"cache")
    (source / "outputs/train/corrected_raw").mkdir(parents=True)
    (source / "outputs/train/corrected_raw/table.csv").write_text("x\n", encoding="utf-8")
    (source / "repair_submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "outputs": {
                    "corrected_raw": "corrected_raw",
                    "repair_candidates": "repair_candidates.jsonl",
                },
                "replay": {
                    "argv": [
                        "python",
                        "pipeline/run.py",
                        "--raw-root",
                        "{raw_root}",
                        "--output-dir",
                        "{output_dir}",
                    ]
                },
            }
        ) + "\n",
        encoding="utf-8",
    )

    destination = tmp_path / "frozen"
    freeze_repair_source(source, destination)

    assert (destination / "pipeline/run.py").is_file()
    assert not (destination / "pipeline/__pycache__").exists()
    assert not (destination / "outputs").exists()


def test_freeze_repair_source_rejects_pipeline_data_assets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pipeline").mkdir(parents=True)
    (source / "pipeline/run.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "pipeline/train_values.csv").write_text("private\n", encoding="utf-8")
    (source / "repair_submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "outputs": {
                    "corrected_raw": "corrected_raw",
                    "repair_candidates": "repair_candidates.jsonl",
                },
                "replay": {
                    "argv": [
                        "python",
                        "pipeline/run.py",
                        "--raw-root",
                        "{raw_root}",
                        "--output-dir",
                        "{output_dir}",
                    ]
                },
            }
        ) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-source artifact"):
        freeze_repair_source(source, tmp_path / "frozen")
