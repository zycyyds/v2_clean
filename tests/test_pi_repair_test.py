from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent.pi_harness import directory_sha256
from agent.pi_repair_test import (
    PiRepairTestConfig,
    PiRepairTestHarness,
)
from agent.pi_train_repair import RepairReplayExecution


def _config(tmp_path: Path) -> PiRepairTestConfig:
    train = tmp_path / "train_experiment"
    snapshot = train / "host/reproducible_snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline").mkdir()
    (snapshot / "pipeline/run.py").write_text("print('ok')\n", encoding="utf-8")
    (snapshot / "repair_submission.json").write_text(
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
    frozen_hash = directory_sha256(snapshot)
    (train / "host/run_report.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS_REPRODUCIBLE",
                "frozen_snapshot_sha256": frozen_hash,
            }
        ),
        encoding="utf-8",
    )
    test_raw = tmp_path / "public/test_raw"
    test_raw.mkdir(parents=True)
    (test_raw / "table.csv").write_text("a\n1\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    gold = private / "test_gold.csv"
    gold.write_text("operation,raw_file,raw_row_index,column,clean_value,dirty_value\n", encoding="utf-8")
    return PiRepairTestConfig(
        project_root=Path(__file__).parents[1],
        train_experiment=train,
        test_experiment=tmp_path / "test_experiment",
        test_raw=test_raw,
        test_gold_log=gold,
        replay_timeout_seconds=10,
    )


def test_one_shot_test_scores_without_agent_and_writes_sanitized_audit(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    calls = 0

    async def replay(**kwargs) -> RepairReplayExecution:
        nonlocal calls
        calls += 1
        output = Path(kwargs["retained_output"])
        output.mkdir(parents=True)
        return RepairReplayExecution(
            "SUCCESS", 0, "", "", 0.1, output, None
        )

    monkeypatch.setattr(
        "agent.pi_repair_test.score_repair_output",
        lambda **_: {
            "status": "SUCCESS",
            "metrics": {
                "exact_repair_f1": 0.4,
                "candidate_recall_at_5": 0.6,
                "clean_preservation": 0.99,
            },
            "repair": {"failure_cases": [{"case_id": "x", "failure": "missed"}]},
        },
    )
    harness = PiRepairTestHarness(config, replay_runner=replay)

    result = asyncio.run(harness.run())

    assert result.status == "SUCCESS"
    assert result.test_execution_count == 1
    assert calls == 1
    run_manifest = json.loads(
        (config.test_experiment / "host/run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["model_access"] is False
    assert run_manifest["api_credentials_required"] is False
    train_snapshot = config.train_experiment / "host/reproducible_snapshot"
    assert run_manifest["train_frozen_snapshot_sha256"] == directory_sha256(
        train_snapshot
    )
    assert not (config.test_experiment / "host/frozen_snapshot/outputs").exists()
    failures = json.loads(
        (config.test_experiment / "host/sanitized_failure_cases.json").read_text(
            encoding="utf-8"
        )
    )
    assert failures["failure_cases"][0]["failure"] == "missed"


def test_one_shot_test_does_not_score_failed_replay(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    scored = False

    async def replay(**kwargs) -> RepairReplayExecution:
        return RepairReplayExecution(
            "EXECUTION_FAILED", 1, "", "boom", 0.1, None, None
        )

    def score(**kwargs):
        nonlocal scored
        scored = True
        return {}

    monkeypatch.setattr("agent.pi_repair_test.score_repair_output", score)
    harness = PiRepairTestHarness(config, replay_runner=replay)

    result = asyncio.run(harness.run())

    assert result.status == "REPLAY_FAILED"
    assert result.test_execution_count == 1
    assert not scored
