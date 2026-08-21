from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from agent.pi_fcorr_synthesis import (
    PiFcorrSynthesisConfig,
    PiFcorrSynthesisHarness,
)


GOOD_RULE = '''def GenerateCandidates(input_string, row_context):
    if input_string == "BAD":
        return [{"value": "GOOD", "rule_id": "train_mapping", "evidence": "paired Train evidence"}]
    return []
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(tmp_path: Path) -> Path:
    root = tmp_path / "evidence"
    root.mkdir()
    field = root / "field.json"
    field.write_text(json.dumps({
        "table": "icu/events",
        "column": "amount",
        "dirty_clean_pairs": [
            {"dirty": "BAD", "clean": "GOOD", "row_context": {"unit": "mg"}},
        ],
        "clean_examples": [
            {"value": "GOOD", "row_context": {"unit": "mg"}},
        ],
    }), encoding="utf-8")
    manifest = {
        "schema_version": 3,
        "status": "SUCCESS",
        "workflow": "gidcl_train_field_pairs",
        "fields": [{
            "field_id": "field1",
            "table": "icu/events",
            "column": "amount",
            "path": "field.json",
            "sha256": _sha256(field),
        }],
    }
    (root / "field_pairs_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


class FakeWorker:
    def __init__(self, workdir: Path, source: str = GOOD_RULE) -> None:
        self.workdir = workdir
        self.source = source
        self.prompts: list[str] = []
        self.started = False
        self.closed = False
        self.cache_clears = 0

    async def start(self) -> None:
        self.started = True

    async def run_turn(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        path = self.workdir / "rules" / "field1" / "correction.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source, encoding="utf-8")
        return {"iterations": 3, "model_calls": 2, "tool_calls": 4}

    async def clear_file_cache(self) -> None:
        self.cache_clears += 1

    async def close(self) -> None:
        self.closed = True


def test_pi_fcorr_freezes_agent_rule_in_existing_registry_format(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    output = tmp_path / "rules"
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# correction\n", encoding="utf-8")
    worker = FakeWorker(output / "agent_workdir")
    harness = PiFcorrSynthesisHarness(
        PiFcorrSynthesisConfig(
            project_root=Path(__file__).parents[1],
            evidence_dir=evidence,
            output_dir=output,
            max_rounds=2,
            skill_dirs=(skill,),
        ),
        worker=worker,
    )

    result = asyncio.run(harness.run())

    assert result.status == "SUCCESS"
    assert result.rounds == 1
    assert result.rule_count == 1
    assert worker.started and worker.closed
    registry = json.loads((output / "rule_registry.json").read_text())
    assert registry["workflow"] == "gidcl_multicandidate_fcorr_rule_registry"
    assert registry["test_time_llm_access"] is False
    assert registry["rules"]["icu/events.amount"]["train_candidate_recall_at_5"] == 1.0
    field_manifest = json.loads(
        (output / "fields" / "field1" / "field_manifest.json").read_text()
    )
    assert field_manifest["synthesis_runtime"] == "persistent_pi_agent"
    assert field_manifest["selected_round"] == 1
    assert field_manifest["validation"]["metrics"]["clean_preservation_rate"] == 1.0


def test_pi_fcorr_rejects_missing_rule_after_bounded_feedback(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    output = tmp_path / "rules"
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# correction\n", encoding="utf-8")
    worker = FakeWorker(
        output / "agent_workdir",
        "def GenerateCandidates(input_string, row_context):\n    return []\n",
    )
    harness = PiFcorrSynthesisHarness(
        PiFcorrSynthesisConfig(
            project_root=Path(__file__).parents[1],
            evidence_dir=evidence,
            output_dir=output,
            max_rounds=2,
            skill_dirs=(skill,),
        ),
        worker=worker,
    )

    result = asyncio.run(harness.run())

    assert result.rounds == 2
    assert result.rule_count == 0
    assert worker.cache_clears == 1
    assert "host_feedback" in worker.prompts[1]
    synthesis = json.loads((output / "synthesis_manifest.json").read_text())
    assert synthesis["status_counts"] == {"FCORR_REJECTED": 1}
    feedback = json.loads(
        (output / "agent_workdir" / "host_feedback" / "round_01.json").read_text()
    )
    assert feedback["accepted_count"] == 0
    assert feedback["fields"][0]["validation"]["metrics"][
        "candidate_recall_at_5"
    ] == 0.0
