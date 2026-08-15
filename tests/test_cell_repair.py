from __future__ import annotations

import asyncio
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from graph.cell_repair import (
    CellRepairError,
    _extract_correction_source,
    _minimax_completion,
    _static_rule_issues,
    apply_repair_plan,
    build_field_pairs,
    build_repair_targets,
    evaluate_candidates,
    evaluate_repairs,
    recover_raw_from_graph,
    run_frozen_rules,
    synthesize_fcorr,
    validate_fcorr,
)


AMOUNT_RULE = '''def GenerateCandidates(input_string, row_context):
    if re.fullmatch(r"[1-5]00", input_string):
        return [{
            "value": input_string[:-2],
            "rule_id": "scale_100",
            "evidence": "remove two trailing zeros",
        }]
    return []
'''

STATUS_RULE = '''def GenerateCandidates(input_string, row_context):
    match = re.fullmatch(r"BAD([0-4])", input_string)
    if match:
        return [{
            "value": "active" + match.group(1),
            "rule_id": "status_suffix",
            "evidence": "preserve numeric suffix",
        }]
    return []
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> dict[str, Path | dict[str, dict[str, int]]]:
    graph = tmp_path / "graph"
    supervision = tmp_path / "private"
    raw = tmp_path / "raw"
    graph.mkdir()
    supervision.mkdir()
    raw.mkdir()
    raw_rows: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    log_rows: list[dict[str, object]] = []
    supervised_indices: list[int] = []
    labels: list[int] = []
    folds: list[int] = []

    def add_cell(
        *,
        fold: int,
        column: str,
        current: str,
        label: int,
        clean: str = "",
        subtype: str = "",
    ) -> int:
        row_number = len(raw_rows) + 1
        row = {
            "stay_id": str(70_000_000 + row_number),
            "amount": "",
            "status": "",
            "note": "public-row-context",
        }
        row[column] = current
        raw_rows.append(row)
        observation_index = -1
        for row_column, raw_value in row.items():
            index = len(observations)
            observations.append({
                "row_id": row_number - 1,
                "table": "icu/events",
                "row_number": row_number,
                "column": row_column,
                "raw_value": raw_value,
                "value_node_id": 1000 + index,
            })
            if row_column == column:
                observation_index = index
        assert observation_index >= 0
        supervised_indices.append(observation_index)
        labels.append(label)
        folds.append(fold)
        if label:
            log_rows.append({
                "raw_file": "icu/events.csv",
                "raw_row_index": row_number - 1,
                "column": column,
                "clean_value": clean,
                "dirty_value": current,
                "error_class": "synthetic",
                "error_subtype": subtype,
                "canonical_stay_id": 30_000_000 + fold,
                "injection_seed": 666,
            })
        return observation_index

    indices: dict[int, dict[str, int]] = {}
    for fold in range(5):
        indices[fold] = {
            "dirty_amount": add_cell(
                fold=fold,
                column="amount",
                current=f"{fold + 1}00",
                clean=str(fold + 1),
                label=1,
                subtype="unit_scale_error",
            ),
            "dirty_status": add_cell(
                fold=fold,
                column="status",
                current=f"BAD{fold}",
                clean=f"active{fold}",
                label=1,
                subtype="enum_violation",
            ),
            "dirty_locator": add_cell(
                fold=fold,
                column="stay_id",
                current=str(99_000_000 + fold),
                clean=str(33_000_000 + fold),
                label=1,
                subtype="orphan_feature_stay_id",
            ),
            "clean_amount": add_cell(
                fold=fold,
                column="amount",
                current="500" if fold == 4 else str(5 + fold),
                label=0,
            ),
            "clean_status": add_cell(
                fold=fold,
                column="status",
                current=f"valid{fold}",
                label=0,
            ),
        }

    _write_csv(raw / "icu" / "events.csv", raw_rows)
    (raw / "unchanged.txt").write_text("untouched\n", encoding="utf-8")
    (graph / "cell_observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8"
    )
    (graph / "graph_manifest.json").write_text(
        json.dumps({
            "raw_root_name": "raw",
            "table_counts": {"icu/events": len(raw_rows)},
            "observation_count": len(observations),
        }) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        supervision / "supervision_masks.npz",
        cell_indices=np.asarray(supervised_indices, dtype=np.int64),
        cell_labels=np.asarray(labels, dtype=np.int8),
        cell_folds=np.asarray(folds, dtype=np.int8),
        cell_source=np.asarray([2 if value else 0 for value in labels], dtype=np.int8),
    )
    log = supervision / "merged_injection_log.csv"
    _write_csv(log, log_rows)

    predictions = tmp_path / "predictions.csv"
    prediction_rows: list[dict[str, object]] = []
    for split, fold in (("validation", 3), ("internal_test", 4)):
        split_predictions = {
            "dirty_amount": 1,
            "dirty_status": 0,
            "dirty_locator": 0,
            "clean_amount": 1,
            "clean_status": 0,
        }
        for name, observation_index in indices[fold].items():
            prediction_rows.append({
                "split": split,
                "observation_index": observation_index,
                "dirty_score": 0.99 if split_predictions[name] else 0.01,
                "threshold": 0.67,
                "prediction": split_predictions[name],
                "label": int(name.startswith("dirty_")),
                "source": "synthetic_dirty" if name.startswith("dirty_") else "sampled_clean",
                "fold": fold,
            })
    _write_csv(predictions, prediction_rows)
    return {
        "graph": graph,
        "supervision": supervision,
        "raw": raw,
        "log": log,
        "predictions": predictions,
        "counts": {
            "train": {"dirty": 9, "clean": 6},
            "validation": {"dirty": 3, "clean": 2},
            "internal_test": {"dirty": 3, "clean": 2},
        },
    }


def _build_pairs(paths: dict[str, object], output: Path) -> Path:
    build_field_pairs(
        graph_dir=paths["graph"],
        supervision_dir=paths["supervision"],
        raw_dir=paths["raw"],
        paired_log=paths["log"],
        output_dir=output,
        expected_counts=paths["counts"],
    )
    return output


def _good_completion(messages, evidence, attempt):
    del messages, attempt
    source = AMOUNT_RULE if evidence["column"] == "amount" else STATUS_RULE
    return f"General pattern.\n```python\n{source}```", "MiniMax-M3"


def _synthesize(paths: dict[str, object], tmp_path: Path) -> Path:
    evidence = _build_pairs(paths, tmp_path / "pairs")
    rules = tmp_path / "rules"
    synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=rules,
        completion=_good_completion,
    )
    return rules


def test_recover_raw_from_graph_round_trips_and_builds_pairs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    recovered = tmp_path / "recovered"
    report = recover_raw_from_graph(
        graph_dir=paths["graph"],
        output_dir=recovered,
    )

    assert report["status"] == "SUCCESS"
    assert report["table_count"] == 1
    assert report["row_count"] == 25
    assert report["observation_count"] == 100
    recovered_table = recovered / "icu" / "events.csv"
    assert recovered_table.read_bytes() == (Path(paths["raw"]) / "icu" / "events.csv").read_bytes()
    assert report["tables"]["icu/events"]["sha256"] == _sha256(recovered_table)

    pair_report = build_field_pairs(
        graph_dir=paths["graph"],
        supervision_dir=paths["supervision"],
        raw_dir=recovered,
        paired_log=paths["log"],
        output_dir=tmp_path / "recovered-pairs",
        expected_counts=paths["counts"],
    )
    assert pair_report["split_counts"] == paths["counts"]
    assert pair_report["exported_train_dirty_pair_count"] == 6


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("column_order", "columns changed"),
        ("non_contiguous_row", "non-contiguous row numbers"),
        ("duplicate_column", "duplicate Cell observation column"),
    ],
)
def test_recover_raw_from_graph_rejects_inconsistent_observations(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _fixture(tmp_path)
    observations_path = Path(paths["graph"]) / "cell_observations.jsonl"
    observations = [json.loads(line) for line in observations_path.read_text().splitlines()]
    if mutation == "column_order":
        observations[4], observations[5] = observations[5], observations[4]
    elif mutation == "non_contiguous_row":
        for observation in observations[4:8]:
            observation["row_number"] = 3
    else:
        observations[1]["column"] = observations[0]["column"]
    observations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in observations),
        encoding="utf-8",
    )

    output = tmp_path / f"bad-recovery-{mutation}"
    with pytest.raises(CellRepairError, match=message):
        recover_raw_from_graph(graph_dir=paths["graph"], output_dir=output)
    assert not output.exists()


def test_build_field_pairs_is_complete_train_only_and_redacted(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = _build_pairs(paths, tmp_path / "pairs")
    manifest = json.loads((output / "field_pairs_manifest.json").read_text())

    assert manifest["split_counts"] == paths["counts"]
    assert manifest["exported_train_dirty_pair_count"] == 6
    assert manifest["exported_train_clean_example_count"] == 6
    assert manifest["excluded_locator_pair_count"] == 3
    assert {entry["column"] for entry in manifest["fields"]} == {"amount", "status"}
    body = "\n".join(
        (output / entry["path"]).read_text() for entry in manifest["fields"]
    )
    for forbidden in (
        "BAD3",
        "BAD4",
        "active3",
        "active4",
        "unit_scale_error",
        "injection_seed",
        "fold",
        "label",
        "row_number",
        "observation_index",
        "99000000",
    ):
        assert forbidden not in body
    for entry in manifest["fields"]:
        evidence = json.loads((output / entry["path"]).read_text())
        for pair in evidence["dirty_clean_pairs"]:
            assert "row_context" in pair
            assert entry["column"] not in pair["row_context"]
            assert "stay_id" not in pair["row_context"]
        for example in evidence["clean_examples"]:
            assert set(example) == {"value", "row_context"}
            assert entry["column"] not in example["row_context"]
            assert "stay_id" not in example["row_context"]
        assert entry["sha256"] == _sha256(output / entry["path"])
        assert entry["serialized_bytes"] == (output / entry["path"]).stat().st_size


def test_build_field_pairs_without_raw_uses_graph_values_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    checked = _build_pairs(paths, tmp_path / "checked-pairs")
    unchecked = tmp_path / "graph-pairs"
    manifest = build_field_pairs(
        graph_dir=paths["graph"],
        supervision_dir=paths["supervision"],
        paired_log=paths["log"],
        output_dir=unchecked,
        expected_counts=paths["counts"],
    )

    assert manifest["status"] == "SUCCESS"
    assert manifest["current_value_source"] == "cell_observations"
    assert manifest["raw_cross_check_enabled"] is False
    assert manifest["inputs"]["raw_table_sha256"] == {}
    for entry in manifest["fields"]:
        assert (unchecked / entry["path"]).read_bytes() == (
            checked / entry["path"]
        ).read_bytes()


def test_build_field_pairs_fails_on_counts_and_dirty_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(CellRepairError, match="frozen protocol"):
        build_field_pairs(
            graph_dir=paths["graph"],
            supervision_dir=paths["supervision"],
            raw_dir=paths["raw"],
            paired_log=paths["log"],
            output_dir=tmp_path / "bad-counts",
            expected_counts={"train": {"dirty": 1, "clean": 1}},
        )

    rows = list(csv.DictReader(Path(paths["log"]).open(encoding="utf-8")))
    rows[0]["dirty_value"] = "not-current"
    bad_log = tmp_path / "bad-log.csv"
    _write_csv(bad_log, rows)
    with pytest.raises(CellRepairError, match="dirty value mismatch"):
        build_field_pairs(
            graph_dir=paths["graph"],
            supervision_dir=paths["supervision"],
            raw_dir=paths["raw"],
            paired_log=bad_log,
            output_dir=tmp_path / "bad-log-output",
            expected_counts=paths["counts"],
        )


def test_validate_fcorr_reports_candidate_recall_and_rejects_invalid_lists() -> None:
    evidence = {
        "table": "t",
        "column": "v",
        "dirty_clean_pairs": [
            {"dirty": "X", "clean": "A", "row_context": {"kind": "alpha"}},
            {"dirty": "X", "clean": "B", "row_context": {"kind": "beta"}},
            {"dirty": "X", "clean": "C", "row_context": {"kind": "gamma"}},
        ],
        "clean_examples": [
            {"value": "A", "row_context": {"kind": "alpha"}},
            {"value": "B", "row_context": {"kind": "beta"}},
        ],
    }
    source = '''def GenerateCandidates(input_string, row_context):
    kind = row_context.get("kind", "")
    if kind == "alpha":
        return [{"value": "A", "rule_id": "context_kind", "evidence": "kind=alpha"}]
    if kind == "beta":
        return [
            {"value": "A", "rule_id": "fallback", "evidence": "common value"},
            {"value": "B", "rule_id": "context_kind", "evidence": "kind=beta"},
        ]
    return [
        {"value": "A", "rule_id": "fallback", "evidence": "common value"},
        {"value": "B", "rule_id": "fallback", "evidence": "alternate value"},
        {"value": "C", "rule_id": "context_kind", "evidence": "kind=gamma"},
    ]
'''
    report = validate_fcorr(source, evidence, minimum_recall_at_5=0.85)
    assert report["status"] == "SUCCESS"
    assert report["metrics"]["candidate_recall_at_1"] == pytest.approx(1 / 3)
    assert report["metrics"]["candidate_recall_at_3"] == 1.0
    assert report["metrics"]["candidate_recall_at_5"] == 1.0
    assert report["metrics"]["mean_reciprocal_rank"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert report["metrics"]["clean_preservation_rate"] == 1.0
    assert report["metrics"]["maximum_candidate_count"] == 3

    report = validate_fcorr(
        "def GenerateCandidates(input_string, row_context):\n    return []\n",
        evidence,
    )
    assert report["status"] == "FAILED"
    assert report["metrics"]["candidate_recall_at_5"] == 0.0
    assert len(report["missing_candidate_pairs"]) == 3

    report = validate_fcorr(
        "def GenerateCandidates(input_string, row_context):\n    return 'A'\n",
        evidence,
    )
    assert report["status"] == "FAILED"
    assert report["metrics"]["runtime_issue_count"] == 5

    too_many = "def GenerateCandidates(input_string, row_context):\n    return " + repr([
        {"value": str(index), "rule_id": "r", "evidence": "e"}
        for index in range(6)
    ]) + "\n"
    report = validate_fcorr(too_many, evidence)
    assert report["status"] == "FAILED"
    assert any("at most 5" in issue for issue in report["issues"])


def test_static_sandbox_rejects_import_ids_dynamic_calls_and_multiple_functions() -> None:
    assert any("imports are forbidden" in issue for issue in _static_rule_issues(
        "def GenerateCandidates(input_string, row_context):\n    import os\n    return []\n"
    ))
    assert any("suspicious hard-coded identifier" in issue for issue in _static_rule_issues(
        "def GenerateCandidates(input_string, row_context):\n"
        "    return [{'value': '33976251', 'rule_id': 'x', 'evidence': 'x'}]\n"
    ))
    assert any("forbidden call 'open'" in issue for issue in _static_rule_issues(
        "def GenerateCandidates(input_string, row_context):\n    return open(input_string)\n"
    ))
    assert not _static_rule_issues(
        "def GenerateCandidates(input_string, row_context):\n"
        "    candidates = []\n"
        "    candidates.append({'value': 'M', 'rule_id': 'r', 'evidence': 'e'})\n"
        "    return candidates\n"
    )
    assert any("unsafe attribute 'compile'" in issue for issue in _static_rule_issues(
        "def GenerateCandidates(input_string, row_context):\n"
        "    pattern = re.compile('x')\n"
        "    return []\n"
    ))
    response = (
        "```python\ndef GenerateCandidates(input_string, row_context):\n    return []\n```\n"
        "```python\ndef GenerateCandidates(input_string, row_context):\n"
        "    return [{'value': input_string, 'rule_id': 'keep', 'evidence': 'keep'}]\n```"
    )
    with pytest.raises(CellRepairError, match="exactly one"):
        _extract_correction_source(response)
    with pytest.raises(CellRepairError, match="exactly one"):
        _extract_correction_source("def GenerateCandidates(:\n    pass")


def test_extract_correction_source_ignores_minimax_thinking_drafts() -> None:
    response = (
        "<think>\n"
        "```python\n"
        "def GenerateCandidates(input_string, row_context):\n"
        "    return [{'value': 'draft', 'rule_id': 'draft', 'evidence': 'draft'}]\n"
        "```\n"
        "</think>\n"
        "Final answer:\n"
        "```python\n"
        "def GenerateCandidates(input_string, row_context):\n"
        "    candidates = []\n"
        "    if input_string == 'UNKNOWN_GENDER_CODE':\n"
        "        candidates.append({'value': 'M', 'rule_id': 'r1', 'evidence': 'train'})\n"
        "    return candidates\n"
        "```\n"
    )

    source = _extract_correction_source(response)

    assert "'draft'" not in source
    assert "candidates.append" in source
    assert validate_fcorr(
        source,
        {
            "table": "hosp/patients",
            "column": "gender",
            "dirty_clean_pairs": [{
                "dirty": "UNKNOWN_GENDER_CODE",
                "clean": "M",
                "row_context": {},
            }],
            "clean_examples": [],
        },
    )["status"] == "SUCCESS"


def test_synthesis_uses_field_isolated_history_and_counterexample_retry(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    evidence = _build_pairs(paths, tmp_path / "pairs")
    calls: list[tuple[str, int, list[str], str]] = []

    def completion(messages, field_evidence, attempt):
        calls.append((
            field_evidence["column"],
            attempt,
            [message["role"] for message in messages],
            messages[-1]["content"],
        ))
        if field_evidence["column"] == "amount" and attempt == 1:
            return (
                "def GenerateCandidates(input_string, row_context):\n    return []\n",
                "MiniMax-M3",
            )
        return _good_completion(messages, field_evidence, attempt)

    output = tmp_path / "rules"
    manifest = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=output,
        completion=completion,
    )
    assert manifest["status_counts"] == {"FROZEN": 2}
    amount_calls = [call for call in calls if call[0] == "amount"]
    assert amount_calls[1][2] == ["system", "user", "assistant", "user"]
    feedback = json.loads(amount_calls[1][3].split("\n\n", 1)[1])
    assert len(feedback["missing_candidate_pairs"]) == 3
    assert feedback["previous_function"].startswith("def GenerateCandidates")
    assert feedback["metrics"]["candidate_recall_at_5"] == 0.0
    status_first = next(call for call in calls if call[0] == "status")
    assert status_first[2] == ["system", "user"]
    assert "BAD3" not in status_first[3] and "BAD4" not in status_first[3]
    assert manifest["registry"]["rule_count"] == 2
    assert manifest["registry"]["test_time_llm_access"] is False


def test_minimax_completion_passes_agentscope_messages_to_model(monkeypatch) -> None:
    agentscope_message = pytest.importorskip("agentscope.message")
    Msg = agentscope_message.Msg
    TextBlock = agentscope_message.TextBlock

    import lib.agent_runtime as runtime

    observed_roles: list[str] = []
    closed = False

    class FakeModel:
        async def __call__(self, messages):
            assert all(isinstance(message, Msg) for message in messages)
            observed_roles.extend(message.role for message in messages)
            return SimpleNamespace(content=[TextBlock(type="text", text=AMOUNT_RULE)])

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    class RejectingFormatter:
        async def format(self, messages):
            del messages
            raise AssertionError("model must own AgentScope message formatting")

    monkeypatch.setattr(
        runtime,
        "create_openai_model_and_formatter",
        lambda *args, **kwargs: (FakeModel(), RejectingFormatter()),
    )
    monkeypatch.setattr(runtime, "resolve_model_name", lambda *args: "MiniMax-M3")

    response, model_name = asyncio.run(_minimax_completion(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ],
        agent_key="react_planner",
    ))

    assert response == AMOUNT_RULE.strip()
    assert model_name == "MiniMax-M3"
    assert observed_roles == ["system", "user", "assistant"]
    assert closed is True


def test_synthesis_stops_at_twelve_or_terminal_generation_failure(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    evidence = _build_pairs(paths, tmp_path / "pairs")
    attempts: list[int] = []

    def never_correct(messages, field_evidence, attempt):
        del messages, field_evidence
        attempts.append(attempt)
        return "def GenerateCandidates(input_string, row_context):\n    return []\n", "MiniMax-M3"

    rejected = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=tmp_path / "rejected",
        fields=["icu/events.amount"],
        max_attempts=12,
        completion=never_correct,
    )
    assert attempts == list(range(1, 13))
    assert rejected["status_counts"] == {"FCORR_REJECTED": 1}
    assert rejected["registry"]["rule_count"] == 0

    def too_large(messages, field_evidence, attempt):
        del messages, field_evidence, attempt
        raise RuntimeError("maximum context length exceeded")

    limited = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=tmp_path / "too-large",
        fields=["icu/events.amount"],
        completion=too_large,
    )
    assert limited["status_counts"] == {"CONTEXT_TOO_LARGE": 1}

    runtime_attempts: list[int] = []

    def runtime_error(messages, field_evidence, attempt):
        del messages, field_evidence
        runtime_attempts.append(attempt)
        raise ModuleNotFoundError("No module named 'httpx'")

    runtime_failure = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=tmp_path / "runtime-error",
        fields=["icu/events.amount"],
        completion=runtime_error,
    )
    assert runtime_attempts == [1]
    assert runtime_failure["status_counts"] == {"SYNTHESIS_RUNTIME_ERROR": 1}
    runtime_field = runtime_failure["fields"][0]
    runtime_manifest = json.loads(
        (tmp_path / "runtime-error" / runtime_field["field_manifest"]).read_text()
    )
    assert runtime_manifest["attempt_count"] == 1

    wrong_model = synthesize_fcorr(
        evidence_dir=evidence,
        output_dir=tmp_path / "wrong-model",
        fields=["icu/events.amount"],
        max_attempts=1,
        completion=lambda messages, field, attempt: (AMOUNT_RULE, "other-model"),
    )
    assert wrong_model["status_counts"] == {"MODEL_MISMATCH": 1}


@pytest.mark.parametrize("tampered", ["source", "evidence", "manifest"])
def test_frozen_registry_rejects_tampering(tmp_path: Path, tampered: str) -> None:
    paths = _fixture(tmp_path)
    rules = _synthesize(paths, tmp_path)
    copied = tmp_path / f"rules-{tampered}"
    shutil.copytree(rules, copied)
    registry = json.loads((copied / "rule_registry.json").read_text())
    entry = next(iter(registry["rules"].values()))
    target = {
        "source": copied / entry["source_path"],
        "evidence": copied / entry["evidence_path"],
        "manifest": copied / "synthesis_manifest.json",
    }[tampered]
    target.write_text(target.read_text() + "\n", encoding="utf-8")
    targets_dir = tmp_path / f"targets-{tampered}"
    build_repair_targets(
        predictions=paths["predictions"],
        graph_dir=paths["graph"],
        raw_dir=paths["raw"],
        split="internal_test",
        output_dir=targets_dir,
    )
    with pytest.raises(CellRepairError, match="hash mismatch"):
        run_frozen_rules(
            targets=targets_dir / "repair_targets.csv",
            rule_dir=copied,
            output_dir=tmp_path / f"run-{tampered}",
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", 2),
        ("workflow", "gidcl_direct_fcorr_rule_registry"),
        ("maximum_candidates", 1),
    ],
)
def test_frozen_registry_rejects_protocol_tampering(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    paths = _fixture(tmp_path)
    rules = _synthesize(paths, tmp_path)
    registry_path = rules / "rule_registry.json"
    registry = json.loads(registry_path.read_text())
    registry[key] = value
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    targets_dir = tmp_path / "targets"
    build_repair_targets(
        predictions=paths["predictions"],
        graph_dir=paths["graph"],
        split="internal_test",
        output_dir=targets_dir,
    )

    with pytest.raises(CellRepairError, match="not frozen for offline execution"):
        run_frozen_rules(
            targets=targets_dir / "repair_targets.csv",
            rule_dir=rules,
            output_dir=tmp_path / "run",
        )


def test_offline_target_run_and_private_candidate_audit_are_deterministic(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rules = _synthesize(paths, tmp_path)
    targets = tmp_path / "targets"
    target_manifest = build_repair_targets(
        predictions=paths["predictions"],
        graph_dir=paths["graph"],
        raw_dir=paths["raw"],
        split="internal_test",
        output_dir=targets,
    )
    assert target_manifest["target_count"] == 2
    target_rows = list(csv.DictReader((targets / "repair_targets.csv").open()))
    assert all("row_context_json" in row for row in target_rows)
    assert all("stay_id" not in json.loads(row["row_context_json"]) for row in target_rows)
    assert all(row["column"] not in json.loads(row["row_context_json"]) for row in target_rows)

    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    first_manifest = run_frozen_rules(
        targets=targets / "repair_targets.csv", rule_dir=rules, output_dir=first
    )
    run_frozen_rules(
        targets=targets / "repair_targets.csv", rule_dir=rules, output_dir=second
    )
    assert first_manifest["candidate_count"] == 2
    assert first_manifest["targets_with_candidates"] == 2
    assert first_manifest["llm_called"] is False
    assert (first / "candidate_values.jsonl").read_bytes() == (
        second / "candidate_values.jsonl"
    ).read_bytes()
    candidate_rows = [
        json.loads(line)
        for line in (first / "candidate_values.jsonl").read_text().splitlines()
    ]
    private_keys = {
        "label",
        "fold",
        "error_class",
        "error_subtype",
        "injection_seed",
        "expected_clean",
        "clean_value",
        "gold",
    }
    assert all(private_keys.isdisjoint(row) for row in candidate_rows)
    assert all(
        private_keys.isdisjoint(candidate)
        for row in candidate_rows
        for candidate in row["candidates"]
    )

    evaluation = evaluate_candidates(
        candidate_values=first / "candidate_values.jsonl",
        injection_log=paths["log"],
        predictions=paths["predictions"],
        graph_dir=paths["graph"],
        output_dir=tmp_path / "evaluation",
    )
    metrics = evaluation["metrics"]
    assert evaluation["evaluation_role"] == "frozen_audit"
    assert metrics["detector_confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 2, "tn": 1}
    assert metrics["rule_candidate_recall_at_1"] == 1.0
    assert metrics["rule_candidate_recall_at_3"] == 1.0
    assert metrics["rule_candidate_recall_at_5"] == 1.0
    assert metrics["joint_candidate_recall_at_5"] == pytest.approx(1 / 3)
    assert metrics["mean_reciprocal_rank"] == 1.0
    assert metrics["clean_preservation_rate"] == 0.5
    assert evaluation["selection_performed"] is False


def test_validation_and_internal_test_use_identical_frozen_audit_policy(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rules = _synthesize(paths, tmp_path)
    reports = {}
    for split in ("validation", "internal_test"):
        targets = tmp_path / f"{split}-targets"
        build_repair_targets(
            predictions=paths["predictions"],
            graph_dir=paths["graph"],
            split=split,
            output_dir=targets,
        )
        candidates = tmp_path / f"{split}-candidates"
        run_frozen_rules(
            targets=targets / "repair_targets.csv",
            rule_dir=rules,
            output_dir=candidates,
        )
        reports[split] = evaluate_candidates(
            candidate_values=candidates / "candidate_values.jsonl",
            injection_log=paths["log"],
            predictions=paths["predictions"],
            graph_dir=paths["graph"],
            output_dir=tmp_path / f"{split}-audit",
        )

    assert reports["validation"]["evaluation_role"] == "frozen_audit"
    assert reports["internal_test"]["evaluation_role"] == "frozen_audit"
    assert reports["validation"]["workflow"] == reports["internal_test"]["workflow"]
    assert set(reports["validation"]["metrics"]) == set(
        reports["internal_test"]["metrics"]
    )
    assert reports["validation"]["selection_performed"] is False
    assert reports["internal_test"]["selection_performed"] is False


def test_apply_supports_gzip_and_changes_only_declared_cell(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "table.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["a", "b"], lineterminator="\n")
        writer.writeheader()
        writer.writerows([{"a": "x", "b": "1"}, {"a": "y", "b": "2"}])
    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps({
        "action": "replace",
        "table": "table",
        "raw_row_index": 1,
        "column": "b",
        "current_value": "2",
        "replacement_value": "20",
    }) + "\n")
    output = tmp_path / "output"
    apply_repair_plan(raw_dir=raw, repair_plan=plan, output_dir=output)
    with gzip.open(output / "table.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "x", "b": "1"}, {"a": "y", "b": "20"}]
