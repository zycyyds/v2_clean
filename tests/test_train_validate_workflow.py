from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from main import parse_args
from workflow.agent_runners import (
    ExplorerRunContext,
    _engineer_attempt_root,
    run_engineer_round,
    run_explorer_analysis,
)
from workflow.orchestrator import (
    EngineerRoundResult,
    TrainValidateConfig,
    TrainValidateWorkflow,
    stage_result_package,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_success_audit(output: Path) -> Path:
    path = output / "skill_usage_report.json"
    _write_json(path, {"status": "SUCCESS", "unexplained_deviations": []})
    return path


def _write_result_workbook(output: Path, row: dict[str, object]) -> tuple[Path, dict[str, object]]:
    path = output / "final_dataset.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([{"case_id": str(row.get("record_id") or "val-a")}]).to_excel(
            writer, sheet_name="_cases", index=False,
        )
        pd.DataFrame(columns=["target_field_id", "skill"]).to_excel(
            writer, sheet_name="_provenance", index=False,
        )
        pd.DataFrame(columns=["target_field_id", "reason"]).to_excel(
            writer, sheet_name="_unsupported", index=False,
        )
        pd.DataFrame([row]).to_excel(writer, sheet_name="main", index=False)
    return path, {
        "alias": "main",
        "path": str(path),
        "sheet_name": "main",
        "case_id_column": "record_id",
    }


def _write_task_execution(output: Path, context, workbook: Path) -> Path:
    task_plan = json.loads(context.task_plan_path.read_text(encoding="utf-8"))
    path = output / "extraction_task_execution.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "status": "completed",
                    "rule_ids": task["rule_ids"],
                    "artifacts": [str(workbook)],
                    "reason": "",
                }
                for task in task_plan["tasks"]
                if task.get("required") is True
            ],
        },
    )
    return path


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    examples = tmp_path / "examples"
    case = examples / "train-a"
    raw = case / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame([{"record_id": "train-a", "answer": 1}]).to_csv(raw / "source.csv", index=False)
    _write_json(case / "gold.json", {"record_id": "train-a", "answer": 1})
    validation_raw = tmp_path / "validation_raw"
    validation_raw.mkdir()
    pd.DataFrame([{"record_id": "val-a", "answer": 2}]).to_csv(validation_raw / "source.csv", index=False)
    validation_gold = tmp_path / "validation_gold.csv"
    pd.DataFrame([{"record_id": "val-a", "answer": 2}]).to_csv(validation_gold, index=False)
    return examples, validation_raw, validation_gold


def test_engineer_retry_uses_an_isolated_attempt_directory(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    candidate = experiment / "candidates" / "round_0001"
    candidate.mkdir(parents=True)
    _write_json(
        experiment / "experiment_state.json",
        {
            "round_attempts": [
                {"round": 1, "status": "failed"},
                {"round": 1, "status": "failed"},
                {"round": 2, "status": "failed"},
            ],
        },
    )

    root = _engineer_attempt_root(candidate, 1)

    assert root == experiment / "agent_runs" / "round_0001" / "attempt_0003"


def test_stage_result_package_preserves_final_dataset_csv_name(tmp_path: Path) -> None:
    output = tmp_path / "engineer"
    output.mkdir()
    workbook, workbook_artifact = _write_result_workbook(
        output,
        {"record_id": "val-a", "answer": 2},
    )
    csv_path = output / "final_dataset.csv"
    pd.DataFrame([{"case_id": "val-a", "answer": 2}]).to_csv(csv_path, index=False)
    manifest = output / "result_manifest.json"
    _write_json(
        manifest,
        {
            "artifacts": [
                workbook_artifact,
                {
                    "alias": "case_level_csv",
                    "path": str(csv_path),
                    "case_id_column": "case_id",
                },
            ]
        },
    )
    mapping = output / "target_field_mapping.json"
    _write_json(mapping, {"mappings": []})

    staged = stage_result_package(
        EngineerRoundResult(manifest, mapping, None),
        tmp_path / "candidate",
    )
    staged_manifest = json.loads(staged.result_manifest_path.read_text(encoding="utf-8"))

    assert workbook.is_file()
    assert any(Path(item["path"]).name == "final_dataset.csv" for item in staged_manifest["artifacts"])
    assert (tmp_path / "candidate" / "results" / "artifacts" / "final_dataset.csv").is_file()


def test_explorer_runner_can_skip_model_agent_and_publish_canonical_handoffs(tmp_path: Path) -> None:
    examples, _validation_raw, _validation_gold = _build_fixture(tmp_path)
    result = asyncio.run(
        run_explorer_analysis(
            ExplorerRunContext(
                train_examples=examples,
                experiment_dir=tmp_path / "experiment_explorer_skip",
                task_text="generic extraction",
            ),
            max_iters=0,
        )
    )

    assert set(result) == {
        "field_extraction_rules",
        "extraction_task_plan",
        "data_analysis_report",
        "explorer_run_record",
    }
    assert all(Path(path).is_file() for path in result.values())
    record = json.loads(Path(result["explorer_run_record"]).read_text(encoding="utf-8"))
    assert record["completion"]["complete"] is True


def test_engineer_runner_can_skip_model_agent_and_use_deterministic_completion(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "deterministic_engineer"
    output.mkdir()
    manifest = output / "result_manifest.json"
    mapping = output / "target_field_mapping.json"
    audit = _write_success_audit(output)
    execution = output / "extraction_task_execution.json"
    _write_json(manifest, {"artifacts": []})
    _write_json(mapping, {"mappings": []})
    _write_json(execution, {"tasks": []})

    def forbidden_model_agent(*args, **kwargs):
        raise AssertionError("model engineer agent should not be created when max_iters=0")

    monkeypatch.setattr("workflow.agent_runners.create_engineer_agent", forbidden_model_agent)
    monkeypatch.setattr(
        "workflow.agent_runners.EngineerToolContext.from_task",
        lambda **kwargs: SimpleNamespace(variant_root=tmp_path / "variants"),
    )
    monkeypatch.setattr("workflow.agent_runners.create_engineer_toolkit", lambda *args, **kwargs: (None, []))
    monkeypatch.setattr("workflow.agent_runners.restore_bundle_variants", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "workflow.agent_runners.complete_engineer_result_package",
        lambda **kwargs: {
            "result_manifest": manifest,
            "target_field_mapping": mapping,
            "skill_usage_report": audit,
            "task_execution": execution,
        },
    )

    candidate = tmp_path / "experiment_engineer_skip" / "candidates" / "round_0001"
    active = tmp_path / "experiment_engineer_skip" / "active_bundle"
    candidate.mkdir(parents=True)
    active.mkdir(parents=True)
    for name in [
        "field_extraction_rules.json",
        "extraction_task_plan.json",
        "data_analysis_report.json",
        "explorer_run_record.json",
    ]:
        _write_json(tmp_path / name, {})

    result = asyncio.run(
        run_engineer_round(
            SimpleNamespace(
                round_index=1,
                validation_raw=tmp_path / "validation_raw",
                task_text="generic extraction",
                active_bundle=active,
                candidate_bundle=candidate,
                rules_path=tmp_path / "field_extraction_rules.json",
                task_plan_path=tmp_path / "extraction_task_plan.json",
                analysis_report_path=tmp_path / "data_analysis_report.json",
                explorer_run_record_path=tmp_path / "explorer_run_record.json",
                public_feedback_path=None,
            ),
            max_iters=0,
        )
    )

    assert result.result_manifest_path == manifest
    assert result.target_mapping_path == mapping


def test_train_validate_loop_hides_gold_and_freezes_best_bundle(tmp_path: Path) -> None:
    examples, validation_raw, validation_gold = _build_fixture(tmp_path)
    calls: list[int] = []

    async def fake_engineer(context) -> EngineerRoundResult:
        calls.append(context.round_index)
        assert not hasattr(context, "validation_gold")
        assert context.validation_raw == validation_raw.resolve()
        output = tmp_path / "engineer" / f"round_{context.round_index}"
        output.mkdir(parents=True)
        data, artifact = _write_result_workbook(output, {"record_id": "val-a", "answer": 2})
        manifest = output / "result_manifest.json"
        _write_json(manifest, {"artifacts": [artifact]})
        rules = json.loads(context.rules_path.read_text(encoding="utf-8"))
        answer_rule = next(rule for rule in rules["rules"] if rule["target_field_path"] == "answer")
        id_rule = next(rule for rule in rules["rules"] if rule["target_field_path"] == "record_id")
        mapping = output / "target_field_mapping.json"
        _write_json(
            mapping,
            {
                "mappings": [
                    {"target_field_id": id_rule["target_field_id"], "artifact": "main", "sheet_name": "main", "source_column": "record_id"},
                    {"target_field_id": answer_rule["target_field_id"], "artifact": "main", "sheet_name": "main", "source_column": "answer"},
                ]
            },
        )
        return EngineerRoundResult(
            manifest,
            mapping,
            None,
            skill_usage_report_path=_write_success_audit(output),
            task_execution_path=_write_task_execution(output, context, data),
        )

    workflow = TrainValidateWorkflow(
        TrainValidateConfig(
            train_examples=examples,
            validation_raw=validation_raw,
            validation_gold=validation_gold,
            experiment_dir=tmp_path / "experiment",
            task_text="generic extraction",
        ),
        engineer_runner=fake_engineer,
    )
    result = workflow.run_sync()

    assert calls == [1, 2, 3]
    assert result["status"] == "frozen"
    assert result["best_score"] == 1.0
    assert Path(result["frozen_bundle"]).is_dir()
    state = json.loads((tmp_path / "experiment" / "experiment_state.json").read_text())
    assert state["rounds"][-1]["consecutive_no_improvement"] == 2
    assert "validation_gold" not in json.dumps(state)

    resumed = workflow.run_sync()
    assert resumed["status"] == "frozen"
    assert calls == [1, 2, 3]


def test_feedback_enricher_failure_does_not_block_round_completion(tmp_path: Path) -> None:
    examples, validation_raw, validation_gold = _build_fixture(tmp_path)
    calls: list[int] = []

    async def fake_engineer(context) -> EngineerRoundResult:
        calls.append(context.round_index)
        output = tmp_path / "engineer_feedback" / f"round_{context.round_index}"
        output.mkdir(parents=True)
        data, artifact = _write_result_workbook(output, {"record_id": "val-a", "answer": 2})
        manifest = output / "result_manifest.json"
        _write_json(manifest, {"artifacts": [artifact]})
        rules = json.loads(context.rules_path.read_text(encoding="utf-8"))
        mappings = [
            {
                "target_field_id": rule["target_field_id"],
                "artifact": "main",
                "sheet_name": "main",
                "source_column": rule["target_field_path"],
            }
            for rule in rules["rules"]
            if rule["target_field_path"] in {"record_id", "answer"}
        ]
        mapping = output / "target_field_mapping.json"
        _write_json(mapping, {"mappings": mappings})
        return EngineerRoundResult(
            manifest,
            mapping,
            None,
            skill_usage_report_path=_write_success_audit(output),
            task_execution_path=_write_task_execution(output, context, data),
        )

    async def broken_feedback_enricher(public_feedback_path: Path, context) -> None:
        assert public_feedback_path.is_file()
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    result = TrainValidateWorkflow(
        TrainValidateConfig(
            train_examples=examples,
            validation_raw=validation_raw,
            validation_gold=validation_gold,
            experiment_dir=tmp_path / "experiment_feedback",
            task_text="generic extraction",
            round_limit=2,
        ),
        engineer_runner=fake_engineer,
        feedback_enricher=broken_feedback_enricher,
    ).run_sync()

    assert result["status"] == "paused"
    assert result["round_count"] == 2
    assert calls == [1, 2]
    state = json.loads((tmp_path / "experiment_feedback" / "experiment_state.json").read_text())
    assert [entry["status"] for entry in state["round_attempts"]] == ["completed", "completed"]
    feedback = json.loads(
        (tmp_path / "experiment_feedback" / "evaluations" / "round_0001" / "public_feedback.json").read_text()
    )
    assert feedback["feedback_enricher"]["status"] == "failed"


def test_cli_accepts_structured_train_validate_paths(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--workflow",
            "train-validate",
            "--train-examples",
            str(tmp_path / "examples"),
            "--validation-raw",
            str(tmp_path / "validation-raw"),
            "--validation-gold",
            str(tmp_path / "validation-gold"),
            "--experiment-dir",
            str(tmp_path / "experiment"),
            "--round-limit",
            "2",
            "task",
        ]
    )

    assert args.workflow == "train-validate"
    assert args.train_examples.endswith("examples")
    assert args.validation_raw.endswith("validation-raw")
    assert args.validation_gold.endswith("validation-gold")
    assert args.experiment_dir.endswith("experiment")
    assert args.round_limit == 2
    assert args.enable_feedback_agent is False
    assert args.enable_explorer_agent is False
    assert args.enable_engineer_agent is False

    enabled_args = parse_args(
        [
            "--workflow",
            "train-validate",
            "--enable-feedback-agent",
            "--enable-explorer-agent",
            "--enable-engineer-agent",
            "--train-examples",
            str(tmp_path / "examples"),
            "--validation-raw",
            str(tmp_path / "validation-raw"),
            "--validation-gold",
            str(tmp_path / "validation-gold"),
            "--experiment-dir",
            str(tmp_path / "experiment"),
            "task",
        ]
    )

    assert enabled_args.enable_feedback_agent is True
    assert enabled_args.enable_explorer_agent is True
    assert enabled_args.enable_engineer_agent is True


def test_validation_missing_rule_can_be_added_in_next_candidate(tmp_path: Path) -> None:
    examples, validation_raw, validation_gold = _build_fixture(tmp_path)
    pd.DataFrame([{"record_id": "val-a", "answer": 2, "validation_only": "found"}]).to_csv(
        validation_raw / "source.csv",
        index=False,
    )
    pd.DataFrame([{"record_id": "val-a", "answer": 2, "validation_only": "found"}]).to_csv(
        validation_gold,
        index=False,
    )
    calls: list[int] = []

    async def fake_engineer(context) -> EngineerRoundResult:
        calls.append(context.round_index)
        output = tmp_path / "engineer_missing" / f"round_{context.round_index}"
        output.mkdir(parents=True)
        include_new = context.round_index >= 2
        row = {"record_id": "val-a", "answer": 2}
        if include_new:
            row["validation_only"] = "found"
        data, artifact = _write_result_workbook(output, row)
        manifest = output / "result_manifest.json"
        _write_json(manifest, {"artifacts": [artifact]})
        rules = json.loads(context.rules_path.read_text())
        mappings = [
            {"target_field_id": rule["target_field_id"], "artifact": "main", "sheet_name": "main", "source_column": rule["target_field_path"]}
            for rule in rules["rules"]
            if rule["target_field_path"] in row
        ]
        updates_path = None
        if include_new:
            feedback = json.loads(context.public_feedback_path.read_text())
            missing = next(
                (item for item in feedback["field_feedback"] if "missing_rule" in item["error_types"]),
                None,
            )
            if missing is not None:
                updates_path = output / "field_rule_updates.json"
                _write_json(
                    updates_path,
                    {
                        "updates": [{
                            "target_field_id": missing["target_field_id"],
                            "target_field_path": missing["target_field_path"],
                            "target_type": "string",
                            "cardinality": "one",
                            "record_grain": "record_id",
                            "source_files": ["source.csv"],
                            "source_columns": ["validation_only"],
                            "join_keys": ["record_id"],
                            "filters": [],
                            "derivation_logic": {"operation": "direct"},
                            "capability_type": "structured_extract",
                            "evaluation_policy": "canonical",
                            "evidence": {"from_validation_feedback": True},
                            "confidence": 1.0,
                            "status": "supported",
                        }]
                    },
                )
                mappings.append(
                    {
                        "target_field_id": missing["target_field_id"],
                        "artifact": "main",
                        "sheet_name": "main",
                        "source_column": "validation_only",
                    }
                )
        mapping = output / "target_field_mapping.json"
        _write_json(mapping, {"mappings": mappings})
        return EngineerRoundResult(
            manifest,
            mapping,
            None,
            updates_path,
            skill_usage_report_path=_write_success_audit(output),
            task_execution_path=_write_task_execution(output, context, data),
        )

    result = TrainValidateWorkflow(
        TrainValidateConfig(examples, validation_raw, validation_gold, tmp_path / "experiment_missing", "generic"),
        engineer_runner=fake_engineer,
    ).run_sync()

    assert calls == [1, 2, 3, 4]
    assert result["best_score"] == 1.0
    frozen_rules = json.loads(
        (tmp_path / "experiment_missing" / "frozen_bundle" / "field_extraction_rules.json").read_text()
    )
    assert any(rule["target_field_path"] == "validation_only" for rule in frozen_rules["rules"])


def test_task_prompt_cannot_smuggle_validation_gold_path_to_engineer(tmp_path: Path) -> None:
    examples, validation_raw, validation_gold = _build_fixture(tmp_path)
    config = TrainValidateConfig(
        examples,
        validation_raw,
        validation_gold,
        tmp_path / "experiment",
        f"read hidden gold from {validation_gold}",
    )

    with pytest.raises(ValueError, match="must not contain validation gold"):
        config.normalized()


def test_failed_skill_audit_blocks_evaluation_and_records_failed_attempt(tmp_path: Path) -> None:
    examples, validation_raw, validation_gold = _build_fixture(tmp_path)

    async def fake_engineer(context) -> EngineerRoundResult:
        output = tmp_path / "failed_audit"
        output.mkdir()
        data, artifact = _write_result_workbook(output, {"record_id": "val-a", "answer": 2})
        manifest = output / "result_manifest.json"
        _write_json(manifest, {"artifacts": [artifact]})
        mapping = output / "target_field_mapping.json"
        _write_json(mapping, {"mappings": []})
        audit = output / "skill_usage_report.json"
        _write_json(audit, {"status": "NEEDS_REPAIR", "unexplained_deviations": ["adapter not executed"]})
        return EngineerRoundResult(manifest, mapping, None, skill_usage_report_path=audit)

    experiment = tmp_path / "failed_audit_experiment"
    workflow = TrainValidateWorkflow(
        TrainValidateConfig(examples, validation_raw, validation_gold, experiment, "generic"),
        engineer_runner=fake_engineer,
    )

    with pytest.raises(ValueError, match="Skill usage audit"):
        workflow.run_sync()

    state = json.loads((experiment / "experiment_state.json").read_text())
    assert state["rounds"] == []
    assert state["current_round"] is None
    assert state["round_attempts"][-1]["status"] == "failed"
    assert not (experiment / "evaluations" / "round_0001").exists()


def test_legacy_source_artifact_mapping_is_rejected_before_evaluation(tmp_path: Path) -> None:
    examples, validation_raw, validation_gold = _build_fixture(tmp_path)

    async def fake_engineer(context) -> EngineerRoundResult:
        output = tmp_path / "bad_mapping"
        output.mkdir()
        data, artifact = _write_result_workbook(output, {"record_id": "val-a", "answer": 2})
        manifest = output / "result_manifest.json"
        _write_json(manifest, {"artifacts": [artifact]})
        mapping = output / "target_field_mapping.json"
        _write_json(mapping, {"mappings": [{"target_field_id": "x", "source_artifact": "main", "source_column": "answer"}]})
        return EngineerRoundResult(manifest, mapping, None, skill_usage_report_path=_write_success_audit(output))

    workflow = TrainValidateWorkflow(
        TrainValidateConfig(examples, validation_raw, validation_gold, tmp_path / "bad_mapping_experiment", "generic"),
        engineer_runner=fake_engineer,
    )

    with pytest.raises(ValueError, match="artifact"):
        workflow.run_sync()
