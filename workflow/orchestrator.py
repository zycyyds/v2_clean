from __future__ import annotations

import asyncio
import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

from workflow.bundle import ExperimentBundleManager
from workflow.evaluator import SemanticScorer, evaluate_result_package
from workflow.gold_analysis import analyze_training_examples
from workflow.skill_adapter import persist_validated_variants
from workflow.task_compiler import compile_extraction_task_plan
from workflow.rule_updates import apply_candidate_rule_updates


@dataclass(frozen=True)
class TrainValidateConfig:
    train_examples: str | Path
    validation_raw: str | Path
    validation_gold: str | Path
    experiment_dir: str | Path
    task_text: str
    round_limit: int | None = None

    def normalized(self) -> "TrainValidateConfig":
        values = {
            "train_examples": Path(self.train_examples).expanduser().resolve(),
            "validation_raw": Path(self.validation_raw).expanduser().resolve(),
            "validation_gold": Path(self.validation_gold).expanduser().resolve(),
            "experiment_dir": Path(self.experiment_dir).expanduser().resolve(),
        }
        for name in ("train_examples", "validation_raw", "validation_gold"):
            if not values[name].exists():
                raise ValueError(f"{name} does not exist: {values[name]}")
        if str(values["validation_gold"]) in str(self.task_text or ""):
            raise ValueError("Task prompt must not contain validation gold path; pass it only via --validation-gold")
        if self.round_limit is not None and self.round_limit < 1:
            raise ValueError("round_limit must be positive when provided")
        return TrainValidateConfig(**values, task_text=self.task_text, round_limit=self.round_limit)


@dataclass(frozen=True)
class EngineerRoundContext:
    round_index: int
    validation_raw: Path
    task_text: str
    active_bundle: Path
    candidate_bundle: Path
    rules_path: Path
    task_plan_path: Path
    analysis_report_path: Path
    explorer_run_record_path: Path
    public_feedback_path: Path | None


@dataclass(frozen=True)
class EngineerRoundResult:
    result_manifest_path: Path
    target_mapping_path: Path
    variant_root: Path | None
    field_rule_updates_path: Path | None = None
    skill_usage_report_path: Path | None = None
    task_execution_path: Path | None = None


EngineerRunner = Callable[[EngineerRoundContext], EngineerRoundResult | Awaitable[EngineerRoundResult]]
FeedbackEnricher = Callable[[Path, EngineerRoundContext], Any | Awaitable[Any]]


@dataclass(frozen=True)
class ExplorerRunContext:
    train_examples: Path
    experiment_dir: Path
    task_text: str


ExplorerRunner = Callable[[ExplorerRunContext], dict[str, str] | Awaitable[dict[str, str]]]


class TrainValidateWorkflow:
    def __init__(
        self,
        config: TrainValidateConfig,
        *,
        engineer_runner: EngineerRunner,
        feedback_enricher: FeedbackEnricher | None = None,
        explorer_runner: ExplorerRunner | None = None,
        semantic_scorer: SemanticScorer | None = None,
    ) -> None:
        self.config = config.normalized()
        self.manager = ExperimentBundleManager(self.config.experiment_dir)
        self.engineer_runner = engineer_runner
        self.feedback_enricher = feedback_enricher
        self.explorer_runner = explorer_runner
        self.semantic_scorer = semantic_scorer

    def run_sync(self) -> dict[str, Any]:
        return asyncio.run(self.run())

    async def run(self) -> dict[str, Any]:
        if self.manager.state_path.exists():
            state = self.manager.load_state()
            if state.get("status") == "frozen":
                return self._final_result(state)
        else:
            if self.explorer_runner is None:
                artifacts = analyze_training_examples(self.config.train_examples, self.manager.explorer_dir)
            else:
                artifacts = await _maybe_await(
                    self.explorer_runner(
                        ExplorerRunContext(
                            train_examples=Path(self.config.train_examples),
                            experiment_dir=Path(self.config.experiment_dir),
                            task_text=self.config.task_text,
                        )
                    )
                )
            self.manager.initialize(artifacts)

        completed_this_call = 0
        while True:
            state = self.manager.load_state()
            round_index = len(state.get("rounds", [])) + 1
            if self.config.round_limit is not None and completed_this_call >= self.config.round_limit:
                return self._paused_result(state)
            candidate = self.manager.begin_round(round_index)
            previous_feedback = self._previous_public_feedback(round_index)
            context = EngineerRoundContext(
                round_index=round_index,
                validation_raw=Path(self.config.validation_raw),
                task_text=self.config.task_text,
                active_bundle=self.manager.active_dir,
                candidate_bundle=candidate,
                rules_path=candidate / "field_extraction_rules.json",
                task_plan_path=candidate / "extraction_task_plan.json",
                analysis_report_path=self.manager.explorer_dir / "data_analysis_report.json",
                explorer_run_record_path=self.manager.explorer_dir / "explorer_run_record.json",
                public_feedback_path=previous_feedback,
            )
            try:
                result = await _maybe_await(self.engineer_runner(context))
                _validate_skill_usage_report(result.skill_usage_report_path)
                staged = stage_result_package(result, candidate)
                _validate_extraction_task_execution(
                    context.task_plan_path,
                    staged.task_execution_path,
                    staged.result_manifest_path,
                    staged.target_mapping_path,
                )
                if result.field_rule_updates_path is not None:
                    apply_candidate_rule_updates(
                        context.rules_path,
                        result.field_rule_updates_path,
                        context.public_feedback_path,
                    )
                    _refresh_candidate_task_plan(context.rules_path, context.task_plan_path)
                    history_dir = candidate / "rule_update_history"
                    history_dir.mkdir(exist_ok=True)
                    shutil.copy2(
                        result.field_rule_updates_path,
                        history_dir / f"round_{round_index:04d}.json",
                    )
                if result.variant_root is not None:
                    persist_validated_variants(result.variant_root, candidate)
                evaluation_dir = self.manager.evaluations_dir / f"round_{round_index:04d}"
                evaluation = evaluate_result_package(
                    gold_path=self.config.validation_gold,
                    rules_path=context.rules_path,
                    result_manifest_path=staged.result_manifest_path,
                    target_mapping_path=staged.target_mapping_path,
                    output_dir=evaluation_dir,
                    semantic_scorer=self.semantic_scorer,
                    source_data_path=self.config.validation_raw,
                )
                public_feedback = Path(evaluation["public_feedback"])
                if self.feedback_enricher is not None:
                    try:
                        await _maybe_await(self.feedback_enricher(public_feedback, context))
                    except Exception as exc:
                        _record_feedback_enricher_failure(public_feedback, exc)
                    _assert_public_feedback_private(public_feedback)
                report = json.loads(Path(evaluation["evaluation_report"]).read_text(encoding="utf-8"))
                evaluation_valid = bool(
                    report.get("status") == "SUCCESS"
                    and (report.get("artifact_validation") or {}).get("valid") is True
                )
                round_record = self.manager.finish_round(
                    round_index,
                    score=float(report["metrics"]["composite_score"]),
                    candidate_dir=candidate,
                    metadata={
                        "evaluation_report": str(Path(evaluation["evaluation_report"])),
                        "public_feedback": str(public_feedback),
                        "result_manifest": str(staged.result_manifest_path),
                        "skill_usage_report": str(staged.skill_usage_report_path or ""),
                    },
                    eligible_for_promotion=evaluation_valid,
                )
            except BaseException as exc:
                self.manager.fail_round(round_index, candidate, str(exc))
                raise
            completed_this_call += 1
            if round_record["stop"]:
                state = self.manager.load_state()
                if int(state.get("best_round", 0)) == 0:
                    return {
                        **self._paused_result(state),
                        "status": "NEEDS_REPAIR",
                        "reason": "No evaluator-valid candidate has been produced.",
                    }
                frozen = self.manager.freeze()
                state = self.manager.load_state()
                return {**self._final_result(state), "frozen_bundle": str(frozen)}

    def _previous_public_feedback(self, round_index: int) -> Path | None:
        if round_index <= 1:
            return None
        path = self.manager.evaluations_dir / f"round_{round_index - 1:04d}" / "public_feedback.json"
        return path if path.is_file() else None

    def _final_result(self, state: dict[str, Any]) -> dict[str, Any]:
        frozen = self.manager.experiment_dir / "frozen_bundle"
        return {
            "status": str(state.get("status") or "frozen"),
            "experiment_dir": str(self.manager.experiment_dir),
            "best_score": state.get("best_score", 0.0),
            "best_round": state.get("best_round", 0),
            "round_count": len(state.get("rounds", [])),
            "frozen_bundle": str(frozen) if frozen.is_dir() else "",
        }

    def _paused_result(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "paused",
            "experiment_dir": str(self.manager.experiment_dir),
            "best_score": state.get("best_score", 0.0),
            "best_round": state.get("best_round", 0),
            "round_count": len(state.get("rounds", [])),
        }


def stage_result_package(result: EngineerRoundResult, candidate_bundle: str | Path) -> EngineerRoundResult:
    manifest_path = Path(result.result_manifest_path).expanduser().resolve()
    mapping_path = Path(result.target_mapping_path).expanduser().resolve()
    if not manifest_path.is_file() or not mapping_path.is_file():
        raise ValueError("Engineer must produce result_manifest.json and target_field_mapping.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), list):
        raise ValueError("result_manifest.json must contain an artifacts list")
    workbook_sources: list[Path] = []
    for artifact in manifest["artifacts"]:
        if not isinstance(artifact, dict) or not artifact.get("path"):
            continue
        source = Path(str(artifact["path"])).expanduser()
        source = (manifest_path.parent / source).resolve() if not source.is_absolute() else source.resolve()
        if source.name == "final_dataset.xlsx":
            workbook_sources.append(source)
    if not workbook_sources:
        raise ValueError("Engineer result package must include final_dataset.xlsx")
    for workbook in set(workbook_sources):
        if not workbook.is_file():
            raise ValueError(f"Final workbook does not exist: {workbook}")
        sheets = set(pd.ExcelFile(workbook).sheet_names)
        missing_sheets = {"_cases", "_provenance", "_unsupported"} - sheets
        if missing_sheets:
            raise ValueError(f"final_dataset.xlsx is missing control sheets: {sorted(missing_sheets)}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not isinstance(mapping.get("mappings"), list):
        raise ValueError("target_field_mapping.json must contain a mappings list")
    aliases = {
        str(item.get("alias") or "")
        for item in manifest["artifacts"]
        if isinstance(item, dict)
    }
    for index, item in enumerate(mapping["mappings"]):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid target field mapping at index {index}")
        if "source_artifact" in item:
            raise ValueError("target field mapping must use artifact, not source_artifact")
        missing = [key for key in ("target_field_id", "artifact", "source_column") if not item.get(key)]
        if missing:
            raise ValueError(f"Target field mapping at index {index} is missing: {', '.join(missing)}")
        if str(item["artifact"]) not in aliases:
            raise ValueError(f"Target field mapping references unknown artifact: {item['artifact']}")
    candidate = Path(candidate_bundle).expanduser().resolve()
    results_dir = candidate / "results"
    temporary = candidate / ".results.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "artifacts").mkdir(parents=True)
    staged_artifacts = []
    staged_sources: dict[Path, Path] = {}
    for index, artifact in enumerate(manifest["artifacts"]):
        if not isinstance(artifact, dict) or not artifact.get("path"):
            raise ValueError(f"Invalid result artifact at index {index}")
        source = Path(str(artifact["path"])).expanduser()
        if not source.is_absolute():
            source = (manifest_path.parent / source).resolve()
        else:
            source = source.resolve()
        if not source.is_file():
            raise ValueError(f"Result artifact does not exist: {source}")
        alias = str(artifact.get("alias") or f"artifact_{index}")
        destination = staged_sources.get(source)
        if destination is None:
            preferred = (
                source.name
                if source.name in {"final_dataset.csv", "final_dataset.xlsx"}
                else f"{alias}{source.suffix.lower()}"
            )
            destination = temporary / "artifacts" / preferred
            if destination.exists():
                destination = temporary / "artifacts" / f"{alias}_{index}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            staged_sources[source] = destination
        staged_artifacts.append({**artifact, "path": str((results_dir / "artifacts" / destination.name).resolve())})
    staged_manifest = {**manifest, "artifacts": staged_artifacts}
    staged_manifest_path = temporary / "result_manifest.json"
    staged_mapping_path = temporary / "target_field_mapping.json"
    staged_audit_path = temporary / "skill_usage_report.json"
    staged_task_execution_path = temporary / "extraction_task_execution.json"
    _write_json(staged_manifest_path, staged_manifest)
    shutil.copy2(mapping_path, staged_mapping_path)
    if result.skill_usage_report_path is not None:
        shutil.copy2(Path(result.skill_usage_report_path).expanduser().resolve(), staged_audit_path)
    if result.task_execution_path is not None:
        shutil.copy2(Path(result.task_execution_path).expanduser().resolve(), staged_task_execution_path)
    if results_dir.exists():
        shutil.rmtree(results_dir)
    temporary.rename(results_dir)
    return EngineerRoundResult(
        result_manifest_path=results_dir / "result_manifest.json",
        target_mapping_path=results_dir / "target_field_mapping.json",
        variant_root=result.variant_root,
        field_rule_updates_path=result.field_rule_updates_path,
        skill_usage_report_path=results_dir / "skill_usage_report.json",
        task_execution_path=results_dir / "extraction_task_execution.json",
    )


def _validate_skill_usage_report(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("Engineer must produce skill_usage_report.json")
    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file():
        raise ValueError(f"Skill usage audit report does not exist: {report_path}")
    value = json.loads(report_path.read_text(encoding="utf-8"))
    issues = value.get("unexplained_deviations") if isinstance(value, dict) else None
    if not isinstance(value, dict) or value.get("status") != "SUCCESS" or issues not in ([], None):
        raise ValueError(f"Skill usage audit failed: {issues or value}")
    return value


def _validate_extraction_task_execution(
    task_plan_path: str | Path,
    execution_path: str | Path | None,
    result_manifest_path: str | Path,
    target_mapping_path: str | Path,
) -> None:
    if execution_path is None or not Path(execution_path).is_file():
        raise ValueError("Engineer must produce extraction_task_execution.json")
    plan = json.loads(Path(task_plan_path).read_text(encoding="utf-8"))
    execution = json.loads(Path(execution_path).read_text(encoding="utf-8"))
    required = {
        str(item.get("task_id")): item
        for item in plan.get("tasks", [])
        if isinstance(item, dict) and item.get("required") is True
    }
    entries = execution.get("tasks") if isinstance(execution, dict) else None
    if not isinstance(entries, list):
        raise ValueError("extraction_task_execution.json must contain a tasks list")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("task_id"):
            raise ValueError("Invalid extraction task execution entry")
        task_id = str(entry["task_id"])
        if task_id in by_id:
            raise ValueError(f"Duplicate extraction task execution: {task_id}")
        by_id[task_id] = entry
    missing_tasks = sorted(set(required) - set(by_id))
    if missing_tasks:
        raise ValueError(f"Required extraction tasks were not executed: {missing_tasks}")

    mapping = json.loads(Path(target_mapping_path).read_text(encoding="utf-8"))
    mapped_ids = {
        str(item.get("target_field_id"))
        for item in mapping.get("mappings", [])
        if isinstance(item, dict) and item.get("target_field_id")
    }
    manifest = json.loads(Path(result_manifest_path).read_text(encoding="utf-8"))
    workbook = next(
        (
            Path(str(item["path"]))
            for item in manifest.get("artifacts", [])
            if isinstance(item, dict) and Path(str(item.get("path") or "")).name == "final_dataset.xlsx"
        ),
        None,
    )
    unsupported_ids: set[str] = set()
    if workbook is not None and workbook.is_file():
        unsupported_frame = pd.read_excel(workbook, sheet_name="_unsupported", dtype=object)
        if "target_field_id" in unsupported_frame.columns:
            unsupported_ids = {
                str(value) for value in unsupported_frame["target_field_id"].dropna().tolist()
            }

    for task_id, task in required.items():
        entry = by_id[task_id]
        status = entry.get("status")
        if status not in {"completed", "unsupported"}:
            raise ValueError(f"Extraction task {task_id} has unresolved status: {status}")
        planned_rules = set(task.get("rule_ids") or [])
        if set(entry.get("rule_ids") or []) != planned_rules:
            raise ValueError(f"Extraction task {task_id} rule coverage does not match the plan")
        uncovered = planned_rules - mapped_ids - unsupported_ids
        if uncovered:
            raise ValueError(f"Extraction task {task_id} silently skipped fields: {sorted(uncovered)}")
        artifacts = entry.get("artifacts") or []
        if status == "completed" and (
            not artifacts or any(not Path(str(path)).expanduser().resolve().is_file() for path in artifacts)
        ):
            raise ValueError(f"Completed extraction task {task_id} has no real artifacts")
        if status == "unsupported" and not str(entry.get("reason") or "").strip():
            raise ValueError(f"Unsupported extraction task {task_id} requires a reason")


def _refresh_candidate_task_plan(rules_path: Path, task_plan_path: Path) -> None:
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    rules = payload.get("rules") if isinstance(payload.get("rules"), list) else []
    plan = compile_extraction_task_plan(
        rules,
        record_grain=str(payload.get("record_grain") or "case"),
    )
    plan["target_categories"] = list(payload.get("target_categories") or [])
    _write_json(task_plan_path, plan)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _assert_public_feedback_private(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    forbidden = {"gold_value", "pred_value", "prediction_value"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            leaked = forbidden & set(value)
            if leaked:
                raise ValueError(f"Public feedback contains private keys: {sorted(leaked)}")
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)


def _record_feedback_enricher_failure(path: Path, exc: Exception) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["feedback_enricher"] = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "fallback": "deterministic_public_feedback",
    }
    _write_json(path, payload)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
