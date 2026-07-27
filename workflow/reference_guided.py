from __future__ import annotations

import asyncio
import csv
import gzip
import hashlib
import json
import os
import re
import signal
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

import pandas as pd

from agent.reference_runtime import (
    DATA_CLEANING_AGENT_PIPELINE_SKILLS,
    close_reference_agent,
    create_reference_agent,
    create_reference_toolkit,
    reference_code_denied_read_roots,
    reference_code_read_roots,
    stream_reference_agent_reply,
)
from agent_tools.context import EngineerToolContext
from agent_tools.codegraph_mcp import (
    CodeGraphRuntime,
    codegraph_manifest_identity,
    open_codegraph_runtime,
    probe_codegraph_connection,
    summarize_codegraph_usage,
)
from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context
from lib.agent_runtime import AllApiKeysRateLimitedError, format_message_content, make_user_msg
from workflow.reference_splits import infer_dataset_profile
from workflow.reference_evaluation import (
    IGNORED_PACKAGE_FILES,
    SCORER_VERSION,
    evaluate_reference_directory_package as evaluate_balanced_reference_directory_package,
)
from workflow.reference_package_gate import (
    MIMIC_EXPECTED_BUSINESS_FILE_COUNT,
    validate_business_result_package,
)
from workflow.reference_pipeline import (
    PIPELINE_MODULES,
    pipeline_directory_sha256,
    pipeline_modules_for_business_targets,
    validate_and_replay_candidate_pipeline,
    validate_pipeline_structure,
)
from workflow.skill_adapter import persist_validated_variants, restore_bundle_variants


KEY_PRIORITY = ("hadm_id", "stay_id", "subject_id", "case_id", "row_id")
LABEL_COLUMNS = {"label", "mortality", "readmission", "los_label", "outcome"}
NUMERIC_NORMALIZATION_PLACES = Decimal("0.000001")
PUBLIC_GATE_MAX_REPAIRS = 2


@dataclass(frozen=True)
class ReferenceGuidedConfig:
    dataset_split: str | Path
    experiment_dir: str | Path
    task_text: str
    round_limit: int | None = None
    patience: int = 2
    max_attempts: int = 100
    target_score: float | None = None
    max_iters: int = 60
    pipeline_skills_enabled: bool = True
    codegraph_enabled: bool = False

    def normalized(self) -> "ReferenceGuidedConfig":
        split = Path(self.dataset_split).expanduser().resolve()
        if not split.exists():
            raise ValueError(f"dataset_split does not exist: {split}")
        required = [split / name for name in ("train", "validation", "test")]
        missing = [str(path) for path in required if not path.is_dir()]
        if missing:
            raise ValueError("dataset_split must contain train/, validation/, and test/: " + ", ".join(missing))
        if self.round_limit is not None and self.round_limit < 1:
            raise ValueError("round_limit must be positive when provided")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.target_score is not None and not 0.0 <= self.target_score <= 1.0:
            raise ValueError("target_score must be between 0 and 1")
        if self.max_iters < 1:
            raise ValueError("max_iters must be positive")
        return ReferenceGuidedConfig(
            dataset_split=split,
            experiment_dir=Path(self.experiment_dir).expanduser().resolve(),
            task_text=self.task_text,
            round_limit=self.round_limit,
            patience=self.patience,
            max_attempts=self.max_attempts,
            target_score=self.target_score,
            max_iters=self.max_iters,
            pipeline_skills_enabled=self.pipeline_skills_enabled,
            codegraph_enabled=bool(self.codegraph_enabled),
        )


def _migrate_reference_experiment_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return the v2 attempt/promotion state without discarding v1 history."""
    migrated = json.loads(json.dumps(state))
    if int(migrated.get("schema_version") or 1) >= 2:
        migrated.setdefault("attempts", [])
        migrated.setdefault("rounds", [])
        migrated.setdefault("checkpoints", [])
        migrated.setdefault("sessions", [])
        migrated.setdefault("abandoned_checkpoints", [])
        migrated.setdefault("best_attempt", 0)
        migrated.setdefault("best_feedback", "")
        migrated.setdefault("consecutive_valid_no_improvement", 0)
        migrated.setdefault("invalid_attempt_count", 0)
        return migrated

    legacy_rounds = [item for item in migrated.get("rounds") or [] if isinstance(item, dict)]
    attempts: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    best_attempt = 0
    best_feedback = ""
    legacy_best_round = int(migrated.get("best_round") or 0)
    invalid_attempt_count = 0
    for index, legacy in enumerate(legacy_rounds, start=1):
        attempt_index = int(legacy.get("round") or index)
        candidate_dir = str(legacy.get("candidate_dir") or "")
        delta_path = Path(candidate_dir) / "candidate_delta.json" if candidate_dir else None
        legacy_valid = True
        gate: dict[str, Any] = {"valid": True, "issues": []}
        if delta_path is not None and delta_path.is_file():
            delta = _load_json(delta_path)
            if str(delta.get("status") or "SUCCESS").upper() != "SUCCESS":
                legacy_valid = False
                invalid_attempt_count += 1
                gate = {
                    "valid": False,
                    "issues": ["legacy candidate delta did not pass"],
                    "candidate_delta": str(delta_path),
                }
        attempt = {
            **legacy,
            "attempt": attempt_index,
            "legacy_round": attempt_index,
            "status": "evaluated" if legacy_valid else "invalid",
            "valid": legacy_valid,
            "gate": gate,
        }
        attempts.append(attempt)
        if legacy.get("improved"):
            promotion = {
                **legacy,
                "round": len(promotions) + 1,
                "attempt": attempt_index,
                "legacy_round": attempt_index,
            }
            promotions.append(promotion)
        if attempt_index == legacy_best_round:
            best_attempt = attempt_index
            best_feedback = str((legacy.get("evaluation") or {}).get("public_feedback") or "")

    migrated.update(
        {
            "schema_version": 2,
            "attempts": attempts,
            "rounds": promotions,
            "best_round": next(
                (int(item["round"]) for item in promotions if int(item.get("attempt") or 0) == best_attempt),
                len(promotions) if promotions else 0,
            ),
            "best_attempt": best_attempt,
            "best_feedback": best_feedback,
            "consecutive_valid_no_improvement": _trailing_valid_non_improvements(attempts),
            "invalid_attempt_count": invalid_attempt_count,
            "checkpoints": [],
            "sessions": [],
            "abandoned_checkpoints": [],
        }
    )
    migrated.pop("consecutive_no_improvement", None)
    return migrated


def _trailing_valid_non_improvements(attempts: list[dict[str, Any]]) -> int:
    count = 0
    for attempt in reversed(attempts):
        if attempt.get("improved"):
            break
        if attempt.get("valid", True):
            count += 1
    return count


def _finish_reference_attempt_state(
    state: dict[str, Any],
    *,
    attempt_index: int,
    score: float | None,
    candidate_dir: str,
    result: dict[str, Any],
    evaluation: dict[str, str],
    gate: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Record an attempt and create a formal round only for a promotion."""
    updated = _migrate_reference_experiment_state(state)
    gate_payload = gate or {"valid": True, "issues": []}
    valid = bool(gate_payload.get("valid", True)) and score is not None
    improved = valid and float(score) > float(updated.get("best_score") or 0.0)
    attempt = {
        "attempt": attempt_index,
        "status": "evaluated" if valid else "invalid",
        "valid": valid,
        "score": round(float(score), 6) if score is not None else None,
        "improved": improved,
        "candidate_dir": candidate_dir,
        "producer": result.get("producer", ""),
        "result": {key: value for key, value in result.items() if key != "producer"},
        "evaluation": evaluation,
        "gate": gate_payload,
    }
    updated.setdefault("attempts", []).append(attempt)
    if not valid:
        updated["invalid_attempt_count"] = int(updated.get("invalid_attempt_count") or 0) + 1
    elif improved:
        formal_round = len(updated.setdefault("rounds", [])) + 1
        updated["rounds"].append(
            {
                **attempt,
                "round": formal_round,
            }
        )
        updated["best_score"] = round(float(score), 6)
        updated["best_round"] = formal_round
        updated["best_attempt"] = attempt_index
        updated["best_feedback"] = str(evaluation.get("public_feedback") or "")
        updated["consecutive_valid_no_improvement"] = 0
    else:
        updated["consecutive_valid_no_improvement"] = int(
            updated.get("consecutive_valid_no_improvement") or 0
        ) + 1
    updated["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return updated, improved


def _reference_stop_reason(
    state: dict[str, Any],
    *,
    promotions_this_run: int,
    attempts_this_run: int,
    round_limit: int | None,
    patience: int,
    max_attempts: int,
    target_score: float | None,
) -> str:
    if target_score is not None and float(state.get("best_score") or 0.0) >= target_score:
        return "target_score"
    if round_limit is not None and promotions_this_run >= round_limit:
        return "round_limit"
    if int(state.get("consecutive_valid_no_improvement") or 0) >= patience:
        return "patience"
    if attempts_this_run >= max_attempts:
        return "max_attempts"
    return ""


def _rate_limit_block_state(
    state: dict[str, Any],
    error: AllApiKeysRateLimitedError,
) -> dict[str, Any]:
    """Pause a run without converting an infrastructure error into an attempt."""
    updated = json.loads(json.dumps(state))
    updated["status"] = "blocked_rate_limited"
    updated["termination_reason"] = "all_api_keys_rate_limited"
    updated["rate_limit_error"] = str(error)
    updated["blocked_at"] = datetime.now().isoformat(timespec="seconds")
    return updated


def _reference_candidate_quality_gate(
    *,
    active: Path,
    candidate: Path,
    contract: dict[str, Any],
    repair_targets: Path | None,
) -> dict[str, Any]:
    """Apply deterministic checks before spending a hidden evaluation."""
    issues: list[str] = []
    active_package = active / "results" / "result_package"
    candidate_package = candidate / "results" / "result_package"
    active_hashes = _package_relative_hashes(active_package)
    candidate_hashes = _package_relative_hashes(candidate_package)
    changed_business = sorted(
        relative for relative in set(active_hashes) | set(candidate_hashes)
        if active_hashes.get(relative) != candidate_hashes.get(relative)
    )
    if not candidate_package.is_dir() or not candidate_hashes:
        issues.append("result_package is missing or contains no structured business file")
    if not changed_business:
        issues.append("no changed business file")

    allowed_targets: set[str] = set()
    has_generic_target = False
    if repair_targets is not None and repair_targets.is_file():
        payload = _load_json(repair_targets)
        target_items = [item for item in payload.get("targets") or [] if isinstance(item, dict)]
        has_generic_target = any(
            not str(item.get("relative_path") or "").strip()
            for item in target_items
        )
        allowed_targets = {
            str(item.get("relative_path") or "").strip().lstrip("/")
            for item in target_items
            if isinstance(item, dict) and str(item.get("relative_path") or "").strip()
        }
        if not has_generic_target:
            unrelated = sorted(set(changed_business) - allowed_targets)
            if unrelated:
                issues.append("unrelated business files changed: " + ", ".join(unrelated[:20]))
            if changed_business and not (set(changed_business) & allowed_targets):
                issues.append("no feedback-linked business file changed")
    allowed_changed_modules = (
        pipeline_modules_for_business_targets(allowed_targets)
        if repair_targets is not None and repair_targets.is_file() and not has_generic_target
        else None
    )

    delta_path = candidate / "candidate_delta.json"
    if delta_path.is_file():
        delta = _load_json(delta_path)
        if str(delta.get("status") or "SUCCESS").upper() != "SUCCESS":
            issues.append("candidate delta did not pass")
        if repair_targets is not None and not delta.get("has_feedback_linked_material_delta"):
            issues.append("candidate delta has no feedback-linked material change")

    train_reference_root_text = str((contract.get("paths") or {}).get("train_reference_root") or "")
    train_reference_root = Path(train_reference_root_text).expanduser() if train_reference_root_text else None
    if train_reference_root is not None and train_reference_root.is_dir():
        expected_files = {
            path.relative_to(train_reference_root).as_posix()
            for path in _structured_package_files(train_reference_root)
            if path.name != "reference.csv"
        }
        missing_files = sorted(expected_files - set(candidate_hashes))
        if missing_files:
            issues.append("result_package is incomplete: " + ", ".join(missing_files[:20]))

    paths = contract.get("paths") or {}
    key_column = str(contract.get("key_column") or contract.get("record_grain") or "")
    required_host_paths = {
        name: Path(str(paths.get(name))).expanduser().resolve()
        for name in (
            "train_raw",
            "train_reference_root",
            "train_keys",
            "validation_raw",
            "validation_keys",
        )
        if str(paths.get(name) or "").strip()
    }
    reference_directory_contract = contract.get("reference_type") == "reference_directory"
    host_path_issues: list[str] = []
    if reference_directory_contract:
        if not key_column:
            host_path_issues.append("reference-directory contract has no key_column")
        for name in ("train_raw", "train_reference_root", "validation_raw"):
            path = required_host_paths.get(name)
            if path is None or not path.is_dir():
                host_path_issues.append(f"required host directory is missing: {name}")
        for name in ("train_keys", "validation_keys"):
            path = required_host_paths.get(name)
            if path is None or not path.is_file():
                host_path_issues.append(f"required host key file is missing: {name}")
        issues.extend(host_path_issues)
    pipeline_required = reference_directory_contract
    business_gate: dict[str, Any] = {}
    pipeline_replay_gate: dict[str, Any] = {}
    if pipeline_required and not host_path_issues:
        business_gate = validate_business_result_package(
            result_package=candidate_package,
            train_reference_root=required_host_paths["train_reference_root"],
            split_keys=required_host_paths["validation_keys"],
            key_column=key_column,
            split_mode="validation",
            required_file_count=MIMIC_EXPECTED_BUSINESS_FILE_COUNT,
        )
        issues.extend(f"business package gate: {item}" for item in business_gate["issues"])
        previous_pipeline = active / "script_bundle" / "pipeline"
        pipeline_replay_gate = validate_and_replay_candidate_pipeline(
            pipeline_dir=candidate / "script_bundle" / "pipeline",
            previous_pipeline_dir=previous_pipeline if previous_pipeline.is_dir() else None,
            candidate_result_package=candidate_package,
            train_raw=required_host_paths["train_raw"],
            train_reference_root=required_host_paths["train_reference_root"],
            train_keys=required_host_paths["train_keys"],
            validation_raw=required_host_paths["validation_raw"],
            validation_keys=required_host_paths["validation_keys"],
            key_column=key_column,
            report_root=candidate / "host_replay",
            allowed_changed_modules=allowed_changed_modules,
        )
        issues.extend(f"pipeline replay gate: {item}" for item in pipeline_replay_gate["issues"])
    elif not pipeline_required:
        regression_path = candidate / "train_regression_report.json"
        if not regression_path.is_file():
            issues.append("train_regression_report.json is missing")
            regression_files: dict[str, bool] = {}
        else:
            regression_payload = _load_json(regression_path)
            regression_files = _train_regression_file_statuses(regression_payload)
            issues.extend(_train_regression_schema_issues(regression_payload, changed_business))
            if str(regression_payload.get("status") or "").upper() != "SUCCESS":
                issues.append("train regression status is not SUCCESS")
            missing_regressions = sorted(path for path in changed_business if path not in regression_files)
            failed_regressions = sorted(path for path in changed_business if regression_files.get(path) is False)
            if missing_regressions:
                issues.append("changed files missing train regression: " + ", ".join(missing_regressions[:20]))
            if failed_regressions:
                issues.append("train regression failed: " + ", ".join(failed_regressions[:20]))

        validation_path = candidate / "result_package_validation_report.json"
        if validation_path.is_file():
            validation = _load_json(validation_path)
            if validation.get("valid") is False or str(validation.get("status") or "SUCCESS").upper() != "SUCCESS":
                issues.append("result package validation did not pass")

        expected_count = _validation_key_count(contract)
        if expected_count and key_column and candidate_package.is_dir():
            cohort = _find_package_cohort_file(candidate_package)
            if cohort is None:
                issues.append("cohort file is missing for exact key coverage validation")
            else:
                try:
                    frame = _read_table_preview(cohort, nrows=None)
                    key_values = frame[key_column].dropna().astype(str) if key_column in frame.columns else pd.Series(dtype=str)
                    if int(key_values.nunique()) != expected_count:
                        issues.append(
                            f"cohort key coverage mismatch: expected={expected_count}, actual={int(key_values.nunique())}"
                        )
                    if bool(key_values.duplicated().any()):
                        issues.append("cohort contains duplicate keys")
                except Exception as exc:
                    issues.append(f"could not validate cohort keys: {exc}")

    return {
        "schema_version": 1,
        "valid": not issues,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "changed_business_files": changed_business,
        "allowed_feedback_files": sorted(allowed_targets),
        "business_gate": business_gate,
        "pipeline_replay_gate": pipeline_replay_gate,
        "issues": issues,
    }


def _public_gate_repair_prompt(
    *,
    gate: dict[str, Any],
    gate_report: Path,
    workspace: Path,
    repair_index: int,
) -> str:
    issues = [str(item) for item in gate.get("issues") or []]
    issue_text = "\n".join(f"- {item}" for item in issues[:100]) or "- candidate gate did not pass"
    if len(issues) > 100:
        issue_text += f"\n- ... {len(issues) - 100} additional issues are recorded in the gate report"
    return f"""\
当前候选未通过宿主公开预提交门禁。这仍是同一个 attempt 的第 {repair_index} 次修复，继续使用当前 Agent 上下文和 workspace。

公开门禁问题：
{issue_text}

完整门禁报告：{gate_report}

请只修复 {workspace / 'pipeline'}、{workspace / 'result_package'} 以及必要的当前 workspace 报告文件。
修复后必须重新运行累计 Pipeline、重新发布完整17文件 result_package，并再次执行 ValidateResultPackage。
不得读取 validation/reference、test reference、历史实验或其他未授权路径；不得手工修改 Pipeline 运行后生成的业务 CSV。
完成修复后再结束本次回复，宿主会在同一个 attempt 内重新归档并复查。
"""


async def _run_public_gate_repair_loop(
    *,
    candidate: Path,
    phase_root: Path,
    attempt_index: int,
    max_repairs: int,
    stage_and_gate: Callable[[int], tuple[dict[str, Any], dict[str, Any]]],
    request_repair: Callable[[str, int], Awaitable[None]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck public candidate gates after bounded repairs in one Agent session."""
    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    phase_root.mkdir(parents=True, exist_ok=True)
    gate_dir = candidate / "draft_gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Any] = {}
    gate: dict[str, Any] = {
        "schema_version": 1,
        "valid": False,
        "status": "NEEDS_REPAIR",
        "issues": ["candidate was not checked"],
    }
    for check_index in range(max_repairs + 1):
        staged, gate = stage_and_gate(check_index)
        gate_report = gate_dir / f"draft_gate_{check_index + 1:02d}.json"
        _write_json(gate_report, gate)
        _append_runtime_event(
            phase_root,
            {
                "event": "public_draft_gate_checked",
                "attempt": attempt_index,
                "check": check_index + 1,
                "valid": bool(gate.get("valid")),
                "gate_report": str(gate_report),
                "issue_count": len(gate.get("issues") or []),
            },
        )
        if gate.get("valid") or check_index == max_repairs:
            return staged, gate
        repair_index = check_index + 1
        prompt = _public_gate_repair_prompt(
            gate=gate,
            gate_report=gate_report,
            workspace=phase_root / "workspace",
            repair_index=repair_index,
        )
        _append_runtime_event(
            phase_root,
            {
                "event": "public_draft_gate_repair_requested",
                "attempt": attempt_index,
                "repair": repair_index,
                "gate_report": str(gate_report),
            },
        )
        await request_repair(prompt, repair_index)
    return staged, gate


def _package_relative_hashes(package_root: Path) -> dict[str, str]:
    if not package_root.is_dir():
        return {}
    return {
        path.relative_to(package_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _structured_package_files(package_root)
    }


def _train_regression_file_statuses(report: dict[str, Any]) -> dict[str, bool]:
    return {
        relative: bool(item["passed"])
        for relative, item in _normalized_train_regression_entries(report).items()
        if isinstance(item.get("passed"), bool)
    }


def _normalized_train_regression_entries(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_files = report.get("files") or []
    if isinstance(raw_files, dict):
        entries = [{"relative_path": path, **(value if isinstance(value, dict) else {})} for path, value in raw_files.items()]
    elif isinstance(raw_files, list):
        entries = [item for item in raw_files if isinstance(item, dict)]
    else:
        entries = []
    normalized: dict[str, dict[str, Any]] = {}
    for item in entries:
        relative = str(item.get("relative_path") or item.get("file") or item.get("name") or "").strip()
        for prefix in ("results/result_package/", "result_package/"):
            if relative.startswith(prefix):
                relative = relative[len(prefix):]
        if relative:
            normalized[relative] = item
    return normalized


def _train_regression_schema_issues(report: dict[str, Any], changed_files: list[str]) -> list[str]:
    issues: list[str] = []
    if int(report.get("schema_version") or 0) != 2:
        issues.append("train regression schema_version must be 2")
    entries = _normalized_train_regression_entries(report)
    required = {
        "train_rows",
        "reference_rows",
        "column_coverage",
        "key_coverage",
        "value_recall",
        "passed",
        "failure_reason",
    }
    for relative in changed_files:
        item = entries.get(relative)
        if item is None:
            continue
        missing = sorted(key for key in required if key not in item)
        if missing:
            issues.append(
                f"train regression fields missing for {relative}: " + ", ".join(missing)
            )
            continue
        if not isinstance(item.get("passed"), bool):
            issues.append(f"train regression passed must be boolean for {relative}")
        if item.get("passed") is False and not str(item.get("failure_reason") or "").strip():
            issues.append(f"train regression failure_reason is required when failed for {relative}")
        for key in ("train_rows", "reference_rows"):
            value = item.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                issues.append(f"train regression {key} must be a non-negative integer for {relative}")
        for key in ("column_coverage", "key_coverage", "value_recall"):
            value = item.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
                issues.append(f"train regression {key} must be between 0 and 1 for {relative}")
    return issues


def infer_reference_contract(dataset_split: str | Path) -> dict[str, Any]:
    split = Path(dataset_split).expanduser().resolve()
    manifest_path = split / "split_manifest.json"
    split_manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    train_raw = split / "train" / "raw"
    validation_raw = split / "validation" / "raw"
    train_reference_root = split / "train" / "reference"
    validation_reference_root = _resolve_reference_root(split, "validation")
    test_reference_root = _resolve_reference_root(split, "test")
    train_reference = _find_reference_file(train_reference_root)
    validation_reference = _find_reference_file(validation_reference_root)
    test_reference = _find_reference_file(test_reference_root)
    if train_reference is None:
        raise ValueError(f"train reference files not found under: {train_reference_root}")
    dataset_profile = str(split_manifest.get("dataset_profile") or infer_dataset_profile(train_raw))
    package_manifest_path = train_reference_root / "package_manifest.json"
    package_manifest = _load_json(package_manifest_path) if package_manifest_path.is_file() else {}
    reference_preview = _read_reference_preview(train_reference)
    manifest_key = str(
        split_manifest.get("split_key")
        or _key_column_from_keys(split / "train" / "keys.csv")
        or _key_column_from_keys(split / "validation" / "keys.csv")
        or ""
    )
    key_column = manifest_key if manifest_key and manifest_key in reference_preview.columns else _infer_key_column(reference_preview)
    feature_columns = [
        column for column in reference_preview.columns
        if column not in set(KEY_PRIORITY) | LABEL_COLUMNS
    ]
    has_nested_reference_tables = train_reference_root.is_dir() and (
        any((train_reference_root / name).is_dir() for name in ("cohort", "features", "csv"))
        or any(path.parent != train_reference_root for path in train_reference_root.rglob("*") if path.is_file())
    )
    reference_type = (
        "reference_directory" if has_nested_reference_tables else
        "feature_package" if feature_columns or (LABEL_COLUMNS & set(reference_preview.columns)) else "field_gold"
    )
    record_grain = key_column or manifest_key
    if dataset_profile == "mimic_iv_3_1" and not record_grain:
        record_grain = "hadm_id"
    if reference_type == "reference_directory":
        validation_reference_private = str(validation_reference_root) if validation_reference_root.is_dir() else ""
        test_reference_private = str(test_reference_root) if test_reference_root.is_dir() else ""
    else:
        validation_reference_private = (
            str(validation_reference)
            if validation_reference is not None
            else str(validation_reference_root) if validation_reference_root.is_dir() else ""
        )
        test_reference_private = (
            str(test_reference)
            if test_reference is not None
            else str(test_reference_root) if test_reference_root.is_dir() else ""
        )
    contract = {
        "schema_version": 1,
        "workflow": "reference-guided-train-validate",
        "dataset_split": str(split),
        "dataset_profile": dataset_profile,
        "reference_type": reference_type,
        "evaluation_mode": "feature_table" if reference_type in {"feature_package", "reference_package", "reference_directory"} else "field_gold",
        "record_grain": record_grain,
        "key_column": key_column,
        "paths": {
            "train_raw": str(train_raw),
            "train_keys": str(split / "train" / "keys.csv"),
            "train_reference_root": str(train_reference_root),
            "train_reference": str(train_reference),
            "validation_raw": str(validation_raw),
            "validation_reference_root": str(validation_reference_root) if validation_reference_root.is_dir() else "",
            "validation_reference_private_root": str(validation_reference_root) if validation_reference_root.is_dir() else "",
            "validation_reference_private": validation_reference_private,
            "validation_keys": str(split / "validation" / "keys.csv"),
            "test_raw": str(split / "test" / "raw"),
            "test_reference_root": str(test_reference_root) if test_reference_root.is_dir() else "",
            "test_reference_private_root": str(test_reference_root) if test_reference_root.is_dir() else "",
            "test_reference_private": test_reference_private,
            "test_keys": str(split / "test" / "keys.csv"),
            "train_reference_package": str(package_manifest_path) if package_manifest else "",
        },
        "reference_schema": {
            "columns": list(reference_preview.columns),
            "feature_columns": feature_columns,
            "label_columns": sorted(LABEL_COLUMNS & set(reference_preview.columns)),
        },
        "reference_package": package_manifest,
        "split_manifest": str(manifest_path) if manifest_path.is_file() else "",
        "status": "supported" if dataset_profile == "mimic_iv_3_1" and reference_type == "reference_directory" else "needs_adapter",
    }
    if contract["status"] != "supported":
        contract["unsupported_reason"] = (
            "当前入口只支持 MIMIC 3.1 的目录型 reference package；"
            "其他 reference 形态已随旧 teacher workflow 移除。"
        )
    return contract


def _ensure_run_manifest(
    config: ReferenceGuidedConfig,
    contract: dict[str, Any],
    experiment_dir: str | Path,
) -> dict[str, Any]:
    experiment = Path(experiment_dir).expanduser().resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment / "run_manifest.json"
    split = Path(config.dataset_split).expanduser().resolve()
    key_hashes: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        path = split / split_name / "keys.csv"
        key_hashes[split_name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    immutable = {
        "dataset_split": str(split),
        "dataset_profile": str(contract.get("dataset_profile") or ""),
        "reference_type": str(contract.get("reference_type") or ""),
        "key_column": str(contract.get("key_column") or ""),
        "key_file_sha256": key_hashes,
        "task_prompt_sha256": hashlib.sha256(config.task_text.encode("utf-8")).hexdigest(),
        "scorer_version": SCORER_VERSION,
        "pipeline_skills_enabled": config.pipeline_skills_enabled,
        **codegraph_manifest_identity(
            enabled=config.codegraph_enabled,
            project_root=Path(__file__).resolve().parent.parent,
            allowed_roots=reference_code_read_roots(
                include_pipeline_skills=config.pipeline_skills_enabled,
                include_error_view_skills=False,
            ),
        ),
    }
    run_id = hashlib.sha256(
        json.dumps(immutable, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    expected = {
        "schema_version": 1,
        "workflow": "reference-guided-train-validate",
        "run_id": run_id,
        **immutable,
    }
    if manifest_path.is_file():
        existing = _load_json(manifest_path)
        mismatches = [key for key, value in expected.items() if existing.get(key) != value]
        if mismatches:
            raise ValueError(
                "run manifest does not match this experiment: " + ", ".join(sorted(mismatches))
            )
        return existing
    existing_entries = [path for path in experiment.iterdir() if path.name != manifest_path.name]
    if existing_entries:
        raise ValueError(
            "historical experiment directories are not imported; choose a new empty experiment directory"
        )
    manifest = {**expected, "created_at": datetime.now().isoformat(timespec="seconds")}
    _write_json(manifest_path, manifest)
    return manifest


class ReferenceGuidedWorkflow:
    def __init__(self, config: ReferenceGuidedConfig) -> None:
        self.config = config.normalized()
        self.experiment_dir = Path(self.config.experiment_dir)
        self.contract_path = self.experiment_dir / "reference_contract.json"

    def run_sync(self) -> dict[str, Any]:
        if self.config.codegraph_enabled:
            asyncio.run(
                probe_codegraph_connection(
                    project_root=Path(__file__).resolve().parent.parent,
                    allowed_roots=reference_code_read_roots(
                        include_pipeline_skills=self.config.pipeline_skills_enabled,
                        include_error_view_skills=False,
                    ),
                ),
            )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        contract = infer_reference_contract(self.config.dataset_split)
        _ensure_run_manifest(self.config, contract, self.experiment_dir)
        _write_json(self.contract_path, contract)
        if contract.get("status") != "supported":
            return {
                "status": "NEEDS_ADAPTER",
                "workflow": "reference-guided-train-validate",
                "experiment_dir": str(self.experiment_dir),
                "reference_contract": str(self.contract_path),
                "reason": contract.get("unsupported_reason", "unsupported reference contract"),
                "contract": contract,
            }
        return {
            **self._run_mimic_reference_agent(contract),
            "workflow": "reference-guided-train-validate",
            "routed_workflow": "data-cleaning-agent",
            "reference_contract": str(self.contract_path),
        }

    def _run_mimic_reference_agent(self, contract: dict[str, Any]) -> dict[str, Any]:
        return DataCleaningAgentRuntime(
            config=self.config,
            contract=contract,
            contract_path=self.contract_path,
        ).run_sync()


def _copy_active_bundle_for_candidate(active_dir: Path, candidate: Path) -> None:
    """Copy only lightweight active state into a candidate round."""
    skip_top_level = {"results", "agent_runs"}
    active_resolved = active_dir.resolve()

    def _ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() == active_resolved:
            return set(names) & skip_top_level
        return set()

    shutil.copytree(active_dir, candidate, ignore=_ignore)


def _inherit_pipeline_to_workspace(active_dir: Path, workspace: Path) -> str:
    source = active_dir / "script_bundle" / "pipeline"
    target = workspace / "pipeline"
    if target.exists():
        shutil.rmtree(target)
    if not source.is_dir():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return pipeline_directory_sha256(source)


def _stage_workspace_pipeline(workspace: Path, candidate: Path) -> Path | None:
    source = workspace / "pipeline"
    target = candidate / "script_bundle" / "pipeline"
    if target.exists():
        shutil.rmtree(target)
    if not source.is_dir():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def _archive_promoted_round(
    *,
    experiment_dir: str | Path,
    candidate: str | Path,
    evaluation: dict[str, Any],
    round_index: int,
    attempt_index: int,
    score: float,
) -> Path:
    experiment = Path(experiment_dir).expanduser().resolve()
    candidate_dir = Path(candidate).expanduser().resolve()
    script_bundle = _update_candidate_script_bundle(
        candidate_dir,
        round_index=round_index,
        attempt_index=attempt_index,
    )
    rounds_dir = experiment / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    round_dir = rounds_dir / f"round_{round_index:04d}"
    prepared_round = rounds_dir / f".round_{round_index:04d}.staging"
    if prepared_round.exists():
        shutil.rmtree(prepared_round)
    prepared_round.mkdir(parents=True, exist_ok=True)
    shutil.copytree(script_bundle, prepared_round / "script_bundle")
    result_package = candidate_dir / "results" / "result_package"
    if not result_package.is_dir():
        raise RuntimeError(f"promoted candidate has no result_package: {result_package}")
    shutil.copytree(result_package, prepared_round / "validation_result")
    evaluation_report = Path(str(evaluation.get("evaluation_report") or "")).expanduser()
    if evaluation_report.is_file():
        shutil.copytree(evaluation_report.resolve().parent, prepared_round / "evaluation")
    else:
        evaluation_dir = prepared_round / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        for name, raw_path in sorted(evaluation.items()):
            source = Path(str(raw_path or "")).expanduser()
            if source.is_file():
                shutil.copy2(source, evaluation_dir / f"{name}{source.suffix}")
    host_replay = candidate_dir / "host_replay"
    if host_replay.is_dir():
        shutil.copytree(host_replay, prepared_round / "host_replay")
    bundle_hash = _directory_sha256(script_bundle)
    result_hash = _directory_sha256(result_package)
    pipeline = script_bundle / "pipeline"
    pipeline_hash = pipeline_directory_sha256(pipeline) if pipeline.is_dir() else ""
    entrypoint = pipeline / "run.py"
    entrypoint_hash = hashlib.sha256(entrypoint.read_bytes()).hexdigest() if entrypoint.is_file() else ""
    pipeline_manifest = _load_json(pipeline / "pipeline_manifest.json") if (pipeline / "pipeline_manifest.json").is_file() else {}
    provenance = {
        "schema_version": 1,
        "status": "SUCCESS",
        "round": round_index,
        "attempt": attempt_index,
        "score": score,
        "candidate": str(candidate_dir),
        "script_bundle_sha256": bundle_hash,
        "pipeline_sha256": pipeline_hash,
        "entrypoint_sha256": entrypoint_hash,
        "parent_pipeline_sha256": str(pipeline_manifest.get("parent_pipeline_sha256") or ""),
        "host_replay_report": str(round_dir / "host_replay/pipeline_replay_report.json") if host_replay.is_dir() else "",
        "validation_result_sha256": result_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(prepared_round / "provenance.json", provenance)
    _commit_prepared_directory(prepared_round, round_dir)
    return round_dir


def _update_candidate_script_bundle(
    candidate: Path,
    *,
    round_index: int,
    attempt_index: int,
) -> Path:
    bundle = candidate / "script_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ("capabilities", "adapter_bundle"):
        source = candidate / name
        if not source.is_dir():
            continue
        target = bundle / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    for name in (
        "skill_bindings.json",
        "skill_registry.json",
        "skill_packaging_plan.json",
        "adapter_bundle_manifest.json",
    ):
        source = candidate / name
        if source.is_file():
            shutil.copy2(source, bundle / name)

    workspace_target = bundle / "workspace"
    workspace_target.mkdir(parents=True, exist_ok=True)
    workspaces = sorted(
        (path for path in (candidate / "agent_runs").rglob("workspace") if path.is_dir())
        if (candidate / "agent_runs").is_dir()
        else [],
        key=lambda path: path.stat().st_mtime_ns,
    )
    if workspaces:
        _merge_script_workspace(workspaces[-1], workspace_target)

    runtime_target = bundle / "runtime" / f"round_{round_index:04d}"
    runtime_target.mkdir(parents=True, exist_ok=True)
    if (candidate / "agent_runs").is_dir():
        for source in sorted(path for path in (candidate / "agent_runs").rglob("*") if path.is_file()):
            if source.name not in {
                "runtime_trace.jsonl",
                "context_summary.md",
                "agent_response.txt",
                "pipeline_skill_usage_check.json",
                "train_regression_report.json",
                "feedback_response.json",
            }:
                continue
            relative = source.relative_to(candidate / "agent_runs")
            target = runtime_target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    file_hashes = _relative_file_hashes(bundle)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "SUCCESS",
            "latest_round": round_index,
            "source_attempt": attempt_index,
            "file_count": len(file_hashes),
            "files": file_hashes,
            "bundle_sha256": _hash_manifest(file_hashes),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return bundle


def _merge_script_workspace(source: Path, target: Path) -> None:
    allowed_suffixes = {
        ".py",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".txt",
        ".sh",
    }
    excluded_parts = {"result_package", "artifacts", "pipeline", "__pycache__", ".pytest_cache"}
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        if excluded_parts.intersection(relative.parts) or path.suffix.casefold() not in allowed_suffixes:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _hash_manifest(file_hashes: dict[str, str]) -> str:
    payload = json.dumps(file_hashes, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _directory_sha256(root: Path) -> str:
    return _hash_manifest(_relative_file_hashes(root))


def _replace_directory_atomically(source: Path, target: Path) -> None:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    prepared = target.parent / f".{target.name}.next"
    backup = target.parent / f".{target.name}.previous"
    for stale in (prepared, backup):
        if stale.exists():
            shutil.rmtree(stale)
    shutil.copytree(source, prepared)
    source_hash = _directory_sha256(source)
    if _directory_sha256(prepared) != source_hash:
        shutil.rmtree(prepared, ignore_errors=True)
        raise RuntimeError(f"prepared directory does not match source: {source}")
    _commit_prepared_directory(prepared, target)


def _commit_prepared_directory(prepared: Path, target: Path) -> None:
    prepared = prepared.expanduser().resolve()
    target = target.expanduser().resolve()
    backup = target.parent / f".{target.name}.previous"
    if backup.exists():
        shutil.rmtree(backup)
    had_target = target.exists()
    try:
        with _defer_sigint():
            if had_target:
                os.replace(target, backup)
            os.replace(prepared, target)
        _fsync_directory(target.parent)
    except BaseException:
        if target.exists():
            shutil.rmtree(target)
        if backup.exists():
            os.replace(backup, target)
        if prepared.exists():
            shutil.rmtree(prepared)
        _fsync_directory(target.parent)
        raise
    if backup.exists():
        shutil.rmtree(backup)


@contextmanager
def _defer_sigint() -> Iterator[None]:
    previous_mask: set[signal.Signals] | None = None
    if hasattr(signal, "pthread_sigmask"):
        try:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        except (OSError, ValueError):
            previous_mask = None
    try:
        yield
    finally:
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _checkpoint_fingerprint(
    frozen_script_bundle: Path,
    contract: dict[str, Any],
    *,
    scorer_version: str,
) -> str:
    test_keys_text = str((contract.get("paths") or {}).get("test_keys") or "")
    test_keys = Path(test_keys_text).expanduser().resolve() if test_keys_text else None
    test_key_hash = (
        hashlib.sha256(test_keys.read_bytes()).hexdigest()
        if test_keys is not None and test_keys.is_file()
        else ""
    )
    payload = {
        "frozen_script_bundle_sha256": _directory_sha256(frozen_script_bundle),
        "test_keys_sha256": test_key_hash,
        "scorer_version": scorer_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _next_checkpoint_index(root: Path, state: dict[str, Any]) -> int:
    indices = {
        int(item.get("checkpoint") or 0)
        for item in state.get("checkpoints") or []
        if int(item.get("checkpoint") or 0) > 0
    }
    if root.is_dir():
        for path in root.glob("checkpoint_*_best_round_*"):
            match = re.fullmatch(r"checkpoint_(\d+)_best_round_\d+", path.name)
            if match:
                indices.add(int(match.group(1)))
    return max(indices, default=0) + 1


def _promote_active_bundle_atomically(
    *,
    active_dir: str | Path,
    candidate: str | Path,
    state_path: str | Path,
    state: dict[str, Any],
    journal_path: str | Path,
) -> None:
    active = Path(active_dir).expanduser().resolve()
    candidate_dir = Path(candidate).expanduser().resolve()
    state_file = Path(state_path).expanduser().resolve()
    journal = Path(journal_path).expanduser().resolve()
    if journal.is_file():
        _recover_interrupted_promotion(journal, state_file)
    prepared = active.parent / f".{active.name}.next"
    backup = active.parent / f".{active.name}.previous"
    for stale in (prepared, backup):
        if stale.exists():
            shutil.rmtree(stale)
    shutil.copytree(candidate_dir, prepared)
    candidate_hash = _directory_sha256(candidate_dir)
    if _directory_sha256(prepared) != candidate_hash:
        shutil.rmtree(prepared, ignore_errors=True)
        raise RuntimeError("prepared active bundle does not match promoted candidate")
    payload = {
        "schema_version": 1,
        "stage": "prepared",
        "active_dir": str(active),
        "backup_dir": str(backup),
        "prepared_dir": str(prepared),
        "candidate_sha256": candidate_hash,
        "best_attempt": int(state.get("best_attempt") or 0),
        "best_round": int(state.get("best_round") or 0),
        "had_active": active.exists(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(journal, payload)
    with _defer_sigint():
        if active.exists():
            os.replace(active, backup)
        payload["stage"] = "old_moved"
        _write_json(journal, payload)
        os.replace(prepared, active)
        payload["stage"] = "new_active"
        _write_json(journal, payload)
        _write_json(state_file, state)
        payload["stage"] = "state_committed"
        _write_json(journal, payload)
    if backup.exists():
        shutil.rmtree(backup)
    journal.unlink(missing_ok=True)
    _fsync_directory(active.parent)


def _recover_interrupted_promotion(journal_path: str | Path, state_path: str | Path) -> None:
    journal = Path(journal_path).expanduser().resolve()
    if not journal.is_file():
        return
    payload = _load_json(journal)
    active = Path(str(payload["active_dir"])).expanduser().resolve()
    backup = Path(str(payload["backup_dir"])).expanduser().resolve()
    prepared = Path(str(payload["prepared_dir"])).expanduser().resolve()
    state_file = Path(state_path).expanduser().resolve()
    state = _load_json(state_file) if state_file.is_file() else {}
    state_committed = int(state.get("best_attempt") or 0) == int(payload.get("best_attempt") or -1)
    active_matches = (
        active.is_dir()
        and str(payload.get("candidate_sha256") or "")
        and _directory_sha256(active) == str(payload.get("candidate_sha256"))
    )
    if state_committed and active_matches:
        if backup.exists():
            shutil.rmtree(backup)
        if prepared.exists():
            shutil.rmtree(prepared)
        journal.unlink(missing_ok=True)
        _fsync_directory(active.parent)
        return

    if backup.is_dir():
        if active.exists():
            shutil.rmtree(active)
        os.replace(backup, active)
    elif not bool(payload.get("had_active")) and active.exists():
        shutil.rmtree(active)
    if prepared.exists():
        shutil.rmtree(prepared)
    journal.unlink(missing_ok=True)
    _fsync_directory(active.parent)


class DataCleaningAgentRuntime:
    """Codex-style single-agent runtime for reference-guided package materialization."""

    def __init__(
        self,
        *,
        config: ReferenceGuidedConfig,
        contract: dict[str, Any],
        contract_path: str | Path,
    ) -> None:
        self.config = config
        self.contract = contract
        self.contract_path = Path(contract_path).expanduser().resolve()
        self.experiment_dir = Path(config.experiment_dir)
        self.active_dir = self.experiment_dir / "active_bundle"
        self.candidates_dir = self.experiment_dir / "candidates"
        self.evaluations_dir = self.experiment_dir / "evaluations"
        self.test_checkpoints_dir = self.experiment_dir / "test_checkpoints"
        self.state_path = self.experiment_dir / "experiment_state.json"
        self.promotion_journal_path = self.experiment_dir / "promotion_journal.json"

    def run_sync(self) -> dict[str, Any]:
        if not self.state_path.exists():
            self._initialize()
        _recover_interrupted_promotion(self.promotion_journal_path, self.state_path)
        raw_state = _load_json(self.state_path)
        if int(raw_state.get("schema_version") or 1) < 2:
            backup = self.experiment_dir / "experiment_state.v1.json"
            if not backup.exists():
                _write_json(backup, raw_state)
        state = _migrate_reference_experiment_state(raw_state)
        pending_text = str(state.pop("pending_checkpoint", "") or "")
        if pending_text:
            pending = Path(pending_text).expanduser().resolve()
            shutil.rmtree(pending / "test_evaluation", ignore_errors=True)
            (pending / "checkpoint_report.json").unlink(missing_ok=True)
            state.setdefault("abandoned_checkpoints", []).append(
                {
                    "checkpoint_dir": str(pending),
                    "status": "abandoned_on_resume",
                    "recovered_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        state["status"] = "active"
        state.pop("termination_reason", None)
        state["consecutive_valid_no_improvement"] = 0
        state.setdefault("sessions", []).append(
            {
                "session": len(state.get("sessions") or []) + 1,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "starting_attempt": len(state.get("attempts") or []) + 1,
                "starting_round": len(state.get("rounds") or []) + 1,
            }
        )
        _write_json(self.state_path, state)
        promotions_this_run = 0
        attempts_this_run = 0
        while True:
            force_validation_attempt = bool(state.pop("force_validation_attempt", False))
            if force_validation_attempt:
                state["status"] = "active"
                _write_json(self.state_path, state)
                stop_reason = ""
            else:
                stop_reason = _reference_stop_reason(
                    state,
                    promotions_this_run=promotions_this_run,
                    attempts_this_run=attempts_this_run,
                    round_limit=self.config.round_limit,
                    patience=self.config.patience,
                    max_attempts=self.config.max_attempts,
                    target_score=self.config.target_score,
                )
            if stop_reason:
                try:
                    state = self._checkpoint_and_test(stop_reason, state)
                except KeyboardInterrupt:
                    state = self._mark_interrupted("second_interrupt")
                if (
                    state.get("status") == "validation_resume_required"
                    and attempts_this_run < self.config.max_attempts
                ):
                    continue
                if state.get("status") == "validation_resume_required":
                    state["status"] = "checkpointed_pipeline_rule_failure"
                    state["termination_reason"] = "max_attempts_after_test_rule_failure"
                state.pop("force_validation_attempt", None)
                _write_json(self.state_path, state)
                break

            attempt_index = len(state.get("attempts", [])) + 1
            candidate = self.candidates_dir / f"attempt_{attempt_index:04d}"
            attempts_this_run += 1
            try:
                candidate = self._begin_attempt(attempt_index)
                state, improved = self._execute_attempt(
                    state=state,
                    candidate=candidate,
                    attempt_index=attempt_index,
                )
            except AllApiKeysRateLimitedError as exc:
                state = _rate_limit_block_state(state, exc)
                _write_json(self.state_path, state)
                break
            except KeyboardInterrupt:
                _recover_interrupted_promotion(self.promotion_journal_path, self.state_path)
                state = _migrate_reference_experiment_state(_load_json(self.state_path))
                state = self._cancel_uncommitted_attempt(state, attempt_index, candidate)
                try:
                    state = self._checkpoint_and_test("cancelled_by_user", state)
                except KeyboardInterrupt:
                    state = self._mark_interrupted("second_interrupt")
                if (
                    state.get("status") == "validation_resume_required"
                    and attempts_this_run < self.config.max_attempts
                ):
                    continue
                break
            if improved:
                promotions_this_run += 1
        return self._result(state)

    def _execute_attempt(
        self,
        *,
        state: dict[str, Any],
        candidate: Path,
        attempt_index: int,
    ) -> tuple[dict[str, Any], bool]:
        public_feedback_text = str(state.get("best_feedback") or "")
        public_feedback = Path(public_feedback_text) if public_feedback_text else None
        if public_feedback is not None and not public_feedback.is_file():
            public_feedback = None
        previous_outcome = self._latest_attempt_outcome(state)
        test_rule_feedback_text = str(state.get("pending_test_rule_feedback", "") or "")
        test_rule_feedback = Path(test_rule_feedback_text) if test_rule_feedback_text else None
        if test_rule_feedback is not None and not test_rule_feedback.is_file():
            test_rule_feedback = None
        package: dict[str, Any] = {}
        try:
            package = asyncio.run(
                self._run_attempt(
                    candidate=candidate,
                    attempt_index=attempt_index,
                    public_feedback=public_feedback,
                    previous_outcome=previous_outcome,
                    test_rule_feedback=test_rule_feedback,
                )
            )
        except AllApiKeysRateLimitedError:
            raise
        except Exception as exc:
            gate = {
                "schema_version": 1,
                "valid": False,
                "status": "NEEDS_REPAIR",
                "issues": [f"agent attempt failed: {type(exc).__name__}: {exc}"],
            }
            state, _ = _finish_reference_attempt_state(
                state,
                attempt_index=attempt_index,
                score=None,
                candidate_dir=str(candidate),
                result=package,
                evaluation={},
                gate=gate,
            )
            self._persist_attempt_state(state, candidate, gate)
            return state, False

        gate = _reference_candidate_quality_gate(
            active=self.active_dir,
            candidate=candidate,
            contract=self.contract,
            repair_targets=(
                candidate / "repair_targets.json"
                if (candidate / "repair_targets.json").is_file()
                else None
            ),
        )
        _write_json(candidate / "candidate_quality_gate.json", gate)
        if not gate["valid"]:
            state, _ = _finish_reference_attempt_state(
                state,
                attempt_index=attempt_index,
                score=None,
                candidate_dir=str(candidate),
                result=package,
                evaluation={},
                gate=gate,
            )
            self._persist_attempt_state(state, candidate, gate)
            return state, False

        evaluation = evaluate_reference_directory_package(
            result_package=package["result_package"],
            validation_reference_root=self.contract["paths"].get("validation_reference_root")
            or self.contract["paths"].get("validation_reference_private_root")
            or self.contract["paths"].get("validation_reference_private")
            or None,
            output_dir=self.evaluations_dir / f"attempt_{attempt_index:04d}",
            key_column=str(self.contract.get("key_column") or self.contract.get("record_grain") or ""),
        )
        report = _load_json(evaluation["evaluation_report"])
        score = float((report.get("metrics") or {}).get("composite_score") or 0.0)
        state, improved = _finish_reference_attempt_state(
            state,
            attempt_index=attempt_index,
            score=score,
            candidate_dir=str(candidate),
            result=package,
            evaluation=evaluation,
            gate=gate,
        )
        if improved:
            state.pop("pending_test_rule_feedback", None)
            round_dir = _archive_promoted_round(
                experiment_dir=self.experiment_dir,
                candidate=candidate,
                evaluation=evaluation,
                round_index=int(state.get("best_round") or len(state.get("rounds", []))),
                attempt_index=attempt_index,
                score=score,
            )
            state["rounds"][-1]["formal_round_dir"] = str(round_dir)
            state["rounds"][-1]["script_bundle"] = str(round_dir / "script_bundle")
            _promote_active_bundle_atomically(
                active_dir=self.active_dir,
                candidate=candidate,
                state_path=self.state_path,
                state=state,
                journal_path=self.promotion_journal_path,
            )
        self._persist_attempt_state(state, candidate, gate)
        return state, improved

    def _initialize(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        for directory in (self.active_dir, self.candidates_dir, self.evaluations_dir):
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.contract_path, self.active_dir / "reference_contract.json")
        _write_json(
            self.active_dir / "manifest.json",
            {
                "schema_version": 1,
                "status": "active",
                "version": 0,
                "runtime": "DataCleaningAgentRuntime",
            },
        )
        _write_json(
            self.state_path,
            {
                "schema_version": 2,
                "workflow": "reference-guided-train-validate",
                "routed_workflow": "data-cleaning-agent",
                "engineer_mode": "agent",
                "status": "active",
                "best_score": 0.0,
                "best_round": 0,
                "best_attempt": 0,
                "best_feedback": "",
                "consecutive_valid_no_improvement": 0,
                "invalid_attempt_count": 0,
                "attempts": [],
                "rounds": [],
                "checkpoints": [],
                "sessions": [],
                "abandoned_checkpoints": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def _begin_attempt(self, attempt_index: int) -> Path:
        candidate = self.candidates_dir / f"attempt_{attempt_index:04d}"
        if candidate.exists():
            shutil.rmtree(candidate)
        _copy_active_bundle_for_candidate(self.active_dir, candidate)
        return candidate

    def _latest_attempt_outcome(self, state: dict[str, Any]) -> Path | None:
        for latest in reversed(state.get("attempts") or []):
            if latest.get("status") == "cancelled_by_user":
                continue
            candidate_dir = str(latest.get("candidate_dir") or "")
            path = Path(candidate_dir) / "attempt_outcome.json" if candidate_dir else None
            if path is not None and path.is_file():
                return path
        return None

    def _cancel_uncommitted_attempt(
        self,
        state: dict[str, Any],
        attempt_index: int,
        candidate: Path,
    ) -> dict[str, Any]:
        state = _migrate_reference_experiment_state(state)
        if any(int(item.get("attempt") or 0) == attempt_index for item in state.get("attempts") or []):
            return state

        next_round_index = len(state.get("rounds") or []) + 1
        orphan_staging = self.experiment_dir / "rounds" / f".round_{next_round_index:04d}.staging"
        shutil.rmtree(orphan_staging, ignore_errors=True)
        orphan_round = self.experiment_dir / "rounds" / f"round_{next_round_index:04d}"
        provenance = orphan_round / "provenance.json"
        if provenance.is_file():
            payload = _load_json(provenance)
            if int(payload.get("attempt") or 0) == attempt_index:
                shutil.rmtree(orphan_round)

        gate = {
            "schema_version": 1,
            "valid": False,
            "status": "CANCELLED",
            "issues": ["validation attempt cancelled by user before atomic commit"],
        }
        state.setdefault("attempts", []).append(
            {
                "attempt": attempt_index,
                "status": "cancelled_by_user",
                "valid": False,
                "score": None,
                "improved": False,
                "candidate_dir": str(candidate),
                "producer": "",
                "result": {},
                "evaluation": {},
                "gate": gate,
            }
        )
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._persist_attempt_state(state, candidate, gate)
        return state

    def _mark_interrupted(self, reason: str) -> dict[str, Any]:
        _recover_interrupted_promotion(self.promotion_journal_path, self.state_path)
        state = _migrate_reference_experiment_state(_load_json(self.state_path))
        pending_text = str(state.pop("pending_checkpoint", "") or "")
        if pending_text:
            pending = Path(pending_text).expanduser().resolve()
            shutil.rmtree(pending / "test_evaluation", ignore_errors=True)
            (pending / "checkpoint_report.json").unlink(missing_ok=True)
            state["interrupted_checkpoint"] = str(pending)
        state["status"] = "interrupted"
        state["termination_reason"] = reason
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(self.state_path, state)
        return state

    async def _run_attempt(
        self,
        *,
        candidate: Path,
        attempt_index: int,
        public_feedback: Path | None,
        previous_outcome: Path | None = None,
        test_rule_feedback: Path | None = None,
    ) -> dict[str, Any]:
        attempt_root = candidate / "agent_runs"
        phase = init_phase_session(attempt_root, "data_cleaning_agent")
        run_id = f"reference_code_attempt_{attempt_index:04d}"
        set_phase_context(run_id, attempt_root, "data_cleaning_agent", phase["phase_root"])
        phase_root = Path(phase["phase_root"])
        paths = self.contract["paths"]
        train_reference_root = Path(
            paths.get("train_reference_root") or Path(paths["train_reference"]).expanduser().resolve().parent
        ).expanduser().resolve()
        validation_keys = Path(paths.get("validation_keys") or "")
        report_paths = {
            "reference_contract": str(self.contract_path),
            "train_reference_root": str(train_reference_root),
        }
        active_result_package = self.active_dir / "results" / "result_package"
        if active_result_package.is_dir():
            report_paths["active_result_package"] = str(active_result_package)
        active_pipeline = self.active_dir / "script_bundle" / "pipeline"
        parent_pipeline_sha256 = (
            pipeline_directory_sha256(active_pipeline) if active_pipeline.is_dir() else ""
        )
        if active_pipeline.is_dir():
            report_paths["current_best_pipeline"] = str(active_pipeline)
        pipeline_contract_path = candidate / "canonical_pipeline_contract.json"
        _write_json(
            pipeline_contract_path,
            {
                "schema_version": 1,
                "entrypoint": "pipeline/run.py",
                "modules": [
                    "cohort.py",
                    "labels.py",
                    "chart.py",
                    "diag.py",
                    "med.py",
                    "out.py",
                    "proc.py",
                    "summary.py",
                ],
                "expected_files": sorted(
                    path.relative_to(train_reference_root).as_posix()
                    for path in _structured_package_files(train_reference_root)
                    if path.name not in IGNORED_PACKAGE_FILES
                ),
                "parent_pipeline_sha256": parent_pipeline_sha256,
                "forbidden_source_markers": [
                    "active_bundle",
                    "validation_result",
                    "reference_private",
                    "test_evaluation",
                    "private_report",
                ],
            },
        )
        report_paths["canonical_pipeline_contract"] = str(pipeline_contract_path)
        if paths.get("train_reference"):
            report_paths["train_reference_sample"] = str(Path(paths["train_reference"]).expanduser().resolve())
        package_manifest = train_reference_root / "package_manifest.json"
        if package_manifest.is_file():
            report_paths["train_reference_manifest"] = str(package_manifest)
        repair_targets_path: Path | None = None
        if public_feedback is not None:
            report_paths["public_feedback"] = str(public_feedback)
            repair_targets_path = candidate / "repair_targets.json"
            _compile_reference_repair_targets(public_feedback, repair_targets_path)
            report_paths["repair_targets"] = str(repair_targets_path)
        if previous_outcome is not None:
            report_paths["previous_attempt_outcome"] = str(previous_outcome)
            previous_payload = _load_json(previous_outcome)
            previous_feedback_text = str(
                ((previous_payload.get("evaluation") or {}).get("public_feedback") or "")
            )
            previous_feedback = Path(previous_feedback_text) if previous_feedback_text else None
            if previous_feedback is not None and previous_feedback.is_file():
                report_paths["previous_attempt_feedback_negative_evidence"] = str(previous_feedback)
        if test_rule_feedback is not None:
            report_paths["test_pipeline_rule_failure"] = str(test_rule_feedback)
            repair_targets_path = repair_targets_path or candidate / "repair_targets.json"
            _merge_test_rule_feedback_into_repair_targets(
                test_rule_feedback,
                repair_targets_path,
            )
            report_paths["repair_targets"] = str(repair_targets_path)
        code_read_roots = reference_code_read_roots(
            include_pipeline_skills=self.config.pipeline_skills_enabled,
            include_error_view_skills=False,
        )
        denied_read_roots = reference_code_denied_read_roots(
            include_pipeline_skills=self.config.pipeline_skills_enabled,
            include_error_view_skills=False,
        )
        read_roots = [
            *code_read_roots,
            paths["train_raw"],
            train_reference_root,
            paths["validation_raw"],
            candidate,
            self.active_dir,
        ]
        if validation_keys.is_file():
            read_roots.append(validation_keys)
        if previous_outcome is not None:
            read_roots.append(previous_outcome)
        if test_rule_feedback is not None:
            read_roots.append(test_rule_feedback)
        previous_feedback_path = report_paths.get("previous_attempt_feedback_negative_evidence", "")
        if previous_feedback_path:
            read_roots.append(previous_feedback_path)
        task_text = self._task_text(candidate, phase_root, report_paths, public_feedback)
        agent: Agent | None = None
        agent_workspace = None
        codegraph_runtime: CodeGraphRuntime | None = None
        try:
            context = EngineerToolContext.from_task(
                task_text=task_text,
                engineer_phase_root=phase_root,
                explorer_phase_root=str(candidate),
                additional_read_roots=read_roots,
                denied_read_roots=denied_read_roots,
                require_skill_plan=False,
                split_mode="validation",
                required_report_paths=report_paths,
            )
            inherited_parent = _inherit_pipeline_to_workspace(
                self.active_dir,
                context.workspace_dir,
            )
            if inherited_parent != parent_pipeline_sha256:
                raise RuntimeError("inherited pipeline hash changed before Agent execution")
            if self.config.codegraph_enabled:
                codegraph_runtime = await open_codegraph_runtime(
                    project_root=Path(__file__).resolve().parent.parent,
                    allowed_roots=code_read_roots,
                    usage_path=phase_root / "codegraph_usage.jsonl",
                    audit_path=context.workspace_dir / "tool_audit.jsonl",
                    attempt=f"attempt_{attempt_index:04d}",
                )
            toolkit, manifest = create_reference_toolkit(
                context,
                include_pipeline_skills=self.config.pipeline_skills_enabled,
                codegraph_tool=(codegraph_runtime.tool if codegraph_runtime is not None else None),
            )
            _remove_fixed_reference_incompatible_tools(toolkit)
            agent_skill_prompt = await toolkit.get_skill_instructions()
            if agent_skill_prompt:
                (phase_root / "agentscope_skill_prompt.txt").write_text(agent_skill_prompt, encoding="utf-8")
            restore_bundle_variants(self.active_dir, context.variant_root)
            _restore_claude_style_capabilities(self.active_dir, context.workspace_dir / "capabilities")
            self._write_context_summary(phase_root, attempt_index, "started")
            _append_runtime_event(
                phase_root,
                {
                    "event": "attempt_started",
                    "attempt": attempt_index,
                    "candidate": str(candidate),
                    "read_roots": [str(Path(root).expanduser().resolve()) for root in read_roots],
                },
            )
            agent, agent_workspace = await create_reference_agent(
                name="Data Cleaning Agent",
                system_prompt=_data_cleaning_agent_system_prompt(
                    "",
                    context,
                    pipeline_skills_enabled=self.config.pipeline_skills_enabled,
                    codegraph_enabled=self.config.codegraph_enabled,
                ),
                toolkit=toolkit,
                workspace_dir=context.workspace_dir,
                max_iters=self.config.max_iters,
            )
            response = await stream_reference_agent_reply(
                agent,
                make_user_msg(name="user", content=task_text),
                skill_usage_report_path=phase_root / "skill_usage_report.json",
                exposed_pipeline_skills=(
                    DATA_CLEANING_AGENT_PIPELINE_SKILLS
                    if self.config.pipeline_skills_enabled
                    else set()
                ),
                exposed_error_view_skills=set(),
            )
            response_text = format_message_content(getattr(response, "content", "")).strip()
            (phase_root / "agent_response.txt").write_text(response_text, encoding="utf-8")
            _append_manifest_steps_to_runtime_trace(phase_root)
            pipeline_check = _write_pipeline_skill_usage_check(
                phase_root,
                required=self.config.pipeline_skills_enabled,
            )
            if self.config.pipeline_skills_enabled and not pipeline_check["has_pipeline_skill_exposure"]:
                self._write_context_summary(phase_root, attempt_index, "needs_repair")
                raise RuntimeError(
                    "Data Cleaning Agent did not expose any pipeline AgentScope skills; "
                    f"see {phase_root / 'pipeline_skill_usage_check.json'}"
                )
            latest_completion_check: Path | None = None
            latest_package: dict[str, Path] | None = None

            def stage_and_gate(_check_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
                nonlocal latest_completion_check, latest_package
                package = _complete_reference_package_paths(phase_root, self.contract)
                if package is None:
                    latest_package = None
                    latest_completion_check = _write_reference_completion_check(phase_root, self.contract)
                    return {}, {
                        "schema_version": 1,
                        "valid": False,
                        "status": "NEEDS_REPAIR",
                        "issues": [
                            "Data Cleaning Agent did not produce a complete reference result package; "
                            f"see {latest_completion_check}"
                        ],
                    }
                latest_package = package
                _stage_workspace_pipeline(context.workspace_dir, candidate)
                staged_package = _stage_reference_agent_package(package, candidate)
                gate = _reference_candidate_quality_gate(
                    active=self.active_dir,
                    candidate=candidate,
                    contract=self.contract,
                    repair_targets=(
                        candidate / "repair_targets.json"
                        if (candidate / "repair_targets.json").is_file()
                        else None
                    ),
                )
                return staged_package, gate

            async def request_gate_repair(prompt: str, repair_index: int) -> None:
                continuation_response = await stream_reference_agent_reply(
                    agent,
                    make_user_msg(name="user", content=prompt),
                )
                continuation_text = format_message_content(
                    getattr(continuation_response, "content", "")
                ).strip()
                (phase_root / f"public_gate_repair_response_{repair_index:02d}.txt").write_text(
                    continuation_text,
                    encoding="utf-8",
                )
                _append_manifest_steps_to_runtime_trace(phase_root)

            staged, draft_gate = await _run_public_gate_repair_loop(
                candidate=candidate,
                phase_root=phase_root,
                attempt_index=attempt_index,
                max_repairs=PUBLIC_GATE_MAX_REPAIRS,
                stage_and_gate=stage_and_gate,
                request_repair=request_gate_repair,
            )
            if not staged:
                self._write_context_summary(phase_root, attempt_index, "needs_repair")
                raise RuntimeError(
                    "Data Cleaning Agent did not produce a complete reference result package; "
                    f"see {latest_completion_check or candidate / 'draft_gates'}"
                )
            staged_pipeline = candidate / "script_bundle" / "pipeline"
            persisted = persist_validated_variants(context.variant_root, candidate)
            packaged_skills = _stage_claude_style_agent_skills(
                phase_root=phase_root,
                candidate=candidate,
                package=latest_package or {},
                contract=self.contract,
            )
            adapter_bundle = _write_adapter_bundle_manifest(candidate, persisted, packaged_skills)
            staged["producer"] = "data_cleaning_agent"
            staged["engineer_phase_root"] = str(phase_root)
            staged["adapter_bundle"] = str(adapter_bundle)
            staged["pipeline"] = str(staged_pipeline) if staged_pipeline.is_dir() else ""
            staged["public_draft_gate"] = str(candidate / "draft_gates")
            staged["public_draft_gate_valid"] = bool(draft_gate.get("valid"))
            feedback_response_path = None
            if repair_targets_path is not None:
                if _latest_named_file(phase_root, "feedback_response.json") is None:
                    _append_runtime_event(
                        phase_root,
                        {
                            "event": "feedback_response_continuation_requested",
                            "attempt": attempt_index,
                            "repair_targets": str(repair_targets_path),
                        },
                    )
                    continuation = _feedback_response_continuation_prompt(
                        repair_targets_path=repair_targets_path,
                        candidate=candidate,
                    )
                    continuation_response = await stream_reference_agent_reply(
                        agent,
                        make_user_msg(name="user", content=continuation),
                    )
                    continuation_text = format_message_content(
                        getattr(continuation_response, "content", "")
                    ).strip()
                    (phase_root / "feedback_response_agent_response.txt").write_text(
                        continuation_text,
                        encoding="utf-8",
                    )
                    _append_manifest_steps_to_runtime_trace(phase_root)
                feedback_response_path = _stage_reference_feedback_response(
                    phase_root=phase_root,
                    candidate=candidate,
                    repair_targets_path=repair_targets_path,
                )
            self._write_context_summary(phase_root, attempt_index, "success")
            _append_runtime_event(
                phase_root,
                {
                    "event": "attempt_completed",
                    "attempt": attempt_index,
                    "package_manifest": staged.get("package_manifest", ""),
                    "adapter_bundle": str(adapter_bundle),
                },
            )
            _copy_runtime_reports_to_candidate(phase_root, candidate)
            delta = _write_reference_candidate_delta(
                active=self.active_dir,
                candidate=candidate,
                public_feedback=public_feedback,
                repair_targets=repair_targets_path,
                feedback_response=feedback_response_path,
            )
            if public_feedback is not None and not delta["has_material_delta"]:
                raise RuntimeError(
                    "Data Cleaning Agent produced no feedback-linked material delta; "
                    f"see {candidate / 'candidate_delta.json'}"
                )
            return staged
        finally:
            await close_reference_agent(agent, agent_workspace)
            if codegraph_runtime is not None:
                await codegraph_runtime.close()
            clear_phase_context()

    def _task_text(
        self,
        candidate: Path,
        phase_root: Path,
        report_paths: dict[str, str],
        public_feedback: Path | None,
    ) -> str:
        paths = self.contract["paths"]
        train_reference_root = Path(
            paths.get("train_reference_root") or Path(paths["train_reference"]).expanduser().resolve().parent
        ).expanduser().resolve()
        active_pipeline = self.active_dir / "script_bundle" / "pipeline"
        expected_parent_pipeline_sha256 = (
            pipeline_directory_sha256(active_pipeline) if active_pipeline.is_dir() else ""
        )
        validation_keys = Path(paths.get("validation_keys") or "")
        expected_count = _validation_key_count(self.contract)
        pipeline_learning_steps = (
            [
                "3. 处理 cohort、diagnosis、procedure、lab、medication、ICU event、clean/package 时，优先阅读或调用对应 pipeline_* skill；如果接口不匹配，记录原因后在 experiment 内创建 adapter/fork。",
                "4. 如果 train/reference 是 `preproc_*_icu.csv` 这类事件明细长表，而对应 pipeline skill 默认输出宽表或列名不匹配，说明这是接口/输出契约差异，不是逻辑不可用。",
                "5. 对接口/输出契约差异，先 inspect_skill，再创建 experiment 内 adapter/fork，复用原 pipeline skill 或其 source_pipeline_files 的核心逻辑；不要新增全局 skill，也不要直接写一个匿名大脚本绕过。",
                "6. 如果某个相关 pipeline skill 的核心逻辑确实不适用，写 no_applicable_skill_reason.json 说明缺口，再创建 adapter/fork 或 standalone 草稿。",
            ]
            if self.config.pipeline_skills_enabled
            else [
                "3. 本次是无 Pipeline Skill 消融：不得读取、调用、复制或依赖任何 pipeline_* skill。",
                "4. 只可根据 train/raw、train/reference、授权的通用代码和工具输出，自行推导文件、字段、key、join、filter、derive 与序列化规则。",
                "5. 将推导出的规则落实到 workspace/pipeline 的明确模块中；不要创建匿名的大型一次性脚本。",
            ]
        )
        lines = [
            _task_text_with_reference_contract(self.config.task_text, self.contract),
            "",
            "【Data Cleaning Agent 目标】",
            "你是一个 Codex-style 本地代码智能体。不要调用固定 workflow/executor 兜底。",
            "你需要自己读取 train/raw 与 train/reference，学习 raw -> reference package 的转换关系，",
            "写 experiment 内 adapter/fork，在 train 上回归验证，再处理 validation raw。",
            "",
            "【路径】",
            f"train raw: {paths['train_raw']}",
            f"train reference root: {train_reference_root}",
            f"validation raw: {paths['validation_raw']}",
            f"validation keys: {validation_keys if validation_keys.is_file() else ''}",
            f"candidate bundle: {candidate}",
            f"workspace: {phase_root / 'workspace'}",
            f"current best result_package (read-only baseline): {self.active_dir / 'results' / 'result_package' if (self.active_dir / 'results' / 'result_package').is_dir() else ''}",
            f"current best canonical pipeline (inherited into workspace/pipeline): {report_paths.get('current_best_pipeline', '')}",
            f"required parent_pipeline_sha256: {expected_parent_pipeline_sha256 or '<root>'}",
            "",
            "【必须先读取】",
            *[f"- {name}: {path}" for name, path in sorted(report_paths.items())],
            "",
            "【固定策略，不固定实现】",
            "1. 用 Read/Glob/InspectDataFile 理解 train/raw 和 train/reference 目录；package_manifest 如存在仅作可选索引，不是必须产物。",
            "2. 分析 raw 到 reference 的文件、字段、key、join、filter、derive、clean 关系。",
            *pipeline_learning_steps,
            "7. 创建或更新 workspace/pipeline 下的累计 Pipeline；唯一入口必须是 workspace/pipeline/run.py。",
            "7a. 本轮不强制包装 Skill。只有当你已经自然整理好稳定能力时，才可选写 workspace/skill_packaging_plan.json 和 workspace/capabilities/<skill_name>/；缺失或格式不完整不得阻塞 result_package 进入评估。",
            "8. 必须用 workspace/pipeline/run.py 在 train raw 上生成预测 reference package，并与 train/reference 做回归检查；Agent 报告只作参考，宿主会在隔离进程重新生成机器报告。",
            "9. train 回归通过或明确 blocked 后，才处理 validation raw。",
            "10. validation 阶段不能读取 hidden validation reference/private report。",
            "11. ExecutePython 只能写本步骤 OUTPUT_DIR，绝不能直接写 workspace/result_package。候选完整 result_package 必须由 workspace/pipeline/run.py 生成，不得执行后手工修改任何业务 CSV。",
            "12. 每轮结束前必须用 PublishDirectoryArtifact 将 OUTPUT_DIR/result_package 发布为 workspace/result_package；宿主只会评估这个已发布的完整候选包。",
            "13. 一个 attempt 内可做多次局部修改和 train 检查；未涉及 repair target 的现有业务文件必须与 current best 保持字节一致。只有门禁通过的候选才会隐藏评估，且 composite_score 严格提升才算完成正式 loop。",
            "13a. 宿主可能在同一个 attempt 内返回公开预提交门禁问题；收到后必须继续修改当前 workspace 中的累计 Pipeline，重新运行、发布并验证完整结果包，不得另起一套规则。",
            "14. workspace/pipeline 必须包含 pipeline_manifest.json、run.py、cohort.py、labels.py、chart.py、diag.py、med.py、out.py、proc.py、summary.py、config.yaml。manifest 必须声明完整业务文件、固定模块以及本轮 required parent_pipeline_sha256。",
            "14a. summary 的实现方式由你自主决定。若 raw 无法稳定推导 train/reference 中的 summary，你可以把从公开 train/reference 学到的 summary 层信息沉淀到 workspace/pipeline/summary_assets/*.csv，并在 manifest 的 learned_summary_assets 中逐项声明 source、path、sha256；宿主会验证它与 train/reference 对应 summary 文件完全一致。不要为了使用该能力机械复制 summary，能够从 raw 稳定计算时仍可直接计算。",
            "15. 后续正式 Round 必须继承 workspace 中已有 Pipeline，只修改反馈涉及模块；入口仍必须生成全部17个业务文件。缺入口、manifest、lineage 或宿主可重放能力时不得返回 SUCCESS。",
            "16. Pipeline 源文件不得引用或写入 active bundle、validation result、private reference、test evaluation 或 private report；这些禁用来源标记也不要出现在 Pipeline 代码、配置和注释中。除已声明且经宿主核验的 summary_assets 外，严禁沉淀 stay_id、subject_id、hadm_id、label、患者级业务行或其他结果数据。",
            "",
            "【最终必须生成的核心结果】",
            "- result_package/ 目录",
            "- result_package 下与 train/reference 同构的 cohort/、features/ 等子文件",
            "- pipeline/ 目录及唯一入口 pipeline/run.py；调用接口固定为 python pipeline/run.py --raw-root <raw> --output-dir <output> --split-mode train|validation|test",
            "- train_regression_report.json，格式必须包含 schema_version=2、status=SUCCESS、files 列表；files 每项必须包含 relative_path、train_rows、reference_rows、column_coverage、key_coverage、value_recall、布尔值 passed 和 failure_reason，并覆盖本轮修改的每个业务文件",
            "- 不要求生成 reference.csv；不要为了凑 reference.csv 把多文件 reference 强行合成一张表",
            "",
            "【可选运行记录】",
            "- package_manifest.json 可作为目录索引/审计文件生成，但不是成功的必要条件。",
            "- reference_shape_contract.json、skill_usage_report.json 可生成，但不要为了补这些可选文件阻塞核心 result_package。",
            "- 如需修改 pipeline skill 行为，只能落实到 workspace/pipeline 对应模块；散落在 workspace 其他位置的临时脚本不会成为 canonical Pipeline。",
            "- Skill 包装是验证 loop 收敛后的独立冻结步骤；本轮如创建了 workspace/capabilities 或 skill_packaging_plan.json，只作为可选记录，不作为成功条件。",
            "- 如果生成 package_manifest，不要把 source/source_reference_root/primary_reference 外部路径写入最终结果包。",
            "",
            f"ValidateResultPackage 的 expected_key_count 应使用 {expected_count}。",
        ]
        if public_feedback is not None:
            repair_targets = report_paths.get("repair_targets", "")
            lines.extend(
                [
                    "",
                    f"【当前 best public feedback】必须读取并据此产生 material delta: {public_feedback}",
                    f"【当前 attempt repair targets】必须逐条回应: {repair_targets}",
                    "必须在 workspace 或当前运行目录生成 feedback_response.json，格式为：",
                    "{",
                    '  "responses": [',
                    '    {"repair_target_id": "...", "decision": "rule_update|adapter_update|standalone_fix|mapping_update|unsupported_reason|no_action_reason", "reason": "...", "changed_files": ["results/result_package/..."]}',
                    "  ]",
                    "}",
                    "每个 repair_target_id 都必须有一条 response；如果不修，必须写 no_action_reason 和明确原因。",
                    "changed_files 必须使用 candidate bundle 相对路径，并指向真实发生变化的能力、mapping、规则或 result_package 文件。",
                ]
            )
        if report_paths.get("previous_attempt_outcome"):
            lines.extend(
                [
                    "",
                    f"【上一候选失败记录】仅作为负面经验读取，不得替代当前 best repair targets: {report_paths['previous_attempt_outcome']}",
                    f"【上一候选反馈】仅用于避免重复失败，不是当前基线反馈: {report_paths.get('previous_attempt_feedback_negative_evidence', '')}",
                ]
            )
        if report_paths.get("test_pipeline_rule_failure"):
            lines.extend(
                [
                    "",
                    f"【Test Pipeline 公开失败反馈】{report_paths['test_pipeline_rule_failure']}",
                    f"该反馈已合并进本轮 repair targets：{report_paths.get('repair_targets', '')}",
                    "只能继承当前 best Pipeline 修复通用业务规则；不得读取 test reference 或评分。",
                    "修复后仍必须让 train/validation 重放、业务变化和严格提分规则全部通过。",
                ]
            )
        return "\n".join(lines)

    def _write_context_summary(self, phase_root: Path, attempt_index: int, status: str) -> None:
        paths = self.contract["paths"]
        workspace = phase_root / "workspace"
        result_package = _latest_result_package_dir_for_summary(phase_root)
        package_manifest = result_package / "package_manifest.json" if result_package else phase_root / "artifacts"
        summary_lines = [
            "# Data Cleaning Agent Context",
            "",
            "## Current State",
            f"- attempt: {attempt_index}",
            f"- status: {status}",
            f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Authorized Inputs",
            f"- train_raw: {paths['train_raw']}",
            f"- train_reference_root: {paths.get('train_reference_root') or Path(paths['train_reference']).parent}",
            f"- validation_raw: {paths['validation_raw']}",
            f"- validation_keys: {paths.get('validation_keys', '')}",
            "- validation_reference_visible_to_agent: no",
            "",
            "## Runtime State Files",
            f"- workspace: {workspace}",
            f"- runtime_trace: {phase_root / 'runtime_trace.jsonl'}",
            f"- manifest: {phase_root / 'manifest.json'}",
            f"- context_report: {phase_root / 'context' / 'context_usage_report.json'}",
            f"- result_package_validation_report: {workspace / 'result_package_validation_report.json'}",
            "",
            "## Latest Result Package",
            f"- result_package: {result_package or ''}",
            f"- package_manifest_optional: {package_manifest if result_package and package_manifest.is_file() else ''}",
            "",
            "## Resume Protocol",
            "- Do not restart from scratch after context compaction.",
            "- First read this file, then inspect the tail of runtime_trace.jsonl and the latest validation/comparison report.",
            "- Continue from the most recent failing artifact, script, manifest, or directory comparison.",
            "- For reference_directory tasks, the target is directory isomorphism with train/reference: same relative CSV files and compatible columns.",
            "- Do not use validation reference/private reports inside the agent runtime.",
        ]
        summary = "\n".join(summary_lines) + "\n"
        (phase_root / "context_summary.md").write_text(summary, encoding="utf-8")

    def _persist_attempt_state(
        self,
        state: dict[str, Any],
        candidate: Path,
        gate: dict[str, Any],
    ) -> None:
        _write_json(self.state_path, state)
        latest = (state.get("attempts") or [{}])[-1]
        _write_json(
            candidate / "attempt_outcome.json",
            {
                "schema_version": 1,
                "attempt": latest.get("attempt", 0),
                "status": latest.get("status", "invalid"),
                "valid": latest.get("valid", False),
                "score": latest.get("score"),
                "improved": latest.get("improved", False),
                "best_score": state.get("best_score", 0.0),
                "best_round": state.get("best_round", 0),
                "best_feedback": state.get("best_feedback", ""),
                "consecutive_valid_no_improvement": state.get(
                    "consecutive_valid_no_improvement", 0
                ),
                "evaluation": latest.get("evaluation") or {},
                "gate": gate,
            },
        )

    def _checkpoint_and_test(self, reason: str, state: dict[str, Any]) -> dict[str, Any]:
        state = _migrate_reference_experiment_state(state)
        state["termination_reason"] = reason
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        best_round = int(state.get("best_round") or 0)
        best_attempt = int(state.get("best_attempt") or 0)
        if best_round < 1 or best_attempt < 1:
            state["status"] = "checkpointed_without_best"
            _write_json(self.state_path, state)
            return state

        state, canonical_pipeline, formal_bundle = self._ensure_canonical_pipeline(state)
        if canonical_pipeline is None or formal_bundle is None:
            state["status"] = "pipeline_consolidation_failed"
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(self.state_path, state)
            return state

        checkpoint_index = _next_checkpoint_index(self.test_checkpoints_dir, state)
        checkpoint_dir = (
            self.test_checkpoints_dir
            / f"checkpoint_{checkpoint_index:04d}_best_round_{best_round:04d}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        state["status"] = "checkpointing"
        state["pending_checkpoint"] = str(checkpoint_dir)
        _write_json(self.state_path, state)
        frozen_script_bundle = checkpoint_dir / "frozen_script_bundle"
        frozen_bundle = self.experiment_dir / "frozen_bundle"
        fingerprint = ""
        try:
            freeze_source = formal_bundle
            formal_pipeline = formal_bundle / "pipeline"
            if not formal_pipeline.is_dir() or formal_pipeline.resolve() != canonical_pipeline.resolve():
                freeze_source = checkpoint_dir / ".canonical_script_bundle_source"
                shutil.copytree(formal_bundle, freeze_source)
                shutil.rmtree(freeze_source / "pipeline", ignore_errors=True)
                shutil.copytree(canonical_pipeline, freeze_source / "pipeline")
            _replace_directory_atomically(freeze_source, frozen_script_bundle)
            if freeze_source != formal_bundle:
                shutil.rmtree(freeze_source, ignore_errors=True)
            _replace_directory_atomically(self.active_dir, frozen_bundle)
            fingerprint = _checkpoint_fingerprint(
                frozen_script_bundle,
                self.contract,
                scorer_version=SCORER_VERSION,
            )
            prior = next(
                (
                    item
                    for item in reversed(state.get("checkpoints") or [])
                    if item.get("fingerprint") == fingerprint
                    and item.get("test_status") in {"success", "reused"}
                ),
                None,
            )
            if prior is not None:
                report = {
                    "schema_version": 1,
                    "workflow": "reference-checkpoint-test",
                    "test_status": "reused",
                    "reused_from": prior.get("checkpoint_dir", ""),
                    "fingerprint": fingerprint,
                    "best_round": best_round,
                    "best_attempt": best_attempt,
                    "validation_best_score": state.get("best_score", 0.0),
                    "metrics": prior.get("metrics") or {},
                    "evaluation": prior.get("evaluation") or {},
                }
                report_path = checkpoint_dir / "checkpoint_report.json"
                _write_json(report_path, report)
                report["report_path"] = str(report_path)
            else:
                from workflow.reference_test_stage import (
                    ReferenceCheckpointTestConfig,
                    ReferenceCheckpointTestRuntime,
                )

                state["status"] = "testing"
                _write_json(self.state_path, state)
                report = ReferenceCheckpointTestRuntime(
                    ReferenceCheckpointTestConfig(
                        dataset_split=self.config.dataset_split,
                        experiment_dir=self.experiment_dir,
                        checkpoint_dir=checkpoint_dir,
                        frozen_script_bundle=frozen_script_bundle,
                        best_round=best_round,
                        best_attempt=best_attempt,
                        best_score=float(state.get("best_score") or 0.0),
                        max_iters=self.config.max_iters,
                    )
                ).run_sync()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            report = {
                "schema_version": 1,
                "workflow": "reference-checkpoint-test",
                "test_status": "test_failed",
                "best_round": best_round,
                "best_attempt": best_attempt,
                "validation_best_score": state.get("best_score", 0.0),
                "metrics": {},
                "evaluation": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
            report_path = checkpoint_dir / "checkpoint_report.json"
            _write_json(report_path, report)
            report["report_path"] = str(report_path)

        checkpoint_record = {
            "checkpoint": checkpoint_index,
            "checkpoint_dir": str(checkpoint_dir),
            "fingerprint": fingerprint,
            "scorer_version": SCORER_VERSION,
            "best_round": best_round,
            "best_attempt": best_attempt,
            "validation_best_score": state.get("best_score", 0.0),
            "test_status": report.get("test_status", "test_failed"),
            "report_path": report.get("report_path", str(checkpoint_dir / "checkpoint_report.json")),
            "metrics": report.get("metrics") or {},
            "evaluation": report.get("evaluation") or {},
            "reused_from": report.get("reused_from", ""),
        }
        state.setdefault("checkpoints", []).append(checkpoint_record)
        if report.get("test_status") == "pipeline_rule_failure":
            feedback_path = checkpoint_dir / "test_pipeline_rule_feedback.json"
            gate = report.get("gate") or {}
            execution = gate.get("execution") or {}
            execution_failure = {
                key: execution.get(key)
                for key in (
                    "status",
                    "returncode",
                    "timed_out",
                    "resource_limit_reason",
                    "error",
                )
                if execution.get(key) not in (None, "")
            }
            log_path_text = str(execution.get("log_path") or "")
            if log_path_text:
                execution_failure["log_tail"] = _read_public_log_tail(
                    Path(log_path_text),
                    max_chars=12000,
                )
            _write_json(
                feedback_path,
                {
                    "schema_version": 1,
                    "status": "PIPELINE_RULE_FAILURE",
                    "source": "host_test_business_gate",
                    "checkpoint": checkpoint_index,
                    "best_round": best_round,
                    "pipeline_sha256": state.get("pipeline_sha256", ""),
                    "issues": gate.get("issues") or [],
                    "diagnostics": gate.get("diagnostics") or [],
                    "execution_failure": execution_failure,
                    "instruction": (
                        "Update the inherited validation pipeline rules and replay train/validation; "
                        "do not inspect test reference or test evaluation."
                    ),
                },
            )
            state["status"] = "validation_resume_required"
            state["pending_test_rule_feedback"] = str(feedback_path)
            state["force_validation_attempt"] = True
            state["consecutive_valid_no_improvement"] = 0
        else:
            state["status"] = "checkpointed"
        state.pop("pending_checkpoint", None)
        state["latest_checkpoint"] = str(checkpoint_dir)
        state["frozen_bundle"] = str(frozen_bundle) if frozen_bundle.is_dir() else ""
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(self.state_path, state)
        return state

    def _ensure_canonical_pipeline(
        self,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], Path | None, Path | None]:
        best_round = int(state.get("best_round") or 0)
        formal_bundle = (
            self.experiment_dir
            / "rounds"
            / f"round_{best_round:04d}"
            / "script_bundle"
        )
        if not formal_bundle.is_dir():
            formal_bundle = self.active_dir / "script_bundle"
        if not formal_bundle.is_dir():
            state["pipeline_consolidation"] = {
                "status": "FAILED",
                "issues": [f"formal best script bundle is missing: {formal_bundle}"],
            }
            return state, None, None

        state_pipeline_text = str(state.get("canonical_pipeline") or "")
        state_pipeline = Path(state_pipeline_text).expanduser().resolve() if state_pipeline_text else None
        formal_pipeline = formal_bundle / "pipeline"
        pipeline_source = ""
        if formal_pipeline.is_dir():
            canonical = formal_pipeline.resolve()
            pipeline_source = "formal_round"
        elif state_pipeline is not None and state_pipeline.is_dir():
            canonical = state_pipeline
            pipeline_source = "legacy_consolidation"
        else:
            canonical = self._consolidate_legacy_pipeline(state, best_round)
            if canonical is None:
                return state, None, formal_bundle
            pipeline_source = "legacy_consolidation"

        entrypoint = canonical / "run.py"
        manifest = canonical / "pipeline_manifest.json"
        if not entrypoint.is_file() or not manifest.is_file():
            state["pipeline_consolidation"] = {
                "status": "FAILED",
                "issues": ["canonical pipeline is missing run.py or pipeline_manifest.json"],
            }
            return state, None, formal_bundle
        expected_files = sorted(
            path.relative_to(Path((self.contract.get("paths") or {})["train_reference_root"])).as_posix()
            for path in _structured_package_files(
                Path((self.contract.get("paths") or {})["train_reference_root"])
            )
            if path.name not in IGNORED_PACKAGE_FILES
        )
        manifest_payload = _load_json(manifest)
        structure = validate_pipeline_structure(
            canonical,
            expected_files=expected_files,
            expected_parent_sha256=str(manifest_payload.get("parent_pipeline_sha256") or ""),
            train_reference_root=(self.contract.get("paths") or {})["train_reference_root"],
        )
        integrity_issues = list(structure.get("issues") or [])
        current_pipeline_hash = pipeline_directory_sha256(canonical)
        current_entrypoint_hash = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
        if pipeline_source == "formal_round":
            provenance_path = formal_bundle.parent / "provenance.json"
            provenance = _load_json(provenance_path) if provenance_path.is_file() else {}
            if not provenance:
                integrity_issues.append("formal round provenance.json is missing")
            if str(provenance.get("pipeline_sha256") or "") != current_pipeline_hash:
                integrity_issues.append("formal pipeline hash differs from promotion provenance")
            if str(provenance.get("entrypoint_sha256") or "") != current_entrypoint_hash:
                integrity_issues.append("formal entrypoint hash differs from promotion provenance")
            if str(provenance.get("script_bundle_sha256") or "") != _directory_sha256(formal_bundle):
                integrity_issues.append("formal script bundle hash differs from promotion provenance")
            replay_report = formal_bundle.parent / "host_replay" / "pipeline_replay_report.json"
            replay = _load_json(replay_report) if replay_report.is_file() else {}
            if not replay or not replay.get("valid") or replay.get("status") != "SUCCESS":
                integrity_issues.append("formal host replay report is missing or unsuccessful")
        else:
            consolidation = state.get("pipeline_consolidation") or {}
            if (
                consolidation.get("status") != "SUCCESS"
                or not consolidation.get("reproducible")
                or str(consolidation.get("pipeline_sha256") or "") != current_pipeline_hash
                or str(consolidation.get("entrypoint_sha256") or "") != current_entrypoint_hash
            ):
                integrity_issues.append("legacy consolidation integrity record does not match pipeline")
        if integrity_issues:
            state["checkpoint_pipeline_validation"] = {
                "status": "FAILED",
                "pipeline": str(canonical),
                "issues": integrity_issues,
            }
            state["reproducible"] = False
            return state, None, formal_bundle
        state["canonical_pipeline"] = str(canonical)
        state["pipeline_sha256"] = current_pipeline_hash
        state["entrypoint_sha256"] = current_entrypoint_hash
        state["reproducible"] = True
        return state, canonical, formal_bundle

    def _consolidate_legacy_pipeline(
        self,
        state: dict[str, Any],
        best_round: int,
    ) -> Path | None:
        round_dir = self.experiment_dir / "rounds" / f"round_{best_round:04d}"
        consolidation_root = round_dir / "pipeline_consolidation"
        consolidation_root.mkdir(parents=True, exist_ok=True)
        report_path = consolidation_root / "consolidation_report.json"
        canonical = consolidation_root / "pipeline"
        try:
            generated = self._run_pipeline_consolidation_agent(
                consolidation_root,
                best_round,
            )
            if canonical.exists():
                shutil.rmtree(canonical)
            shutil.copytree(generated, canonical)
            paths = self.contract.get("paths") or {}
            replay_gate = validate_and_replay_candidate_pipeline(
                pipeline_dir=canonical,
                previous_pipeline_dir=None,
                candidate_result_package=round_dir / "validation_result",
                train_raw=paths["train_raw"],
                train_reference_root=paths["train_reference_root"],
                train_keys=paths["train_keys"],
                validation_raw=paths["validation_raw"],
                validation_keys=paths["validation_keys"],
                key_column=str(self.contract.get("key_column") or "stay_id"),
                report_root=consolidation_root / "host_replay",
            )
            reproducible = bool(replay_gate.get("valid"))
            report = {
                "schema_version": 1,
                "status": "SUCCESS" if reproducible else "FAILED",
                "best_round": best_round,
                "best_score": state.get("best_score", 0.0),
                "canonical_pipeline": str(canonical),
                "pipeline_sha256": pipeline_directory_sha256(canonical),
                "entrypoint_sha256": (
                    hashlib.sha256((canonical / "run.py").read_bytes()).hexdigest()
                    if (canonical / "run.py").is_file()
                    else ""
                ),
                "reproducible": reproducible,
                "host_replay_report": str(
                    consolidation_root / "host_replay/pipeline_replay_report.json"
                ),
                "issues": replay_gate.get("issues") or [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            _write_json(report_path, report)
            state["pipeline_consolidation"] = report
            if not reproducible:
                state["reproducible"] = False
                return None
            return canonical
        except Exception as exc:
            report = {
                "schema_version": 1,
                "status": "FAILED",
                "best_round": best_round,
                "best_score": state.get("best_score", 0.0),
                "canonical_pipeline": "",
                "pipeline_sha256": "",
                "entrypoint_sha256": "",
                "reproducible": False,
                "issues": [f"{type(exc).__name__}: {exc}"],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            _write_json(report_path, report)
            state["pipeline_consolidation"] = report
            state["reproducible"] = False
            return None

    def _run_pipeline_consolidation_agent(
        self,
        consolidation_root: Path,
        best_round: int,
    ) -> Path:
        phase = init_phase_session(consolidation_root / "agent_runs", "data_cleaning_agent")
        phase_root = Path(phase["phase_root"])
        run_id = f"reference_code_pipeline_consolidation_round_{best_round:04d}"
        set_phase_context(
            run_id,
            consolidation_root / "agent_runs",
            "data_cleaning_agent",
            phase_root,
        )
        paths = self.contract.get("paths") or {}
        round_bundles = [
            self.experiment_dir / "rounds" / f"round_{index:04d}" / "script_bundle"
            for index in range(1, best_round + 1)
        ]
        round_bundles = [path for path in round_bundles if path.is_dir()]
        expected_files = sorted(
            path.relative_to(Path(paths["train_reference_root"])).as_posix()
            for path in _structured_package_files(Path(paths["train_reference_root"]))
            if path.name not in IGNORED_PACKAGE_FILES
        )
        workspace = phase_root / "workspace"
        bundle_list = "\n".join(f"- {path}" for path in round_bundles)
        manifest_contract = {
            "schema_version": 1,
            "entrypoint": "run.py",
            "modules": list(PIPELINE_MODULES),
            "expected_files": expected_files,
            "parent_pipeline_sha256": "",
        }
        task_text = f"""\
请把当前同一次实验 Round 1 到 Round {best_round} 已成功晋升的脚本规则整理成唯一累计 Pipeline。

你只能读取下面列出的正式 script_bundle、train/raw、train/reference 和 validation/raw；不得读取任何 validation_result、active_bundle、private reference、评分报告或其他实验目录。

【唯一可写 workspace】
{workspace}

【正式脚本包，只读】
{bundle_list}

【公开数据，只读】
- train/raw: {paths['train_raw']}
- train/reference: {paths['train_reference_root']}
- validation/raw: {paths['validation_raw']}

【工具纪律】
- 使用 Read/Glob/Grep 时必须传上面列出的绝对路径，不得猜测项目根目录或 `/workspace`。
- 使用 Write/Edit 时只能写 {workspace}。
- ExecutePython 只能执行已经先用 Write 创建在 {workspace} 内的 Python 文件。
- 读取 Markdown 或代码必须使用 Read，不要用 ExecutePython。

输出只能写到 workspace/pipeline，固定包含：
- pipeline_manifest.json
- run.py
- cohort.py, labels.py, chart.py, diag.py, med.py, out.py, proc.py, summary.py
- config.yaml

除可选的 `summary_assets/*.csv` 外，workspace/pipeline 内不得出现其他文件或子目录。尤其禁止 `_static_summary/`、其他 CSV/TSV/Parquet/JSONL、第二个入口、临时诊断脚本或缓存文件。
Pipeline 的代码、配置和注释中也不得出现 active bundle、validation result、private reference、test evaluation 或 private report 等禁用来源标记。

固定接口：
python pipeline/run.py --raw-root <raw> --output-dir <output> --split-mode train|validation|test

pipeline_manifest.json 必须包含以下字段、字段名和值；不要使用 output_files，不要把 run.py 或 config.yaml 放进 modules：
{json.dumps(manifest_contract, ensure_ascii=False, indent=2)}

summary 的实现方式由你自主判断。若 raw 无法稳定推导公开 train/reference 中的 summary，可以把必要的 summary 层学习结果写入 `summary_assets/*.csv`，并在 manifest 增加 `learned_summary_assets` 列表；每项只包含 `source`、`path`、`sha256`，且 source 必须是对应的 `summary/*.csv`。宿主会核验资产与 train/reference 完全一致。

除上述受控 summary_assets 外，不得把 train/reference 的业务数据、历史结果 CSV 或 validation 输出复制或转写进 Pipeline，也不得把这些数据伪装进 Python/YAML。严禁硬编码 stay_id、subject_id、hadm_id、label 或患者级业务行。最终17个文件必须由同一入口生成；你可以在 workspace 临时诊断，但最终只有 workspace/pipeline 会被宿主收取并隔离重放。
"""
        reports = {
            "train_reference_root": str(paths["train_reference_root"]),
            **{
                f"formal_round_{index:04d}_script_bundle": str(bundle)
                for index, bundle in enumerate(round_bundles, start=1)
            },
        }
        code_read_roots = reference_code_read_roots(
            include_pipeline_skills=True,
            include_error_view_skills=False,
        )
        read_roots: list[str | Path] = [
            *code_read_roots,
            *round_bundles,
            paths["train_raw"],
            paths["train_reference_root"],
            paths["validation_raw"],
            consolidation_root,
        ]
        try:
            context = EngineerToolContext.from_task(
                task_text=task_text,
                engineer_phase_root=phase_root,
                explorer_phase_root=str(consolidation_root),
                additional_read_roots=read_roots,
                denied_read_roots=reference_code_denied_read_roots(
                    include_pipeline_skills=True,
                    include_error_view_skills=False,
                ),
                require_skill_plan=False,
                split_mode="validation",
                required_report_paths=reports,
            )
            read_roots_text = "\n".join(
                f"- {Path(root).expanduser().resolve()}" for root in read_roots
            )
            codegraph_guidance = (
                "5. 理解授权项目源码时优先使用 CodeGraphExplore；返回 stale、DENIED 或 ERROR 时改用 Read/Grep。\n"
                if self.config.codegraph_enabled
                else ""
            )
            system_prompt = f"""\
你是 Data Cleaning Agent 的一次性 pipeline consolidation 上下文。
只整理本次实验已晋升脚本，不创建新业务规则，不读取隐藏结果。

workspace: {context.workspace_dir}

授权只读根：
{read_roots_text}

工具规则：
1. Read/Glob/Grep 只能访问授权只读根或 workspace，并使用提示中的真实绝对路径。
2. Write/Edit 只能写 workspace；ExecutePython 只能执行 workspace 内已存在的 .py 文件。
3. 不得搜索项目根、experiments 总目录、active_bundle、validation_result 或 private reference。
4. 最终必须生成 {context.workspace_dir / 'pipeline'}；是否可复现由宿主机器验收。
{codegraph_guidance}
"""

            async def run_agent():
                codegraph_runtime: CodeGraphRuntime | None = None
                agent = None
                workspace = None
                try:
                    if self.config.codegraph_enabled:
                        codegraph_runtime = await open_codegraph_runtime(
                            project_root=Path(__file__).resolve().parent.parent,
                            allowed_roots=code_read_roots,
                            usage_path=phase_root / "codegraph_usage.jsonl",
                            audit_path=context.workspace_dir / "tool_audit.jsonl",
                            attempt=f"consolidation_round_{best_round:04d}",
                        )
                    toolkit, _ = create_reference_toolkit(
                        context,
                        codegraph_tool=(
                            codegraph_runtime.tool if codegraph_runtime is not None else None
                        ),
                    )
                    _remove_fixed_reference_incompatible_tools(toolkit)
                    agent, workspace = await create_reference_agent(
                        name="Data Cleaning Agent",
                        system_prompt=system_prompt,
                        toolkit=toolkit,
                        workspace_dir=context.workspace_dir,
                        max_iters=self.config.max_iters,
                    )
                    return await stream_reference_agent_reply(
                        agent,
                        make_user_msg(name="user", content=task_text),
                    )
                finally:
                    await close_reference_agent(agent, workspace)
                    if codegraph_runtime is not None:
                        await codegraph_runtime.close()

            response = asyncio.run(run_agent())
            (phase_root / "agent_response.txt").write_text(
                format_message_content(getattr(response, "content", "")).strip(),
                encoding="utf-8",
            )
            pipeline = context.workspace_dir / "pipeline"
            if not pipeline.is_dir():
                raise RuntimeError("consolidation Agent did not create workspace/pipeline")
            return pipeline
        finally:
            clear_phase_context()

    def _result(self, state: dict[str, Any]) -> dict[str, Any]:
        latest_attempt = (state.get("attempts") or [{}])[-1]
        latest_round = (state.get("rounds") or [{}])[-1]
        frozen = self.experiment_dir / "frozen_bundle"
        latest_checkpoint = (state.get("checkpoints") or [{}])[-1]
        return {
            "status": state.get("status", "active"),
            "termination_reason": state.get("termination_reason", ""),
            "experiment_dir": str(self.experiment_dir),
            "best_score": state.get("best_score", 0.0),
            "best_round": state.get("best_round", 0),
            "best_attempt": state.get("best_attempt", 0),
            "round_count": len(state.get("rounds", [])),
            "attempt_count": len(state.get("attempts", [])),
            "invalid_attempt_count": state.get("invalid_attempt_count", 0),
            "consecutive_valid_no_improvement": state.get(
                "consecutive_valid_no_improvement", 0
            ),
            "latest_round": latest_round,
            "latest_attempt": latest_attempt,
            "active_bundle": str(self.active_dir),
            "frozen_bundle": str(frozen) if frozen.is_dir() else "",
            "checkpoint_count": len(state.get("checkpoints", [])),
            "latest_checkpoint": latest_checkpoint,
            "test_status": latest_checkpoint.get("test_status", ""),
            "codegraph_usage": {
                "enabled": self.config.codegraph_enabled,
                **summarize_codegraph_usage(self.experiment_dir),
            },
        }


def _data_cleaning_agent_system_prompt(
    skill_manifest_text: str,
    context: EngineerToolContext,
    *,
    pipeline_skills_enabled: bool = True,
    codegraph_enabled: bool = False,
) -> str:
    read_roots = "\n".join(f"- {path}" for path in context.read_roots)
    reports = "\n".join(f"- {name}: {path}" for name, path in sorted(context.required_report_paths.items()))
    codegraph_guidance = (
        """
# CodeGraph

理解已索引的授权项目源码和调用链时，优先使用 CodeGraphExplore。
数据文件、配置、文档和当前 attempt workspace 继续使用现有 Read/Grep/InspectDataFile。
CodeGraphExplore 返回 stale、DENIED 或 ERROR 时，使用 Read/Grep 检查对应授权文件。
不得用 CodeGraph 寻找历史实验、隐藏答案、其他项目或被关闭的 Skill。
"""
        if codegraph_enabled
        else ""
    )
    return f"""\
你是 Data Cleaning Agent，一个 Codex-style 本地代码智能体。

# Skill 策略

当前训练/验证 loop 必须维护 workspace/pipeline 下的累计 canonical Pipeline。
你可以读取授权的 skills/lib/workflow 代码作为参考，但候选 result_package 必须由 workspace/pipeline/run.py 生成。
Prompt 只负责指导；宿主会隔离重放 Pipeline，并逐文件比较业务 CSV 的 SHA-256。

# 原子工具

- Read / Glob / Grep：发现和读取授权文件。
- InspectDataFile：检查结构化文件 schema、样本、行数和缺失率。
- Write / Edit：只在当前 workspace 写脚本、adapter 和报告。
- ExecutePython：只执行 workspace 内 Python，OUTPUT_DIR 是本次 step 唯一输出目录。
- CompareArtifact：只用于公开 train reference 回归对比。
- ValidateResultPackage：只做 validation 结果包自洽检查，不读取 hidden reference。
- PublishDirectoryArtifact：把 step OUTPUT_DIR 中的 result_package 目录发布到 workspace。
{codegraph_guidance}

# 工作区

- workspace: `{context.workspace_dir}`
- phase root: `{context.engineer_phase_root}`

# 必读文件

{reports}

# 授权只读根

{read_roots}

# 执行纪律

1. 必须先读取 reference_contract，并用 Glob/InspectDataFile 探索 train/reference 目录；package_manifest 如存在只能作为可选结构索引，不是必须产物。
2. 分析 raw 到 reference 的文件、字段、key、join、filter、derive、clean 关系。
3. 如果 train/reference 中出现 `preproc_chart_icu.csv`、`preproc_med_icu.csv`、`preproc_out_icu.csv`、`preproc_proc_icu.csv`、`preproc_diag_icu.csv`、cohort 明细文件，直接学习并生成同构目录包；不要强行合并成宽表。
4. {"如果现有 pipeline skill 或其接口不匹配 train/reference 形态，允许在当前 experiment 内创建 adapter/fork；不要新增全局 skill。" if pipeline_skills_enabled else "本次是无 Pipeline Skill 消融运行：不得读取、调用或依赖 pipeline_* Skill；只可根据 train/raw、train/reference 和通用代码自行推导规则。"}
5. Round 1 创建完整 Pipeline；后续 Round 继承上一正式 best Pipeline，只修改反馈涉及模块。不得以 build_package.py、standalone 脚本或手工 CSV 替代 canonical Pipeline。
6. 原始 skills/、lib/、workflow/ 和 teacher pipeline 始终只读；需要实现时只在 workspace 写脚本或 adapter。
7. 不要调用 execute_current_extraction_task；不要调用 finalize_result_package；不要把固定 workflow 当兜底。
8. train reference 是公开示例，可用 CompareArtifact 回归；validation reference/private report 禁止读取。
9. 最终核心产物是完整 result_package、workspace/pipeline 和可选的 Agent train_regression_report.json；最终 train regression 由宿主重放生成，不信任文字总结。
10. ExecutePython 绝不能直接写回 workspace/result_package。必须调用 workspace/pipeline/run.py 在 step OUTPUT_DIR 生成完整包，然后用 PublishDirectoryArtifact 发布整个目录；发布后不得手工修改业务 CSV。
11. validation result_package 必须通过 ValidateResultPackage，或至少生成可由宿主基础校验的同构目录文件；不要求 reference.csv 或 package_manifest.json。
12. 未涉及本轮 repair target 的 current best 业务文件不得改变；只改 manifest、报告或 adapter 而不改反馈相关业务文件的候选不会进入隐藏评估。
13. pipeline_manifest.json 必须声明唯一入口 run.py、固定模块、完整17文件和宿主给出的 parent_pipeline_sha256。没有完整 result_package、manifest、唯一入口或可重放 Pipeline 时不得返回 SUCCESS。
14. summary 的实现方式由你自主决定；若必须沉淀公开 train/reference 的 summary 层学习结果，只能使用 manifest 已声明的 summary_assets/*.csv。严禁沉淀患者标识、label、患者级业务行或任何 validation/test private 信息。
"""


def _remove_fixed_reference_incompatible_tools(toolkit) -> None:
    # The parity Toolkit is constructed from an explicit allowlist, so none of
    # the fixed extraction workflow tools can be present.
    return None


def _append_runtime_event(phase_root: Path, payload: dict[str, Any]) -> None:
    path = phase_root / "runtime_trace.jsonl"
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _append_manifest_steps_to_runtime_trace(phase_root: Path) -> None:
    manifest_path = phase_root / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _load_json(manifest_path)
    for step in manifest.get("steps") or []:
        if isinstance(step, dict):
            _append_runtime_event(phase_root, {"event": "tool_step", "step": step})


def _pipeline_skill_call_counts(phase_root: Path) -> dict[str, int]:
    trace_path = phase_root / "runtime_trace.jsonl"
    counts: dict[str, int] = {}
    if not trace_path.is_file():
        return counts
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        step = event.get("step") if isinstance(event, dict) else None
        if not isinstance(step, dict):
            continue
        skill_name = str(step.get("skill_name") or "")
        if skill_name.startswith("pipeline_"):
            counts[skill_name] = counts.get(skill_name, 0) + 1
    return dict(sorted(counts.items()))


def _write_pipeline_skill_usage_check(phase_root: Path, *, required: bool = True) -> dict[str, Any]:
    counts = _pipeline_skill_call_counts(phase_root)
    prompt_path = phase_root / "agentscope_skill_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
    expected = sorted(DATA_CLEANING_AGENT_PIPELINE_SKILLS)
    registered_agent_skills = [name for name in expected if name in prompt]
    has_agent_skills = bool(registered_agent_skills)
    has_any_exposure = bool(counts) or has_agent_skills
    payload = {
        "schema_version": 1,
        "status": "SUCCESS" if has_any_exposure or not required else "NEEDS_REPAIR",
        "pipeline_skills_required": required,
        "has_pipeline_skill_exposure": has_any_exposure,
        "has_pipeline_skill_calls": bool(counts),
        "pipeline_skill_call_counts": counts,
        "has_agentscope_agent_skills": has_agent_skills,
        "agentscope_agent_skill_prompt": str(prompt_path) if prompt_path.is_file() else "",
        "agentscope_agent_skill_names": registered_agent_skills,
        "missing_agentscope_agent_skill_names": [name for name in expected if name not in registered_agent_skills],
        "required_prefix": "pipeline_",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "issues": [] if has_any_exposure or not required else ["No pipeline AgentScope skill or pipeline_* tool was exposed to Data Cleaning Agent"],
    }
    _write_json(phase_root / "pipeline_skill_usage_check.json", payload)
    return payload


def _write_adapter_bundle_manifest(
    candidate: Path,
    persisted_variants: list[str],
    packaged_skills: list[str] | None = None,
) -> Path:
    bundle = candidate / "adapter_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    capabilities = candidate / "capabilities"
    copied: list[str] = []
    if capabilities.is_dir():
        target = bundle / "capabilities"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(capabilities, target)
        copied = [path.name for path in target.iterdir() if path.is_dir()]
    manifest = {
        "schema_version": 1,
        "status": "SUCCESS",
        "adapter_count": len(copied),
        "adapters": copied,
        "persisted_variants": persisted_variants,
        "packaged_skills": packaged_skills or [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(bundle / "manifest.json", manifest)
    return bundle


def _restore_claude_style_capabilities(active: Path, workspace_capabilities: Path) -> list[str]:
    source = active / "capabilities"
    restored: list[str] = []
    if not source.is_dir():
        return restored
    workspace_capabilities.mkdir(parents=True, exist_ok=True)
    for capability in sorted(path for path in source.iterdir() if path.is_dir()):
        if not (capability / "SKILL.md").is_file() or (capability / "variant.json").is_file():
            continue
        target = workspace_capabilities / capability.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(capability, target)
        restored.append(capability.name)
    return restored


def _stage_claude_style_agent_skills(
    *,
    phase_root: Path,
    candidate: Path,
    package: dict[str, Path],
    contract: dict[str, Any],
) -> list[str]:
    workspace = phase_root / "workspace"
    capabilities_root = workspace / "capabilities"
    plan_path = _latest_named_file(phase_root, "skill_packaging_plan.json")
    if plan_path is None:
        return []
    plan = _load_json(plan_path)
    skills = (
        plan.get("skills")
        or plan.get("skills_to_package")
        or plan.get("experiment_local_skills")
        or plan.get("capabilities")
        or plan.get("variants")
        or []
    )
    if not isinstance(skills, list) or not skills:
        check_path = phase_root / "skill_packaging_completion_check.json"
        _write_json(
            check_path,
            {
                "schema_version": 1,
                "status": "SKIPPED",
                "reason": "skill_packaging_plan.json did not contain a non-empty skills list; packaging is optional during validation loop",
                "packaging_plan": str(plan_path),
                "workspace": str(workspace),
                "capabilities_root": str(capabilities_root),
                "written_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return []
    package_dir = package.get("result_package")
    if not isinstance(package_dir, Path) or not package_dir.is_dir():
        check_path = phase_root / "skill_packaging_completion_check.json"
        _write_json(
            check_path,
            {
                "schema_version": 1,
                "status": "SKIPPED",
                "reason": "Cannot stage optional packaged Skills without a result_package directory",
                "packaging_plan": str(plan_path),
                "workspace": str(workspace),
                "capabilities_root": str(capabilities_root),
                "written_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return []
    candidate_capabilities = candidate / "capabilities"
    candidate_capabilities.mkdir(parents=True, exist_ok=True)
    registry_entries: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    receipts_dir = candidate / "execution_receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    packaged: list[str] = []
    for raw_entry in skills:
        if not isinstance(raw_entry, dict):
            check_path = phase_root / "skill_packaging_completion_check.json"
            _write_json(
                check_path,
                {
                    "schema_version": 1,
                    "status": "SKIPPED",
                    "reason": "Each skill_packaging_plan skill must be an object; packaging is optional during validation loop",
                    "packaging_plan": str(plan_path),
                    "workspace": str(workspace),
                    "capabilities_root": str(capabilities_root),
                    "written_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            return []
        try:
            entry = _validate_claude_style_skill_entry(raw_entry, capabilities_root, package_dir)
        except RuntimeError as exc:
            check_path = phase_root / "skill_packaging_completion_check.json"
            _write_json(
                check_path,
                {
                    "schema_version": 1,
                    "status": "SKIPPED",
                    "reason": str(exc),
                    "packaging_plan": str(plan_path),
                    "workspace": str(workspace),
                    "capabilities_root": str(capabilities_root),
                    "written_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            return []
        source_dir = entry["source_dir"]
        target_dir = candidate_capabilities / entry["name"]
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        file_hashes = _relative_file_hashes(target_dir)
        output_hashes = {
            output: hashlib.sha256((package_dir / output).read_bytes()).hexdigest()
            for output in entry["outputs"]
            if output not in {".", "result_package"} and (package_dir / output).is_file()
        }
        receipt_path = receipts_dir / f"{entry['name']}.json"
        receipt = {
            "schema_version": 1,
            "status": "SUCCESS",
            "skill_name": entry["name"],
            "result_package": str(package_dir),
            "outputs": entry["outputs"],
            "output_hashes": output_hashes,
            "skill_hashes": file_hashes,
            "validation_key_count": _validation_key_count(contract),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(receipt_path, receipt)
        registry_entries.append(
            {
                "name": entry["name"],
                "description": entry["description"],
                "capability_path": f"capabilities/{entry['name']}",
                "source_scripts": entry["source_scripts"],
                "outputs": entry["outputs"],
                "base_skills": entry["base_skills"],
                "skill_hashes": file_hashes,
                "execution_receipt": str(receipt_path),
                "created_at": receipt["created_at"],
            }
        )
        bindings.append(
            {
                "binding_type": "claude_style_skill",
                "skill_name": entry["name"],
                "status": "validated",
                "capability_path": f"capabilities/{entry['name']}",
                "reference_artifacts": entry["outputs"],
                "source_scripts": entry["source_scripts"],
                "base_skills": entry["base_skills"],
                "execution_receipt": str(receipt_path),
                "updated_at": receipt["created_at"],
            }
        )
        packaged.append(entry["name"])
    registry = {
        "schema_version": 1,
        "status": "SUCCESS",
        "packaging_plan": str(plan_path),
        "strategy": str(plan.get("strategy") or ""),
        "skill_count": len(registry_entries),
        "skills": registry_entries,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(candidate / "skill_registry.json", registry)
    shutil.copy2(plan_path, candidate / "skill_packaging_plan.json")
    _merge_reference_skill_bindings(candidate / "skill_bindings.json", bindings)
    return packaged


def _workspace_has_reference_adapter_scripts(workspace: Path) -> bool:
    candidates = [
        workspace / "build_package.py",
        workspace / "adapter",
    ]
    if any(path.is_file() for path in candidates):
        return True
    adapter_dir = workspace / "adapter"
    return adapter_dir.is_dir() and any(path.suffix == ".py" for path in adapter_dir.rglob("*.py"))


def _validate_claude_style_skill_entry(
    entry: dict[str, Any],
    capabilities_root: Path,
    package_dir: Path,
) -> dict[str, Any]:
    name = str(entry.get("name") or entry.get("skill_name") or "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,80}", name):
        raise RuntimeError(f"Invalid packaged Skill name: {name!r}")
    source_dir = (capabilities_root / name).resolve()
    if not source_dir.is_dir() or capabilities_root.resolve() not in source_dir.parents:
        raise RuntimeError(f"Packaged Skill directory is missing: {source_dir}")
    skill_md = source_dir / "SKILL.md"
    if not skill_md.is_file() or not skill_md.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Packaged Skill must include a non-empty SKILL.md: {source_dir}")
    scripts_dir = source_dir / "scripts"
    script_files = (
        sorted(
            path
            for path in scripts_dir.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )
        if scripts_dir.is_dir()
        else []
    )
    if not script_files:
        raise RuntimeError(f"Packaged Skill must include at least one script under scripts/: {source_dir}")
    outputs = [
        str(value).strip().lstrip("/")
        for value in (
            entry.get("outputs")
            or entry.get("reference_artifacts")
            or entry.get("reference_artifacts_owned")
            or entry.get("responsible_reference_artifacts")
            or []
        )
    ]
    outputs = [value.removeprefix("result_package/") for value in outputs if value]
    if not outputs:
        outputs = _infer_reference_artifacts_from_skill_md(skill_md.read_text(encoding="utf-8"), package_dir)
    if not outputs:
        raise RuntimeError(f"Packaged Skill must declare outputs/reference_artifacts: {name}")
    missing_outputs = [
        output for output in outputs
        if output not in {".", "result_package"} and not (package_dir / output).exists()
    ]
    if missing_outputs:
        raise RuntimeError(f"Packaged Skill {name} declares missing outputs: {missing_outputs}")
    source_scripts = [
        path.relative_to(source_dir).as_posix()
        for path in script_files
    ]
    base_skills = [str(value) for value in entry.get("base_skills") or entry.get("base_skill_names") or []]
    return {
        "name": name,
        "description": str(entry.get("description") or "").strip(),
        "source_dir": source_dir,
        "source_scripts": source_scripts,
        "outputs": outputs,
        "base_skills": base_skills,
    }


def _infer_reference_artifacts_from_skill_md(text: str, package_dir: Path) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?:result_package/)?((?:cohort|features)/[A-Za-z0-9_.\-/]+\.csv)", text):
        artifact = match.group(1).strip().lstrip("/")
        if artifact in seen:
            continue
        if (package_dir / artifact).exists():
            candidates.append(artifact)
            seen.add(artifact)
    return candidates


def _relative_file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _merge_reference_skill_bindings(path: Path, new_bindings: list[dict[str, Any]]) -> None:
    payload: dict[str, Any] = {"schema_version": 1, "bindings": []}
    if path.is_file():
        try:
            loaded = _load_json(path)
            if isinstance(loaded.get("bindings"), list):
                payload = loaded
        except Exception:
            pass
    existing = [item for item in payload.get("bindings") or [] if isinstance(item, dict)]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in existing:
        key = (str(item.get("binding_type") or ""), str(item.get("skill_name") or item.get("variant_name") or ""))
        if key[1]:
            by_key[key] = item
    for item in new_bindings:
        key = (str(item.get("binding_type") or ""), str(item.get("skill_name") or ""))
        by_key[key] = item
    payload["bindings"] = [by_key[key] for key in sorted(by_key)]
    _write_json(path, payload)


def _copy_runtime_reports_to_candidate(phase_root: Path, candidate: Path) -> None:
    for name in (
        "runtime_trace.jsonl",
        "context_summary.md",
        "reference_shape_contract.json",
        "train_regression_report.json",
        "result_package_validation_report.json",
        "run_record.json",
    ):
        source = _latest_named_file(phase_root, name)
        if source is not None and source.is_file():
            shutil.copy2(source, candidate / name)


def _write_reference_candidate_delta(
    *,
    active: Path,
    candidate: Path,
    public_feedback: Path | None,
    repair_targets: Path | None = None,
    feedback_response: Path | None = None,
) -> dict[str, Any]:
    base_hashes = _material_file_hashes(active)
    candidate_hashes = _material_file_hashes(candidate)
    changed = sorted(
        path for path in set(base_hashes) | set(candidate_hashes)
        if base_hashes.get(path) != candidate_hashes.get(path)
    )
    feedback_audit = _audit_reference_feedback_response(
        changed_files=changed,
        repair_targets=repair_targets,
        feedback_response=feedback_response,
    )
    if public_feedback is None:
        status = "SUCCESS"
    elif feedback_audit["target_count"] == 0:
        status = "SUCCESS" if changed else "NEEDS_REPAIR"
    else:
        status = "SUCCESS" if changed and feedback_audit["is_complete"] and feedback_audit["has_linked_delta"] else "NEEDS_REPAIR"
    payload = {
        "schema_version": 1,
        "status": status,
        "base_bundle": str(active),
        "candidate_bundle": str(candidate),
        "public_feedback": str(public_feedback) if public_feedback else "",
        "repair_targets": str(repair_targets) if repair_targets else "",
        "feedback_response": str(feedback_response) if feedback_response else "",
        "has_material_delta": bool(changed),
        "has_feedback_linked_material_delta": bool(feedback_audit["has_linked_delta"]),
        "changed_files": changed,
        "feedback_targets_addressed": feedback_audit["addressed_target_ids"],
        "feedback_targets_unaddressed": feedback_audit["unaddressed_target_ids"],
        "feedback_linked_changed_files": feedback_audit["linked_changed_files"],
        "feedback_response_issues": feedback_audit["issues"],
        "checked_scopes": [
            "adapter_bundle",
            "capabilities",
            "reference_shape_contract.json",
            "results/result_manifest.json",
            "results/target_field_mapping.json",
            "results/result_package",
        ],
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(candidate / "candidate_delta.json", payload)
    return payload


def _compile_reference_repair_targets(public_feedback: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Convert value-free public feedback into stable repair targets for the next agent loop."""
    feedback_path = Path(public_feedback).expanduser().resolve()
    feedback = _load_json(feedback_path)
    targets: list[dict[str, Any]] = []
    for index, item in enumerate(feedback.get("repair_targets") or [], start=1):
        if not isinstance(item, dict):
            continue
        target_type = str(item.get("type") or "repair").strip() or "repair"
        relative_path = str(item.get("relative_path") or "").strip()
        columns = [str(column) for column in item.get("columns") or []]
        column = str(item.get("column") or "").strip()
        if column:
            columns = [column]
        repair_target_id = _reference_repair_target_id(target_type, relative_path, columns, index)
        target = {
            "repair_target_id": repair_target_id,
            "type": target_type,
            "relative_path": relative_path,
            "columns": columns,
            "metrics": _public_metric_subset(item),
            "suggestion": str(item.get("suggestion") or ""),
            "required_decision": True,
        }
        if item.get("key_columns"):
            target["key_columns"] = [str(column) for column in item.get("key_columns") or []]
        targets.append(target)
    payload = {
        "schema_version": 1,
        "status": "SUCCESS",
        "public_feedback": str(feedback_path),
        "target_count": len(targets),
        "targets": targets,
        "privacy": "value_free_repair_targets",
    }
    _write_json(Path(output_path), payload)
    return payload


def _merge_test_rule_feedback_into_repair_targets(
    test_feedback: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    feedback_path = Path(test_feedback).expanduser().resolve()
    feedback = _load_json(feedback_path)
    output = Path(output_path)
    payload = _load_json(output) if output.is_file() else {
        "schema_version": 1,
        "status": "SUCCESS",
        "public_feedback": "",
        "targets": [],
        "privacy": "value_free_repair_targets",
    }
    targets = [item for item in payload.get("targets") or [] if isinstance(item, dict)]
    issue_texts = [str(item) for item in feedback.get("issues") or []]
    execution_failure = feedback.get("execution_failure") or {}
    if isinstance(execution_failure, dict) and execution_failure.get("log_tail"):
        issue_texts.extend(str(execution_failure["log_tail"]).splitlines())
    relative_paths: set[str] = set()
    for issue in issue_texts:
        relative_paths.update(
            match.group(0).strip().lstrip("/")
            for match in re.finditer(
                r"(?:cohort|features|summary|csv)/[^\s,:;]+\.csv",
                issue,
                flags=re.IGNORECASE,
            )
        )
    existing = {
        (str(item.get("type") or ""), str(item.get("relative_path") or ""))
        for item in targets
    }
    candidates = sorted(relative_paths) or [""]
    for relative_path in candidates:
        key = ("test_pipeline_rule_failure", relative_path)
        if key in existing:
            continue
        index = len(targets) + 1
        targets.append(
            {
                "repair_target_id": _reference_repair_target_id(
                    key[0], relative_path, [], index
                ),
                "type": key[0],
                "relative_path": relative_path,
                "columns": [],
                "metrics": {},
                "suggestion": "修复冻结 Pipeline 在 test raw 上暴露的通用规则或路径问题。",
                "required_decision": True,
                "source_feedback": str(feedback_path),
            }
        )
    payload["targets"] = targets
    payload["target_count"] = len(targets)
    payload["test_pipeline_rule_feedback"] = str(feedback_path)
    _write_json(output, payload)
    return payload


def _reference_repair_target_id(target_type: str, relative_path: str, columns: list[str], index: int) -> str:
    readable = "__".join(part for part in [target_type, relative_path.replace("/", "_"), "_".join(columns)] if part)
    digest = hashlib.sha256(f"{target_type}|{relative_path}|{','.join(columns)}|{index}".encode("utf-8")).hexdigest()[:10]
    stem = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in readable)[:80].strip("_")
    return f"{stem or 'repair'}__{digest}"


def _public_metric_subset(item: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "result_rows",
        "reference_rows",
        "exact_row_precision",
        "missing_reference_rows",
        "key_row_recall",
        "reference_non_empty_cells",
        "matched_non_empty_cells",
        "recall",
    }
    return {key: item[key] for key in sorted(allowed) if key in item}


def _feedback_response_continuation_prompt(*, repair_targets_path: Path, candidate: Path) -> str:
    return "\n".join(
        [
            "上一轮 public feedback 已编译为 repair targets，但你还没有生成 feedback_response.json。",
            "",
            "现在只执行一个补交任务：读取 repair_targets.json，写出 feedback_response.json。",
            "",
            "严格要求：",
            "1. 不要重新诊断 train/reference。",
            "2. 不要重新生成 result_package。",
            "3. 不要读取 validation/reference 或任何 hidden/private report。",
            "4. 必须逐条回应 repair_targets.json 里的每个 repair_target_id。",
            "5. 如果本轮已经修改了 result_package、capabilities、adapter 或 build 脚本，必须在 changed_files 中列出对应相对路径。",
            "6. 如果没有实质修复，也必须写明 no_action_reason 或 decision=carry_forward。",
            "7. 不得包含 gold_value、reference_value、pred_value、prediction_value 等具体隐藏值字段。",
            "",
            f"repair_targets.json 路径：{repair_targets_path}",
            f"candidate 目录：{candidate}",
            "",
            "请在当前 workspace 或当前运行目录写出 feedback_response.json，格式必须是：",
            "{",
            '  "schema_version": 1,',
            '  "status": "SUCCESS",',
            '  "responses": [',
            "    {",
            '      "repair_target_id": "...",',
            '      "decision": "rule_update | adapter_update | mapping_update | unsupported_reason | no_action | carry_forward",',
            '      "reason": "...",',
            '      "changed_files": ["results/result_package/...", "capabilities/..."],',
            '      "no_action_reason": ""',
            "    }",
            "  ]",
            "}",
            "",
            "写完该文件后直接结束，不要继续做其他任务。",
        ]
    )


def _stage_reference_feedback_response(
    *,
    phase_root: Path,
    candidate: Path,
    repair_targets_path: Path,
) -> Path:
    repair_targets = _load_json(repair_targets_path)
    target_ids = [
        str(item.get("repair_target_id"))
        for item in repair_targets.get("targets") or []
        if isinstance(item, dict) and item.get("repair_target_id")
    ]
    if not target_ids:
        path = candidate / "feedback_response.json"
        _write_json(
            path,
            {
                "schema_version": 1,
                "status": "SUCCESS",
                "repair_targets": str(repair_targets_path),
                "responses": [],
                "addressed_target_ids": [],
                "unaddressed_target_ids": [],
                "issues": [],
            },
        )
        return path
    targets_by_id = {
        str(item.get("repair_target_id")): item
        for item in repair_targets.get("targets") or []
        if isinstance(item, dict) and item.get("repair_target_id")
    }
    source = _latest_named_file(phase_root, "feedback_response.json")
    if source is None or not source.is_file():
        raw = {
            "schema_version": 1,
            "status": "SUCCESS",
            "source": "",
            "generated_by": "host_fallback",
            "fallback_reason": "Data Cleaning Agent did not write feedback_response.json before package staging.",
            "responses": _host_reference_feedback_responses(targets_by_id),
        }
    else:
        raw = _load_json(source)
    responses = raw.get("responses") or []
    if not isinstance(responses, list):
        responses = _host_reference_feedback_responses(targets_by_id)
        raw["responses"] = responses
        raw["generated_by"] = "host_fallback"
        raw["fallback_reason"] = "Existing feedback_response.json did not contain a responses list."
    addressed = {
        str(item.get("repair_target_id"))
        for item in responses
        if isinstance(item, dict) and item.get("repair_target_id")
    }
    auto_filled = sorted(set(target_ids) - addressed)
    if auto_filled:
        responses.extend(
            _host_reference_feedback_responses(
                {target_id: targets_by_id[target_id] for target_id in auto_filled if target_id in targets_by_id}
            )
        )
        addressed = {
            str(item.get("repair_target_id"))
            for item in responses
            if isinstance(item, dict) and item.get("repair_target_id")
        }
    unaddressed = sorted(set(target_ids) - addressed)
    issues: list[str] = []
    if unaddressed:
        issues.append("missing responses for repair targets: " + ", ".join(unaddressed[:20]))
    text = json.dumps(raw, ensure_ascii=False)
    for forbidden in ("gold_value", "pred_value", "reference_value", "prediction_value"):
        if forbidden in text:
            issues.append(f"feedback_response leaks forbidden key: {forbidden}")
    payload = {
        "schema_version": 1,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "source": str(source) if source else "",
        "generated_by": raw.get("generated_by", "agent"),
        "fallback_reason": raw.get("fallback_reason", ""),
        "repair_targets": str(repair_targets_path),
        "responses": responses,
        "addressed_target_ids": sorted(addressed & set(target_ids)),
        "unaddressed_target_ids": unaddressed,
        "auto_filled_target_ids": auto_filled,
        "issues": issues,
    }
    path = candidate / "feedback_response.json"
    _write_json(path, payload)
    if issues:
        raise RuntimeError(f"Invalid feedback_response.json; see {path}")
    return path


def _host_reference_feedback_responses(targets_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for target_id, target in sorted(targets_by_id.items()):
        relative_path = str(target.get("relative_path") or "").strip().lstrip("/")
        changed_files = [f"results/result_package/{relative_path}"] if relative_path else ["results/result_package"]
        responses.append(
            {
                "repair_target_id": target_id,
                "decision": "host_generated_carry_forward",
                "repair_type": target.get("type", ""),
                "relative_path": relative_path,
                "columns": target.get("columns") or [],
                "changed_files": changed_files,
                "reason": (
                    "Data Cleaning Agent did not provide an explicit response for this public feedback target. "
                    "The host generated this value-free carry-forward response so the candidate can still be "
                    "evaluated without reading or leaking hidden reference values."
                ),
                "metrics": target.get("metrics") or {},
                "next_action": "carry_forward_if_score_does_not_improve",
            }
        )
    return responses


def _audit_reference_feedback_response(
    *,
    changed_files: list[str],
    repair_targets: Path | None,
    feedback_response: Path | None,
) -> dict[str, Any]:
    if repair_targets is None or not repair_targets.is_file():
        return {
            "target_count": 0,
            "is_complete": True,
            "has_linked_delta": False,
            "addressed_target_ids": [],
            "unaddressed_target_ids": [],
            "linked_changed_files": [],
            "issues": [],
        }
    target_payload = _load_json(repair_targets)
    target_ids = [
        str(item.get("repair_target_id"))
        for item in target_payload.get("targets") or []
        if isinstance(item, dict) and item.get("repair_target_id")
    ]
    generic_target_ids = {
        str(item.get("repair_target_id"))
        for item in target_payload.get("targets") or []
        if isinstance(item, dict)
        and item.get("repair_target_id")
        and not str(item.get("relative_path") or "").strip()
    }
    issues: list[str] = []
    if feedback_response is None or not feedback_response.is_file():
        return {
            "target_count": len(target_ids),
            "is_complete": False,
            "has_linked_delta": False,
            "addressed_target_ids": [],
            "unaddressed_target_ids": target_ids,
            "linked_changed_files": [],
            "issues": ["feedback_response.json is missing"],
        }
    response_payload = _load_json(feedback_response)
    responses = response_payload.get("responses") or []
    if not isinstance(responses, list):
        responses = []
        issues.append("feedback_response.responses is not a list")
    changed_set = set(changed_files)
    addressed: set[str] = set()
    response_changed_files: set[str] = set()
    for response in responses:
        if not isinstance(response, dict):
            continue
        target_id = str(response.get("repair_target_id") or "")
        if target_id in target_ids:
            addressed.add(target_id)
        for changed_file in response.get("changed_files") or []:
            response_changed_files.add(str(changed_file))
    linked_changed = sorted(changed_set & response_changed_files)
    generic_addressed = bool(generic_target_ids & addressed)
    generic_package_link = any(
        path.rstrip("/") == "results/result_package"
        for path in response_changed_files
    )
    if generic_addressed and generic_package_link:
        linked_changed = sorted(
            path for path in changed_set if path.startswith("results/result_package/")
        )
    unaddressed = sorted(set(target_ids) - addressed)
    if unaddressed:
        issues.append("feedback targets not addressed: " + ", ".join(unaddressed[:20]))
    if target_ids and not linked_changed:
        issues.append("feedback responses do not point to any changed material file")
    return {
        "target_count": len(target_ids),
        "is_complete": not unaddressed and not issues,
        "has_linked_delta": bool(linked_changed),
        "addressed_target_ids": sorted(addressed),
        "unaddressed_target_ids": unaddressed,
        "linked_changed_files": linked_changed,
        "issues": issues,
    }


def _material_file_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    candidates: list[Path] = []
    for relative in (
        "adapter_bundle",
        "capabilities",
    ):
        directory = root / relative
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    for relative in (
        "reference_shape_contract.json",
        "results/result_manifest.json",
        "results/target_field_mapping.json",
    ):
        path = root / relative
        if path.is_file():
            candidates.append(path)
    result_package = root / "results" / "result_package"
    if result_package.is_dir():
        candidates.extend(path for path in result_package.rglob("*") if path.is_file())
    hashes: dict[str, str] = {}
    for path in sorted(set(candidates)):
        try:
            relative = str(path.relative_to(root))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            continue
    return hashes


def _complete_reference_package_paths(root: Path, contract: dict[str, Any]) -> dict[str, Path] | None:
    reference_shape_contract = _latest_named_file(root, "reference_shape_contract.json")
    train_regression = _latest_named_file(root, "train_regression_report.json")
    result_manifest = _latest_named_file(root, "result_manifest.json")
    mapping = _latest_named_file(root, "target_field_mapping.json")
    skill_usage = _latest_named_file(root, "skill_usage_report.json")
    package_dir = root / "workspace" / "result_package"
    if not package_dir.is_dir():
        return None
    package_manifest_path = package_dir / "package_manifest.json"
    package_manifest = package_manifest_path if package_manifest_path.is_file() else None
    package_reference = package_dir / "reference.csv"
    if not package_reference.is_file():
        package_reference = None
    package_files = _structured_package_files(package_dir)
    if not package_files:
        return None
    cohort = _latest_named_file(root, "cohort.csv") or _find_package_cohort_file(package_dir) or _first_keyed_package_file(package_files, contract)
    features = _latest_named_file(root, "features_wide.csv") or _find_package_feature_file(package_dir) or package_reference or cohort
    final_dataset = _latest_named_file(root, "final_dataset.csv") or package_reference or features
    key_column = str(contract.get("key_column") or contract.get("record_grain") or "")
    try:
        features_frame = pd.read_csv(features, dtype=object)
        cohort_frame = pd.read_csv(cohort, dtype=object)
    except Exception:
        return None
    if features_frame.empty or cohort_frame.empty:
        return None
    if key_column and key_column not in features_frame.columns and key_column not in cohort_frame.columns:
        return None
    expected_count = _validation_key_count(contract)
    if expected_count and not _package_has_expected_key_coverage(package_files, key_column, expected_count):
        return None
    package_validation = (
        _latest_named_file(root, "result_package_validation_report.json")
        or _write_lightweight_result_package_validation(
            root=root,
            package_dir=package_dir,
            key_column=key_column,
            expected_count=expected_count,
        )
    )
    run_record = _write_reference_run_record(
        root=root,
        contract=contract,
        package_dir=package_dir,
        package_reference=package_reference,
        cohort=cohort,
        features=features,
        final_dataset=final_dataset,
        result_manifest=result_manifest,
        mapping=mapping,
        skill_usage=skill_usage,
    )
    package = {
        "cohort": cohort,
        "features_wide": features,
        "final_dataset": final_dataset,
        "result_package": package_dir,
        "result_package_validation_report": package_validation,
        "run_record": run_record,
        **_optional_latest_paths(
            root,
            {
                "reference_shape_contract": "reference_shape_contract.json",
                "train_regression_report": "train_regression_report.json",
                "result_manifest": "result_manifest.json",
                "target_field_mapping": "target_field_mapping.json",
                "skill_usage_report": "skill_usage_report.json",
                "feature_manifest": "feature_manifest.json",
                "feature_provenance": "feature_provenance.json",
                "extraction_task_execution": "extraction_task_execution.json",
            },
        ),
    }
    if package_manifest is not None:
        package["package_manifest"] = package_manifest
    if package_reference is not None:
        package["package_reference"] = package_reference
    return package


def _stage_reference_agent_package(package: dict[str, Path], candidate: Path) -> dict[str, Any]:
    results = candidate / "results"
    if results.exists():
        shutil.rmtree(results)
    results.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Any] = {}
    for key, source in package.items():
        if key == "result_package" or not isinstance(source, Path):
            continue
        if not source.exists() or not source.is_file():
            continue
        target = results / source.name
        if key == "package_manifest":
            target = results / "result_package" / "package_manifest.json"
            target.parent.mkdir(parents=True, exist_ok=True)
        elif key == "package_reference":
            target = results / "result_package" / "reference.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        staged[key] = str(target)
    package_dir = package.get("result_package")
    if isinstance(package_dir, Path) and package_dir.is_dir():
        target_dir = results / "result_package"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(package_dir, target_dir)
        staged["result_package"] = str(target_dir)
        if (target_dir / "package_manifest.json").is_file():
            staged["package_manifest"] = str(target_dir / "package_manifest.json")
        if (target_dir / "reference.csv").is_file():
            staged["package_reference"] = str(target_dir / "reference.csv")
    _write_json(
        candidate / "agent_result_package.json",
        {
            "schema_version": 1,
            "status": "SUCCESS",
            "staged_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": staged,
        },
    )
    return staged


def _write_reference_completion_check(root: Path, contract: dict[str, Any]) -> Path:
    required = {
        "result_package": _latest_result_package_dir(root),
    }
    optional = {
        "package_manifest": _latest_result_package_manifest(root),
        "package_reference": None,
        "reference_shape_contract": _latest_named_file(root, "reference_shape_contract.json"),
        "train_regression_report": _latest_named_file(root, "train_regression_report.json"),
        "result_package_validation_report": _latest_named_file(root, "result_package_validation_report.json"),
        "result_manifest": _latest_named_file(root, "result_manifest.json"),
        "target_field_mapping": _latest_named_file(root, "target_field_mapping.json"),
        "skill_usage_report": _latest_named_file(root, "skill_usage_report.json"),
        "cohort": _latest_named_file(root, "cohort.csv"),
        "features_wide": _latest_named_file(root, "features_wide.csv"),
        "final_dataset": _latest_named_file(root, "final_dataset.csv"),
    }
    if optional["package_manifest"]:
        reference = optional["package_manifest"].parent / "reference.csv"
        optional["package_reference"] = reference if reference.is_file() else None
    elif required["result_package"]:
        reference = required["result_package"] / "reference.csv"
        optional["package_reference"] = reference if reference.is_file() else None
    payload = {
        "schema_version": 1,
        "status": "NEEDS_REPAIR",
        "required_core": {name: str(path) if path else "" for name, path in required.items()},
        "optional_records": {name: str(path) if path else "" for name, path in optional.items()},
        "validation_key_count": _validation_key_count(contract),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = root / "reference_engineer_completion_check.json"
    _write_json(path, payload)
    return path


def evaluate_reference_directory_package(
    *,
    result_package: str | Path,
    validation_reference_root: str | Path | None,
    output_dir: str | Path,
    key_column: str = "",
) -> dict[str, str]:
    """Evaluate with the fixed-weight, row-aligned schema-v2 scorer."""
    return evaluate_balanced_reference_directory_package(
        result_package=result_package,
        validation_reference_root=validation_reference_root,
        output_dir=output_dir,
        key_column=key_column,
    )


def _reference_directory_score_report(package_root: Path, reference_root: Path, *, key_column: str) -> dict[str, Any]:
    reference_files = _structured_package_files(reference_root)
    comparable_files = [
        path for path in reference_files
        if path.name not in {
            "package_manifest.json",
            "result_manifest.json",
            "target_field_mapping.json",
            "reference.csv",
        }
    ]
    if not comparable_files:
        return {
            "schema_version": 1,
            "status": "NEEDS_REPAIR",
            "mode": "reference_directory",
            "metrics": {
                "file_coverage": 0.0,
                "column_coverage": 0.0,
                "row_coverage": 0.0,
                "value_consistency": 0.0,
                "composite_score": 0.0,
            },
            "issues": ["reference directory contains no comparable structured files"],
        }

    file_reports: list[dict[str, Any]] = []
    matched_files = 0
    expected_columns_total = 0
    matched_columns_total = 0
    reference_rows_total = 0
    exact_row_matches_total = 0
    key_row_matches_total = 0
    result_rows_total = 0
    reference_cells_total = 0
    matched_cells_total = 0

    for reference_file in comparable_files:
        relative = reference_file.relative_to(reference_root).as_posix()
        result_file = package_root / relative
        file_report: dict[str, Any] = {
            "relative_path": relative,
            "reference_file": str(reference_file),
            "result_file": str(result_file) if result_file.is_file() else "",
            "status": "missing_file",
        }
        if not result_file.is_file():
            file_reports.append(file_report)
            continue
        try:
            comparison = _compare_reference_table_streaming(
                reference_file=reference_file,
                result_file=result_file,
                relative_path=relative,
                key_column=key_column,
            )
        except Exception as exc:
            file_report.update({"status": "read_error", "issue": str(exc)})
            file_reports.append(file_report)
            continue
        matched_files += 1
        expected_columns_total += comparison["expected_column_count"]
        matched_columns_total += comparison["matched_column_count"]
        reference_rows_total += comparison["reference_rows"]
        result_rows_total += comparison["result_rows"]
        exact_row_matches_total += comparison["exact_row_matches"]
        key_row_matches_total += comparison["key_row_matches"]
        reference_cells_total += comparison["reference_non_empty_cell_count"]
        matched_cells_total += comparison["matched_non_empty_cell_count"]
        file_report.update(
            {
                "status": "compared",
                **comparison,
            }
        )
        file_reports.append(file_report)

    file_coverage = matched_files / max(len(comparable_files), 1)
    column_coverage = matched_columns_total / max(expected_columns_total, 1)
    exact_row_recall = exact_row_matches_total / max(reference_rows_total, 1)
    key_row_recall = key_row_matches_total / max(reference_rows_total, 1)
    exact_row_precision = exact_row_matches_total / max(result_rows_total, 1)
    cell_recall = matched_cells_total / max(reference_cells_total, 1)
    row_coverage = exact_row_recall
    value_consistency = cell_recall
    composite = (
        0.10 * file_coverage
        + 0.20 * column_coverage
        + 0.20 * row_coverage
        + 0.40 * value_consistency
        + 0.10 * exact_row_precision
    )
    missing_files = [item["relative_path"] for item in file_reports if item["status"] == "missing_file"]
    missing_columns = {
        item["relative_path"]: item.get("missing_columns", [])
        for item in file_reports
        if item.get("missing_columns")
    }
    return {
        "schema_version": 1,
        "status": "SUCCESS",
        "mode": "reference_directory",
        "metrics": {
            "file_coverage": round(file_coverage, 4),
            "column_coverage": round(column_coverage, 4),
            "row_coverage": round(row_coverage, 4),
            "value_consistency": round(value_consistency, 4),
            "cell_recall": round(cell_recall, 4),
            "exact_row_recall": round(exact_row_recall, 4),
            "key_row_recall": round(key_row_recall, 4),
            "exact_row_precision": round(exact_row_precision, 4),
            "composite_score": round(composite, 4),
        },
        "detail_metrics": {
            "reference_file_count": len(comparable_files),
            "matched_file_count": matched_files,
            "expected_column_count": expected_columns_total,
            "matched_column_count": matched_columns_total,
            "reference_row_count": reference_rows_total,
            "result_row_count": result_rows_total,
            "exact_row_match_count": exact_row_matches_total,
            "key_row_match_count": key_row_matches_total,
            "reference_non_empty_cell_count": reference_cells_total,
            "matched_non_empty_cell_count": matched_cells_total,
            "expected_row_count": reference_rows_total,
            "matched_row_count": exact_row_matches_total,
            "expected_value_count": reference_cells_total,
            "matched_value_count": matched_cells_total,
        },
        "missing_files": missing_files,
        "missing_columns": missing_columns,
        "file_reports": file_reports,
    }


def _compare_reference_table_streaming(
    *,
    reference_file: Path,
    result_file: Path,
    relative_path: str,
    key_column: str,
) -> dict[str, Any]:
    reference_columns = _table_columns(reference_file)
    result_columns = _table_columns(result_file)
    if not reference_columns:
        return {
            "comparison_mode": "empty_reference",
            "key_columns": [],
            "expected_column_count": 0,
            "matched_column_count": 0,
            "missing_columns": [],
            "reference_rows": 0,
            "result_rows": _count_table_rows(result_file),
            "exact_row_matches": 0,
            "key_row_matches": 0,
            "missing_reference_rows": 0,
            "extra_result_rows": 0,
            "reference_non_empty_cell_count": 0,
            "matched_non_empty_cell_count": 0,
            "per_column": {},
        }
    common_columns = [column for column in reference_columns if column in result_columns]
    missing_columns = [column for column in reference_columns if column not in result_columns]
    key_columns = _streaming_comparison_key_columns(
        relative_path=relative_path,
        reference_columns=reference_columns,
        result_columns=result_columns,
        key_column=key_column,
    )
    if not key_columns:
        key_columns = common_columns[:1]
    if _should_use_fast_large_table_comparison(reference_file, result_file):
        return _compare_reference_table_fast_large(
            reference_file=reference_file,
            result_file=result_file,
            reference_columns=reference_columns,
            result_columns=result_columns,
            common_columns=common_columns,
            missing_columns=missing_columns,
            key_columns=key_columns,
        )

    reference_rows = _build_reference_row_counters(reference_file, reference_columns, key_columns)
    ref_exact = reference_rows["exact"]
    ref_keys = reference_rows["keys"]
    ref_cells = reference_rows["cells"]
    per_column = reference_rows["per_column"]
    reference_row_count = reference_rows["row_count"]
    reference_cell_count = reference_rows["cell_count"]

    remaining_exact = Counter(ref_exact)
    remaining_keys = Counter(ref_keys)
    remaining_cells = Counter(ref_cells)
    column_index = {column: index for index, column in enumerate(reference_columns)}
    exact_matches = 0
    key_matches = 0
    matched_cells = 0
    per_column_matches: Counter[str] = Counter()
    result_rows = 0
    exact_non_empty_columns = reference_rows["exact_non_empty_columns"]
    # Full per-cell partial matching is useful for small tables, but it becomes
    # prohibitively slow for long event tables with tens of millions of rows.
    # For large references, count cells only when the whole normalized row
    # matches exactly. This is stricter than partial matching and keeps the
    # evaluator bounded enough for validation-scale MIMIC event tables.
    allow_partial_cell_matching = reference_cell_count <= 2_000_000

    for row in _iter_table_dicts(result_file):
        result_rows += 1
        exact_signature = _row_signature_from_mapping(row, reference_columns)
        key_signature = _row_signature_from_mapping(row, key_columns) if key_columns else ()
        exact_row_matched = False
        if remaining_exact[exact_signature] > 0:
            exact_matches += 1
            remaining_exact[exact_signature] -= 1
            exact_row_matched = True
            for column in exact_non_empty_columns.get(exact_signature, ()):
                matched_cells += 1
                per_column_matches[column] += 1
                cell_signature = (column, key_signature, exact_signature[column_index[column]])
                if remaining_cells[cell_signature] > 0:
                    remaining_cells[cell_signature] -= 1
        if key_columns:
            if remaining_keys[key_signature] > 0:
                key_matches += 1
                remaining_keys[key_signature] -= 1
            if not allow_partial_cell_matching or exact_row_matched:
                continue
            for column in common_columns:
                value = row.get(column, "")
                if _is_empty_value(value):
                    continue
                cell_signature = (column, key_signature, _normalize_directory_value(value))
                if remaining_cells[cell_signature] > 0:
                    matched_cells += 1
                    per_column_matches[column] += 1
                    remaining_cells[cell_signature] -= 1
        else:
            if exact_signature in ref_exact and not exact_row_matched:
                for column in common_columns:
                    value = row.get(column, "")
                    if not _is_empty_value(value) and per_column_matches[column] < per_column[column]:
                        matched_cells += 1
                        per_column_matches[column] += 1

    per_column_report: dict[str, dict[str, Any]] = {}
    for column in reference_columns:
        expected = int(per_column[column])
        matched = int(per_column_matches[column])
        status = "matched" if column in result_columns else "missing_column"
        per_column_report[column] = {
            "status": status,
            "reference_non_empty_cells": expected,
            "matched_non_empty_cells": matched,
            "recall": round(matched / max(expected, 1), 4) if expected else 1.0,
        }

    return {
        "comparison_mode": "cell_level_keyed" if key_columns else "cell_level_rowset",
        "key_columns": key_columns,
        "expected_column_count": len(reference_columns),
        "matched_column_count": len(common_columns),
        "missing_columns": missing_columns,
        "reference_rows": int(reference_row_count),
        "result_rows": int(result_rows),
        "exact_row_matches": int(exact_matches),
        "key_row_matches": int(key_matches),
        "missing_reference_rows": int(reference_row_count - exact_matches),
        "extra_result_rows": int(max(result_rows - exact_matches, 0)),
        "reference_non_empty_cell_count": int(reference_cell_count),
        "matched_non_empty_cell_count": int(matched_cells),
        "exact_row_recall": round(exact_matches / max(reference_row_count, 1), 4),
        "key_row_recall": round(key_matches / max(reference_row_count, 1), 4),
        "exact_row_precision": round(exact_matches / max(result_rows, 1), 4),
        "cell_recall": round(matched_cells / max(reference_cell_count, 1), 4),
        "per_column": per_column_report,
    }


def _should_use_fast_large_table_comparison(reference_file: Path, result_file: Path) -> bool:
    try:
        return reference_file.stat().st_size + result_file.stat().st_size > 128 * 1024 * 1024
    except OSError:
        return False


def _compare_reference_table_fast_large(
    *,
    reference_file: Path,
    result_file: Path,
    reference_columns: list[str],
    result_columns: list[str],
    common_columns: list[str],
    missing_columns: list[str],
    key_columns: list[str],
) -> dict[str, Any]:
    """Compare large CSV-like tables with bounded Python work per row.

    The detailed evaluator normalizes every cell with broad type inference and
    can be too slow for MIMIC event tables. This path uses simpler normalization
    and counts cell matches only for exact normalized row matches. It is stricter
    than partial cell matching, but it finishes reliably on validation-scale
    long tables and still reports row/key/cell recall.
    """
    result_column_positions = {column: index for index, column in enumerate(result_columns)}
    reference_column_positions = {column: index for index, column in enumerate(reference_columns)}
    result_positions = [result_column_positions.get(column) for column in reference_columns]
    reference_key_positions = [reference_column_positions[column] for column in key_columns]
    result_key_positions = [result_column_positions.get(column) for column in key_columns]

    ref_exact: Counter[tuple[str, ...]] = Counter()
    ref_keys: Counter[tuple[str, ...]] = Counter()
    per_column: Counter[str] = Counter()
    reference_row_count = 0
    reference_cell_count = 0

    for values in _iter_table_rows(reference_file):
        reference_row_count += 1
        normalized = tuple(_fast_normalize_directory_value(_row_value(values, index)) for index in range(len(reference_columns)))
        ref_exact[normalized] += 1
        key_signature = tuple(normalized[index] for index in reference_key_positions)
        ref_keys[key_signature] += 1
        for index, column in enumerate(reference_columns):
            if normalized[index] == "":
                continue
            per_column[column] += 1
            reference_cell_count += 1

    remaining_exact = Counter(ref_exact)
    remaining_keys = Counter(ref_keys)
    exact_matches = 0
    key_matches = 0
    matched_cells = 0
    per_column_matches: Counter[str] = Counter()
    result_rows = 0

    for values in _iter_table_rows(result_file):
        result_rows += 1
        normalized = tuple(
            _fast_normalize_directory_value(_row_value(values, position))
            if position is not None
            else "__missing_column__"
            for position in result_positions
        )
        if remaining_exact[normalized] > 0:
            exact_matches += 1
            remaining_exact[normalized] -= 1
            for index, column in enumerate(reference_columns):
                if normalized[index] == "":
                    continue
                matched_cells += 1
                per_column_matches[column] += 1
        if key_columns:
            key_signature = tuple(
                _fast_normalize_directory_value(_row_value(values, position))
                if position is not None
                else "__missing_column__"
                for position in result_key_positions
            )
            if remaining_keys[key_signature] > 0:
                key_matches += 1
                remaining_keys[key_signature] -= 1

    per_column_report: dict[str, dict[str, Any]] = {}
    for column in reference_columns:
        expected = int(per_column[column])
        matched = int(per_column_matches[column])
        status = "matched" if column in result_columns else "missing_column"
        per_column_report[column] = {
            "status": status,
            "reference_non_empty_cells": expected,
            "matched_non_empty_cells": matched,
            "recall": round(matched / max(expected, 1), 4) if expected else 1.0,
        }

    return {
        "comparison_mode": "fast_large_table_exact",
        "key_columns": key_columns,
        "expected_column_count": len(reference_columns),
        "matched_column_count": len(common_columns),
        "missing_columns": missing_columns,
        "reference_rows": int(reference_row_count),
        "result_rows": int(result_rows),
        "exact_row_matches": int(exact_matches),
        "key_row_matches": int(key_matches),
        "missing_reference_rows": int(reference_row_count - exact_matches),
        "extra_result_rows": int(max(result_rows - exact_matches, 0)),
        "reference_non_empty_cell_count": int(reference_cell_count),
        "matched_non_empty_cell_count": int(matched_cells),
        "exact_row_recall": round(exact_matches / max(reference_row_count, 1), 4),
        "key_row_recall": round(key_matches / max(reference_row_count, 1), 4),
        "exact_row_precision": round(exact_matches / max(result_rows, 1), 4),
        "cell_recall": round(matched_cells / max(reference_cell_count, 1), 4),
        "per_column": per_column_report,
    }


def _iter_table_rows(path: Path) -> Iterator[list[str]]:
    name = path.name.lower()
    if not (name.endswith(".csv") or name.endswith(".csv.gz") or name.endswith(".tsv") or name.endswith(".tsv.gz")):
        frame = _read_table_preview(path, nrows=None)
        frame.columns = [str(column) for column in frame.columns]
        for row in frame.astype(object).itertuples(index=False, name=None):
            yield ["" if _is_empty_value(value) else str(value) for value in row]
        return
    with _open_text_table(path) as handle:
        delimiter = "\t" if name.endswith(".tsv") or name.endswith(".tsv.gz") else ","
        reader = csv.reader(handle, delimiter=delimiter)
        next(reader, None)
        for row in reader:
            yield row


def _row_value(values: list[str], index: int | None) -> str:
    if index is None or index >= len(values):
        return ""
    return values[index]


def _fast_normalize_directory_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "" or text.casefold() in {"nan", "nat", "none", "null"}:
        return ""
    if len(text) <= 64 and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        try:
            return _normalize_decimal(Decimal(text))
        except (InvalidOperation, ValueError):
            pass
    return " ".join(text.casefold().split())


def _build_reference_row_counters(path: Path, columns: list[str], key_columns: list[str]) -> dict[str, Any]:
    row_count = 0
    exact_counter: Counter[tuple[str, ...]] = Counter()
    key_counter: Counter[tuple[str, ...]] = Counter()
    cell_counter: Counter[tuple[str, tuple[str, ...], str]] = Counter()
    exact_non_empty_columns: dict[tuple[str, ...], tuple[str, ...]] = {}
    per_column: Counter[str] = Counter()
    cell_count = 0
    for row in _iter_table_dicts(path):
        row_count += 1
        exact_signature = _row_signature_from_mapping(row, columns)
        exact_counter[exact_signature] += 1
        if exact_signature not in exact_non_empty_columns:
            exact_non_empty_columns[exact_signature] = tuple(
                column for column in columns if not _is_empty_value(row.get(column, ""))
            )
        key_signature = _row_signature_from_mapping(row, key_columns)
        key_counter[key_signature] += 1
        for column in columns:
            value = row.get(column, "")
            if _is_empty_value(value):
                continue
            per_column[column] += 1
            cell_count += 1
            cell_counter[(column, key_signature, _normalize_directory_value(value))] += 1
    return {
        "row_count": row_count,
        "exact": exact_counter,
        "keys": key_counter,
        "cells": cell_counter,
        "exact_non_empty_columns": exact_non_empty_columns,
        "per_column": per_column,
        "cell_count": cell_count,
    }


def _streaming_comparison_key_columns(
    *,
    relative_path: str,
    reference_columns: list[str],
    result_columns: list[str],
    key_column: str,
) -> list[str]:
    common = set(reference_columns) & set(result_columns)
    lower_path = relative_path.casefold()
    candidates: list[list[str]] = []
    if "preproc_chart" in lower_path or "chart" in lower_path:
        candidates.extend([
            ["stay_id", "itemid", "event_time_from_admit"],
            ["stay_id", "itemid", "charttime"],
            ["subject_id", "hadm_id", "stay_id", "itemid", "charttime"],
        ])
    if "preproc_med" in lower_path or "med" in lower_path:
        candidates.extend([
            ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime", "orderid"],
            ["stay_id", "itemid", "starttime", "endtime"],
        ])
    if "preproc_out" in lower_path or "output" in lower_path:
        candidates.extend([
            ["subject_id", "hadm_id", "stay_id", "itemid", "charttime"],
            ["stay_id", "itemid", "charttime"],
        ])
    if "preproc_proc" in lower_path or "procedure" in lower_path or "proc" in lower_path:
        candidates.extend([
            ["subject_id", "hadm_id", "stay_id", "itemid", "starttime"],
            ["stay_id", "itemid", "starttime"],
            ["subject_id", "hadm_id", "stay_id", "icd_code"],
        ])
    if "preproc_diag" in lower_path or "diagnos" in lower_path or "diag" in lower_path:
        candidates.extend([
            ["subject_id", "hadm_id", "stay_id", "icd_code"],
            ["hadm_id", "icd_code"],
        ])
    if "cohort" in lower_path:
        candidates.extend([
            ["stay_id"],
            ["hadm_id"],
            ["subject_id"],
        ])
    if key_column:
        candidates.append([key_column])
    candidates.extend([[column] for column in KEY_PRIORITY])
    for candidate in candidates:
        if candidate and all(column in common for column in candidate):
            return candidate
    return []


def _table_columns(path: Path) -> list[str]:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz") or name.endswith(".tsv") or name.endswith(".tsv.gz"):
        with _open_text_table(path) as handle:
            delimiter = "\t" if name.endswith(".tsv") or name.endswith(".tsv.gz") else ","
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                return [str(column) for column in next(reader)]
            except StopIteration:
                return []
    frame = _read_table_preview(path, nrows=0)
    return [str(column) for column in frame.columns]


def _count_table_rows(path: Path) -> int:
    return sum(1 for _ in _iter_table_dicts(path))


def _iter_table_dicts(path: Path) -> Iterator[dict[str, Any]]:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz") or name.endswith(".tsv") or name.endswith(".tsv.gz"):
        with _open_text_table(path) as handle:
            delimiter = "\t" if name.endswith(".tsv") or name.endswith(".tsv.gz") else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                yield {str(key): value for key, value in row.items() if key is not None}
        return
    frame = _read_table_preview(path, nrows=None)
    frame.columns = [str(column) for column in frame.columns]
    for row in frame.to_dict(orient="records"):
        yield row


def _open_text_table(path: Path):
    if path.name.lower().endswith((".gz",)):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _row_signature_from_mapping(row: dict[str, Any], columns: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_directory_value(row.get(column, "__MISSING_COLUMN__")) for column in columns)


def _compare_reference_frames(reference: pd.DataFrame, result: pd.DataFrame, *, key_column: str) -> dict[str, Any]:
    ref_columns = [str(column) for column in reference.columns]
    result_columns = [str(column) for column in result.columns]
    reference = reference.copy()
    result = result.copy()
    reference.columns = ref_columns
    result.columns = result_columns
    keys = _comparison_key_columns(reference, result, key_column=key_column)
    for key in keys:
        reference[key] = reference[key].astype(str)
        result[key] = result[key].astype(str)
    compare_columns = [column for column in ref_columns if column not in keys]
    matched_columns = [column for column in compare_columns if column in result.columns]
    missing_columns = [column for column in compare_columns if column not in result.columns]
    if keys and not reference.duplicated(subset=keys).any() and not result.duplicated(subset=keys).any():
        return _compare_unique_key_frame(reference, result, keys, compare_columns, matched_columns, missing_columns)
    return _compare_rowset_frame(reference, result, compare_columns, matched_columns, missing_columns)


def _comparison_key_columns(reference: pd.DataFrame, result: pd.DataFrame, *, key_column: str) -> list[str]:
    if key_column and key_column in reference.columns and key_column in result.columns:
        return [key_column]
    keys = [column for column in KEY_PRIORITY if column in reference.columns and column in result.columns]
    if "stay_id" in keys:
        return ["stay_id"]
    if "hadm_id" in keys:
        return ["hadm_id"]
    return keys[:1]


def _compare_unique_key_frame(
    reference: pd.DataFrame,
    result: pd.DataFrame,
    keys: list[str],
    compare_columns: list[str],
    matched_columns: list[str],
    missing_columns: list[str],
) -> dict[str, Any]:
    result_keyed = result.drop_duplicates(subset=keys).set_index(keys, drop=False)
    reference_keyed = reference.drop_duplicates(subset=keys).set_index(keys, drop=False)
    result_index = {_index_key(value) for value in result_keyed.index}
    reference_index = [_index_key(value) for value in reference_keyed.index]
    matched_rows = sum(1 for value in reference_index if value in result_index)
    expected_values = 0
    matched_values = 0
    for key_value in reference_keyed.index:
        normalized_key = _index_key(key_value)
        if normalized_key not in result_index:
            row = reference_keyed.loc[key_value]
            expected_values += sum(not _is_empty_value(row[column]) for column in compare_columns)
            continue
        ref_row = reference_keyed.loc[key_value]
        res_row = result_keyed.loc[key_value]
        for column in compare_columns:
            ref_value = ref_row[column]
            if _is_empty_value(ref_value):
                continue
            expected_values += 1
            if column in matched_columns and _directory_values_match(ref_value, res_row[column]):
                matched_values += 1
    return {
        "comparison_mode": "unique_key",
        "key_columns": keys,
        "expected_column_count": len(compare_columns),
        "matched_column_count": len(matched_columns),
        "missing_columns": missing_columns,
        "expected_row_count": int(len(reference_keyed)),
        "matched_row_count": int(matched_rows),
        "expected_value_count": int(expected_values),
        "matched_value_count": int(matched_values),
    }


def _compare_rowset_frame(
    reference: pd.DataFrame,
    result: pd.DataFrame,
    compare_columns: list[str],
    matched_columns: list[str],
    missing_columns: list[str],
) -> dict[str, Any]:
    if matched_columns:
        result_rows = {_row_signature(row, matched_columns) for _, row in result[matched_columns].iterrows()}
        matched_rows = sum(
            1
            for _, row in reference[matched_columns].iterrows()
            if _row_signature(row, matched_columns) in result_rows
        )
    else:
        matched_rows = 0
    expected_values = int(
        sum(not _is_empty_value(row[column]) for _, row in reference.iterrows() for column in compare_columns)
    )
    matched_values = int(matched_rows * max(len(matched_columns), 1)) if matched_columns else 0
    return {
        "comparison_mode": "row_set",
        "key_columns": [],
        "expected_column_count": len(compare_columns),
        "matched_column_count": len(matched_columns),
        "missing_columns": missing_columns,
        "expected_row_count": int(len(reference)),
        "matched_row_count": int(matched_rows),
        "expected_value_count": int(expected_values),
        "matched_value_count": min(matched_values, expected_values),
    }


def _index_key(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def _row_signature(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_directory_value(row[column]) for column in columns)


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    try:
        if value is pd.NA:
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.casefold() in {"nan", "nat", "none", "null"}


def _normalize_directory_value(value: Any) -> str:
    if _is_empty_value(value):
        return ""
    text = str(value).strip()
    duration_hours = _duration_text_to_hours(text)
    if duration_hours is not None:
        return _normalize_decimal(duration_hours)
    try:
        numeric = Decimal(text)
        if numeric.is_nan():
            return ""
        return _normalize_decimal(numeric)
    except (InvalidOperation, ValueError):
        pass
    return " ".join(text.casefold().split())


def _normalize_decimal(value: Decimal) -> str:
    normalized_decimal = value.quantize(NUMERIC_NORMALIZATION_PLACES, rounding=ROUND_HALF_UP)
    normalized = format(normalized_decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _duration_text_to_hours(text: str) -> Decimal | None:
    normalized = text.strip().casefold()
    match = re.fullmatch(
        r"(?:(?P<days>-?\d+)\s+days?\s+)?(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2}(?:\.\d+)?)",
        normalized,
    )
    if not match:
        return None
    days = Decimal(match.group("days") or "0")
    hours = Decimal(match.group("hours"))
    minutes = Decimal(match.group("minutes"))
    seconds = Decimal(match.group("seconds"))
    sign = Decimal("-1") if days < 0 else Decimal("1")
    return days * Decimal(24) + sign * (hours + minutes / Decimal(60) + seconds / Decimal(3600))


def _directory_values_match(left: Any, right: Any) -> bool:
    if _is_empty_value(left) and _is_empty_value(right):
        return True
    if _is_empty_value(left) or _is_empty_value(right):
        return False
    return _normalize_directory_value(left) == _normalize_directory_value(right)


def _public_feedback_from_directory_report(report: dict[str, Any]) -> dict[str, Any]:
    repair_targets: list[dict[str, Any]] = []
    for relative in report.get("missing_files") or []:
        repair_targets.append({"type": "missing_file", "relative_path": relative})
    for relative, columns in (report.get("missing_columns") or {}).items():
        repair_targets.append({"type": "missing_columns", "relative_path": relative, "columns": list(columns)})
    for item in report.get("file_reports") or []:
        if item.get("status") != "compared":
            continue
        if (
            int(item.get("result_rows", 0) or 0) > int(item.get("reference_rows", 0) or 0)
            and item.get("exact_row_precision", 1.0) < 0.98
        ):
            repair_targets.append(
                {
                    "type": "over_extraction",
                    "relative_path": item.get("relative_path", ""),
                    "result_rows": item.get("result_rows", 0),
                    "reference_rows": item.get("reference_rows", 0),
                    "exact_row_precision": item.get("exact_row_precision", 0.0),
                    "suggestion": "收紧 cohort 过滤、时间窗、item/code 过滤或 join 条件，减少 reference 外的多余行。",
                }
            )
        if item.get("missing_reference_rows", 0) > 0 and item.get("key_row_recall", 1.0) < 0.98:
            repair_targets.append(
                {
                    "type": "missing_reference_rows",
                    "relative_path": item.get("relative_path", ""),
                    "key_columns": item.get("key_columns", []),
                    "missing_reference_rows": item.get("missing_reference_rows", 0),
                    "key_row_recall": item.get("key_row_recall", 0.0),
                    "suggestion": "检查 key 生成、case 过滤、事件表来源和 join 逻辑。",
                }
            )
        for column, column_report in (item.get("per_column") or {}).items():
            if column_report.get("status") == "missing_column":
                continue
            expected = int(column_report.get("reference_non_empty_cells", 0) or 0)
            recall = float(column_report.get("recall", 1.0) or 0.0)
            if expected and recall < 0.98:
                repair_targets.append(
                    {
                        "type": "low_column_recall",
                        "relative_path": item.get("relative_path", ""),
                        "column": column,
                        "reference_non_empty_cells": expected,
                        "matched_non_empty_cells": int(column_report.get("matched_non_empty_cells", 0) or 0),
                        "recall": recall,
                        "suggestion": "优先检查该字段的来源列、单位/格式规范化、聚合逻辑和缺失值处理。",
                    }
                )
    return {
        "status": report.get("status", "SUCCESS"),
        "mode": report.get("mode", "reference_directory"),
        "metrics": report.get("metrics") or {},
        "repair_targets": repair_targets[:200],
    }


def _latest_result_package_manifest(root: Path) -> Path | None:
    manifests = sorted(
        (path for path in root.rglob("package_manifest.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in manifests:
        if path.parent.name == "result_package" or "result_package" in path.parts:
            return path
    return manifests[0] if manifests else None


def _latest_result_package_dir(root: Path) -> Path | None:
    dirs = sorted(
        (path for path in root.rglob("result_package") if path.is_dir()),
        key=_directory_latest_mtime_ns,
        reverse=True,
    )
    if dirs:
        return dirs[0]
    manifest = _latest_result_package_manifest(root)
    if manifest is not None:
        return manifest.parent
    return None


def _directory_latest_mtime_ns(path: Path) -> int:
    latest = path.stat().st_mtime_ns
    for child in path.rglob("*"):
        try:
            latest = max(latest, child.stat().st_mtime_ns)
        except OSError:
            continue
    return latest


def _latest_result_package_dir_for_summary(root: Path) -> Path | None:
    return _latest_result_package_dir(root)


def _find_package_cohort_file(package_dir: Path) -> Path | None:
    cohort_dir = package_dir / "cohort"
    if cohort_dir.is_dir():
        for suffix in ("*.csv", "*.tsv", "*.parquet", "*.jsonl", "*.json", "*.xlsx", "*.xls"):
            matches = sorted(cohort_dir.glob(suffix))
            if matches:
                return matches[0]
    return None


def _find_package_feature_file(package_dir: Path) -> Path | None:
    feature_dir = package_dir / "features"
    if feature_dir.is_dir():
        for suffix in ("*.csv", "*.tsv", "*.parquet", "*.jsonl", "*.json", "*.xlsx", "*.xls"):
            matches = sorted(feature_dir.glob(suffix))
            if matches:
                return matches[0]
    return None


def _structured_package_files(package_dir: Path) -> list[Path]:
    suffixes = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".parquet", ".jsonl", ".json", ".xlsx", ".xls")
    return sorted(
        path for path in package_dir.rglob("*")
        if path.is_file()
        and path.name != "package_manifest.json"
        and path.name != "result_manifest.json"
        and path.name != "target_field_mapping.json"
        and path.name.lower().endswith(suffixes)
    )


def _first_keyed_package_file(files: list[Path], contract: dict[str, Any]) -> Path | None:
    key_column = str(contract.get("key_column") or contract.get("record_grain") or "")
    for path in files:
        try:
            frame = _read_table_preview(path, nrows=20)
        except Exception:
            continue
        if key_column and key_column in frame.columns and not frame.empty:
            return path
    return files[0] if files else None


def _package_has_expected_key_coverage(files: list[Path], key_column: str, expected_count: int) -> bool:
    if not key_column or not expected_count:
        return True
    best_unique = 0
    for path in files:
        try:
            frame = _read_table_preview(path, nrows=None)
        except Exception:
            continue
        if key_column not in frame.columns or frame.empty:
            continue
        unique_count = int(frame[key_column].dropna().astype(str).nunique())
        best_unique = max(best_unique, unique_count)
        if unique_count == expected_count:
            return True
    return best_unique >= max(1, int(expected_count * 0.5))


def _read_table_preview(path: Path, nrows: int | None = 50) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path, dtype=object, nrows=nrows)
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", dtype=object, nrows=nrows)
    if name.endswith(".parquet"):
        frame = pd.read_parquet(path)
        return frame.head(nrows) if nrows is not None else frame
    if name.endswith(".xlsx") or name.endswith(".xls"):
        frame = pd.read_excel(path, dtype=object)
        return frame.head(nrows) if nrows is not None else frame
    if name.endswith(".jsonl"):
        return pd.read_json(path, lines=True, dtype=object, nrows=nrows)
    if name.endswith(".json"):
        frame = pd.read_json(path, dtype=object)
        return frame.head(nrows) if nrows is not None else frame
    raise ValueError(f"Unsupported table format: {path}")


def _write_lightweight_result_package_validation(
    *,
    root: Path,
    package_dir: Path,
    key_column: str,
    expected_count: int,
) -> Path:
    issues: list[str] = []
    checked_files: list[dict[str, Any]] = []
    key_coverages: list[int] = []
    for path in _structured_package_files(package_dir):
        try:
            frame = _read_table_preview(path, nrows=None)
        except Exception as exc:
            issues.append(f"could not read package table {path}: {exc}")
            continue
        row_count = int(len(frame))
        key_unique_count = int(frame[key_column].dropna().astype(str).nunique()) if key_column in frame.columns else 0
        if key_unique_count:
            key_coverages.append(key_unique_count)
        if row_count == 0:
            issues.append(f"package table is empty: {path}")
        checked_files.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "row_count": row_count,
                "column_count": int(len(frame.columns)),
                "key_unique_count": key_unique_count,
            }
        )
    if not checked_files:
        issues.append("result_package contains no structured tables")
    if expected_count and key_column and (not key_coverages or max(key_coverages) < max(1, int(expected_count * 0.5))):
        issues.append(
            f"no package table covers enough expected keys: expected={expected_count}, "
            f"best={max(key_coverages) if key_coverages else 0}"
        )
    payload = {
        "schema_version": 1,
        "valid": not issues,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "package_root": str(package_dir),
        "key_column": key_column,
        "expected_key_count": expected_count,
        "best_key_unique_count": max(key_coverages) if key_coverages else 0,
        "checked_files": checked_files,
        "issues": issues,
        "written_by": "host_lightweight_validation",
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = root / "result_package_validation_report.json"
    _write_json(path, payload)
    return path


def _write_reference_run_record(
    *,
    root: Path,
    contract: dict[str, Any],
    package_dir: Path,
    package_reference: Path | None,
    cohort: Path,
    features: Path,
    final_dataset: Path,
    result_manifest: Path | None,
    mapping: Path | None,
    skill_usage: Path | None,
) -> Path:
    payload = {
        "schema_version": 1,
        "status": "SUCCESS",
        "workflow": "reference-guided-train-validate",
        "dataset_split": contract.get("dataset_split", ""),
        "record_grain": contract.get("record_grain", ""),
        "key_column": contract.get("key_column", ""),
        "artifacts": {
            "result_package": str(package_dir),
            "package_reference": str(package_reference) if package_reference else "",
            "cohort": str(cohort),
            "features_wide": str(features),
            "final_dataset": str(final_dataset),
            "result_manifest": str(result_manifest) if result_manifest else "",
            "target_field_mapping": str(mapping) if mapping else "",
            "skill_usage_report": str(skill_usage) if skill_usage else "",
        },
        "written_by": "host",
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = root / "run_record.json"
    _write_json(path, payload)
    return path


def _latest_named_file(root: Path, filename: str) -> Path | None:
    matches = sorted(
        (path for path in root.rglob(filename) if path.is_file()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return matches[0] if matches else None


def _optional_latest_paths(root: Path, names: dict[str, str]) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for key, filename in names.items():
        path = _latest_named_file(root, filename)
        if path is not None:
            values[key] = path
    return values


def _validation_key_count(contract: dict[str, Any]) -> int:
    path = Path(str((contract.get("paths") or {}).get("validation_keys") or ""))
    if not path.is_file():
        return 0
    try:
        return int(pd.read_csv(path, dtype=object).shape[0])
    except Exception:
        return 0


def _find_reference_file(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    preferred = root / "reference.csv"
    if preferred.is_file():
        return preferred
    for suffix in ("*.csv", "*.csv.gz", "*.tsv", "*.tsv.gz", "*.parquet", "*.jsonl", "*.json", "*.xlsx", "*.xls"):
        matches = sorted(root.glob(suffix))
        if matches:
            return matches[0]
    preferred_nested = root / "cohort"
    if preferred_nested.is_dir():
        for suffix in ("*.csv", "*.csv.gz", "*.tsv", "*.tsv.gz", "*.parquet", "*.jsonl", "*.json", "*.xlsx", "*.xls"):
            matches = sorted(preferred_nested.glob(suffix))
            if matches:
                return matches[0]
    for suffix in ("*.csv", "*.csv.gz", "*.tsv", "*.tsv.gz", "*.parquet", "*.jsonl", "*.json", "*.xlsx", "*.xls"):
        matches = sorted(root.rglob(suffix))
        if matches:
            return matches[0]
    return None


def _resolve_reference_root(split: Path, split_name: str) -> Path:
    """Return the hidden/public reference root for a split.

    Validation/test references are normally stored under reference_private, but
    materialized local experiments may use reference when no privacy boundary is
    needed. Prefer reference_private when both exist.
    """
    split_root = split / split_name
    for dirname in ("reference_private", "reference"):
        candidate = split_root / dirname
        if _find_reference_file(candidate) is not None:
            return candidate
    return split_root / "reference_private"


def _key_column_from_keys(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        frame = pd.read_csv(path, nrows=0)
    except Exception:
        return ""
    return str(frame.columns[0]) if len(frame.columns) else ""


def _task_text_with_reference_contract(task_text: str, contract: dict[str, Any]) -> str:
    """Add deterministic reference-package hints without exposing validation reference."""
    hints: list[str] = []
    record_grain = str(contract.get("record_grain") or contract.get("key_column") or "")
    if record_grain:
        hints.append(f"record_grain={record_grain}")
    if record_grain == "stay_id":
        hints.append("care_setting=ICU")
    elif record_grain == "hadm_id":
        hints.append("care_setting=Non-ICU")
    outcome = _infer_outcome_from_contract(contract)
    if outcome:
        hints.append(f"outcome={outcome}")
    if not hints:
        return task_text
    return (
        f"{task_text}\n\n"
        "[reference contract inferred from training split]\n"
        + "\n".join(f"- {hint}" for hint in hints)
    )


def _infer_outcome_from_contract(contract: dict[str, Any]) -> str:
    text_parts = [
        str(contract.get("dataset_split") or ""),
        str(contract.get("split_manifest") or ""),
    ]
    package = contract.get("reference_package")
    if isinstance(package, dict):
        text_parts.append(str(package.get("primary_reference") or ""))
        for section in ("cohort_files", "feature_files"):
            for item in package.get(section, []) or []:
                if isinstance(item, dict):
                    text_parts.extend([str(item.get("source") or ""), str(item.get("path") or "")])
    text = " ".join(text_parts).casefold()
    if "readmission" in text:
        return "Readmission"
    if "mortality" in text or "death" in text:
        return "Mortality"
    if "los" in text or "length_of_stay" in text or "length-of-stay" in text:
        return "Length of Stay"
    return ""


def _read_reference_preview(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, nrows=50, dtype=object)
    if name.endswith(".parquet"):
        return pd.read_parquet(path).head(50)
    if name.endswith(".jsonl"):
        return pd.read_json(path, lines=True, nrows=50, dtype=object)
    if name.endswith(".json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return pd.DataFrame(value).head(50)
        if isinstance(value, dict):
            rows = value.get("rows") if isinstance(value.get("rows"), list) else [value]
            return pd.DataFrame(rows).head(50)
    sep = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
    return pd.read_csv(path, sep=sep, nrows=50, dtype=object)


def _infer_key_column(frame: pd.DataFrame) -> str:
    for column in KEY_PRIORITY:
        if column in frame.columns:
            return column
    return str(frame.columns[0]) if len(frame.columns) else ""


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_public_log_tail(path: Path, *, max_chars: int) -> str:
    try:
        text = path.expanduser().resolve().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
