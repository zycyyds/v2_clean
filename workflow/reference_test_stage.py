from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agentscope.agent import ReActAgent

from agent.reference_runtime import (
    ENGINEER_CODE_READ_ROOTS,
    create_reference_memory,
    create_reference_toolkit,
    make_reference_model,
)
from agent_tools.context import EngineerToolContext
from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context
from lib.agent_runtime import format_message_content, make_user_msg
from workflow.reference_evaluation import (
    _iter_table_dicts,
    _key_columns,
    _normalize_value,
    _structured_package_files,
    _table_columns,
)
from workflow.reference_guided import (
    _directory_sha256,
    _latest_result_package_dir,
    _remove_fixed_reference_incompatible_tools,
    _restore_claude_style_capabilities,
    _write_json,
    evaluate_reference_directory_package,
    infer_reference_contract,
)
from workflow.skill_adapter import restore_bundle_variants


@dataclass(frozen=True)
class ReferenceCheckpointTestConfig:
    dataset_split: str | Path
    experiment_dir: str | Path
    checkpoint_dir: str | Path
    frozen_script_bundle: str | Path
    best_round: int
    best_attempt: int
    best_score: float
    max_iters: int = 60

    def normalized(self) -> "ReferenceCheckpointTestConfig":
        split = Path(self.dataset_split).expanduser().resolve()
        experiment = Path(self.experiment_dir).expanduser().resolve()
        checkpoint = Path(self.checkpoint_dir).expanduser().resolve()
        bundle = Path(self.frozen_script_bundle).expanduser().resolve()
        if not split.is_dir():
            raise ValueError(f"dataset_split does not exist: {split}")
        if not bundle.is_dir():
            raise ValueError(f"frozen_script_bundle does not exist: {bundle}")
        if self.best_round < 1 or self.best_attempt < 1:
            raise ValueError("checkpoint test requires a committed formal best")
        if self.max_iters < 1:
            raise ValueError("max_iters must be positive")
        checkpoint.mkdir(parents=True, exist_ok=True)
        return ReferenceCheckpointTestConfig(
            dataset_split=split,
            experiment_dir=experiment,
            checkpoint_dir=checkpoint,
            frozen_script_bundle=bundle,
            best_round=self.best_round,
            best_attempt=self.best_attempt,
            best_score=float(self.best_score),
            max_iters=self.max_iters,
        )


class ReferenceCheckpointTestRuntime:
    """Run a fresh ReferenceCodeAgent in test mode, then evaluate after it exits."""

    def __init__(self, config: ReferenceCheckpointTestConfig) -> None:
        self.config = config.normalized()
        self.contract = infer_reference_contract(self.config.dataset_split)
        self.sanitized_contract = _build_sanitized_test_contract(self.contract)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.bundle = Path(self.config.frozen_script_bundle)
        self.agent_runs_dir = self.checkpoint_dir / "agent_runs"
        self.test_run_dir = self.checkpoint_dir / "test_run"
        self.test_evaluation_dir = self.checkpoint_dir / "test_evaluation"
        self.report_path = self.checkpoint_dir / "checkpoint_report.json"
        self._last_gate: dict[str, Any] | None = None

    def run_sync(self) -> dict[str, Any]:
        bundle_hash_before = _directory_sha256(self.bundle)
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            agent_package = self._run_agent_session()
            gate = self._last_gate or self._quality_gate(agent_package, bundle_hash_before)
            bundle_hash_after = _directory_sha256(self.bundle)
            if bundle_hash_after != bundle_hash_before:
                gate = {
                    **gate,
                    "valid": False,
                    "status": "NEEDS_REPAIR",
                    "issues": [*(gate.get("issues") or []), "frozen script bundle changed during test"],
                }
            if not gate.get("valid"):
                return self._write_report(
                    test_status="test_failed",
                    started_at=started_at,
                    gate=gate,
                    evaluation={},
                    metrics={},
                    bundle_hash_before=bundle_hash_before,
                    bundle_hash_after=bundle_hash_after,
                )

            output_package = self.test_run_dir / "result_package"
            if output_package.exists():
                shutil.rmtree(output_package)
            output_package.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(agent_package, output_package)

            # The Agent session has returned and clear_phase_context() has run.
            # Only the host process can resolve and read the private test reference.
            private_reference = _private_test_reference(self.contract)
            from workflow import reference_guided

            evaluation = reference_guided.evaluate_reference_directory_package(
                result_package=output_package,
                validation_reference_root=private_reference,
                output_dir=self.test_evaluation_dir,
                key_column=str(self.contract.get("key_column") or ""),
            )
            evaluation_report = json.loads(
                Path(evaluation["evaluation_report"]).read_text(encoding="utf-8")
            )
            metrics = evaluation_report.get("metrics") or {}
            return self._write_report(
                test_status="success",
                started_at=started_at,
                gate=gate,
                evaluation=evaluation,
                metrics=metrics,
                bundle_hash_before=bundle_hash_before,
                bundle_hash_after=_directory_sha256(self.bundle),
                result_package=output_package,
            )
        except Exception as exc:
            return self._write_report(
                test_status="test_failed",
                started_at=started_at,
                gate=self._last_gate or {},
                evaluation={},
                metrics={},
                bundle_hash_before=bundle_hash_before,
                bundle_hash_after=_directory_sha256(self.bundle),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _run_agent_session(self) -> Path:
        phase = init_phase_session(self.agent_runs_dir, "reference_code_agent")
        phase_root = Path(phase["phase_root"])
        run_id = f"reference_code_test_checkpoint_{self.config.best_round:04d}"
        set_phase_context(run_id, self.agent_runs_dir, "reference_code_agent", phase_root)
        sanitized_contract_path = phase_root / "sanitized_test_contract.json"
        _write_json(sanitized_contract_path, self.sanitized_contract)
        read_roots = self._read_roots(sanitized_contract_path)
        reports = {
            "sanitized_test_contract": str(sanitized_contract_path),
            "frozen_script_bundle": str(self.bundle),
            "train_reference_root": self.sanitized_contract["paths"]["train_reference_root"],
        }
        task_text = self._task_text(phase_root, sanitized_contract_path)
        try:
            context = EngineerToolContext.from_task(
                task_text=task_text,
                engineer_phase_root=phase_root,
                explorer_phase_root=str(self.bundle),
                additional_read_roots=read_roots,
                require_skill_plan=False,
                split_mode="test",
                required_report_paths=reports,
            )
            toolkit, _ = create_reference_toolkit(context)
            _remove_fixed_reference_incompatible_tools(toolkit)
            restore_bundle_variants(self.bundle, context.variant_root)
            _restore_claude_style_capabilities(self.bundle, context.workspace_dir / "capabilities")
            model, formatter = make_reference_model()
            agent = ReActAgent(
                name="ReferenceCodeAgent",
                sys_prompt=_checkpoint_test_system_prompt(context),
                model=model,
                formatter=formatter,
                toolkit=toolkit,
                memory=create_reference_memory(
                    {
                        "phase_name": "reference_code_agent",
                        "phase_root": str(phase_root),
                        "run_id": run_id,
                        "run_root": str(self.agent_runs_dir),
                    },
                    model,
                ),
                parallel_tool_calls=False,
                max_iters=self.config.max_iters,
                print_hint_msg=False,
            )
            prompt = task_text
            package: Path | None = None
            for repair_index in range(3):
                response = asyncio.run(agent(make_user_msg(name="user", content=prompt)))
                response_text = format_message_content(getattr(response, "content", "")).strip()
                (phase_root / f"agent_response_{repair_index + 1}.txt").write_text(
                    response_text,
                    encoding="utf-8",
                )
                package = _latest_result_package_dir(phase_root)
                if package is None:
                    gate = {
                        "schema_version": 1,
                        "valid": False,
                        "status": "NEEDS_REPAIR",
                        "issues": ["ReferenceCodeAgent did not publish workspace/result_package"],
                    }
                else:
                    gate = self._quality_gate(package, _directory_sha256(self.bundle))
                self._last_gate = gate
                _write_json(phase_root / f"test_quality_gate_{repair_index + 1}.json", gate)
                if gate.get("valid") or repair_index == 2:
                    break
                prompt = _test_repair_prompt(gate, context.workspace_dir)
            if package is None:
                raise RuntimeError("ReferenceCodeAgent produced no test result_package")
            return package
        finally:
            clear_phase_context()

    def _read_roots(self, sanitized_contract_path: Path) -> list[str | Path]:
        paths = self.sanitized_contract["paths"]
        roots: list[str | Path] = [
            *ENGINEER_CODE_READ_ROOTS,
            self.bundle,
            paths["train_raw"],
            paths["train_reference_root"],
            paths["test_raw"],
            sanitized_contract_path,
        ]
        for name in ("train_keys", "test_keys"):
            path = Path(str(paths.get(name) or ""))
            if path.is_file():
                roots.append(path)
        return roots

    def _task_text(self, phase_root: Path, sanitized_contract_path: Path) -> str:
        paths = self.sanitized_contract["paths"]
        return f"""\
请以 ReferenceCodeAgent 的 test 模式完成当前 checkpoint 的一次测试数据处理。

【唯一允许的当前实验输入】
- 冻结脚本包（只读）：{self.bundle}
- 脱敏 test contract：{sanitized_contract_path}
- train raw：{paths['train_raw']}
- train reference：{paths['train_reference_root']}
- train keys：{paths['train_keys']}
- test raw：{paths['test_raw']}
- test keys：{paths['test_keys']}

【执行要求】
1. 这是一个统一任务，不要判断或声明 standalone/non-standalone 模式。
2. 自己阅读冻结脚本，决定直接执行、参数化路径、补 runner、调整调用顺序或衔接已有脚本。
3. 不得修改冻结脚本包；新增 runner 和适配代码只能写在本次 workspace。
4. 可以先用 train/raw -> train/reference 检查衔接是否正确，再用完全相同的 runner 处理 test/raw。
5. 不得搜索或读取 experiments 根目录、其他实验、validation 数据、validation/test reference 或任何评分报告。
6. 最终必须通过 PublishDirectoryArtifact 发布完整目录到 {phase_root / 'workspace' / 'result_package'}。
7. 结果必须包含 train/reference 中全部同名业务文件和 schema，并覆盖 test keys。
"""

    def _quality_gate(self, result_package: Path, bundle_hash_before: str) -> dict[str, Any]:
        paths = self.sanitized_contract["paths"]
        train_reference = Path(paths["train_reference_root"])
        test_keys = Path(paths["test_keys"])
        validation_package = Path(self.config.experiment_dir) / "active_bundle" / "results" / "result_package"
        return _test_package_quality_gate(
            result_package=result_package,
            train_reference_root=train_reference,
            test_keys=test_keys,
            key_column=str(self.sanitized_contract.get("key_column") or "stay_id"),
            frozen_script_bundle=self.bundle,
            frozen_hash_before=bundle_hash_before,
            validation_result_package=validation_package if validation_package.is_dir() else None,
        )

    def _write_report(
        self,
        *,
        test_status: str,
        started_at: str,
        gate: dict[str, Any],
        evaluation: dict[str, Any],
        metrics: dict[str, Any],
        bundle_hash_before: str,
        bundle_hash_after: str,
        result_package: Path | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        report = {
            "schema_version": 1,
            "workflow": "reference-checkpoint-test",
            "test_status": test_status,
            "best_round": self.config.best_round,
            "best_attempt": self.config.best_attempt,
            "validation_best_score": self.config.best_score,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_dir": str(self.checkpoint_dir),
            "frozen_script_bundle": str(self.bundle),
            "frozen_script_bundle_sha256_before": bundle_hash_before,
            "frozen_script_bundle_sha256_after": bundle_hash_after,
            "result_package": str(result_package) if result_package else "",
            "gate": gate,
            "evaluation": evaluation,
            "metrics": metrics,
            "error": error,
        }
        _write_json(self.report_path, report)
        report["report_path"] = str(self.report_path)
        return report


def _build_sanitized_test_contract(contract: dict[str, Any]) -> dict[str, Any]:
    source_paths = contract.get("paths") or {}
    train_raw = Path(str(source_paths.get("train_raw") or "")).expanduser().resolve()
    paths = {
        "train_raw": str(train_raw),
        "train_reference_root": str(
            Path(str(source_paths.get("train_reference_root") or "")).expanduser().resolve()
        ),
        "train_keys": str(train_raw.parent / "keys.csv"),
        "test_raw": str(Path(str(source_paths.get("test_raw") or "")).expanduser().resolve()),
        "test_keys": str(Path(str(source_paths.get("test_keys") or "")).expanduser().resolve()),
    }
    return {
        "schema_version": 1,
        "workflow": "reference-checkpoint-test",
        "split_mode": "test",
        "dataset_profile": contract.get("dataset_profile", ""),
        "reference_type": contract.get("reference_type", "reference_directory"),
        "record_grain": contract.get("record_grain", ""),
        "key_column": contract.get("key_column", ""),
        "paths": paths,
    }


def _checkpoint_test_system_prompt(context: EngineerToolContext) -> str:
    read_roots = "\n".join(f"- {path}" for path in context.read_roots)
    return f"""\
你是 ReferenceCodeAgent。当前是 test checkpoint 阶段，使用与 validation attempt 相同的 Agent 实现，但拥有全新的 memory、toolkit 和 workspace。

你只能读取以下授权路径：
{read_roots}

你必须自行理解并衔接冻结脚本，不得搜索历史实验。ExecutePython 只能写本步骤 OUTPUT_DIR；最终用 PublishDirectoryArtifact 发布完整 result_package。任何 validation/test 金标准和评分结果都不可见。
"""


def _test_repair_prompt(gate: dict[str, Any], workspace: Path) -> str:
    issues = "\n".join(f"- {issue}" for issue in gate.get("issues") or [])
    return f"""\
当前 test 结果未通过公开结构门禁。以下信息只来自 train schema 和 test keys，不含 test 金标准：
{issues}

请继续在同一个 Agent 会话中修复 runner 或输出，并重新发布完整目录到 {workspace / 'result_package'}。不得修改冻结脚本包。
"""


def _test_package_quality_gate(
    *,
    result_package: str | Path,
    train_reference_root: str | Path,
    test_keys: str | Path,
    key_column: str,
    frozen_script_bundle: str | Path,
    frozen_hash_before: str,
    validation_result_package: str | Path | None = None,
) -> dict[str, Any]:
    package = Path(result_package).expanduser().resolve()
    reference = Path(train_reference_root).expanduser().resolve()
    keys_path = Path(test_keys).expanduser().resolve()
    bundle = Path(frozen_script_bundle).expanduser().resolve()
    issues: list[str] = []
    files: list[dict[str, Any]] = []
    expected_files = [
        path
        for path in _structured_package_files(reference)
        if path.name not in {"package_manifest.json", "result_manifest.json", "target_field_mapping.json", "reference.csv"}
    ]
    test_key_values = _read_key_values(keys_path, key_column)
    for reference_file in expected_files:
        relative = reference_file.relative_to(reference).as_posix()
        result_file = package / relative
        file_report: dict[str, Any] = {"relative_path": relative, "passed": True, "issues": []}
        if not result_file.is_file():
            file_report["passed"] = False
            file_report["issues"].append("missing file")
        else:
            try:
                expected_columns = _table_columns(reference_file)
                result_columns = _table_columns(result_file)
                if result_columns != expected_columns:
                    file_report["passed"] = False
                    file_report["issues"].append("schema differs from train reference")
                business_key_columns = _key_columns(relative, expected_columns, key_column)
                result_business_duplicates = _duplicate_business_key_count(
                    result_file,
                    business_key_columns,
                )
                reference_business_duplicates = _duplicate_business_key_count(
                    reference_file,
                    business_key_columns,
                )
                row_count, observed_keys, duplicate_keys, placeholder_only = _scan_gate_file(
                    result_file,
                    result_columns,
                    key_column=key_column,
                    require_unique_key=relative.startswith("cohort/") or relative.endswith("/labels.csv"),
                )
                file_report.update(
                    {
                        "row_count": row_count,
                        "observed_key_count": len(observed_keys),
                        "duplicate_key_count": duplicate_keys,
                        "business_key_columns": business_key_columns,
                        "duplicate_business_key_count": result_business_duplicates,
                        "train_duplicate_business_key_count": reference_business_duplicates,
                    }
                )
                if result_business_duplicates and not reference_business_duplicates:
                    file_report["passed"] = False
                    file_report["issues"].append(
                        f"contains {result_business_duplicates} duplicate business keys"
                    )
                if row_count == 0 and _count_file_rows(reference_file) > 0:
                    file_report["passed"] = False
                    file_report["issues"].append("empty output file")
                if placeholder_only:
                    file_report["passed"] = False
                    file_report["issues"].append("output contains only placeholder values")
                if key_column in result_columns:
                    unknown_keys = sorted(observed_keys - test_key_values)
                    if unknown_keys:
                        file_report["passed"] = False
                        file_report["issues"].append(
                            f"contains {len(unknown_keys)} unknown test keys"
                        )
                if (relative.startswith("cohort/") or relative.endswith("/labels.csv")) and key_column in result_columns:
                    missing_keys = sorted(test_key_values - observed_keys)
                    if missing_keys:
                        file_report["passed"] = False
                        file_report["issues"].append(f"missing {len(missing_keys)} test keys")
                    if duplicate_keys:
                        file_report["passed"] = False
                        file_report["issues"].append(f"contains {duplicate_keys} duplicate primary keys")
            except Exception as exc:
                file_report["passed"] = False
                file_report["issues"].append(f"read error: {type(exc).__name__}: {exc}")
        if not file_report["passed"]:
            issues.extend(f"{relative}: {item}" for item in file_report["issues"])
        files.append(file_report)

    current_bundle_hash = _directory_sha256(bundle)
    if current_bundle_hash != frozen_hash_before:
        issues.append("frozen script bundle hash changed")
    if validation_result_package:
        validation_package = Path(validation_result_package).expanduser().resolve()
        if validation_package.is_dir() and package.is_dir() and _directory_sha256(validation_package) == _directory_sha256(package):
            issues.append("test output is a direct copy of the validation result package")
    return {
        "schema_version": 1,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "valid": not issues,
        "expected_file_count": len(expected_files),
        "test_key_count": len(test_key_values),
        "frozen_script_bundle_sha256": current_bundle_hash,
        "issues": issues,
        "files": files,
    }


def _scan_gate_file(
    path: Path,
    columns: list[str],
    *,
    key_column: str,
    require_unique_key: bool,
) -> tuple[int, set[str], int, bool]:
    observed_keys: set[str] = set()
    duplicate_keys = 0
    row_count = 0
    non_placeholder_value = False
    placeholders = {"placeholder", "todo", "unknown", "n/a", "not available"}
    for row in _iter_table_dicts(path):
        row_count += 1
        if key_column in columns:
            value = str(row.get(key_column, "")).strip()
            if require_unique_key and value in observed_keys:
                duplicate_keys += 1
            observed_keys.add(value)
        if row_count <= 1000:
            for column in columns:
                if column == key_column:
                    continue
                value = str(row.get(column, "")).strip().casefold()
                if value and value not in placeholders:
                    non_placeholder_value = True
    placeholder_only = row_count > 0 and len(columns) > int(key_column in columns) and not non_placeholder_value
    return row_count, observed_keys, duplicate_keys, placeholder_only


def _duplicate_business_key_count(path: Path, key_columns: list[str]) -> int:
    if not key_columns:
        return 0
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in _iter_table_dicts(path):
        key = tuple(_normalize_value(row.get(column, "")) for column in key_columns)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def _read_key_values(path: Path, key_column: str) -> set[str]:
    if not path.is_file():
        raise ValueError(f"test keys do not exist: {path}")
    values = {
        str(row.get(key_column, "")).strip()
        for row in _iter_table_dicts(path)
        if str(row.get(key_column, "")).strip()
    }
    if not values:
        raise ValueError(f"test keys contain no {key_column} values: {path}")
    return values


def _count_file_rows(path: Path) -> int:
    return sum(1 for _ in _iter_table_dicts(path))


def _private_test_reference(contract: dict[str, Any]) -> Path:
    paths = contract.get("paths") or {}
    for name in ("test_reference_private_root", "test_reference_root", "test_reference_private"):
        raw = str(paths.get(name) or "").strip()
        if raw:
            path = Path(raw).expanduser().resolve()
            if path.is_dir():
                return path
    raise ValueError("dataset split does not expose a private test reference directory")


@dataclass(frozen=True)
class ReferenceTestStageConfig:
    dataset_split: str | Path
    experiment_dir: str | Path
    adapter_script: str | Path

    def normalized(self) -> "ReferenceTestStageConfig":
        split = Path(self.dataset_split).expanduser().resolve()
        if not split.is_dir():
            raise ValueError(f"dataset_split does not exist: {split}")
        experiment = Path(self.experiment_dir).expanduser().resolve()
        adapter = Path(self.adapter_script).expanduser().resolve()
        if not adapter.is_file():
            raise ValueError(f"adapter_script does not exist: {adapter}")
        return ReferenceTestStageConfig(
            dataset_split=split,
            experiment_dir=experiment,
            adapter_script=adapter,
        )


def run_reference_test_stage(config: ReferenceTestStageConfig) -> dict[str, Any]:
    cfg = config.normalized()
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    contract = infer_reference_contract(cfg.dataset_split)
    paths = contract.get("paths") or {}
    test_raw = _required_path(paths, "test_raw")
    test_keys = _required_path(paths, "test_keys")
    train_reference_root = _required_path(paths, "train_reference_root")
    test_reference_root = (
        _optional_existing_path(paths.get("test_reference_private_root"))
        or _optional_existing_path(paths.get("test_reference_root"))
        or _optional_existing_path(paths.get("test_reference_private"))
    )
    if test_reference_root is None:
        raise ValueError("dataset_split does not expose a test reference directory")

    run_root = cfg.experiment_dir / "test_run"
    output_package = run_root / "result_package"
    if output_package.exists():
        shutil.rmtree(output_package)
    run_root.mkdir(parents=True, exist_ok=True)

    adapter_snapshot = cfg.experiment_dir / "adapter_script.py"
    if cfg.adapter_script.resolve() != adapter_snapshot.resolve():
        shutil.copy2(cfg.adapter_script, adapter_snapshot)
    run_log = cfg.experiment_dir / "test_adapter_run.log"
    env = os.environ.copy()
    env.update(
        {
            "RAW_ROOT": str(test_raw),
            "KEYS_CSV": str(test_keys),
            "TRAIN_REF_ROOT": str(train_reference_root),
            "OUTPUT_DIR": str(run_root),
        }
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    with run_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            [sys.executable, str(adapter_snapshot)],
            cwd=str(cfg.experiment_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finished_at = datetime.now().isoformat(timespec="seconds")

    evaluation_paths: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    if process.returncode == 0:
        evaluation_paths = evaluate_reference_directory_package(
            result_package=output_package,
            validation_reference_root=test_reference_root,
            output_dir=cfg.experiment_dir / "test_evaluation",
            key_column=str(contract.get("key_column") or ""),
        )
        report_path = Path(evaluation_paths["evaluation_report"])
        if report_path.is_file():
            evaluation_report = json.loads(report_path.read_text(encoding="utf-8"))
            metrics = evaluation_report.get("metrics") or {}

    report = {
        "schema_version": 1,
        "status": "SUCCESS" if process.returncode == 0 and metrics else "FAILED",
        "workflow": "reference-test-evaluate",
        "started_at": started_at,
        "finished_at": finished_at,
        "dataset_split": str(cfg.dataset_split),
        "experiment_dir": str(cfg.experiment_dir),
        "adapter_script": str(cfg.adapter_script),
        "adapter_snapshot": str(adapter_snapshot),
        "run_log": str(run_log),
        "returncode": process.returncode,
        "result_package": str(output_package),
        "evaluation": evaluation_paths,
        "metrics": metrics,
        "contract_paths": {
            "test_raw": str(test_raw),
            "test_keys": str(test_keys),
            "train_reference_root": str(train_reference_root),
            "test_reference_root": str(test_reference_root),
        },
    }
    report_path = cfg.experiment_dir / "reference_test_stage_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _required_path(paths: dict[str, Any], key: str) -> Path:
    value = paths.get(key)
    path = Path(str(value or "")).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"contract path {key} does not exist: {path}")
    return path


def _optional_existing_path(value: Any) -> Path | None:
    if not str(value or "").strip():
        return None
    path = Path(str(value)).expanduser().resolve()
    return path if path.exists() else None
