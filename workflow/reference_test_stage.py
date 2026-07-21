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
    create_reference_compression_config,
    create_reference_memory,
    create_reference_toolkit,
    make_reference_model,
)
from agent_tools.context import EngineerToolContext
from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context
from lib.agent_runtime import format_message_content, make_user_msg
from workflow.reference_evaluation import (
    _structured_package_files,
)
from workflow.reference_package_gate import (
    MIMIC_EXPECTED_BUSINESS_FILE_COUNT,
    validate_business_result_package,
)
from workflow.reference_guided import (
    _directory_sha256,
    _remove_fixed_reference_incompatible_tools,
    _write_json,
    evaluate_reference_directory_package,
    infer_reference_contract,
)
from workflow.reference_pipeline import (
    pipeline_directory_sha256,
    run_pipeline_isolated,
    validate_pipeline_structure,
)


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
            runner_spec = self._run_agent_session()
            runner_gate = self._last_gate or {}
            bundle_hash_after = _directory_sha256(self.bundle)
            if bundle_hash_after != bundle_hash_before:
                runner_gate = {
                    **runner_gate,
                    "valid": False,
                    "status": "NEEDS_REPAIR",
                    "issues": [*(runner_gate.get("issues") or []), "frozen script bundle changed during test"],
                }
            if not runner_gate.get("valid"):
                return self._write_report(
                    test_status="test_failed",
                    started_at=started_at,
                    gate=runner_gate,
                    evaluation={},
                    metrics={},
                    bundle_hash_before=bundle_hash_before,
                    bundle_hash_after=bundle_hash_after,
                )

            output_package = self.test_run_dir / "result_package"
            execution = run_pipeline_isolated(
                pipeline_dir=self.bundle / "pipeline",
                raw_root=self.sanitized_contract["paths"]["test_raw"],
                output_dir=output_package,
                split_mode="test",
                log_path=self.test_run_dir / "pipeline_run.log",
            )
            if execution["returncode"] != 0:
                gate = {
                    **runner_gate,
                    "valid": False,
                    "status": "PIPELINE_RULE_FAILURE",
                    "issues": [*(runner_gate.get("issues") or []), "frozen pipeline failed on test raw"],
                    "execution": execution,
                }
                self._last_gate = gate
                return self._write_report(
                    test_status="pipeline_rule_failure",
                    started_at=started_at,
                    gate=gate,
                    evaluation={},
                    metrics={},
                    bundle_hash_before=bundle_hash_before,
                    bundle_hash_after=_directory_sha256(self.bundle),
                    result_package=output_package if output_package.is_dir() else None,
                )

            gate = self._quality_gate(output_package, bundle_hash_before)
            gate["runner_spec"] = str(runner_spec)
            gate["execution"] = execution
            self._last_gate = gate
            if not gate.get("valid"):
                gate["status"] = "PIPELINE_RULE_FAILURE"
                return self._write_report(
                    test_status="pipeline_rule_failure",
                    started_at=started_at,
                    gate=gate,
                    evaluation={},
                    metrics={},
                    bundle_hash_before=bundle_hash_before,
                    bundle_hash_after=_directory_sha256(self.bundle),
                    result_package=output_package,
                )

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
                compression_config=create_reference_compression_config(model),
                parallel_tool_calls=False,
                max_iters=self.config.max_iters,
                print_hint_msg=False,
            )
            prompt = task_text
            runner_spec: Path | None = None
            for repair_index in range(3):
                response = asyncio.run(agent(make_user_msg(name="user", content=prompt)))
                response_text = format_message_content(getattr(response, "content", "")).strip()
                (phase_root / f"agent_response_{repair_index + 1}.txt").write_text(
                    response_text,
                    encoding="utf-8",
                )
                runner_spec = context.workspace_dir / "runner_spec.json"
                gate = _runner_spec_quality_gate(
                    runner_spec=runner_spec,
                    workspace=context.workspace_dir,
                    frozen_script_bundle=self.bundle,
                    test_raw=Path(self.sanitized_contract["paths"]["test_raw"]),
                    output_dir=self.test_run_dir / "result_package",
                    train_reference_root=Path(
                        self.sanitized_contract["paths"]["train_reference_root"]
                    ),
                )
                self._last_gate = gate
                _write_json(phase_root / f"runner_spec_gate_{repair_index + 1}.json", gate)
                if gate.get("valid") or repair_index == 2:
                    break
                prompt = _test_repair_prompt(gate, context.workspace_dir)
            if runner_spec is None or not runner_spec.is_file():
                raise RuntimeError("ReferenceCodeAgent produced no runner_spec.json")
            return runner_spec
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
- 冻结 Pipeline SHA-256：{pipeline_directory_sha256(self.bundle / 'pipeline') if (self.bundle / 'pipeline').is_dir() else ''}
- 脱敏 test contract：{sanitized_contract_path}
- train raw：{paths['train_raw']}
- train reference：{paths['train_reference_root']}
- train keys：{paths['train_keys']}
- test raw：{paths['test_raw']}
- test keys：{paths['test_keys']}

【执行要求】
1. 冻结入口唯一固定为：{self.bundle / 'pipeline/run.py'}。
2. 你只能在 workspace 写 runner_spec.json，字段必须包含 schema_version、entrypoint、pipeline_sha256、raw_root、output_dir、split_mode。
3. raw_root 固定为 {paths['test_raw']}；output_dir 固定为 {self.test_run_dir / 'result_package'}；split_mode 固定为 test。
4. 不得创建 build_test.py、runner.py、shell 脚本、adapter 或任何业务转换代码，也不得生成或修改业务 CSV。
5. 不得执行清洗规则；runner_spec 通过后由宿主隔离执行冻结 Pipeline。
6. 不得修改冻结脚本包，不得搜索历史实验、validation 数据、private reference 或评分报告。
7. 如果你确认冻结 Pipeline 的业务规则本身不足，只能在 runner_spec.json 的 notes 中记录；宿主执行失败后会标记 pipeline_rule_failure 并返回 Validation。
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

你只负责提交 runner_spec.json，不得编写或改写任何业务转换代码，也不得生成 result_package。宿主会执行冻结的唯一入口。任何 validation/test 金标准和评分结果都不可见。
"""


def _runner_spec_quality_gate(
    *,
    runner_spec: str | Path,
    workspace: str | Path,
    frozen_script_bundle: str | Path,
    test_raw: str | Path,
    output_dir: str | Path,
    train_reference_root: str | Path,
) -> dict[str, Any]:
    spec_path = Path(runner_spec).expanduser().resolve()
    workspace_path = Path(workspace).expanduser().resolve()
    bundle = Path(frozen_script_bundle).expanduser().resolve()
    pipeline = bundle / "pipeline"
    issues: list[str] = []
    forbidden_scripts = sorted(
        path.relative_to(workspace_path).as_posix()
        for path in workspace_path.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".py", ".sh"}
    ) if workspace_path.is_dir() else []
    if forbidden_scripts:
        issues.append(
            "test workspace contains forbidden business/runner scripts: "
            + ", ".join(forbidden_scripts[:20])
        )

    expected_files = sorted(
        path.relative_to(Path(train_reference_root).expanduser().resolve()).as_posix()
        for path in _structured_package_files(Path(train_reference_root).expanduser().resolve())
        if path.name not in {
            "package_manifest.json",
            "result_manifest.json",
            "target_field_mapping.json",
            "reference.csv",
        }
    )
    manifest_path = pipeline / "pipeline_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    structure = validate_pipeline_structure(
        pipeline,
        expected_files=expected_files,
        expected_parent_sha256=str(manifest.get("parent_pipeline_sha256") or ""),
        train_reference_root=train_reference_root,
    )
    issues.extend(f"frozen pipeline: {item}" for item in structure["issues"])

    spec: dict[str, Any] = {}
    if not spec_path.is_file():
        issues.append("runner_spec.json is missing")
    else:
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"runner_spec.json is invalid JSON: {type(exc).__name__}: {exc}")
    if spec:
        expected_values = {
            "schema_version": 1,
            "entrypoint": str((pipeline / "run.py").resolve()),
            "pipeline_sha256": pipeline_directory_sha256(pipeline),
            "raw_root": str(Path(test_raw).expanduser().resolve()),
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "split_mode": "test",
        }
        for key, expected in expected_values.items():
            if spec.get(key) != expected:
                issues.append(
                    f"runner_spec {key} mismatch: expected={expected!r}, actual={spec.get(key)!r}"
                )
        allowed_fields = {*expected_values, "notes"}
        unexpected = sorted(set(spec) - allowed_fields)
        if unexpected:
            issues.append("runner_spec contains unsupported fields: " + ", ".join(unexpected))

    return {
        "schema_version": 1,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "valid": not issues,
        "runner_spec": str(spec_path),
        "pipeline_structure": structure,
        "forbidden_scripts": forbidden_scripts,
        "issues": issues,
    }


def _test_repair_prompt(gate: dict[str, Any], workspace: Path) -> str:
    issues = "\n".join(f"- {issue}" for issue in gate.get("issues") or [])
    return f"""\
当前 test 结果未通过公开结构门禁。以下信息只来自 train schema 和 test keys，不含 test 金标准：
{issues}

请继续在同一个 Agent 会话中只修复 {workspace / 'runner_spec.json'}。不得创建 Python/Shell 脚本、不得生成业务 CSV、不得修改冻结脚本包。
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
    bundle = Path(frozen_script_bundle).expanduser().resolve()
    report = validate_business_result_package(
        result_package=package,
        train_reference_root=train_reference_root,
        split_keys=test_keys,
        key_column=key_column,
        split_mode="test",
        required_file_count=MIMIC_EXPECTED_BUSINESS_FILE_COUNT,
    )
    issues = list(report["issues"])

    current_bundle_hash = _directory_sha256(bundle)
    if current_bundle_hash != frozen_hash_before:
        issues.append("frozen script bundle hash changed")
    if validation_result_package:
        validation_package = Path(validation_result_package).expanduser().resolve()
        if validation_package.is_dir() and package.is_dir() and _directory_sha256(validation_package) == _directory_sha256(package):
            issues.append("test output is a direct copy of the validation result package")
    return {
        **report,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "valid": not issues,
        "test_key_count": report["split_key_count"],
        "frozen_script_bundle_sha256": current_bundle_hash,
        "issues": issues,
    }


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
