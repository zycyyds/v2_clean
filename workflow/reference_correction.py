from __future__ import annotations

import asyncio
import csv
import gzip
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.reference_runtime import (
    ENGINEER_CODE_READ_ROOTS,
    create_reference_agent,
    create_reference_toolkit,
    load_agent_state,
    save_agent_state,
)
from agent_tools.context import EngineerToolContext
from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context
from lib.agent_runtime import format_message_content, make_user_msg
from workflow.correction_evaluation import evaluate_correction_package
from workflow.reference_guided import (
    _latest_result_package_dir,
)


@dataclass(frozen=True)
class ReferenceCorrectionConfig:
    dataset_split: str | Path
    experiment_dir: str | Path
    task_text: str
    max_iters: int = 10000
    resume: bool = False

    def normalized(self) -> "ReferenceCorrectionConfig":
        dataset = Path(self.dataset_split).expanduser().resolve()
        experiment = Path(self.experiment_dir).expanduser().resolve()
        if not dataset.is_dir():
            raise ValueError(f"correction dataset does not exist: {dataset}")
        for relative in (
            "train/raw",
            "train/reference",
            "correction/raw",
            "correction/reference_private",
            "host_private/correction_modification_log.csv",
        ):
            if not (dataset / relative).exists():
                raise ValueError(f"correction dataset is missing: {relative}")
        if not self.task_text.strip():
            raise ValueError("correction task prompt must not be empty")
        if self.max_iters < 1:
            raise ValueError("max_iters must be positive")
        return ReferenceCorrectionConfig(
            dataset_split=dataset,
            experiment_dir=experiment,
            task_text=self.task_text,
            max_iters=self.max_iters,
            resume=bool(self.resume),
        )


class ReferenceCorrectionWorkflow:
    def __init__(self, config: ReferenceCorrectionConfig) -> None:
        self.config = config.normalized()
        self.dataset = Path(self.config.dataset_split)
        self.experiment = Path(self.config.experiment_dir)
        self.source_manifest = json.loads(
            (self.dataset / "split_manifest.json").read_text(encoding="utf-8")
        )
        self.sanitized_contract = self._build_sanitized_contract()
        self.resume_context: dict[str, Any] | None = None

    def run_sync(self) -> dict[str, Any]:
        self.experiment.mkdir(parents=True, exist_ok=True)
        report_path = self.experiment / "correction_run_report.json"
        self.resume_context = self._prepare_experiment(report_path)
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            agent_package = self._run_agent_session()
            gate = validate_correction_result_package(
                result_package=agent_package,
                input_package=self.dataset / "correction/raw",
                split_keys=self.dataset / "correction/keys.csv",
                key_column=str(self.source_manifest.get("split_key") or "stay_id"),
            )
            _write_json(self.experiment / "correction_result_gate.json", gate)
            if not gate["valid"]:
                report = {
                    "schema_version": 1,
                    "status": "correction_failed",
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "gate": gate,
                    "metrics": {},
                }
                _write_json(report_path, report)
                return report
            final_package = self.experiment / "result_package"
            if final_package.exists():
                shutil.rmtree(final_package)
            shutil.copytree(agent_package, final_package)

            evaluation = evaluate_correction_package(
                dirty_root=self.dataset / "correction/raw",
                result_root=final_package,
                clean_root=self.dataset / "correction/reference_private",
                modification_log=self.dataset
                / "host_private/correction_modification_log.csv",
                output_dir=self.experiment / "evaluation",
            )
            report = {
                "schema_version": 1,
                "status": "SUCCESS",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_split": str(self.dataset),
                "experiment_dir": str(self.experiment),
                "result_package": str(final_package),
                "gate": gate,
                "metrics": evaluation["metrics"],
                "by_error_class": evaluation["by_error_class"],
                "by_error_subtype": evaluation["by_error_subtype"],
                "evaluation_report": evaluation["evaluation_report"],
            }
            _write_json(report_path, report)
            report["report_path"] = str(report_path)
            return report
        except Exception as exc:
            report = {
                "schema_version": 1,
                "status": "correction_failed",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "metrics": {},
                "error": f"{type(exc).__name__}: {exc}",
                "resume_count": int((self.resume_context or {}).get("resume_index") or 0),
            }
            _write_json(report_path, report)
            report["report_path"] = str(report_path)
            return report

    def _manifest_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow": "reference-guided-correct",
            "dataset_split": str(self.dataset),
            "source_archive_sha256": self.source_manifest.get("source_archive_sha256", ""),
            "task_text_sha256": _sha256_text(self.config.task_text),
            "max_iters": self.config.max_iters,
        }

    def _prepare_experiment(self, report_path: Path) -> dict[str, Any] | None:
        manifest_path = self.experiment / "run_manifest.json"
        expected = self._manifest_identity()
        if not self.config.resume:
            if report_path.exists() or manifest_path.exists():
                raise ValueError(f"correction experiment already initialized: {self.experiment}")
            _write_json(
                manifest_path,
                {
                    **expected,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "resume_count": 0,
                    "resumes": [],
                },
            )
            return None

        if not manifest_path.is_file():
            raise ValueError(f"cannot resume correction experiment without manifest: {self.experiment}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise ValueError(
                "correction resume manifest does not match: " + ", ".join(sorted(mismatches))
            )

        previous_report: dict[str, Any]
        if report_path.is_file():
            previous_report = json.loads(report_path.read_text(encoding="utf-8"))
            if previous_report.get("status") != "correction_failed":
                raise ValueError(
                    f"only a failed correction experiment can resume: {self.experiment}"
                )
        else:
            previous_report = {
                "status": "correction_failed",
                "error": "previous process ended without a final report",
            }

        phase_manifest_path = (
            self.experiment / "agent_runs/data_cleaning_agent/manifest.json"
        )
        phase_manifest = (
            json.loads(phase_manifest_path.read_text(encoding="utf-8"))
            if phase_manifest_path.is_file()
            else {}
        )
        prior_steps = list(phase_manifest.get("steps") or [])
        resume_index = int(manifest.get("resume_count") or 0) + 1
        history_dir = self.experiment / "resume_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"resume_{resume_index:04d}_previous_report.json"
        if report_path.is_file():
            os.replace(report_path, history_path)
        else:
            _write_json(history_path, previous_report)

        resumed_at = datetime.now().isoformat(timespec="seconds")
        resume_record = {
            "resume_index": resume_index,
            "resumed_at": resumed_at,
            "previous_error": _previous_failure_summary(previous_report),
            "prior_step_count": len(prior_steps),
            "previous_report": str(history_path),
        }
        manifest["resume_count"] = resume_index
        manifest.setdefault("resumes", []).append(resume_record)
        _write_json(manifest_path, manifest)
        return {
            **resume_record,
            "phase_root": str(phase_manifest_path.parent),
            "recent_steps": [_resume_step_summary(step) for step in prior_steps[-8:]],
            "existing_files": _resume_existing_files(phase_manifest_path.parent),
        }

    def _build_sanitized_contract(self) -> dict[str, Any]:
        key_column = str(self.source_manifest.get("split_key") or "stay_id")
        return {
            "schema_version": 1,
            "workflow": "reference-guided-correct",
            "split_mode": "correction",
            "key_column": key_column,
            "counts": dict(self.source_manifest.get("counts") or {}),
            "paths": {
                "train_raw": str(self.dataset / "train/raw"),
                "train_reference": str(self.dataset / "train/reference"),
                "correction_raw": str(self.dataset / "correction/raw"),
            },
        }

    def _agent_read_roots(self) -> list[str | Path]:
        paths = self.sanitized_contract["paths"]
        return [
            *ENGINEER_CODE_READ_ROOTS,
            paths["train_raw"],
            paths["train_reference"],
            paths["correction_raw"],
        ]

    def _task_text(
        self,
        phase_root: Path,
        resume_context_path: Path | None = None,
    ) -> str:
        paths = self.sanitized_contract["paths"]
        train_count = int(self.sanitized_contract.get("counts", {}).get("train") or 0)
        correction_count = int(
            self.sanitized_contract.get("counts", {}).get("correction") or 0
        )
        key_column = str(self.sanitized_contract.get("key_column") or "stay_id")
        resume_block = ""
        if resume_context_path is not None:
            resume_block = f"""

【同一实验恢复模式】
- 上一次进程因外部异常退出；这不是新实验，也不是历史实验导入。
- 首先读取恢复说明：{resume_context_path}
- 继续检查当前 phase_root 中已有的 rules、清洗脚本、manifest 最近步骤及其工具输出。
- 优先从最近已验证的发现继续，不要重新进行完整的初始目录扫描。
- 既有候选仍是未提交中间产物；必须继续验证、更新执行代码并重新生成完整结果包。
"""
        return f"""\
{self.config.task_text.strip()}

【任务目标】
根据{train_count}个成对标准示例，自主学习 dirty package 到 clean package 的纠错关系，然后修复 correction package。不要假设存在预先给定的错误清单。{resume_block}

【结果包主键契约】
- 本任务的样本主键是 `{key_column}`，correction 共 {correction_count} 个主键。
- cohort 中同一 `hadm_id` 可以合法对应多个不同的 ICU `stay_id`，不得按 `hadm_id` 去重或删除合法 stay。
- 生成前后必须自行检查 `{key_column}` 覆盖和重复情况；宿主最终会按 {correction_count} 个 correction key 执行统一门禁。

【可读输入】
- train dirty examples: {paths['train_raw']}
- train clean references: {paths['train_reference']}
- correction dirty package: {paths['correction_raw']}

【执行要求】
1. 自主比较 train dirty 与 train clean，归纳哪些差异需要修复以及如何从表内、实体和跨表关系推导修复。
2. 不得搜索数据集根目录、历史 experiments、隐藏答案、评分结果或任何未列出的路径。
3. 只在有证据时修改；必须尽量保持正常数据、目录结构、文件名、schema 和 gzip 格式不变。
4. correction 的完整输出必须包含输入中的全部业务文件，不能只输出发生修改的文件。
5. 允许在当前 workspace 编写并执行纠错脚本，但不得修改任何输入目录。
6. 最后一次 RunAnalysisPython 必须在该工具分配的 OUTPUT_DIR/result_package 生成完整结果包；宿主会接管最新的完整输出目录。
7. 不要在最终回答中声称读取了不可见答案；宿主会在会话完全结束后独立评分。
"""

    def _run_agent_session(self) -> Path:
        return asyncio.run(self._run_agent_session_async())

    async def _run_agent_session_async(self) -> Path:
        agent_runs = self.experiment / "agent_runs"
        phase = init_phase_session(agent_runs, "data_cleaning_agent")
        phase_root = Path(phase["phase_root"])
        run_id = "reference_guided_correction"
        set_phase_context(run_id, agent_runs, "data_cleaning_agent", phase_root)
        contract_path = phase_root / "sanitized_correction_contract.json"
        _write_json(contract_path, self.sanitized_contract)
        resume_context_path: Path | None = None
        if self.resume_context is not None:
            resume_context_path = phase_root / "resume_context.json"
            _write_json(resume_context_path, self.resume_context)
        task_text = self._task_text(phase_root, resume_context_path)
        try:
            context = EngineerToolContext.from_task(
                task_text=task_text,
                engineer_phase_root=phase_root,
                explorer_phase_root=str(self.dataset / "train/reference"),
                additional_read_roots=[
                    *self._agent_read_roots(),
                    contract_path,
                    *([resume_context_path] if resume_context_path is not None else []),
                ],
                require_skill_plan=False,
                split_mode="correction",
                required_report_paths={
                    "sanitized_correction_contract": str(contract_path),
                    "train_reference": str(self.dataset / "train/reference"),
                },
            )
            toolkit, _ = create_reference_toolkit(context)
            state_path = phase_root / "agent_state.json"
            agent, workspace = await create_reference_agent(
                name="Data Cleaning Agent",
                system_prompt=_correction_system_prompt(context),
                toolkit=toolkit,
                workspace_dir=context.workspace_dir,
                max_iters=self.config.max_iters,
                state=load_agent_state(state_path),
            )
            try:
                response = await agent.reply(make_user_msg(name="user", content=task_text))
                response_text = format_message_content(getattr(response, "content", "")).strip()
                (phase_root / "agent_response.txt").write_text(response_text, encoding="utf-8")
                save_agent_state(state_path, agent.state)
                package = _latest_result_package_dir(phase_root)
                if package is None:
                    raise RuntimeError(
                        "Data Cleaning Agent did not generate OUTPUT_DIR/result_package"
                    )
                return package
            finally:
                save_agent_state(state_path, agent.state)
                await workspace.close()
        finally:
            clear_phase_context()


def validate_correction_result_package(
    *,
    result_package: str | Path,
    input_package: str | Path,
    split_keys: str | Path | None = None,
    key_column: str = "stay_id",
) -> dict[str, Any]:
    result = Path(result_package).expanduser().resolve()
    source = Path(input_package).expanduser().resolve()
    issues: list[str] = []
    expected_files = _relative_files(source)
    result_files = _relative_files(result) if result.is_dir() else []
    missing = sorted(set(expected_files) - set(result_files))
    extra = sorted(set(result_files) - set(expected_files))
    if missing:
        issues.append("missing business files: " + ", ".join(missing[:20]))
    if extra:
        issues.append("unexpected business files: " + ", ".join(extra[:20]))
    expected_keys = _read_split_keys(split_keys, key_column) if split_keys else set()
    file_reports: list[dict[str, Any]] = []
    for relative in expected_files:
        source_file = source / relative
        result_file = result / relative
        item = {"relative_path": relative, "passed": True, "issues": []}
        if not result_file.is_file():
            item["passed"] = False
            item["issues"].append("missing file")
        elif relative.endswith(".csv") or relative.endswith(".csv.gz"):
            source_header, source_rows = _csv_shape(source_file)
            result_header, result_rows = _csv_shape(result_file)
            item.update({"source_rows": source_rows, "result_rows": result_rows})
            if source_header != result_header:
                item["passed"] = False
                item["issues"].append("schema differs from correction input")
            if source_rows and not result_rows:
                item["passed"] = False
                item["issues"].append("non-empty input became empty")
            if result_header == source_header and key_column in result_header:
                observed_keys, duplicate_keys = _csv_key_profile(result_file, key_column)
                item.update(
                    {
                        "key_column": key_column,
                        "observed_key_count": len(observed_keys),
                        "duplicate_key_count": duplicate_keys,
                    }
                )
                unknown_keys = observed_keys - expected_keys if expected_keys else set()
                if unknown_keys:
                    item["passed"] = False
                    item["issues"].append(
                        f"contains {len(unknown_keys)} unknown {key_column} values"
                    )
                if relative.startswith("cohort/"):
                    missing_keys = expected_keys - observed_keys if expected_keys else set()
                    if missing_keys:
                        item["passed"] = False
                        item["issues"].append(
                            f"missing {len(missing_keys)} correction {key_column} values"
                        )
                    if duplicate_keys:
                        item["passed"] = False
                        item["issues"].append(
                            f"contains {duplicate_keys} duplicate {key_column} values"
                        )
        if not item["passed"]:
            issues.extend(f"{relative}: {message}" for message in item["issues"])
        file_reports.append(item)
    return {
        "schema_version": 1,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "valid": not issues,
        "expected_file_count": len(expected_files),
        "issues": issues,
        "files": file_reports,
    }


def _correction_system_prompt(context: EngineerToolContext) -> str:
    roots = "\n".join(f"- {path}" for path in context.read_roots)
    return f"""\
你是 Data Cleaning Agent，当前执行一次独立的 reference-guided 数据纠错任务。

你只能读取：
{roots}

你必须从成对训练示例中自主发现纠错规则。禁止猜测或搜索未授权路径。RunAnalysisPython 只能写当前步骤 OUTPUT_DIR；最终在 OUTPUT_DIR/result_package 生成完整结果包，由宿主接管。
"""


def _relative_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    )


def _csv_shape(path: Path) -> tuple[list[str], int]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        return header, sum(1 for _ in reader)


def _read_split_keys(path: str | Path, key_column: str) -> set[str]:
    split_keys = Path(path).expanduser().resolve()
    opener = gzip.open if split_keys.name.endswith(".gz") else open
    with opener(split_keys, "rt", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get(key_column, "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get(key_column, "")).strip()
        }


def _csv_key_profile(path: Path, key_column: str) -> tuple[set[str], int]:
    opener = gzip.open if path.name.endswith(".gz") else open
    values: list[str] = []
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get(key_column, "")).strip()
            if value:
                values.append(value)
    return set(values), len(values) - len(set(values))


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resume_step_summary(step: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(step.get("metadata") or {})
    details: dict[str, Any] = {}
    for key in ("script_path", "returncode", "stdout", "stderr", "operation", "diff"):
        value = metadata.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str) and len(value) > 6_000:
            value = value[-6_000:]
        details[key] = value
    return {
        "step_index": step.get("step_index"),
        "skill_name": step.get("skill_name"),
        "status": step.get("status"),
        "summary": step.get("summary"),
        "artifacts": list(step.get("artifacts") or []),
        "details": details,
        "created_at": step.get("created_at"),
    }


def _resume_existing_files(phase_root: Path) -> list[str]:
    preferred = [
        phase_root / "rules.json",
        phase_root / "rules.md",
        phase_root / "clean_correction.py",
        phase_root / "manifest.json",
        phase_root / "context/context_events.jsonl",
    ]
    recent_scripts = sorted(
        phase_root.glob("*.py"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:20]
    result: list[str] = []
    for path in [*preferred, *recent_scripts]:
        value = str(path)
        if path.is_file() and value not in result:
            result.append(value)
    return result


def _previous_failure_summary(report: dict[str, Any]) -> str:
    error = str(report.get("error") or "").strip()
    if error:
        return error
    gate = report.get("gate")
    if isinstance(gate, dict):
        issues = [str(item).strip() for item in gate.get("issues") or [] if str(item).strip()]
        if issues:
            return "Host result gate failed: " + "; ".join(issues[:20])
    return "previous process ended without a final report"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
