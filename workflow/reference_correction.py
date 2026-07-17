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
from workflow.correction_evaluation import evaluate_correction_package
from workflow.reference_guided import (
    _latest_result_package_dir,
    _remove_fixed_reference_incompatible_tools,
)


@dataclass(frozen=True)
class ReferenceCorrectionConfig:
    dataset_split: str | Path
    experiment_dir: str | Path
    task_text: str
    max_iters: int = 10000

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

    def run_sync(self) -> dict[str, Any]:
        self.experiment.mkdir(parents=True, exist_ok=True)
        report_path = self.experiment / "correction_run_report.json"
        if report_path.exists():
            raise ValueError(f"correction experiment already completed: {self.experiment}")
        _write_json(
            self.experiment / "run_manifest.json",
            {
                "schema_version": 1,
                "workflow": "reference-guided-correct",
                "dataset_split": str(self.dataset),
                "source_archive_sha256": self.source_manifest.get("source_archive_sha256", ""),
                "task_text_sha256": _sha256_text(self.config.task_text),
                "max_iters": self.config.max_iters,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        started_at = datetime.now().isoformat(timespec="seconds")
        try:
            agent_package = self._run_agent_session()
            gate = validate_correction_result_package(
                result_package=agent_package,
                input_package=self.dataset / "correction/raw",
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
            }
            _write_json(report_path, report)
            report["report_path"] = str(report_path)
            return report

    def _build_sanitized_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow": "reference-guided-correct",
            "split_mode": "correction",
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

    def _task_text(self, phase_root: Path) -> str:
        paths = self.sanitized_contract["paths"]
        return f"""\
{self.config.task_text.strip()}

【任务目标】
根据10个成对标准示例，自主学习 dirty package 到 clean package 的纠错关系，然后修复 correction package。不要假设存在预先给定的错误清单。

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
6. 最终必须使用 PublishDirectoryArtifact 发布完整结果目录到 {phase_root / 'workspace/result_package'}。
7. 不要在最终回答中声称读取了不可见答案；宿主会在会话完全结束后独立评分。
"""

    def _run_agent_session(self) -> Path:
        agent_runs = self.experiment / "agent_runs"
        phase = init_phase_session(agent_runs, "reference_code_agent")
        phase_root = Path(phase["phase_root"])
        run_id = "reference_guided_correction"
        set_phase_context(run_id, agent_runs, "reference_code_agent", phase_root)
        contract_path = phase_root / "sanitized_correction_contract.json"
        _write_json(contract_path, self.sanitized_contract)
        task_text = self._task_text(phase_root)
        try:
            context = EngineerToolContext.from_task(
                task_text=task_text,
                engineer_phase_root=phase_root,
                explorer_phase_root=str(self.dataset / "train/reference"),
                additional_read_roots=[*self._agent_read_roots(), contract_path],
                require_skill_plan=False,
                split_mode="correction",
                required_report_paths={
                    "sanitized_correction_contract": str(contract_path),
                    "train_reference": str(self.dataset / "train/reference"),
                },
            )
            toolkit, _ = create_reference_toolkit(context)
            _remove_fixed_reference_incompatible_tools(toolkit)
            model, formatter = make_reference_model()
            agent = ReActAgent(
                name="ReferenceCodeAgent",
                sys_prompt=_correction_system_prompt(context),
                model=model,
                formatter=formatter,
                toolkit=toolkit,
                memory=create_reference_memory(
                    {
                        "phase_name": "reference_code_agent",
                        "phase_root": str(phase_root),
                        "run_id": run_id,
                        "run_root": str(agent_runs),
                    },
                    model,
                ),
                parallel_tool_calls=False,
                max_iters=self.config.max_iters,
                print_hint_msg=False,
            )
            response = asyncio.run(agent(make_user_msg(name="user", content=task_text)))
            response_text = format_message_content(getattr(response, "content", "")).strip()
            (phase_root / "agent_response.txt").write_text(response_text, encoding="utf-8")
            package = _latest_result_package_dir(phase_root)
            if package is None:
                raise RuntimeError("ReferenceCodeAgent did not publish workspace/result_package")
            return package
        finally:
            clear_phase_context()


def validate_correction_result_package(
    *,
    result_package: str | Path,
    input_package: str | Path,
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
你是 ReferenceCodeAgent，当前执行一次独立的 reference-guided 数据纠错任务。

你只能读取：
{roots}

你必须从成对训练示例中自主发现纠错规则。禁止猜测或搜索未授权路径。ExecutePython 只能写当前步骤 OUTPUT_DIR；最终通过 PublishDirectoryArtifact 发布完整结果包。
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


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
