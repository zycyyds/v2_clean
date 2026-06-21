from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from lib.agent_artifacts import get_phase_context, record_step
from workflow.skill_adapter import verify_variant_execution_receipt

from .context import EngineerToolContext


_VARIANT_NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")
_BLOCKED_IMPORTS = {"requests", "httpx", "socket", "subprocess", "urllib"}


def _response(status: str, summary: str, artifacts=None, issues=()) -> ToolResponse:
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "status": status,
                        "summary": summary,
                        "artifacts": artifacts or {},
                        "issues": list(issues),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
        ],
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class EngineerSkillLifecycleTools:
    """Plan, derive, validate, and audit run-local Engineer Skill variants."""

    def __init__(
        self,
        context: EngineerToolContext,
        *,
        skill_names: Iterable[str],
        skills_root: str | Path,
    ) -> None:
        self.context = context
        self.skill_names = frozenset(skill_names)
        self.skills_root = Path(skills_root).expanduser().resolve()
        self.inspected_skill_hashes: dict[str, dict[str, str]] = {}

    def initialize_skill_usage_plan(self) -> ToolResponse:
        """Create a complete plan skeleton from canonical reports and Skill metadata."""
        try:
            report_paths = dict(self.context.required_report_paths)
            if not report_paths:
                analysis = next(
                    (path for path in self.context.fingerprints if path.name == "data_analysis_report.json"),
                    None,
                )
                if analysis is None:
                    raise ValueError("Read data_analysis_report.json before initializing the Skill plan")
                report_paths = {"analysis_report": str(analysis)}
            report_hashes: dict[str, dict[str, str]] = {}
            for name, raw_path in report_paths.items():
                path = self.context.resolve_read_path(raw_path)
                if not path.is_file() or not self.context.was_read_and_unchanged(path):
                    raise ValueError(f"Read canonical handoff before planning: {path}")
                report_hashes[name] = {"path": str(path), "sha256": _sha256(path)}

            required_capabilities = self._required_capabilities(report_paths.get("extraction_task_plan"))
            matched_capabilities: set[str] = set()
            decisions: list[dict[str, Any]] = []
            for skill_name in sorted(self.skill_names):
                metadata = self._read_skill_metadata(self._base_skill_dir(skill_name) / "skill.py")
                capabilities = {str(value) for value in metadata.get("capability_types") or []}
                matched = sorted(required_capabilities & capabilities)
                if matched:
                    matched_capabilities.update(matched)
                    decisions.append({
                        "skill_name": skill_name,
                        "decision": "use",
                        "reason": f"Matches required capabilities: {', '.join(matched)}.",
                        "parameters": {},
                        "capability_gap": "",
                        "why_parameters_insufficient": "",
                    })
                else:
                    decisions.append({
                        "skill_name": skill_name,
                        "decision": "skip",
                        "reason": "No required extraction capability matches this Skill.",
                        "parameters": {},
                        "capability_gap": "",
                        "why_parameters_insufficient": "",
                    })
            missing = sorted(required_capabilities - matched_capabilities)
            if missing:
                gap = ", ".join(missing)
                decisions.append({
                    "skill_name": "standalone_python",
                    "decision": "use",
                    "reason": f"No registered Skill covers required capabilities: {gap}.",
                    "parameters": {},
                    "capability_gap": gap,
                    "why_parameters_insufficient": "No registered Skill exposes these capability types.",
                })
            payload = {
                "created_run_id": (get_phase_context() or {}).get("run_id", self.context.engineer_phase_root.name),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "report_hashes": report_hashes,
                "required_capabilities": sorted(required_capabilities),
                "decisions": self._validate_decisions(decisions),
            }
            _write_json(self.context.skill_usage_plan_path, payload)
            record_step(
                "initialize_skill_usage_plan",
                "Initialized Skill decisions from the extraction task plan.",
                [self.context.skill_usage_plan_path],
                metadata={"decision_count": len(decisions), "required_capabilities": sorted(required_capabilities)},
            )
            return _response(
                "SUCCESS",
                "Skill usage plan initialized.",
                {"plan_path": str(self.context.skill_usage_plan_path), "decision_count": len(decisions)},
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Skill usage plan initialization failed.", issues=[str(exc)])

    def revise_skill_usage(
        self,
        skill_name: str,
        decision: str,
        reason: str,
        capability_gap: str = "",
        why_parameters_insufficient: str = "",
        parameters_json: str = "{}",
    ) -> ToolResponse:
        """Revise one initialized Skill decision without resubmitting the whole plan."""
        try:
            plan = self.context.load_skill_plan()
            if not plan:
                raise ValueError("Call initialize_skill_usage_plan first")
            parameters = json.loads(parameters_json)
            if not isinstance(parameters, dict):
                raise ValueError("parameters_json must contain a JSON object")
            replacement = {
                "skill_name": skill_name,
                "decision": decision,
                "reason": reason,
                "parameters": parameters,
                "capability_gap": capability_gap,
                "why_parameters_insufficient": why_parameters_insufficient,
            }
            decisions = [
                item for item in plan.get("decisions", [])
                if item.get("skill_name") != skill_name
            ]
            decisions.append(replacement)
            plan["decisions"] = self._validate_decisions(decisions)
            plan["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(self.context.skill_usage_plan_path, plan)
            return _response(
                "SUCCESS",
                f"Revised Skill decision for {skill_name}.",
                {"plan_path": str(self.context.skill_usage_plan_path), "skill_name": skill_name},
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Skill usage revision failed.", issues=[str(exc)])

    def plan_skill_usage(self, report_paths_json: str, decisions_json: str) -> ToolResponse:
        """Record report fingerprints and use/adapt/skip decisions before Python work."""
        try:
            report_paths = json.loads(report_paths_json)
            decisions = json.loads(decisions_json)
            if not isinstance(report_paths, dict) or not report_paths:
                raise ValueError("report_paths_json must be a non-empty JSON object")
            if "analysis_report" not in report_paths:
                raise ValueError("report_paths_json must include analysis_report")
            if not isinstance(decisions, list):
                raise ValueError("decisions_json must be a JSON list")

            for name, required_path in self.context.required_report_paths.items():
                supplied = report_paths.get(name)
                if not supplied:
                    raise ValueError(f"report_paths_json must include required handoff: {name}")
                if Path(str(supplied)).expanduser().resolve() != Path(required_path).expanduser().resolve():
                    raise ValueError(f"Handoff path does not match canonical path for {name}")

            report_hashes: dict[str, dict[str, str]] = {}
            for name, raw_path in report_paths.items():
                if not raw_path:
                    continue
                path = self.context.resolve_read_path(str(raw_path))
                if not path.is_file():
                    raise ValueError(f"Report does not exist: {path}")
                if not self.context.was_read_and_unchanged(path):
                    raise ValueError(f"Read report before planning and keep it unchanged: {path}")
                report_hashes[str(name)] = {"path": str(path), "sha256": _sha256(path)}

            normalized = self._validate_decisions(decisions)
            phase_context = get_phase_context() or {}
            payload = {
                "created_run_id": phase_context.get("run_id", Path(self.context.engineer_phase_root).name),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "report_hashes": report_hashes,
                "decisions": normalized,
            }
            _write_json(self.context.skill_usage_plan_path, payload)
            record = record_step(
                "plan_skill_usage",
                "Recorded report fingerprints and Skill usage decisions.",
                [self.context.skill_usage_plan_path],
                metadata={"decision_count": len(normalized), "report_count": len(report_hashes)},
            )
            return _response(
                "SUCCESS",
                "Skill usage plan recorded.",
                {
                    "plan_path": str(self.context.skill_usage_plan_path),
                    "report_hashes": report_hashes,
                    "decision_count": len(normalized),
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Skill usage planning failed.", issues=[str(exc)])

    def inspect_skill(self, skill_name: str) -> ToolResponse:
        """Inspect one read-only base Skill and return its source hashes."""
        try:
            source_dir = self._base_skill_dir(skill_name)
            sources = self._source_hashes(source_dir)
            self.inspected_skill_hashes[skill_name] = sources
            skill_md = source_dir / "SKILL.md"
            return _response(
                "SUCCESS",
                f"Inspected base Skill {skill_name}.",
                {
                    "skill_name": skill_name,
                    "source_dir": str(source_dir),
                    "source_paths": list(sources),
                    "source_hashes": sources,
                    "skill_markdown": skill_md.read_text(encoding="utf-8") if skill_md.exists() else "",
                    "metadata": self._read_skill_metadata(source_dir / "skill.py"),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Skill inspection failed.", issues=[str(exc)])

    def create_skill_variant(
        self,
        base_skill: str,
        variant_name: str,
        mode: str,
        change_spec: str,
        contracts: str,
    ) -> ToolResponse:
        """Create a run-local adapter or fork scaffold without modifying the base Skill."""
        try:
            self._require_plan_decision(base_skill, "adapt")
            source_dir = self._base_skill_dir(base_skill)
            current_source_hashes = self._source_hashes(source_dir)
            if self.inspected_skill_hashes.get(base_skill) != current_source_hashes:
                raise ValueError(f"Call inspect_skill for {base_skill} immediately before creating a variant")
            if mode not in {"adapter", "fork"}:
                raise ValueError("mode must be adapter or fork")
            if not _VARIANT_NAME_RE.fullmatch(variant_name):
                raise ValueError("variant_name must be a 3-64 character snake_case identifier")
            if not str(change_spec).strip():
                raise ValueError("change_spec is required")
            contract_value = json.loads(contracts)
            if not isinstance(contract_value, dict):
                raise ValueError("contracts must be a JSON object")
            input_contract = contract_value.get("input_contract")
            output_contract = contract_value.get("output_contract")
            if not isinstance(input_contract, dict) or not isinstance(output_contract, dict):
                raise ValueError("contracts must contain input_contract and output_contract objects")

            variant_dir = (self.context.variant_root / variant_name).resolve()
            if variant_dir.exists():
                raise ValueError(f"Variant already exists in this run: {variant_name}")
            variant_dir.mkdir(parents=True)
            phase_context = get_phase_context() or {}
            source_hashes = current_source_hashes
            metadata = {
                "protocol_version": 1,
                "base_skill": base_skill,
                "mode": mode,
                "base_source_hashes": source_hashes,
                "change_spec": str(change_spec).strip(),
                "input_contract": input_contract,
                "output_contract": output_contract,
                "status": "draft",
                "created_run_id": phase_context.get("run_id", Path(self.context.engineer_phase_root).name),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            _write_json(variant_dir / "variant.json", metadata)
            _write_json(
                variant_dir / "request.json",
                {
                    "rule_ids": [],
                    "input_artifacts": [],
                    "parameters": {},
                    "output_contract": output_contract,
                },
            )
            (variant_dir / "SKILL.md").write_text(
                f"# {variant_name}\n\n"
                f"Run-local {mode} derived from `{base_skill}`.\n\n"
                f"## Change specification\n\n{change_spec.strip()}\n",
                encoding="utf-8",
            )
            (variant_dir / "variant.py").write_text(
                self._variant_scaffold(base_skill, mode, output_contract),
                encoding="utf-8",
            )
            record = record_step(
                "create_skill_variant",
                f"Created draft {mode} variant {variant_name} from {base_skill}.",
                list(variant_dir.iterdir()),
                metadata={"variant_name": variant_name, "base_skill": base_skill, "mode": mode},
            )
            return _response(
                "SUCCESS",
                f"Created draft Skill variant {variant_name}.",
                {
                    "variant_dir": str(variant_dir),
                    "variant_path": str(variant_dir / "variant.py"),
                    "request_path": str(variant_dir / "request.json"),
                    "metadata_path": str(variant_dir / "variant.json"),
                    "variant_status": "draft",
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Skill variant creation failed.", issues=[str(exc)])

    def validate_skill_variant(self, variant_name: str, artifact_path: str = "") -> ToolResponse:
        """Validate variant files, safety, base hashes, and optionally its primary artifact."""
        try:
            variant_dir = self._variant_dir(variant_name)
            metadata_path = variant_dir / "variant.json"
            metadata = _load_json(metadata_path)
            required = [variant_dir / name for name in ("SKILL.md", "variant.json", "variant.py", "request.json")]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ValueError(f"Variant files are missing: {missing}")
            self._verify_source_hashes(metadata)
            self._validate_variant_source(variant_dir / "variant.py")
            _load_json(variant_dir / "request.json")

            if artifact_path:
                artifact = self.context.resolve_read_path(artifact_path)
                self._validate_artifact(artifact, metadata.get("output_contract") or {})
                receipt = verify_variant_execution_receipt(variant_dir)
                receipt_paths = {
                    str(Path(path).expanduser().resolve())
                    for path in receipt.get("output_files", [])
                }
                result_paths = {
                    str(Path(str(item.get("path"))).expanduser().resolve())
                    for item in (receipt.get("result", {}).get("artifacts") or [])
                    if isinstance(item, dict) and item.get("path")
                }
                if str(artifact.resolve()) not in receipt_paths | result_paths:
                    raise ValueError("Validated artifact was not produced by this variant execution")
                metadata["status"] = "validated"
                metadata["validated_artifact"] = str(artifact)
            else:
                metadata["status"] = "ready"
            metadata["validated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(metadata_path, metadata)
            record = record_step(
                "validate_skill_variant",
                f"Variant {variant_name} is {metadata['status']}.",
                [metadata_path, *([artifact_path] if artifact_path else [])],
                metadata={"variant_name": variant_name, "variant_status": metadata["status"]},
            )
            return _response(
                "SUCCESS",
                f"Skill variant {variant_name} is {metadata['status']}.",
                {
                    "variant_dir": str(variant_dir),
                    "variant_status": metadata["status"],
                    "artifact_path": str(artifact_path or ""),
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            self._mark_variant_failed(variant_name, str(exc))
            return _response("NEEDS_REPAIR", "Skill variant validation failed.", issues=[str(exc)])

    def execute_skill_variant(self, variant_name: str, timeout_seconds: int = 300) -> ToolResponse:
        """Execute a ready variant with its lifecycle-managed request.json."""
        try:
            variant_dir = self._variant_dir(variant_name)
            metadata = _load_json(variant_dir / "variant.json")
            if metadata.get("status") not in {"ready", "validated"}:
                raise ValueError("Derived Skill must be ready before execution")
            # Local import avoids a module cycle while keeping one execution sandbox.
            from .tools import EngineerTools

            response = EngineerTools(self.context).ExecutePython(
                str(variant_dir / "variant.py"),
                args=[str(variant_dir / "request.json")],
                timeout_seconds=timeout_seconds,
            )
            payload = json.loads(response.content[0]["text"])
            self.context.record_skill_call(
                str(metadata.get("base_skill") or ""),
                f"variant:{variant_name}",
                str(payload.get("status") or "UNKNOWN"),
                artifacts=(payload.get("artifacts") or {}).get("output_files") or [],
            )
            return response
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Skill variant execution failed.", issues=[str(exc)])

    def record_extraction_task(
        self,
        task_id: str,
        status: str,
        rule_ids_json: str,
        artifact_paths_json: str = "[]",
        reason: str = "",
    ) -> ToolResponse:
        """Record one required task after inspecting its real output artifacts."""
        try:
            task_plan_path = self.context.required_report_paths.get("extraction_task_plan")
            if not task_plan_path:
                raise ValueError("No canonical extraction_task_plan was configured")
            plan = _load_json(Path(task_plan_path).expanduser().resolve())
            tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
            task = next((item for item in tasks if isinstance(item, dict) and item.get("task_id") == task_id), None)
            if task is None:
                raise ValueError(f"Unknown extraction task: {task_id}")
            if status not in {"completed", "unsupported"}:
                raise ValueError("status must be completed or unsupported")
            rule_ids = json.loads(rule_ids_json)
            artifacts = json.loads(artifact_paths_json)
            if not isinstance(rule_ids, list) or not all(isinstance(value, str) for value in rule_ids):
                raise ValueError("rule_ids_json must be a JSON list of strings")
            if set(rule_ids) != set(task.get("rule_ids") or []):
                raise ValueError(f"rule_ids do not match extraction task {task_id}")
            if not isinstance(artifacts, list) or not all(isinstance(value, str) for value in artifacts):
                raise ValueError("artifact_paths_json must be a JSON list of paths")
            resolved_artifacts: list[str] = []
            for raw_path in artifacts:
                path = self.context.resolve_read_path(raw_path)
                if not path.is_file() or not self.context.was_read_and_unchanged(path):
                    raise ValueError(f"Read and inspect task artifact before recording it: {path}")
                resolved_artifacts.append(str(path))
            if status == "completed" and not resolved_artifacts:
                raise ValueError("completed task must include at least one inspected artifact")
            if status == "unsupported" and not str(reason).strip():
                raise ValueError("unsupported task requires a reason")
            if status == "completed":
                self._require_matching_skill_output(task, resolved_artifacts)

            payload = {"schema_version": 1, "tasks": []}
            if self.context.task_execution_path.is_file():
                payload = _load_json(self.context.task_execution_path)
                if not isinstance(payload.get("tasks"), list):
                    payload["tasks"] = []
            entry = {
                "task_id": task_id,
                "status": status,
                "rule_ids": rule_ids,
                "artifacts": resolved_artifacts,
                "reason": str(reason).strip(),
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
            payload["tasks"] = [item for item in payload["tasks"] if item.get("task_id") != task_id]
            payload["tasks"].append(entry)
            _write_json(self.context.task_execution_path, payload)
            record_step(
                "record_extraction_task",
                f"Recorded extraction task {task_id} as {status}.",
                [self.context.task_execution_path, *resolved_artifacts],
                metadata={"task_id": task_id, "status": status, "rule_count": len(rule_ids)},
            )
            return _response(
                "SUCCESS",
                f"Extraction task {task_id} recorded.",
                {"task_execution_path": str(self.context.task_execution_path), "task": entry},
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Extraction task recording failed.", issues=[str(exc)])

    def _require_matching_skill_output(
        self,
        task: dict[str, Any],
        resolved_artifacts: list[str],
    ) -> None:
        capability = str(task.get("capability_type") or "")
        if not capability:
            return
        plan = self.context.load_skill_plan()
        decisions = {
            str(item.get("skill_name") or ""): str(item.get("decision") or "")
            for item in plan.get("decisions", [])
            if isinstance(item, dict)
        }
        matching_skills = {
            skill_name
            for skill_name in self.skill_names
            if decisions.get(skill_name) in {"use", "adapt"}
            and capability in {
                str(value)
                for value in self._read_skill_metadata(
                    self._base_skill_dir(skill_name) / "skill.py",
                ).get("capability_types") or []
            }
        }
        if not matching_skills:
            return
        successful_calls = [
            event
            for event in self.context.load_skill_calls()
            if str(event.get("skill_name") or "") in matching_skills
            and str(event.get("status") or "").upper() == "SUCCESS"
        ]
        if not successful_calls:
            raise ValueError(
                "Task capability "
                f"{capability} must call its planned Skill first: {', '.join(sorted(matching_skills))}",
            )
        skill_artifacts = {
            str(Path(path).expanduser().resolve())
            for event in successful_calls
            for path in event.get("artifacts") or []
        }
        if not set(resolved_artifacts) & skill_artifacts:
            raise ValueError(
                "任务产物必须包含匹配 Skill 的真实输出: "
                f"{', '.join(sorted(matching_skills))}",
            )

    def audit_skill_usage(self) -> ToolResponse:
        """Compare the Skill plan with actual manifest activity and variant states."""
        issues: list[str] = []
        report_path = self.context.workspace_dir / "skill_usage_report.json"
        try:
            plan = self.context.load_skill_plan()
            if not plan:
                issues.append("skill_usage_plan.json is missing or invalid")
            current_hashes: dict[str, dict[str, str]] = {}
            variants: list[dict[str, Any]] = []
            if self.context.variant_root.exists():
                for variant_dir in sorted(self.context.variant_root.iterdir()):
                    metadata_path = variant_dir / "variant.json"
                    if not metadata_path.is_file():
                        issues.append(f"Variant metadata is missing: {variant_dir.name}")
                        continue
                    metadata = _load_json(metadata_path)
                    variants.append({"name": variant_dir.name, **metadata})
                    if metadata.get("status") != "validated":
                        issues.append(f"Variant {variant_dir.name} is not validated")
                    try:
                        self._verify_source_hashes(metadata)
                        current_hashes[variant_dir.name] = self._source_hashes(
                            self._base_skill_dir(str(metadata.get("base_skill"))),
                        )
                    except Exception as exc:
                        issues.append(str(exc))

            phase_context = get_phase_context() or {}
            manifest_path = Path(phase_context.get("manifest_path", self.context.engineer_phase_root / "manifest.json"))
            manifest = _load_json(manifest_path) if manifest_path.exists() else {}
            actual_steps = [
                {
                    "skill_name": entry.get("skill_name"),
                    "status": entry.get("status"),
                    "metadata": entry.get("metadata") or {},
                }
                for entry in manifest.get("steps", [])
            ]
            skill_calls = self.context.load_skill_calls()
            called_skills = {str(event.get("skill_name") or "") for event in skill_calls}
            executed_variants = {
                str((entry.get("metadata") or {}).get("variant_name") or "")
                for entry in manifest.get("steps", [])
                if entry.get("skill_name") == "ExecutePython"
            }
            variants_by_name = {str(item.get("name") or ""): item for item in variants}
            executed_variant_bases = {
                str(variants_by_name[name].get("base_skill") or "")
                for name in executed_variants
                if name in variants_by_name
            }
            standalone_executions = [
                entry
                for entry in manifest.get("steps", [])
                if entry.get("skill_name") == "ExecutePython"
                and not (entry.get("metadata") or {}).get("variant_name")
            ]

            task_plan_path = self.context.required_report_paths.get("extraction_task_plan")
            task_executions: list[dict[str, Any]] = []
            if task_plan_path:
                task_plan = _load_json(Path(task_plan_path).expanduser().resolve())
                required_tasks = {
                    str(item.get("task_id")): item
                    for item in task_plan.get("tasks", [])
                    if isinstance(item, dict) and item.get("required") is True
                }
                execution_payload = (
                    _load_json(self.context.task_execution_path)
                    if self.context.task_execution_path.is_file()
                    else {}
                )
                task_executions = [
                    item for item in execution_payload.get("tasks", []) if isinstance(item, dict)
                ]
                executed_by_id = {str(item.get("task_id")): item for item in task_executions}
                for task_id, task in required_tasks.items():
                    execution = executed_by_id.get(task_id)
                    if execution is None:
                        issues.append(f"Required extraction task was not recorded: {task_id}")
                        continue
                    if execution.get("status") not in {"completed", "unsupported"}:
                        issues.append(f"Required extraction task is unresolved: {task_id}")
                    if set(execution.get("rule_ids") or []) != set(task.get("rule_ids") or []):
                        issues.append(f"Extraction task rule coverage does not match plan: {task_id}")

            for decision in plan.get("decisions", []):
                name = str(decision.get("skill_name") or "")
                action = decision.get("decision")
                if name == "standalone_python":
                    if action == "use" and not standalone_executions:
                        issues.append("standalone_python was planned but not executed")
                    continue
                if action == "use" and name not in called_skills:
                    issues.append(f"Planned Skill was not called: {name}")
                if action == "adapt" and name not in called_skills and name not in executed_variant_bases:
                    issues.append(f"Planned adaptation was not implemented: {name}")
                if action == "skip" and (name in called_skills or name in executed_variant_bases):
                    issues.append(f"Skipped Skill was used or adapted: {name}")

            for variant in variants:
                name = str(variant.get("name") or "")
                if (
                    variant.get("status") == "validated"
                    and name not in executed_variants
                    and not variant.get("restored_from_bundle")
                ):
                    issues.append(f"Validated variant was not executed through ExecutePython: {name}")

            for name, fingerprint in (plan.get("report_hashes") or {}).items():
                try:
                    path = Path(str(fingerprint["path"])).resolve()
                    if not path.is_file() or _sha256(path) != fingerprint.get("sha256"):
                        issues.append(f"Report changed after planning: {name}")
                except Exception:
                    issues.append(f"Invalid report fingerprint in plan: {name}")

            if not (manifest.get("latest_handoffs") or {}):
                issues.append("No final artifact has been published")
            if self.context.split_mode == "test":
                for entry in manifest.get("steps", []):
                    if entry.get("skill_name") == "load_learned_rules":
                        statuses = set((entry.get("metadata") or {}).get("include_statuses") or [])
                        if statuses != {"frozen"}:
                            issues.append("Test mode must load frozen rules only")
                    if entry.get("skill_name") in {
                        "promote_validation_feedback", "freeze_learned_rules",
                    }:
                        issues.append(f"Test mode forbids rule mutation: {entry.get('skill_name')}")
            report = {
                "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
                "plan_path": str(self.context.skill_usage_plan_path),
                "report_hashes": plan.get("report_hashes", {}),
                "planned_decisions": plan.get("decisions", []),
                "actual_steps": actual_steps,
                "actual_skill_calls": skill_calls,
                "task_executions": task_executions,
                "variants": variants,
                "base_source_hashes_after": current_hashes,
                "unexplained_deviations": issues,
            }
            _write_json(report_path, report)
            record_step(
                "audit_skill_usage",
                "Skill usage audit passed." if not issues else "Skill usage audit found unresolved deviations.",
                [report_path],
                status=report["status"],
                issues=issues,
            )
            return _response(
                report["status"],
                "Skill usage audit passed." if not issues else "Skill usage audit needs repair.",
                {"report_path": str(report_path), "variant_count": len(variants)},
                issues,
            )
        except Exception as exc:
            issues.append(str(exc))
            _write_json(report_path, {"status": "NEEDS_REPAIR", "unexplained_deviations": issues})
            return _response(
                "NEEDS_REPAIR",
                "Skill usage audit failed.",
                {"report_path": str(report_path)},
                issues,
            )

    def _validate_decisions(self, decisions: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        by_name: dict[str, dict[str, Any]] = {}
        for raw in decisions:
            if not isinstance(raw, dict):
                raise ValueError("Each Skill decision must be a JSON object")
            name = str(raw.get("skill_name") or "")
            if name in by_name:
                raise ValueError(f"Duplicate Skill decision: {name}")
            if name not in self.skill_names and name != "standalone_python":
                raise ValueError(f"Unknown Engineer Skill in plan: {name}")
            decision = str(raw.get("decision") or "")
            if decision not in {"use", "adapt", "skip"}:
                raise ValueError(f"Invalid decision for {name}: {decision}")
            reason = str(raw.get("reason") or "").strip()
            if not reason:
                raise ValueError(f"Decision reason is required for {name}")
            gap = str(raw.get("capability_gap") or "").strip()
            why = str(raw.get("why_parameters_insufficient") or "").strip()
            if decision == "adapt" and (not gap or not why):
                raise ValueError(
                    f"adapt decision for {name} requires capability_gap and why_parameters_insufficient",
                )
            if name == "standalone_python" and (decision != "use" or not gap or not why):
                raise ValueError(
                    "standalone_python requires decision=use, capability_gap, and why_parameters_insufficient",
                )
            item = {
                "skill_name": name,
                "decision": decision,
                "reason": reason,
                "parameters": raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {},
                "capability_gap": gap,
                "why_parameters_insufficient": why,
            }
            by_name[name] = item
            normalized.append(item)
        missing = sorted(self.skill_names - set(by_name))
        if missing:
            raise ValueError(f"Missing decisions for Engineer Skills: {', '.join(missing)}")
        return normalized

    @staticmethod
    def _required_capabilities(task_plan_path: str | None) -> set[str]:
        if not task_plan_path:
            return set()
        path = Path(task_plan_path).expanduser().resolve()
        if not path.is_file():
            return set()
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks = [
            task
            for task in payload.get("tasks", [])
            if isinstance(task, dict) and task.get("required") is True
        ]
        capabilities = {
            str(task.get("capability_type"))
            for task in tasks
            if task.get("capability_type")
        }
        if tasks:
            capabilities.update({"reversible_cleaning", "result_package_build"})
        if any(len(task.get("input_artifacts") or []) > 1 for task in tasks):
            capabilities.add("structured_join")
        return capabilities

    def _base_skill_dir(self, skill_name: str) -> Path:
        if skill_name not in self.skill_names:
            raise ValueError(f"Skill is not available to Engineer: {skill_name}")
        path = (self.skills_root / skill_name).resolve()
        if path.parent != self.skills_root or not path.is_dir():
            raise ValueError(f"Base Skill directory does not exist: {path}")
        return path

    @staticmethod
    def _source_hashes(source_dir: Path) -> dict[str, str]:
        paths = [path for path in (source_dir / "SKILL.md", source_dir / "skill.py") if path.is_file()]
        if not paths:
            raise ValueError(f"Base Skill has no inspectable source files: {source_dir}")
        return {str(path.resolve()): _sha256(path) for path in paths}

    @staticmethod
    def _read_skill_metadata(skill_py: Path) -> dict[str, Any]:
        if not skill_py.exists():
            return {}
        try:
            tree = ast.parse(skill_py.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "SKILL" for target in node.targets
                ):
                    value = ast.literal_eval(node.value)
                    return value if isinstance(value, dict) else {}
        except Exception:
            return {}
        return {}

    def _require_plan_decision(self, skill_name: str, expected: str) -> None:
        plan = self.context.load_skill_plan()
        decision = next(
            (item for item in plan.get("decisions", []) if item.get("skill_name") == skill_name),
            None,
        )
        if not decision or decision.get("decision") != expected:
            raise ValueError(f"plan_skill_usage must mark {skill_name} as {expected}")

    def _variant_dir(self, variant_name: str) -> Path:
        if not _VARIANT_NAME_RE.fullmatch(variant_name):
            raise ValueError("Invalid variant_name")
        path = (self.context.variant_root / variant_name).resolve()
        if path.parent != self.context.variant_root.resolve() or not path.is_dir():
            raise ValueError(f"Variant does not exist in this run: {variant_name}")
        return path

    def _verify_source_hashes(self, metadata: dict[str, Any]) -> None:
        expected = metadata.get("base_source_hashes")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Variant is missing base_source_hashes")
        current = self._source_hashes(self._base_skill_dir(str(metadata.get("base_skill") or "")))
        if current != expected:
            raise ValueError(f"Base Skill source changed after variant creation: {metadata.get('base_skill')}")

    @staticmethod
    def _validate_variant_source(path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        if "IMPLEMENTATION_REQUIRED" in source:
            raise ValueError("variant.py still contains IMPLEMENTATION_REQUIRED; implement the change_spec first")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                if names & _BLOCKED_IMPORTS:
                    raise ValueError(f"Forbidden import in variant.py: {sorted(names & _BLOCKED_IMPORTS)}")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _BLOCKED_IMPORTS:
                raise ValueError(f"Forbidden import in variant.py: {node.module}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr in {
                    "system", "popen", "spawnl", "spawnlp", "spawnv", "spawnvp",
                }:
                    raise ValueError(f"Forbidden process call in variant.py: os.{node.func.attr}")
        required_tokens = ("OUTPUT_DIR", "sys.argv", "VARIANT_RESULT_JSON")
        missing = [token for token in required_tokens if token not in source]
        if missing:
            raise ValueError(f"variant.py is missing protocol tokens: {', '.join(missing)}")

    @staticmethod
    def _validate_artifact(path: Path, contract: dict[str, Any]) -> None:
        if not path.is_file():
            raise ValueError(f"Variant artifact does not exist: {path}")
        expected_name = str(contract.get("filename") or "")
        if expected_name and path.name != expected_name:
            raise ValueError(f"Expected artifact filename {expected_name}, got {path.name}")
        fmt = str(contract.get("format") or path.suffix.lstrip(".")).lower()
        allow_empty = bool(contract.get("allow_empty", False))
        required_columns = [str(value) for value in contract.get("required_columns") or []]
        primary_key = str(contract.get("primary_key") or "")
        if fmt == "csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                rows = list(reader)
            missing = [column for column in required_columns if column not in columns]
            if missing:
                raise ValueError(f"Artifact is missing required columns: {missing}")
            if not allow_empty and not rows:
                raise ValueError("Artifact must contain at least one data row")
            if primary_key:
                values = [row.get(primary_key, "") for row in rows]
                if any(value in {None, ""} for value in values):
                    raise ValueError(f"Primary key {primary_key} contains missing values")
                if len(values) != len(set(values)):
                    raise ValueError(f"Primary key {primary_key} contains duplicates")
        elif fmt == "json":
            json.loads(path.read_text(encoding="utf-8"))
        elif fmt == "jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not allow_empty and not rows:
                raise ValueError("Artifact must contain at least one JSONL row")
        else:
            raise ValueError(f"Unsupported output contract format: {fmt}")

    def _mark_variant_failed(self, variant_name: str, issue: str) -> None:
        try:
            metadata_path = self._variant_dir(variant_name) / "variant.json"
            metadata = _load_json(metadata_path)
            metadata["status"] = "failed"
            metadata["last_error"] = issue
            _write_json(metadata_path, metadata)
        except Exception:
            return

    @staticmethod
    def _variant_scaffold(
        base_skill: str,
        mode: str,
        output_contract: dict[str, Any],
    ) -> str:
        contract_literal = repr(output_contract)
        return f'''from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# IMPLEMENTATION_REQUIRED
# Base Skill: {base_skill}
# Mode: {mode}
# Output contract: {contract_literal}
# Replace this scaffold with an implementation derived from the inspected base Skill.
request_path = Path(sys.argv[1]).resolve()
request = json.loads(request_path.read_text(encoding="utf-8"))
output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
raise NotImplementedError("IMPLEMENTATION_REQUIRED")
# Successful implementations must print: VARIANT_RESULT_JSON={{...}}
'''


def register_engineer_skill_lifecycle_tools(
    toolkit: Toolkit,
    context: EngineerToolContext,
    *,
    skill_names: Iterable[str],
    skills_root: str | Path,
) -> EngineerSkillLifecycleTools:
    tools = EngineerSkillLifecycleTools(
        context,
        skill_names=skill_names,
        skills_root=skills_root,
    )
    for tool in (
        tools.initialize_skill_usage_plan,
        tools.revise_skill_usage,
        tools.plan_skill_usage,
        tools.inspect_skill,
        tools.create_skill_variant,
        tools.validate_skill_variant,
        tools.execute_skill_variant,
        tools.record_extraction_task,
        tools.audit_skill_usage,
    ):
        toolkit.register_tool_function(tool)
    return tools
