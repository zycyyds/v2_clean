from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


class EngineerToolPermissionError(ValueError):
    """Raised when an atomic tool accesses a path outside its capability."""


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _normalize_roots(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path not in roots:
            roots.append(path)
    return tuple(roots)


@dataclass
class EngineerToolContext:
    """Path permissions and read fingerprints for Engineer atomic tools."""

    engineer_phase_root: str | Path
    read_roots: Iterable[str | Path] = field(default_factory=tuple)
    require_skill_plan: bool = False
    split_mode: str = "training"
    required_report_paths: dict[str, str] = field(default_factory=dict)
    workspace_dir: Path = field(init=False)
    variant_root: Path = field(init=False)
    skill_usage_plan_path: Path = field(init=False)
    skill_usage_events_path: Path = field(init=False)
    task_execution_path: Path = field(init=False)
    fingerprints: dict[Path, tuple[int, int]] = field(default_factory=dict)
    executable_skill_names: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.engineer_phase_root = Path(self.engineer_phase_root).expanduser().resolve()
        self.engineer_phase_root.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = self.engineer_phase_root / "workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.variant_root = self.workspace_dir / "skill_variants"
        self.skill_usage_plan_path = self.workspace_dir / "skill_usage_plan.json"
        self.skill_usage_events_path = self.workspace_dir / "skill_usage_events.jsonl"
        self.task_execution_path = self.workspace_dir / "extraction_task_execution.json"
        self.read_roots = _normalize_roots(
            [*self.read_roots, self.engineer_phase_root],
        )

    @classmethod
    def from_task(
        cls,
        *,
        task_text: str,
        engineer_phase_root: str | Path,
        explorer_phase_root: str | Path | None = None,
        additional_read_roots: Iterable[str | Path] = (),
        require_skill_plan: bool = False,
        split_mode: str = "training",
        required_report_paths: dict[str, str] | None = None,
    ) -> "EngineerToolContext":
        roots: list[str | Path] = list(additional_read_roots)
        if explorer_phase_root:
            roots.append(explorer_phase_root)
        return cls(
            engineer_phase_root=engineer_phase_root,
            read_roots=roots,
            require_skill_plan=require_skill_plan,
            split_mode=split_mode,
            required_report_paths=required_report_paths or {},
        )

    def require_python_plan(self, path: str | Path) -> None:
        """Enforce the run-local Skill plan before Python authoring/execution."""
        if not self.require_skill_plan:
            return
        if not self.skill_usage_plan_path.exists():
            raise EngineerToolPermissionError(
                "Call initialize_skill_usage_plan (or legacy plan_skill_usage) before writing or executing Python.",
            )
        resolved = Path(path).resolve()
        if _is_within(resolved, self.variant_root):
            return
        plan = self.load_skill_plan()
        standalone = next(
            (
                item
                for item in plan.get("decisions", [])
                if item.get("skill_name") == "standalone_python"
            ),
            None,
        )
        if not standalone or standalone.get("decision") != "use":
            raise EngineerToolPermissionError(
                "Standalone Python was not approved by the Skill usage plan.",
            )

    def validate_python_rule_ids(self, content: str) -> None:
        """Reject target field IDs that are absent from the canonical rule file."""
        rules_value = self.required_report_paths.get("field_extraction_rules")
        if not rules_value:
            return
        rules_path = Path(rules_value).expanduser().resolve()
        if not rules_path.is_file():
            return
        try:
            payload = json.loads(rules_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise EngineerToolPermissionError(
                f"Cannot validate Python target_field_id values: {exc}",
            ) from exc
        known_ids = {
            str(item.get("target_field_id"))
            for item in payload.get("rules", [])
            if isinstance(item, dict) and item.get("target_field_id")
        }
        referenced_ids = set(re.findall(r"\bfield_[0-9a-f]{16}\b", str(content)))
        unknown_ids = sorted(referenced_ids - known_ids)
        if unknown_ids:
            raise EngineerToolPermissionError(
                "Python references target_field_id values absent from canonical rules: "
                + ", ".join(unknown_ids),
            )

    def validate_standalone_task(self, task_id: str, content: str) -> None:
        """Allow standalone data production only for a declared capability gap."""
        task_plan_value = self.required_report_paths.get("extraction_task_plan")
        if not self.require_skill_plan or not task_plan_value:
            return
        task_plan = json.loads(
            Path(task_plan_value).expanduser().resolve().read_text(encoding="utf-8"),
        )
        task = next(
            (
                item
                for item in task_plan.get("tasks", [])
                if isinstance(item, dict) and str(item.get("task_id") or "") == task_id
            ),
            None,
        )
        if task is None:
            raise EngineerToolPermissionError(f"Unknown standalone extraction task: {task_id}")
        plan = self.load_skill_plan()
        standalone = next(
            (
                item
                for item in plan.get("decisions", [])
                if item.get("skill_name") == "standalone_python"
                and item.get("decision") == "use"
            ),
            None,
        )
        gap_capabilities = {
            value.strip()
            for value in re.split(r"[,，]", str((standalone or {}).get("capability_gap") or ""))
            if value.strip()
        }
        capability = str(task.get("capability_type") or "")
        if standalone is None or capability not in gap_capabilities:
            raise EngineerToolPermissionError(
                f"Task {task_id} capability {capability} is covered by a planned domain Skill; "
                "call that Skill instead of standalone Python.",
            )
        task_rule_ids = {str(value) for value in task.get("rule_ids") or []}
        referenced_ids = set(re.findall(r"\bfield_[0-9a-f]{16}\b", str(content)))
        unrelated = sorted(referenced_ids - task_rule_ids)
        if unrelated:
            raise EngineerToolPermissionError(
                f"Standalone script for {task_id} references rule IDs from another task: "
                + ", ".join(unrelated),
            )

    def load_skill_plan(self) -> dict:
        if not self.skill_usage_plan_path.exists():
            return {}
        try:
            value = json.loads(self.skill_usage_plan_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def record_skill_call(
        self,
        skill_name: str,
        tool_name: str,
        status: str,
        *,
        artifacts: Iterable[str | Path] = (),
    ) -> None:
        event = {
            "skill_name": skill_name,
            "tool_name": tool_name,
            "status": status,
            "artifacts": [str(Path(path).expanduser().resolve()) for path in artifacts],
        }
        with self.skill_usage_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def register_executable_skills(self, names: Iterable[str]) -> None:
        self.executable_skill_names.update(str(name) for name in names if str(name))

    def load_skill_calls(self) -> list[dict]:
        if not self.skill_usage_events_path.exists():
            return []
        events: list[dict] = []
        for line in self.skill_usage_events_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events

    def resolve_read_path(self, value: str | Path) -> Path:
        path = self._resolve(value)
        if not any(self._read_root_allows(path, root) for root in self.read_roots):
            raise EngineerToolPermissionError(
                f"Read path is outside authorized roots: {path}",
            )
        return path

    def resolve_write_path(self, value: str | Path) -> Path:
        path = self._resolve(value)
        phase_root = Path(self.engineer_phase_root)
        if not _is_within(path, phase_root):
            raise EngineerToolPermissionError(
                f"Write path is outside the Engineer phase: {path}",
            )
        return path

    def mark_read(self, path: str | Path) -> None:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        self.fingerprints[resolved] = (int(stat.st_mtime_ns), int(stat.st_size))

    def was_read_and_unchanged(self, path: str | Path) -> bool:
        resolved = Path(path).resolve()
        previous = self.fingerprints.get(resolved)
        if previous is None or not resolved.exists():
            return False
        stat = resolved.stat()
        return previous == (int(stat.st_mtime_ns), int(stat.st_size))

    def forget_read(self, path: str | Path) -> None:
        self.fingerprints.pop(Path(path).resolve(), None)

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.workspace_dir / path
        return path.resolve()

    @staticmethod
    def _read_root_allows(path: Path, root: Path) -> bool:
        if root.is_file():
            return path == root
        return _is_within(path, root)
