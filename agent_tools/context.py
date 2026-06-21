from __future__ import annotations

import os
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
        roots.extend(_existing_absolute_paths(task_text))
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


class ExplorerToolContext(EngineerToolContext):
    """Read-oriented atomic-tool context bound to one Explorer phase."""

    @classmethod
    def from_task(
        cls,
        *,
        task_text: str,
        explorer_phase_root: str | Path,
        additional_read_roots: Iterable[str | Path] = (),
    ) -> "ExplorerToolContext":
        roots: list[str | Path] = list(additional_read_roots)
        roots.extend(_existing_absolute_paths(task_text))
        return cls(
            engineer_phase_root=explorer_phase_root,
            read_roots=roots,
            require_skill_plan=False,
        )

    @property
    def explorer_phase_root(self) -> Path:
        return Path(self.engineer_phase_root)


_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.-])(/[^\s，。；;：:'\"<>]+)")


def _existing_absolute_paths(text: str) -> list[Path]:
    paths: list[Path] = []
    for match in _ABSOLUTE_PATH_RE.finditer(str(text or "")):
        raw = match.group(1).rstrip(")]}>,.，。")
        path = Path(os.path.expanduser(raw))
        if path.exists():
            resolved = path.resolve()
            if resolved not in paths:
                paths.append(resolved)
    return paths
