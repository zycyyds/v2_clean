from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from workflow.task_compiler import compile_extraction_task_plan


class ExperimentBundleManager:
    """Copy-on-write extraction bundle lifecycle for one experiment."""

    improvement_threshold = 0.001

    def __init__(self, experiment_dir: str | Path):
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.explorer_dir = self.experiment_dir / "explorer"
        self.active_dir = self.experiment_dir / "active_bundle"
        self.candidates_dir = self.experiment_dir / "candidates"
        self.evaluations_dir = self.experiment_dir / "evaluations"
        self.versions_dir = self.experiment_dir / "versions"
        self.state_path = self.experiment_dir / "experiment_state.json"

    def initialize(self, explorer_artifacts: dict[str, str]) -> Path:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        for directory in (self.explorer_dir, self.candidates_dir, self.evaluations_dir, self.versions_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists() and self.active_dir.is_dir():
            return self.active_dir
        required = {
            "field_extraction_rules": "field_extraction_rules.json",
            "data_analysis_report": "data_analysis_report.json",
            "explorer_run_record": "explorer_run_record.json",
        }
        self.active_dir.mkdir(parents=True, exist_ok=True)
        for alias, filename in required.items():
            source = Path(str(explorer_artifacts.get(alias) or "")).expanduser().resolve()
            if not source.is_file():
                raise ValueError(f"Missing Explorer artifact {alias}: {source}")
            explorer_target = (self.explorer_dir / filename).resolve()
            if source != explorer_target:
                shutil.copy2(source, explorer_target)
            if alias == "field_extraction_rules":
                shutil.copy2(source, self.active_dir / filename)
        task_plan_source = Path(str(explorer_artifacts.get("extraction_task_plan") or "")).expanduser()
        task_plan_target = self.explorer_dir / "extraction_task_plan.json"
        if task_plan_source.is_file():
            task_plan_source = task_plan_source.resolve()
            if task_plan_source != task_plan_target.resolve():
                shutil.copy2(task_plan_source, task_plan_target)
        else:
            rules_payload = _load_json(self.active_dir / "field_extraction_rules.json")
            rules = rules_payload.get("rules") if isinstance(rules_payload.get("rules"), list) else []
            complete_rules = [
                rule for rule in rules
                if isinstance(rule, dict) and rule.get("target_field_id") and rule.get("target_field_path")
            ]
            task_plan = compile_extraction_task_plan(
                complete_rules,
                record_grain=str(rules_payload.get("record_grain") or "case"),
            )
            task_plan["target_categories"] = list(rules_payload.get("target_categories") or [])
            _write_json(task_plan_target, task_plan)
        shutil.copy2(task_plan_target, self.active_dir / "extraction_task_plan.json")
        _write_json(self.active_dir / "skill_bindings.json", {"bindings": []})
        _write_json(self.active_dir / "target_field_mapping.json", {"mappings": []})
        (self.active_dir / "capabilities").mkdir()
        (self.active_dir / "tests").mkdir()
        (self.active_dir / "evaluation_history.jsonl").touch()
        self._write_manifest(self.active_dir, status="active", version=0)
        state = {
            "schema_version": 1,
            "status": "active",
            "best_score": 0.0,
            "best_round": 0,
            "consecutive_no_improvement": 0,
            "active_bundle_hash": _bundle_hash(self.active_dir),
            "rounds": [],
            "current_round": None,
            "round_attempts": [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(self.state_path, state)
        return self.active_dir

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise ValueError(f"Experiment state does not exist: {self.state_path}")
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Experiment state must be a JSON object")
        return value

    def create_candidate(self, round_index: int) -> Path:
        if round_index < 1:
            raise ValueError("round_index must be positive")
        if not self.active_dir.is_dir():
            raise ValueError("Active bundle has not been initialized")
        candidate = self.candidates_dir / f"round_{round_index:04d}"
        if candidate.exists():
            return candidate
        temporary = candidate.with_name(candidate.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(self.active_dir, temporary)
        manifest = _load_json(temporary / "manifest.json")
        manifest.update(
            {
                "status": "candidate",
                "round": round_index,
                "parent_bundle_hash": _bundle_hash(self.active_dir),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(candidate)
        return candidate

    def begin_round(self, round_index: int) -> Path:
        """Start a transactional candidate attempt and recover stale partial work."""
        state = self.load_state()
        current = state.get("current_round")
        if isinstance(current, dict) and current.get("status") == "running":
            if int(current.get("round", -1)) != round_index:
                raise ValueError(f"Round {current.get('round')} is already running")
            state.setdefault("round_attempts", []).append(
                {
                    "round": round_index,
                    "status": "recovered",
                    "candidate_dir": str(current.get("candidate_dir") or ""),
                    "error": "Recovered a stale running candidate after interruption.",
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        candidate = self.candidates_dir / f"round_{round_index:04d}"
        if candidate.exists():
            shutil.rmtree(candidate)
        candidate = self.create_candidate(round_index)
        state["current_round"] = {
            "round": round_index,
            "status": "running",
            "candidate_dir": str(candidate),
            "active_bundle_hash": state.get("active_bundle_hash", ""),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        state.setdefault("round_attempts", [])
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(self.state_path, state)
        return candidate

    def fail_round(self, round_index: int, candidate_dir: str | Path, error: str) -> dict[str, Any]:
        """Record an orchestration failure without mutating the active bundle."""
        state = self.load_state()
        candidate = Path(candidate_dir).expanduser().resolve()
        expected = (self.candidates_dir / f"round_{round_index:04d}").resolve()
        if candidate != expected:
            raise ValueError(f"Invalid candidate directory for round {round_index}: {candidate}")
        record = {
            "round": round_index,
            "status": "failed",
            "candidate_dir": str(candidate),
            "error": str(error),
            "failed_at": datetime.now().isoformat(timespec="seconds"),
        }
        state.setdefault("round_attempts", []).append(record)
        state["current_round"] = None
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(self.state_path, state)
        return record

    def finish_round(
        self,
        round_index: int,
        *,
        score: float,
        candidate_dir: str | Path,
        metadata: dict[str, Any] | None = None,
        eligible_for_promotion: bool = True,
    ) -> dict[str, Any]:
        state = self.load_state()
        existing = next((item for item in state["rounds"] if item.get("round") == round_index), None)
        if existing:
            return existing
        candidate = Path(candidate_dir).expanduser().resolve()
        expected = (self.candidates_dir / f"round_{round_index:04d}").resolve()
        if candidate != expected or not candidate.is_dir():
            raise ValueError(f"Invalid candidate directory for round {round_index}: {candidate}")
        score_value = round(float(score), 6)
        improved = bool(eligible_for_promotion) and (
            not state.get("rounds")
            or score_value >= float(state["best_score"]) + self.improvement_threshold
        )
        if improved:
            self._promote_candidate(candidate, round_index)
            state["best_score"] = score_value
            state["best_round"] = round_index
            state["consecutive_no_improvement"] = 0
            state["active_bundle_hash"] = _bundle_hash(self.active_dir)
        else:
            state["consecutive_no_improvement"] = int(state["consecutive_no_improvement"]) + 1
        record = {
            "round": round_index,
            "score": score_value,
            "improved": improved,
            "eligible_for_promotion": bool(eligible_for_promotion),
            "candidate_dir": str(candidate),
            "active_bundle_hash": state["active_bundle_hash"],
            "consecutive_no_improvement": state["consecutive_no_improvement"],
            "stop": state["consecutive_no_improvement"] >= 2,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            **(metadata or {}),
        }
        state["rounds"].append(record)
        state.setdefault("round_attempts", []).append({
            "round": round_index,
            "status": "completed",
            "candidate_dir": str(candidate),
            "score": score_value,
            "finished_at": record["finished_at"],
        })
        state["current_round"] = None
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(self.state_path, state)
        return record

    def freeze(self) -> Path:
        state = self.load_state()
        frozen = self.experiment_dir / "frozen_bundle"
        temporary = self.experiment_dir / "frozen_bundle.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(self.active_dir, temporary)
        self._write_manifest(temporary, status="frozen", version=int(state.get("best_round", 0)))
        if frozen.exists():
            shutil.rmtree(frozen)
        temporary.rename(frozen)
        state["status"] = "frozen"
        state["frozen_bundle_hash"] = _bundle_hash(frozen)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(self.state_path, state)
        return frozen

    def _promote_candidate(self, candidate: Path, round_index: int) -> None:
        temporary = self.experiment_dir / "active_bundle.next"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(candidate, temporary)
        self._write_manifest(temporary, status="active", version=round_index)
        backup = self.versions_dir / f"before_round_{round_index:04d}"
        if backup.exists():
            shutil.rmtree(backup)
        self.active_dir.rename(backup)
        try:
            temporary.rename(self.active_dir)
        except Exception:
            backup.rename(self.active_dir)
            raise

    @staticmethod
    def _write_manifest(bundle_dir: Path, *, status: str, version: int) -> None:
        previous = _load_json(bundle_dir / "manifest.json") if (bundle_dir / "manifest.json").exists() else {}
        previous.update(
            {
                "schema_version": 1,
                "status": status,
                "version": version,
                "bundle_hash": _bundle_hash(bundle_dir),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        _write_json(bundle_dir / "manifest.json", previous)


def _bundle_hash(bundle_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in bundle_dir.rglob("*") if item.is_file() and item.name != "manifest.json"):
        digest.update(path.relative_to(bundle_dir).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
