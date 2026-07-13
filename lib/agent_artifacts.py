from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

V2_DIR = Path(__file__).resolve().parents[1]
AGENT_RUNS_DIR = V2_DIR / "program" / "output" / "agent_runs"
LEGACY_CODE_OUTPUT_DIR = V2_DIR / "output" / "code_outputs"

ENV_RUN_ID = "MEDICAL_AGENT_RUN_ID"
ENV_RUN_ROOT = "MEDICAL_AGENT_RUN_ROOT"
ENV_PHASE_NAME = "MEDICAL_AGENT_PHASE_NAME"
ENV_PHASE_ROOT = "MEDICAL_AGENT_PHASE_ROOT"

_HANDOFF_FILENAMES = {
    "analysis_report": "data_analysis_report.json",
    "field_extraction_rules": "field_extraction_rules.json",
    "extraction_task_plan": "extraction_task_plan.json",
    "explorer_run_record": "explorer_run_record.json",
    "ml_dataset_csv": "ml_dataset_final.csv",
    "final_dataset_workbook": "final_dataset.xlsx",
    "final_report": "final_report.json",
}


def _slugify(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z一-鿿]+", "_", str(value or "").strip()).strip("_")
    return (text or "artifact")[:80]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if isinstance(data, dict):
        return data
    return default


def _phase_relpath(phase_name: str) -> Path:
    return Path(_slugify(phase_name))


def create_run_session(mode: str) -> dict[str, str]:
    AGENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = AGENT_RUNS_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    session_payload = {
        "run_id": run_id,
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
    }
    _write_json(run_root / "session.json", session_payload)
    return session_payload


def init_phase_session(run_root: str | Path, phase_name: str) -> dict[str, str]:
    run_root_path = Path(run_root)
    phase_root = run_root_path / _phase_relpath(phase_name)
    artifacts_dir = phase_root / "artifacts"
    next_input_dir = phase_root / "next_input"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    next_input_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = phase_root / "manifest.json"
    manifest = _read_json(
        manifest_path,
        {
            "run_id": run_root_path.name,
            "phase": phase_name,
            "phase_root": str(phase_root),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "steps": [],
            "latest_handoffs": {},
        },
    )
    manifest.setdefault("steps", [])
    manifest.setdefault("latest_handoffs", {})
    manifest["phase_root"] = str(phase_root)
    _write_json(manifest_path, manifest)
    return {
        "phase_name": phase_name,
        "phase_root": str(phase_root),
        "artifacts_dir": str(artifacts_dir),
        "next_input_dir": str(next_input_dir),
        "manifest_path": str(manifest_path),
    }


def set_phase_context(run_id: str, run_root: str | Path, phase_name: str, phase_root: str | Path) -> None:
    os.environ[ENV_RUN_ID] = str(run_id)
    os.environ[ENV_RUN_ROOT] = str(run_root)
    os.environ[ENV_PHASE_NAME] = str(phase_name)
    os.environ[ENV_PHASE_ROOT] = str(phase_root)


def clear_phase_context() -> None:
    for key in (ENV_RUN_ID, ENV_RUN_ROOT, ENV_PHASE_NAME, ENV_PHASE_ROOT):
        os.environ.pop(key, None)


def get_phase_context() -> dict[str, str] | None:
    run_id = os.environ.get(ENV_RUN_ID)
    run_root = os.environ.get(ENV_RUN_ROOT)
    phase_name = os.environ.get(ENV_PHASE_NAME)
    phase_root = os.environ.get(ENV_PHASE_ROOT)
    if not all((run_id, run_root, phase_name, phase_root)):
        return None
    manifest_path = Path(phase_root) / "manifest.json"
    return {
        "run_id": run_id,
        "run_root": run_root,
        "phase_name": phase_name,
        "phase_root": phase_root,
        "manifest_path": str(manifest_path),
        "artifacts_dir": str(Path(phase_root) / "artifacts"),
        "next_input_dir": str(Path(phase_root) / "next_input"),
    }


def begin_step(skill_name: str) -> dict[str, Any] | None:
    context = get_phase_context()
    if context is None:
        return None
    manifest_path = Path(context["manifest_path"])
    manifest = _read_json(
        manifest_path,
        {
            "run_id": context["run_id"],
            "phase": context["phase_name"],
            "phase_root": context["phase_root"],
            "steps": [],
            "latest_handoffs": {},
        },
    )
    steps = manifest.setdefault("steps", [])
    step_index = len(steps) + 1
    step_name = f"step{step_index:02d}_{_slugify(skill_name)}"
    step_dir = Path(context["artifacts_dir"]) / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    return {
        **context,
        "step_index": step_index,
        "skill_name": skill_name,
        "step_name": step_name,
        "step_dir": str(step_dir),
    }


def _normalize_artifact_paths(paths: list[str | Path]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = str(Path(path).resolve())
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def publish_handoffs(handoffs: dict[str, str | Path]) -> dict[str, dict[str, str]]:
    context = get_phase_context()
    if context is None:
        return {}
    next_input_dir = Path(context["next_input_dir"])
    next_input_dir.mkdir(parents=True, exist_ok=True)
    published: dict[str, dict[str, str]] = {}
    for alias, source in handoffs.items():
        source_path = Path(source)
        if not source_path.exists() or not source_path.is_file():
            continue
        filename = _HANDOFF_FILENAMES.get(alias, source_path.name)
        target_path = next_input_dir / filename
        shutil.copy2(source_path, target_path)
        record = {
            "source_path": str(source_path.resolve()),
            "canonical_path": str(target_path.resolve()),
        }
        if alias == "analysis_report":
            LEGACY_CODE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            legacy_path = LEGACY_CODE_OUTPUT_DIR / "data_analysis_report.json"
            shutil.copy2(source_path, legacy_path)
            record["legacy_path"] = str(legacy_path.resolve())
        published[alias] = record
    return published


def record_step(
    skill_name: str,
    summary: str,
    artifact_paths: list[str | Path],
    *,
    handoffs: dict[str, str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
    status: str = "SUCCESS",
    issues: list[str] | None = None,
    _step_ctx: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    context = _step_ctx if _step_ctx is not None else begin_step(skill_name)
    if context is None:
        return None
    manifest_path = Path(context["manifest_path"])
    manifest = _read_json(
        manifest_path,
        {
            "run_id": context["run_id"],
            "phase": context["phase_name"],
            "phase_root": context["phase_root"],
            "steps": [],
            "latest_handoffs": {},
        },
    )
    normalized_artifacts = _normalize_artifact_paths(artifact_paths)
    published = publish_handoffs(handoffs or {}) if handoffs else {}
    entry = {
        "step_index": context["step_index"],
        "skill_name": skill_name,
        "status": status,
        "summary": summary,
        "artifact_dir": context["step_dir"],
        "artifacts": normalized_artifacts,
        "handoffs": published,
        "metadata": metadata or {},
        "issues": issues or [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest.setdefault("steps", []).append(entry)
    latest_handoffs = manifest.setdefault("latest_handoffs", {})
    for alias, record in published.items():
        latest_handoffs[alias] = record["canonical_path"]
    _write_json(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path.resolve()),
        "step_index": context["step_index"],
        "step_artifact_dir": context["step_dir"],
        "handoffs": published,
    }


def resolve_phase_handoff(run_root: str | Path, phase_name: str, alias: str) -> str | None:
    phase_root = Path(run_root) / _phase_relpath(phase_name)
    manifest_path = phase_root / "manifest.json"
    manifest = _read_json(manifest_path, {})
    latest_handoffs = manifest.get("latest_handoffs") or {}
    handoff_path = latest_handoffs.get(alias)
    if handoff_path and Path(handoff_path).exists():
        return str(Path(handoff_path).resolve())
    for entry in reversed(manifest.get("steps") or []):
        entry_handoffs = entry.get("handoffs") or {}
        record = entry_handoffs.get(alias)
        if isinstance(record, dict):
            candidate = record.get("canonical_path")
            if candidate and Path(candidate).exists():
                return str(Path(candidate).resolve())
    if alias == "analysis_report":
        # scan artifacts/ subtree first
        artifacts_dir = phase_root / "artifacts"
        if artifacts_dir.exists():
            exact_candidates = sorted(artifacts_dir.rglob("data_analysis_report.json"), reverse=True)
            if exact_candidates:
                return str(exact_candidates[0].resolve())
            for candidate in sorted(artifacts_dir.rglob("*.json"), reverse=True):
                if _is_analysis_report_candidate(candidate):
                    return str(candidate.resolve())
        # fallback: any report-like JSON directly under phase_root
        exact_phase_candidates = sorted(phase_root.glob("data_analysis_report.json"), reverse=True)
        if exact_phase_candidates:
            return str(exact_phase_candidates[0].resolve())
        for candidate in sorted(phase_root.glob("*.json"), reverse=True):
            if _is_analysis_report_candidate(candidate):
                return str(candidate.resolve())
    return None


def resolve_canonical_phase_handoff(
    run_root: str | Path,
    phase_name: str,
    alias: str,
) -> str | None:
    """Resolve only an explicitly published canonical handoff, without scanning artifacts."""
    phase_root = Path(run_root) / _phase_relpath(phase_name)
    manifest = _read_json(phase_root / "manifest.json", {})
    candidate = (manifest.get("latest_handoffs") or {}).get(alias)
    if not candidate:
        return None
    path = Path(candidate).expanduser().resolve()
    canonical_root = (phase_root / "next_input").resolve()
    if not path.is_file() or (path != canonical_root and canonical_root not in path.parents):
        return None
    return str(path)


def _is_analysis_report_candidate(path: Path) -> bool:
    name = path.name
    if name == "data_analysis_report.json":
        return True
    excluded = {
        "gold_field_provenance_report.json",
        "planner_extraction_brief.json",
        "gold_schema_summary.json",
    }
    if name in excluded:
        return False
    return name.endswith("report.json")
