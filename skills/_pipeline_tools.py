from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from agentscope.tool import ToolResponse

from skills._rule_tools import record, response, step_output


PIPELINE_ROOT = Path("/Users/mkbk/PycharmProjects/MIMIC-IV-Data-Pipeline-main")


def parse_spec(spec_json: str) -> dict[str, Any]:
    payload = json.loads(spec_json or "{}")
    if not isinstance(payload, dict):
        raise ValueError("spec_json must be a JSON object")
    return payload


def mimic_root_from_spec(spec: dict[str, Any]) -> str:
    root = str(spec.get("mimic_root") or spec.get("raw_root") or "")
    if not root:
        raise ValueError("spec_json requires mimic_root or raw_root")
    return root


def task_spec_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    task_spec = spec.get("task_spec") or {}
    if not isinstance(task_spec, dict):
        raise ValueError("spec_json.task_spec must be an object when provided")
    result = dict(task_spec)
    result.setdefault("care_setting", spec.get("care_setting") or "ICU")
    result.setdefault("outcome_type", spec.get("outcome_type") or "Mortality")
    result.setdefault("record_grain", spec.get("record_grain") or ("stay_id" if result.get("care_setting") == "ICU" else "hadm_id"))
    if spec.get("time_window_days") is not None:
        result["time_window_days"] = spec["time_window_days"]
    return result


def run_pipeline_step(
    skill_name: str,
    spec_json: str,
    runner: Callable[[dict[str, Any], Path], dict[str, Any]],
    *,
    summary: str,
    artifact_keys: tuple[str, ...],
    source_pipeline_files: list[str],
    expected_raw_files: list[str],
) -> ToolResponse:
    step, output = step_output(skill_name)
    try:
        spec = parse_spec(spec_json)
        result = runner(spec, output)
        artifacts = {
            key: result.get(key, "")
            for key in artifact_keys
            if key in result
        }
        paths = [Path(str(value)) for value in artifacts.values() if str(value)]
        manifest = record(
            skill_name,
            summary,
            paths,
            step=step,
            metadata={
                "row_counts": result.get("row_counts") or {},
                "columns": result.get("columns") or {},
                "source_files": result.get("source_files") or [],
                "source_pipeline_files": source_pipeline_files,
                "expected_raw_files": expected_raw_files,
                **{key: value for key, value in result.items() if key not in set(artifact_keys)},
            },
            status=str(result.get("status") or "SUCCESS"),
            issues=result.get("issues") or [],
        )
        payload = {
            **artifacts,
            "status": str(result.get("status") or "SUCCESS"),
            "artifacts": artifacts,
            "row_counts": result.get("row_counts") or _row_counts(paths),
            "columns": result.get("columns") or _columns(paths),
            "source_files": result.get("source_files") or [],
            "source_pipeline_files": source_pipeline_files,
            "issues": result.get("issues") or [],
            "manifest_path": (manifest or {}).get("manifest_path", ""),
        }
        payload.update({key: value for key, value in result.items() if key not in payload and key not in set(artifact_keys)})
        return response(payload["status"], summary, payload, payload["issues"])
    except Exception as exc:
        return response(
            "NEEDS_REPAIR",
            f"{skill_name} failed.",
            artifacts={
                "source_pipeline_files": source_pipeline_files,
                "expected_raw_files": expected_raw_files,
            },
            issues=[str(exc)],
        )


def maybe_copy(path: str | Path, target: Path) -> Path:
    source = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target.resolve():
        shutil.copy2(source, target)
    return target


def table_info(path: str | Path) -> tuple[int, list[str]]:
    source = Path(path).expanduser().resolve()
    name = source.name.lower()
    if name.endswith((".csv", ".csv.gz")):
        frame = pd.read_csv(source, dtype=object)
    elif name.endswith((".tsv", ".tsv.gz")):
        frame = pd.read_csv(source, sep="\t", dtype=object)
    elif name.endswith(".parquet"):
        frame = pd.read_parquet(source)
    elif name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(source, dtype=object)
    else:
        return 0, []
    return int(len(frame)), [str(column) for column in frame.columns]


def _row_counts(paths: list[Path]) -> dict[str, int]:
    values: dict[str, int] = {}
    for path in paths:
        try:
            rows, _cols = table_info(path)
            values[path.name] = rows
        except Exception:
            continue
    return values


def _columns(paths: list[Path]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for path in paths:
        try:
            _rows, cols = table_info(path)
            values[path.name] = cols
        except Exception:
            continue
    return values


def raw_sources(root: str | Path, relatives: list[str]) -> list[str]:
    base = Path(root).expanduser().resolve()
    found: list[str] = []
    for relative in relatives:
        path = base / relative
        gz = base / f"{relative}.gz"
        if path.exists():
            found.append(str(path))
        elif gz.exists():
            found.append(str(gz))
    return found


def safe_target(package_root: Path, relative: str) -> Path:
    target = package_root / relative
    resolved_root = package_root.resolve()
    resolved_target = target.resolve()
    if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
        raise ValueError(f"Unsafe package target path: {relative}")
    return target
