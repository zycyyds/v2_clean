from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from workflow.reference_evaluation import (
    IGNORED_PACKAGE_FILES,
    _structured_package_files,
    score_reference_directory,
)
from workflow.reference_package_gate import (
    MIMIC_EXPECTED_BUSINESS_FILE_COUNT,
    validate_business_result_package,
)


PIPELINE_MODULES = (
    "cohort.py",
    "labels.py",
    "chart.py",
    "diag.py",
    "med.py",
    "out.py",
    "proc.py",
    "summary.py",
)
PIPELINE_REQUIRED_FILES = (
    "pipeline_manifest.json",
    "run.py",
    *PIPELINE_MODULES,
    "config.yaml",
)
PIPELINE_MANIFEST_SCHEMA_VERSION = 1
MAX_PIPELINE_FILE_BYTES = 2 * 1024 * 1024
MAX_PIPELINE_TOTAL_BYTES = 10 * 1024 * 1024
MAX_PIPELINE_LITERAL_BYTES = 4096
MAX_PIPELINE_COLLECTION_ITEMS = 256
MAX_PIPELINE_TOTAL_LITERAL_BYTES = 128 * 1024
DEFAULT_PIPELINE_TIMEOUT_SECONDS = 3600
MAX_PIPELINE_LOG_BYTES = 16 * 1024 * 1024
MAX_PIPELINE_ADDRESS_SPACE_BYTES = 24 * 1024 * 1024 * 1024
MAX_PIPELINE_OUTPUT_FILE_BYTES = 8 * 1024 * 1024 * 1024
MAX_PIPELINE_OUTPUT_TOTAL_BYTES = 20 * 1024 * 1024 * 1024
MAX_PIPELINE_OUTPUT_FILES = 256
FORBIDDEN_PIPELINE_SOURCE_MARKERS = (
    "active_bundle",
    "validation_result",
    "reference_private",
    "test_evaluation",
    "private_report",
)
SUMMARY_ASSET_ROOT = "summary_assets"
FORBIDDEN_SUMMARY_ASSET_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "patient_id",
    "person_id",
    "label",
    "mortality",
    "dod",
}
_FORBIDDEN_DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".xlsx",
    ".xls",
    ".jsonl",
}


def pipeline_directory_sha256(pipeline_dir: str | Path) -> str:
    return _hash_manifest(_relative_file_hashes(Path(pipeline_dir).expanduser().resolve()))


def business_file_hashes(package_root: str | Path) -> dict[str, str]:
    root = Path(package_root).expanduser().resolve()
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _structured_package_files(root)
        if path.name not in IGNORED_PACKAGE_FILES
    }


def pipeline_modules_for_business_targets(relative_paths: set[str]) -> set[str]:
    modules: set[str] = set()
    for raw_relative in relative_paths:
        relative = raw_relative.strip().lstrip("/").casefold()
        name = Path(relative).name
        if relative.startswith("cohort/"):
            modules.add("cohort.py")
        elif relative.startswith("summary/"):
            modules.add("summary.py")
        elif "label" in name or name == "reference.csv":
            modules.add("labels.py")
        else:
            for stem in ("chart", "diag", "med", "out", "proc"):
                if stem in name:
                    modules.add(f"{stem}.py")
                    break
    return modules


def validate_pipeline_structure(
    pipeline_dir: str | Path,
    *,
    expected_files: list[str],
    expected_parent_sha256: str,
    previous_pipeline_dir: str | Path | None = None,
    allowed_changed_modules: set[str] | None = None,
    train_reference_root: str | Path | None = None,
) -> dict[str, Any]:
    pipeline = Path(pipeline_dir).expanduser().resolve()
    issues: list[str] = []
    missing = [name for name in PIPELINE_REQUIRED_FILES if not (pipeline / name).is_file()]
    if missing:
        issues.append("missing pipeline files: " + ", ".join(missing))

    actual_files = sorted(
        path.relative_to(pipeline).as_posix()
        for path in pipeline.rglob("*")
        if path.is_file()
    ) if pipeline.is_dir() else []

    manifest_path = pipeline / "pipeline_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                issues.append("pipeline_manifest.json root must be a JSON object")
            else:
                manifest = loaded
        except Exception as exc:
            issues.append(f"pipeline_manifest.json is not valid JSON: {type(exc).__name__}: {exc}")
    learned_summary_assets, asset_issues = _validate_learned_summary_assets(
        pipeline=pipeline,
        manifest=manifest,
        expected_files=expected_files,
        train_reference_root=train_reference_root,
    )
    issues.extend(asset_issues)
    allowed_pipeline_files = {*PIPELINE_REQUIRED_FILES, *learned_summary_assets}
    unexpected_files = sorted(set(actual_files) - allowed_pipeline_files)
    if unexpected_files:
        issues.append(
            "pipeline contains undeclared files: " + ", ".join(unexpected_files[:20])
        )
    oversized_files = sorted(
        relative
        for relative in actual_files
        if (pipeline / relative).stat().st_size > MAX_PIPELINE_FILE_BYTES
    )
    total_pipeline_bytes = sum((pipeline / relative).stat().st_size for relative in actual_files)
    if oversized_files:
        issues.append("pipeline files exceed size limit: " + ", ".join(oversized_files[:20]))
    if total_pipeline_bytes > MAX_PIPELINE_TOTAL_BYTES:
        issues.append(
            f"pipeline total size exceeds {MAX_PIPELINE_TOTAL_BYTES} bytes: {total_pipeline_bytes}"
        )
    forbidden_source_hits: list[str] = []
    for relative in actual_files:
        path = pipeline / relative
        if path.suffix.casefold() not in {".py", ".json", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").casefold()
        markers = [marker for marker in FORBIDDEN_PIPELINE_SOURCE_MARKERS if marker in text]
        if markers:
            forbidden_source_hits.append(f"{relative} ({', '.join(markers)})")
    if forbidden_source_hits:
        issues.append(
            "pipeline source references forbidden result/private locations: "
            + ", ".join(forbidden_source_hits[:20])
        )
    embedded_literal_hits = _embedded_literal_issues(pipeline, actual_files)
    if embedded_literal_hits:
        issues.append(
            "pipeline source contains suspicious embedded literal result data: "
            + ", ".join(embedded_literal_hits[:20])
        )

    if manifest:
        try:
            schema_version = int(manifest.get("schema_version") or 0)
        except (TypeError, ValueError):
            schema_version = 0
        if schema_version != PIPELINE_MANIFEST_SCHEMA_VERSION:
            issues.append(
                f"pipeline manifest schema_version must be {PIPELINE_MANIFEST_SCHEMA_VERSION}"
            )
        if str(manifest.get("entrypoint") or "") != "run.py":
            issues.append("pipeline manifest entrypoint must be run.py")
        modules_value = manifest.get("modules")
        if not isinstance(modules_value, list):
            issues.append("pipeline manifest modules must be a list")
            modules_value = []
        declared_modules = [str(item) for item in modules_value]
        if sorted(declared_modules) != sorted(PIPELINE_MODULES):
            issues.append("pipeline manifest modules must declare the canonical module list")
        expected_files_value = manifest.get("expected_files")
        if not isinstance(expected_files_value, list):
            issues.append("pipeline manifest expected_files must be a list")
            expected_files_value = []
        declared_files = sorted(str(item).lstrip("/") for item in expected_files_value)
        if declared_files != sorted(expected_files):
            issues.append("pipeline manifest must declare the complete expected business file list")
        actual_parent = str(manifest.get("parent_pipeline_sha256") or "")
        if actual_parent != expected_parent_sha256:
            issues.append(
                "parent_pipeline_sha256 mismatch: "
                f"expected={expected_parent_sha256 or '<root>'}, actual={actual_parent or '<root>'}"
            )

    embedded_data = sorted(
        relative
        for relative in actual_files
        if Path(relative).suffix.casefold() in _FORBIDDEN_DATA_SUFFIXES
        and relative not in learned_summary_assets
    )
    if embedded_data:
        issues.append("pipeline contains embedded result data: " + ", ".join(embedded_data[:20]))

    previous = (
        Path(previous_pipeline_dir).expanduser().resolve()
        if previous_pipeline_dir is not None
        else None
    )
    previous_hashes = _relative_file_hashes(previous) if previous is not None and previous.is_dir() else {}
    current_hashes = _relative_file_hashes(pipeline)
    changed_files = sorted(
        relative
        for relative in set(previous_hashes) | set(current_hashes)
        if previous_hashes.get(relative) != current_hashes.get(relative)
    )
    changed_modules = sorted(set(changed_files) & set(PIPELINE_MODULES))
    if previous_hashes and allowed_changed_modules is not None:
        unrelated_modules = sorted(set(changed_modules) - set(allowed_changed_modules))
        if unrelated_modules:
            issues.append(
                "unrelated pipeline modules changed: " + ", ".join(unrelated_modules)
            )
        allowed_rule_files = set(allowed_changed_modules)
        changed_rule_files = sorted(
            set(changed_files) & {"run.py", "config.yaml", *PIPELINE_MODULES}
        )
        unrelated_rule_files = sorted(set(changed_rule_files) - allowed_rule_files)
        if unrelated_rule_files:
            issues.append(
                "unrelated pipeline rule files changed: "
                + ", ".join(unrelated_rule_files)
            )
    return {
        "schema_version": 1,
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "valid": not issues,
        "pipeline_dir": str(pipeline),
        "pipeline_sha256": _hash_manifest(current_hashes),
        "parent_pipeline_sha256": str(manifest.get("parent_pipeline_sha256") or ""),
        "expected_parent_pipeline_sha256": expected_parent_sha256,
        "changed_pipeline_files": changed_files,
        "changed_pipeline_modules": changed_modules,
        "allowed_changed_modules": sorted(allowed_changed_modules or []),
        "learned_summary_assets": sorted(learned_summary_assets),
        "issues": issues,
        "manifest": manifest,
    }


def run_pipeline_isolated(
    *,
    pipeline_dir: str | Path,
    raw_root: str | Path,
    output_dir: str | Path,
    split_mode: str,
    log_path: str | Path,
    timeout_seconds: int = DEFAULT_PIPELINE_TIMEOUT_SECONDS,
    max_output_bytes: int = MAX_PIPELINE_OUTPUT_TOTAL_BYTES,
    max_output_files: int = MAX_PIPELINE_OUTPUT_FILES,
    max_rss_bytes: int = MAX_PIPELINE_ADDRESS_SPACE_BYTES,
) -> dict[str, Any]:
    pipeline = Path(pipeline_dir).expanduser().resolve()
    raw = Path(raw_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    log = Path(log_path).expanduser().resolve()
    if split_mode not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split_mode: {split_mode}")
    if not pipeline.is_dir() or not (pipeline / "run.py").is_file():
        raise ValueError(f"canonical pipeline entrypoint is missing: {pipeline / 'run.py'}")
    if not raw.is_dir():
        raise ValueError(f"raw root does not exist: {raw}")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    if min(max_output_bytes, max_output_files, max_rss_bytes) < 1:
        raise ValueError("pipeline resource limits must be positive")
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    with tempfile.TemporaryDirectory(prefix="reference-pipeline-replay-") as temp_text:
        sandbox = Path(temp_text).resolve()
        sandbox_pipeline = sandbox / "pipeline"
        shutil.copytree(pipeline, sandbox_pipeline)
        (sandbox / "home").mkdir()
        (sandbox / "tmp").mkdir()
        wrapper = sandbox / "isolated_runner.py"
        wrapper.write_text(_ISOLATED_RUNNER, encoding="utf-8")
        runtime_prefixes = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
        allowed_runtime_roots = sorted(
            {
                str(Path(path).expanduser().resolve())
                for path in sys.path
                if path
                and Path(path).exists()
                and any(
                    Path(path).expanduser().resolve() == prefix
                    or prefix in Path(path).expanduser().resolve().parents
                    for prefix in runtime_prefixes
                )
            }
            | {str(prefix) for prefix in runtime_prefixes}
            | {
                "/dev/null",
                "/dev/random",
                "/dev/urandom",
                "/System",
                "/usr/lib",
                "/usr/share",
            }
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if not any(marker in key.upper() for marker in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))
        }
        env.update(
            {
                "HOME": str(sandbox / "home"),
                "TMPDIR": str(sandbox / "tmp"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": "",
                "REFERENCE_PIPELINE_ALLOWED_READS": json.dumps(
                    [str(sandbox), str(raw), *allowed_runtime_roots]
                ),
                "REFERENCE_PIPELINE_ALLOWED_WRITES": json.dumps([str(output)]),
            }
        )
        python_command = [
            sys.executable,
            str(wrapper),
            str(sandbox_pipeline / "run.py"),
            "--raw-root",
            str(raw),
            "--output-dir",
            str(output),
            "--split-mode",
            split_mode,
        ]
        command = python_command
        sandbox_exec = Path("/usr/bin/sandbox-exec")
        if sys.platform == "darwin" and not sandbox_exec.is_file():
            raise RuntimeError("macOS sandbox-exec is required for isolated pipeline replay")
        if sys.platform == "darwin":
            profile = sandbox / "pipeline.sb"
            profile.write_text(
                _macos_sandbox_profile(
                    executable=Path(sys.executable).resolve(),
                    read_roots=[sandbox, raw, *(Path(item) for item in allowed_runtime_roots)],
                    write_roots=[output, sandbox / "home", sandbox / "tmp", Path("/dev/null")],
                ),
                encoding="utf-8",
            )
            command = [str(sandbox_exec), "-f", str(profile), *python_command]
        stdout_path = sandbox / "stdout.log"
        stderr_path = sandbox / "stderr.log"
        timed_out = False
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=str(sandbox),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                preexec_fn=lambda: _apply_pipeline_resource_limits(timeout_seconds),
            )
            returncode, timed_out, resource_limit_reason = _wait_for_pipeline_process(
                process,
                output=output,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                max_output_files=max_output_files,
                max_rss_bytes=max_rss_bytes,
            )
        stdout = _read_bounded_log(stdout_path)
        stderr = _read_bounded_log(stderr_path)
        log.write_text(
            "$ " + " ".join(command) + "\n\n[stdout]\n" + stdout
            + "\n[stderr]\n" + stderr,
            encoding="utf-8",
        )
    return {
        "schema_version": 1,
        "status": (
            "RESOURCE_LIMIT"
            if resource_limit_reason
            else ("TIMEOUT" if timed_out else ("SUCCESS" if returncode == 0 else "FAILED"))
        ),
        "returncode": 125 if resource_limit_reason else (124 if timed_out else returncode),
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "resource_limit_reason": resource_limit_reason,
        "split_mode": split_mode,
        "pipeline_sha256": pipeline_directory_sha256(pipeline),
        "raw_root": str(raw),
        "output_dir": str(output),
        "log_path": str(log),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def validate_and_replay_candidate_pipeline(
    *,
    pipeline_dir: str | Path,
    previous_pipeline_dir: str | Path | None,
    candidate_result_package: str | Path,
    train_raw: str | Path,
    train_reference_root: str | Path,
    train_keys: str | Path,
    validation_raw: str | Path,
    validation_keys: str | Path,
    key_column: str,
    report_root: str | Path,
    required_file_count: int | None = MIMIC_EXPECTED_BUSINESS_FILE_COUNT,
    allowed_changed_modules: set[str] | None = None,
) -> dict[str, Any]:
    pipeline = Path(pipeline_dir).expanduser().resolve()
    reference = Path(train_reference_root).expanduser().resolve()
    report_dir = Path(report_root).expanduser().resolve()
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    expected_files = sorted(
        path.relative_to(reference).as_posix()
        for path in _structured_package_files(reference)
        if path.name not in IGNORED_PACKAGE_FILES
    )
    previous = (
        Path(previous_pipeline_dir).expanduser().resolve()
        if previous_pipeline_dir is not None
        else None
    )
    expected_parent = (
        pipeline_directory_sha256(previous)
        if previous is not None and previous.is_dir()
        else ""
    )
    structure = validate_pipeline_structure(
        pipeline,
        expected_files=expected_files,
        expected_parent_sha256=expected_parent,
        previous_pipeline_dir=previous,
        allowed_changed_modules=allowed_changed_modules,
        train_reference_root=reference,
    )
    issues = list(structure["issues"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "NEEDS_REPAIR",
        "valid": False,
        "pipeline_sha256": structure["pipeline_sha256"],
        "parent_pipeline_sha256": structure["parent_pipeline_sha256"],
        "structure_gate": structure,
        "executions": {},
        "train_regression": {},
        "business_gates": {},
        "hash_consistency": {
            "matched": False,
            "mismatched_files": [],
            "missing_replay_files": [],
            "extra_replay_files": [],
        },
        "issues": issues,
    }
    if not structure["valid"]:
        _write_json(report_dir / "pipeline_replay_report.json", report)
        return report

    train_output = report_dir / "train_result_package"
    validation_output = report_dir / "validation_result_package"
    train_run = _run_pipeline_for_gate(
        pipeline_dir=pipeline,
        raw_root=train_raw,
        output_dir=train_output,
        split_mode="train",
        log_path=report_dir / "train_replay.log",
    )
    validation_run = _run_pipeline_for_gate(
        pipeline_dir=pipeline,
        raw_root=validation_raw,
        output_dir=validation_output,
        split_mode="validation",
        log_path=report_dir / "validation_replay.log",
    )
    report["executions"] = {"train": train_run, "validation": validation_run}
    if train_run["returncode"] != 0:
        issues.append("isolated train replay failed")
    if validation_run["returncode"] != 0:
        issues.append("isolated validation replay failed")

    if train_run["returncode"] == 0:
        train_gate = validate_business_result_package(
            result_package=train_output,
            train_reference_root=reference,
            split_keys=train_keys,
            key_column=key_column,
            split_mode="train",
            required_file_count=required_file_count,
        )
        train_regression = _build_train_regression_report(
            result_package=train_output,
            train_reference_root=reference,
            key_column=key_column,
        )
        report["business_gates"]["train"] = train_gate
        report["train_regression"] = train_regression
        if not train_gate["valid"]:
            issues.extend(f"train replay gate: {item}" for item in train_gate["issues"])
        if train_regression["status"] != "SUCCESS":
            issues.append("host train regression did not exactly match train reference")

    if validation_run["returncode"] == 0:
        validation_gate = validate_business_result_package(
            result_package=validation_output,
            train_reference_root=reference,
            split_keys=validation_keys,
            key_column=key_column,
            split_mode="validation",
            required_file_count=required_file_count,
        )
        report["business_gates"]["validation"] = validation_gate
        if not validation_gate["valid"]:
            issues.extend(
                f"validation replay gate: {item}" for item in validation_gate["issues"]
            )

        candidate_hashes = business_file_hashes(candidate_result_package)
        replay_hashes = business_file_hashes(validation_output)
        expected_set = set(expected_files)
        mismatched = sorted(
            relative
            for relative in expected_set & set(candidate_hashes) & set(replay_hashes)
            if candidate_hashes[relative] != replay_hashes[relative]
        )
        missing = sorted(expected_set - set(replay_hashes))
        extra = sorted(set(replay_hashes) - expected_set)
        candidate_missing = sorted(expected_set - set(candidate_hashes))
        if candidate_missing:
            issues.append("candidate package is incomplete for hash replay: " + ", ".join(candidate_missing[:20]))
        if mismatched or missing or extra:
            issues.append("pipeline replay package does not match candidate business files")
        report["hash_consistency"] = {
            "matched": not (mismatched or missing or extra or candidate_missing),
            "mismatched_files": mismatched,
            "missing_replay_files": missing,
            "extra_replay_files": extra,
            "missing_candidate_files": candidate_missing,
            "candidate_hashes": candidate_hashes,
            "replay_hashes": replay_hashes,
        }

    report["issues"] = issues
    report["valid"] = not issues
    report["status"] = "SUCCESS" if not issues else "NEEDS_REPAIR"
    _write_json(report_dir / "pipeline_replay_report.json", report)
    if report.get("train_regression"):
        _write_json(report_dir / "train_regression_report.json", report["train_regression"])
    return report


def _run_pipeline_for_gate(**kwargs: Any) -> dict[str, Any]:
    try:
        return run_pipeline_isolated(**kwargs)
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "FAILED",
            "returncode": 1,
            "timed_out": False,
            "split_mode": str(kwargs.get("split_mode") or ""),
            "pipeline_sha256": pipeline_directory_sha256(kwargs["pipeline_dir"]),
            "raw_root": str(Path(kwargs["raw_root"]).expanduser().resolve()),
            "output_dir": str(Path(kwargs["output_dir"]).expanduser().resolve()),
            "log_path": str(Path(kwargs["log_path"]).expanduser().resolve()),
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }


def _build_train_regression_report(
    *,
    result_package: Path,
    train_reference_root: Path,
    key_column: str,
) -> dict[str, Any]:
    score = score_reference_directory(
        result_package,
        train_reference_root,
        key_column=key_column,
    )
    files: list[dict[str, Any]] = []
    for item in score.get("file_reports") or []:
        metrics = item.get("metrics") or {}
        reference_rows = int(item.get("reference_rows") or 0)
        result_rows = int(item.get("result_rows") or 0)
        per_column = item.get("per_column") or {}
        passed = (
            item.get("status") == "compared"
            and item.get("reference_columns") == item.get("result_columns")
            and reference_rows == result_rows
            and int(item.get("matched_key_rows") or 0) == reference_rows
            and int(item.get("exact_row_matches") or 0) == reference_rows
            and all(
                int(column_report.get("matched_cells") or 0) == reference_rows
                for column_report in per_column.values()
            )
            and len(per_column) == len(item.get("reference_columns") or [])
        )
        files.append(
            {
                "relative_path": str(item.get("relative_path") or ""),
                "train_rows": result_rows,
                "reference_rows": reference_rows,
                "column_coverage": float(metrics.get("schema_f1") or 0.0),
                "key_coverage": float(metrics.get("key_structure_f1") or 0.0),
                "value_recall": float(metrics.get("row_aligned_cell_f1") or 0.0),
                "passed": passed,
                "failure_reason": "" if passed else "host replay differs from train reference",
            }
        )
    success = bool(files) and all(item["passed"] for item in files)
    return {
        "schema_version": 2,
        "status": "SUCCESS" if success else "NEEDS_REPAIR",
        "written_by": "host",
        "files": files,
    }


def _relative_file_hashes(root: Path | None) -> dict[str, str]:
    if root is None or not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _hash_manifest(file_hashes: dict[str, str]) -> str:
    payload = json.dumps(file_hashes, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_learned_summary_assets(
    *,
    pipeline: Path,
    manifest: dict[str, Any],
    expected_files: list[str],
    train_reference_root: str | Path | None,
) -> tuple[set[str], list[str]]:
    raw_assets = manifest.get("learned_summary_assets", []) if manifest else []
    if raw_assets is None:
        raw_assets = []
    if not isinstance(raw_assets, list):
        return set(), ["pipeline manifest learned_summary_assets must be a list"]

    reference = (
        Path(train_reference_root).expanduser().resolve()
        if train_reference_root is not None
        else None
    )
    expected = set(expected_files)
    allowed_paths: set[str] = set()
    seen_sources: set[str] = set()
    issues: list[str] = []
    for index, raw_asset in enumerate(raw_assets):
        prefix = f"learned_summary_assets[{index}]"
        if not isinstance(raw_asset, dict):
            issues.append(f"{prefix} must be an object")
            continue
        source = str(raw_asset.get("source") or "").strip().lstrip("/")
        relative = str(raw_asset.get("path") or "").strip().lstrip("/")
        declared_hash = str(raw_asset.get("sha256") or "").strip().casefold()
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 2
            or relative_path.parts[0] != SUMMARY_ASSET_ROOT
            or relative_path.suffix.casefold() != ".csv"
        ):
            issues.append(f"{prefix} path must be summary_assets/<name>.csv")
            continue
        allowed_paths.add(relative)
        if source in seen_sources:
            issues.append(f"{prefix} duplicates summary source {source}")
        seen_sources.add(source)
        if not source.startswith("summary/") or source not in expected:
            issues.append(f"{prefix} source must name an expected summary business file")
        if Path(source).name != relative_path.name:
            issues.append(f"{prefix} path basename must match its summary source")

        asset_path = pipeline / relative
        if not asset_path.is_file():
            issues.append(f"{prefix} asset file is missing: {relative}")
            continue
        actual_hash = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        if declared_hash != actual_hash:
            issues.append(f"{prefix} sha256 does not match {relative}")

        try:
            with asset_path.open("r", encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle), [])
        except (OSError, UnicodeError, csv.Error) as exc:
            issues.append(f"{prefix} is not a readable CSV: {type(exc).__name__}: {exc}")
            columns = []
        forbidden_columns = sorted(
            {column.strip().casefold() for column in columns}
            & FORBIDDEN_SUMMARY_ASSET_COLUMNS
        )
        if forbidden_columns:
            issues.append(
                f"{prefix} contains patient-level columns: {', '.join(forbidden_columns)}"
            )

        if reference is None:
            issues.append(f"{prefix} requires train_reference_root for provenance verification")
            continue
        source_path = (reference / source).resolve()
        if reference not in source_path.parents or not source_path.is_file():
            issues.append(f"{prefix} train/reference summary source is missing")
            continue
        reference_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != reference_hash:
            issues.append(f"{prefix} does not match its train/reference summary source")

    return allowed_paths, issues


def _embedded_literal_issues(pipeline: Path, actual_files: list[str]) -> list[str]:
    issues: list[str] = []
    for relative in actual_files:
        path = pipeline / relative
        suffix = path.suffix.casefold()
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".py":
            try:
                tree = ast.parse(text, filename=relative)
            except SyntaxError:
                continue
            total_literal_bytes = 0
            total_collection_items = 0
            file_issue = ""
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
                    size = len(node.value.encode("utf-8")) if isinstance(node.value, str) else len(node.value)
                    total_literal_bytes += size
                    if size > MAX_PIPELINE_LITERAL_BYTES:
                        file_issue = f"{relative}: literal has {size} bytes"
                        break
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
                    total_literal_bytes += len(repr(node.value).encode("utf-8"))
                if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                    total_collection_items += len(node.elts)
                    if len(node.elts) > MAX_PIPELINE_COLLECTION_ITEMS:
                        file_issue = f"{relative}: collection has {len(node.elts)} literal items"
                        break
                if isinstance(node, ast.Dict):
                    total_collection_items += len(node.keys)
                    if len(node.keys) > MAX_PIPELINE_COLLECTION_ITEMS:
                        file_issue = f"{relative}: mapping has {len(node.keys)} literal items"
                        break
            if not file_issue and total_literal_bytes > MAX_PIPELINE_TOTAL_LITERAL_BYTES:
                file_issue = (
                    f"{relative}: total literal payload has {total_literal_bytes} bytes"
                )
            if not file_issue and total_collection_items > MAX_PIPELINE_COLLECTION_ITEMS * 4:
                file_issue = (
                    f"{relative}: total literal collections contain {total_collection_items} items"
                )
            if file_issue:
                issues.append(file_issue)
        elif suffix in {".yaml", ".yml"}:
            longest = max((len(line.encode("utf-8")) for line in text.splitlines()), default=0)
            if longest > MAX_PIPELINE_LITERAL_BYTES:
                issues.append(f"{relative}: scalar line has {longest} bytes")
    return issues


def _wait_for_pipeline_process(
    process: subprocess.Popen[bytes],
    *,
    output: Path,
    timeout_seconds: int,
    max_output_bytes: int,
    max_output_files: int,
    max_rss_bytes: int,
) -> tuple[int, bool, str]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        resource_issue = _pipeline_resource_issue(
            process.pid,
            output=output,
            max_output_bytes=max_output_bytes,
            max_output_files=max_output_files,
            max_rss_bytes=max_rss_bytes,
        )
        returncode = process.poll()
        if resource_issue:
            if returncode is None:
                _kill_pipeline_process(process)
            return process.wait(), False, resource_issue
        if returncode is not None:
            return returncode, False, ""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_pipeline_process(process)
            return process.wait(), True, ""
        try:
            process.wait(timeout=min(2.0, remaining))
        except subprocess.TimeoutExpired:
            pass


def _pipeline_resource_issue(
    pid: int,
    *,
    output: Path,
    max_output_bytes: int,
    max_output_files: int,
    max_rss_bytes: int,
) -> str:
    file_count = 0
    total_bytes = 0
    if output.exists():
        for path in output.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            if file_count > max_output_files:
                return f"output file count exceeded {max_output_files}"
            try:
                total_bytes += path.stat().st_size
            except OSError:
                continue
            if total_bytes > max_output_bytes:
                return f"output size exceeded {max_output_bytes} bytes"
    rss = _process_rss_bytes(pid)
    if rss is not None and rss > max_rss_bytes:
        return f"process RSS exceeded {max_rss_bytes} bytes"
    return ""


def _process_rss_bytes(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        text = result.stdout.strip()
        return int(text) * 1024 if text else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _kill_pipeline_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, 9)
    except ProcessLookupError:
        process.kill()


def _apply_pipeline_resource_limits(timeout_seconds: int) -> None:
    cpu_limit = max(1, int(timeout_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 5))
    if sys.platform != "darwin":
        resource.setrlimit(
            resource.RLIMIT_AS,
            (MAX_PIPELINE_ADDRESS_SPACE_BYTES, MAX_PIPELINE_ADDRESS_SPACE_BYTES),
        )
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (MAX_PIPELINE_OUTPUT_FILE_BYTES, MAX_PIPELINE_OUTPUT_FILE_BYTES),
    )
    soft_nofile, hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    nofile = min(512, hard_nofile if hard_nofile != resource.RLIM_INFINITY else 512)
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))


def _read_bounded_log(path: Path) -> str:
    size = path.stat().st_size if path.is_file() else 0
    with path.open("rb") as handle:
        payload = handle.read(MAX_PIPELINE_LOG_BYTES)
    text = payload.decode("utf-8", errors="replace")
    if size > MAX_PIPELINE_LOG_BYTES:
        text += f"\n[truncated {size - MAX_PIPELINE_LOG_BYTES} bytes]\n"
    return text


def _macos_sandbox_profile(
    *,
    executable: Path,
    read_roots: list[Path],
    write_roots: list[Path],
) -> str:
    def literal(path: Path) -> str:
        return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        "(allow process*)",
        "(deny network*)",
        "(deny process-fork)",
        '(allow file-read-metadata (literal "/"))',
        f'(allow process-exec (literal "{literal(executable)}"))',
    ]
    lines.extend(
        f'(allow file-read* (literal "{literal(root)}") (subpath "{literal(root)}"))'
        for root in [*read_roots, *write_roots]
    )
    lines.extend(
        f'(allow file-read-metadata file-test-existence (path-ancestors "{literal(root)}"))'
        for root in [*read_roots, *write_roots]
    )
    lines.extend(
        f'(allow file-write* (literal "{literal(root)}") (subpath "{literal(root)}"))'
        for root in write_roots
    )
    return "\n".join(lines) + "\n"


_ISOLATED_RUNNER = r'''\
import json
import os
import runpy
import sys
from pathlib import Path

READ_ROOTS = [Path(item).resolve() for item in json.loads(os.environ["REFERENCE_PIPELINE_ALLOWED_READS"])]
WRITE_ROOTS = [Path(item).resolve() for item in json.loads(os.environ["REFERENCE_PIPELINE_ALLOWED_WRITES"])]


def within(path, roots):
    try:
        resolved = Path(path).resolve()
    except (TypeError, ValueError, OSError):
        return True
    return any(resolved == root or root in resolved.parents for root in roots)


def audit(event, args):
    if event in {
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "socket.connect",
        "socket.bind",
    }:
        raise PermissionError(f"isolated pipeline blocked event: {event}")
    if event == "ctypes.dlopen" and args and args[0] is not None:
        if not within(args[0], READ_ROOTS):
            raise PermissionError(f"isolated pipeline blocked native library: {args[0]}")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1] if len(args) > 1 else "r"
        writing = (
            isinstance(mode, str) and any(flag in mode for flag in "wax+")
        ) or (
            isinstance(mode, int) and bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        )
        roots = WRITE_ROOTS if writing else READ_ROOTS + WRITE_ROOTS
        if not within(args[0], roots):
            raise PermissionError(f"isolated pipeline denied {'write' if writing else 'read'}: {args[0]}")
    if event in {"os.listdir", "os.scandir"} and args and args[0] is not None:
        if not within(args[0], READ_ROOTS + WRITE_ROOTS):
            raise PermissionError(f"isolated pipeline denied directory read: {args[0]}")
    if event in {
        "os.remove",
        "os.rmdir",
        "os.mkdir",
        "os.rename",
        "os.link",
        "os.symlink",
    } and args:
        paths = [item for item in args[:2] if isinstance(item, (str, bytes, os.PathLike))]
        if event == "os.mkdir" and paths and Path(paths[0]).is_dir():
            return
        if any(not within(path, WRITE_ROOTS) for path in paths):
            raise PermissionError(f"isolated pipeline denied filesystem mutation: {paths}")


sys.addaudithook(audit)
entrypoint = Path(sys.argv[1]).resolve()
sys.argv = [str(entrypoint), *sys.argv[2:]]
sys.path.insert(0, str(entrypoint.parent))
runpy.run_path(str(entrypoint), run_name="__main__")
'''
