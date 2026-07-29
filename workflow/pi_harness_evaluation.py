from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from workflow.reference_evaluation import (
    IGNORED_PACKAGE_FILES,
    _compare_file,
    _missing_file_report,
    _rounded,
    _structured_package_files,
    _table_columns,
)


SCORER_VERSION = "equal_weight_f1_v1"


def load_evaluation_manifest(
    manifest_path: str | Path,
    reference_root: str | Path,
) -> dict[str, tuple[str, ...]]:
    path = Path(manifest_path).expanduser().resolve()
    gold_root = Path(reference_root).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("evaluation manifest must have schema_version=1 and a files object")

    gold_files = [
        item
        for item in _structured_package_files(gold_root)
        if item.name not in IGNORED_PACKAGE_FILES
    ]
    configured = payload["files"]
    result: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    for gold_file in gold_files:
        relative = gold_file.relative_to(gold_root).as_posix()
        entry = configured.get(relative)
        if not isinstance(entry, dict):
            missing.append(relative)
            continue
        keys = entry.get("key_columns")
        if not isinstance(keys, list) or not keys or not all(isinstance(item, str) and item for item in keys):
            raise ValueError(f"invalid key_columns for {relative}")
        columns = _table_columns(gold_file)
        unknown = [column for column in keys if column not in columns]
        if unknown:
            raise ValueError(
                f"unknown key columns for {relative}: {', '.join(unknown)}",
            )
        result[relative] = tuple(keys)
    if missing:
        raise ValueError("missing key configuration: " + ", ".join(sorted(missing)))

    unknown_entries = sorted(set(configured) - set(result))
    if unknown_entries:
        raise ValueError("manifest references unknown gold files: " + ", ".join(unknown_entries))
    return result


def score_equal_weight_reference_directory(
    result_package: str | Path,
    reference_root: str | Path,
    key_columns_by_file: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    package_input = Path(result_package).expanduser().absolute()
    if package_input.exists():
        _validate_result_tree(package_input)
    package_root = package_input.resolve()
    gold_root = Path(reference_root).expanduser().resolve()
    gold_files = [
        item
        for item in _structured_package_files(gold_root)
        if item.name not in IGNORED_PACKAGE_FILES
    ]
    if not gold_files:
        raise ValueError("reference directory contains no comparable structured files")

    result_files = [
        item
        for item in _structured_package_files(package_root)
        if item.name not in IGNORED_PACKAGE_FILES
    ] if package_root.is_dir() else []
    by_name: dict[str, list[Path]] = {}
    for item in result_files:
        by_name.setdefault(item.name, []).append(item)

    weight = 1.0 / len(gold_files)
    file_reports: list[dict[str, Any]] = []
    metric_names = (
        "schema_f1",
        "key_structure_f1",
        "row_aligned_cell_f1",
        "exact_row_f1",
    )
    metric_totals = {name: 0.0 for name in metric_names}
    score_total = 0.0
    matched_files = 0
    used_result_files: set[Path] = set()
    matched_by_gold: dict[Path, tuple[Path, str]] = {}

    # Reserve exact relative-path matches first so a basename fallback cannot
    # consume a file that belongs to another Gold path.
    for gold_file in gold_files:
        relative = gold_file.relative_to(gold_root).as_posix()
        exact_path = package_root / relative
        if exact_path.is_file():
            matched_by_gold[gold_file] = (exact_path, "relative_path")
            used_result_files.add(exact_path.resolve())
    for gold_file in gold_files:
        if gold_file in matched_by_gold:
            continue
        candidates = [
            item
            for item in by_name.get(gold_file.name, [])
            if item.resolve() not in used_result_files
        ]
        if len(candidates) == 1:
            matched_by_gold[gold_file] = (candidates[0], "basename")
            used_result_files.add(candidates[0].resolve())

    for gold_file in gold_files:
        relative = gold_file.relative_to(gold_root).as_posix()
        key_columns = key_columns_by_file.get(relative)
        if not key_columns:
            raise ValueError(f"missing key configuration: {relative}")
        exact_path = package_root / relative
        matched = matched_by_gold.get(gold_file)
        result_file = matched[0] if matched else None
        match_mode = matched[1] if matched else ""
        candidates = [
            item
            for item in by_name.get(gold_file.name, [])
            if item.resolve() not in used_result_files
        ]

        if result_file is None:
            report = _missing_file_report(
                relative,
                gold_file,
                exact_path,
                "equal_weight",
                weight,
                key_columns,
            )
            report["match_mode"] = "ambiguous" if len(candidates) > 1 else "missing"
            if len(candidates) > 1:
                report["status"] = "ambiguous_file"
                report["candidate_count"] = len(candidates)
        else:
            try:
                report = _compare_file(
                    reference_file=gold_file,
                    result_file=result_file,
                    relative_path=relative,
                    category="equal_weight",
                    weight=weight,
                    fallback_key_column="",
                    key_columns_override=key_columns,
                )
                report["match_mode"] = match_mode
                matched_files += 1
            except Exception as exc:
                report = _missing_file_report(
                    relative,
                    gold_file,
                    result_file,
                    "equal_weight",
                    weight,
                    key_columns,
                )
                report.update(
                    {
                        "status": "read_error",
                        "match_mode": match_mode,
                        "issue": f"{type(exc).__name__}: {exc}",
                    },
                )

        raw_metrics = report.pop("_raw_metrics", report["metrics"])
        score_total += weight * float(raw_metrics["file_score"])
        for name in metric_names:
            metric_totals[name] += weight * float(raw_metrics[name])
        file_reports.append(report)

    extras = sorted(
        item.relative_to(package_root).as_posix()
        for item in result_files
        if item.resolve() not in used_result_files
    )
    return {
        "schema_version": 1,
        "scorer_version": SCORER_VERSION,
        "status": "SUCCESS",
        "mode": "equal_weight_reference_directory",
        "metrics": {
            **{name: _rounded(value) for name, value in metric_totals.items()},
            "file_coverage": _rounded(matched_files / len(gold_files)),
            "composite_score": _rounded(score_total),
        },
        "weighting": {
            "per_file": _rounded(weight),
            "within_file": {
                "schema_f1": 0.10,
                "key_structure_f1": 0.20,
                "row_aligned_cell_f1": 0.40,
                "exact_row_f1": 0.30,
            },
        },
        "detail_metrics": {
            "reference_file_count": len(gold_files),
            "matched_file_count": matched_files,
        },
        "file_reports": file_reports,
        "extra_result_files": extras,
    }


def _validate_result_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("result package must be a real directory")
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(mode):
            raise ValueError(f"result package contains a symbolic link: {relative}")
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"result package contains a special file: {relative}")
        resolved = path.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValueError(f"result package path escapes its root: {relative}")


def public_feedback_from_equal_weight_report(
    report: dict[str, Any],
    *,
    round_index: int,
    best_score: float,
    promoted: bool,
    restored_best: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "round": int(round_index),
        "current_score": float((report.get("metrics") or {}).get("composite_score") or 0.0),
        "best_score": float(best_score),
        "promoted": bool(promoted),
        "restored_best": bool(restored_best),
        "files": [
            {
                "filename": Path(str(item.get("relative_path") or "")).name,
                "file_score": float((item.get("metrics") or {}).get("file_score") or 0.0),
                "gold_rows": int(item.get("reference_rows") or 0),
                "result_rows": int(item.get("result_rows") or 0),
                "status": str(item.get("status") or "unknown"),
            }
            for item in report.get("file_reports") or []
        ],
    }
