from __future__ import annotations

import difflib
import fnmatch
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from lib.agent_artifacts import begin_step, record_step
from workflow.skill_adapter import parse_variant_result, write_variant_execution_receipt

from .context import EngineerToolContext


def _response(
    status: str,
    summary: str,
    artifacts: dict[str, Any] | None = None,
    issues: Iterable[str] = (),
) -> ToolResponse:
    payload = {
        "status": status,
        "summary": summary,
        "artifacts": artifacts or {},
        "issues": list(issues),
    }
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, indent=2),
            ),
        ],
    )


def _truncate(text: str, limit: int = 8_000) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = limit // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:], True


def _diff(before: str, after: str, path: Path) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        ),
    )


def _load_structured_frame(path: Path):
    import pandas as pd

    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False), "csv"
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", low_memory=False), "tsv"
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path), "excel"
    if name.endswith(".parquet"):
        return pd.read_parquet(path), "parquet"
    if name.endswith(".jsonl"):
        return pd.read_json(path, lines=True), "jsonl"
    if name.endswith(".json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return pd.DataFrame(value), "json"
        if isinstance(value, dict):
            rows = value.get("rows") if isinstance(value.get("rows"), list) else [value]
            return pd.DataFrame(rows), "json"
    raise ValueError(
        f"Unsupported structured file format for {path.name}; "
        "supported: csv/csv.gz, tsv/tsv.gz, xlsx/xls, parquet, jsonl, json"
    )


def _load_json_string_list(value: str) -> list[str]:
    """Parse a JSON list of strings, tolerating common tool-call tag leakage."""
    text = str(value or "[]").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.match(r"^\s*(\[[\s\S]*?\])", text)
        if not match:
            raise
        parsed = json.loads(match.group(1))
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("key_columns_json must be a JSON list of strings")
    return parsed


def _tool_payload(response: ToolResponse) -> dict[str, Any]:
    try:
        block = response.content[0]
        text = block["text"] if isinstance(block, dict) else getattr(block, "text")
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_phase_artifact_paths(value: Any, phase_root: Path) -> list[Path]:
    paths: list[Path] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)
            return
        if not isinstance(item, str) or "/" not in item:
            return
        try:
            candidate = Path(item).expanduser()
            if not candidate.is_absolute():
                return
            resolved = candidate.resolve()
        except Exception:
            return
        if not resolved.exists():
            return
        if resolved == phase_root or phase_root in resolved.parents:
            paths.append(resolved)

    visit(value)
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def _inspect_structured_file(path: Path, *, sample_rows: int, max_columns: int) -> dict[str, Any]:
    import pandas as pd

    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        sep = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
        header = pd.read_csv(path, sep=sep, nrows=0)
        all_columns = [str(column) for column in header.columns]
        selected_columns = all_columns[:max_columns]
        sample = pd.read_csv(path, sep=sep, nrows=sample_rows, usecols=selected_columns, dtype=object)
        row_count = 0
        missing_counts = {column: 0 for column in selected_columns}
        unique_values = {column: set() for column in selected_columns}
        unique_capped = {column: False for column in selected_columns}
        dtypes = {column: "object" for column in selected_columns}
        for chunk in pd.read_csv(
            path,
            sep=sep,
            usecols=selected_columns,
            dtype=object,
            chunksize=100_000,
            low_memory=False,
        ):
            row_count += int(len(chunk))
            for column in selected_columns:
                series = chunk[column]
                missing_counts[column] += int(series.isna().sum())
                dtypes[column] = str(series.dtype)
                if unique_capped[column]:
                    continue
                values = {
                    str(value)
                    for value in series.dropna().head(2_000).tolist()
                    if str(value) != ""
                }
                unique_values[column].update(values)
                if len(unique_values[column]) > 2_000:
                    unique_capped[column] = True
                    unique_values[column] = set(list(unique_values[column])[:2_000])
        columns = [
            {
                "name": column,
                "dtype": dtypes[column],
                "missing_count": int(missing_counts[column]),
                "missing_rate": round(float(missing_counts[column] / row_count), 6) if row_count else 0.0,
                "unique_count": int(len(unique_values[column])),
                "unique_count_capped": bool(unique_capped[column]),
            }
            for column in selected_columns
        ]
        return {
            "format": "tsv" if sep == "\t" else "csv",
            "row_count": int(row_count),
            "column_count": int(len(all_columns)),
            "columns": columns,
            "columns_truncated": len(selected_columns) < len(all_columns),
            "sample_frame": sample,
        }

    frame, format_name = _load_structured_frame(path)
    selected_names = list(frame.columns[:max_columns])
    columns = []
    for name in selected_names:
        series = frame[name]
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": round(float(series.isna().mean()), 6) if len(series) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    return {
        "format": format_name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
        "columns_truncated": len(selected_names) < len(frame.columns),
        "sample_frame": frame[selected_names].head(sample_rows),
    }


def _normalizable_artifacts(root: Path) -> list[Path]:
    suffixes = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".json", ".jsonl", ".parquet")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name.lower().endswith(suffixes))


def _compare_artifact_directories(
    expected: Path,
    actual: Path,
    *,
    key_columns: list[str],
    numeric_tolerance: float,
    ignore_row_order: bool,
    normalize_empty: bool,
    limit: int,
) -> dict[str, Any]:
    expected_files = {str(path.relative_to(expected)): path for path in _normalizable_artifacts(expected)}
    actual_files = {str(path.relative_to(actual)): path for path in _normalizable_artifacts(actual)}
    missing = sorted(set(expected_files) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected_files))
    compared: list[dict[str, Any]] = []
    issues: list[str] = []
    for relative in sorted(set(expected_files) & set(actual_files)):
        result = _compare_artifact_files(
            expected_files[relative],
            actual_files[relative],
            key_columns=key_columns,
            numeric_tolerance=numeric_tolerance,
            ignore_row_order=ignore_row_order,
            normalize_empty=normalize_empty,
            limit=limit,
        )
        compared.append({"relative_path": relative, **result})
        if not result["passed"]:
            issues.append(f"{relative}: " + "; ".join(result.get("issues") or ["content differs"]))
    if missing:
        issues.append("Missing files: " + ", ".join(missing[:limit]))
    if extra:
        issues.append("Extra files: " + ", ".join(extra[:limit]))
    return {
        "mode": "directory",
        "passed": not issues,
        "expected_file_count": len(expected_files),
        "actual_file_count": len(actual_files),
        "missing_files": missing[:limit],
        "extra_files": extra[:limit],
        "compared_files": compared[:limit],
        "issues": issues[:limit],
    }


def _compare_artifact_files(
    expected: Path,
    actual: Path,
    *,
    key_columns: list[str],
    numeric_tolerance: float,
    ignore_row_order: bool,
    normalize_empty: bool,
    limit: int,
) -> dict[str, Any]:
    import pandas as pd

    left, left_format = _load_structured_frame(expected)
    right, right_format = _load_structured_frame(actual)
    issues: list[str] = []
    left_columns = [str(column) for column in left.columns]
    right_columns = [str(column) for column in right.columns]
    missing_columns = sorted(set(left_columns) - set(right_columns))
    extra_columns = sorted(set(right_columns) - set(left_columns))
    common_columns = [column for column in left_columns if column in right_columns]
    if missing_columns:
        issues.append("Missing columns: " + ", ".join(missing_columns[:limit]))
    if extra_columns:
        issues.append("Extra columns: " + ", ".join(extra_columns[:limit]))
    selected_keys = [column for column in key_columns if column in common_columns]
    absent_keys = [column for column in key_columns if column not in common_columns]
    if absent_keys:
        issues.append("Key columns absent from one or both files: " + ", ".join(absent_keys))

    left_norm = _normalize_compare_frame(left[common_columns].copy(), normalize_empty=normalize_empty)
    right_norm = _normalize_compare_frame(right[common_columns].copy(), normalize_empty=normalize_empty)
    if ignore_row_order and common_columns:
        sort_columns = selected_keys or common_columns
        left_norm = left_norm.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
        right_norm = right_norm.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    else:
        left_norm = left_norm.reset_index(drop=True)
        right_norm = right_norm.reset_index(drop=True)

    row_count_equal = len(left_norm) == len(right_norm)
    if not row_count_equal:
        issues.append(f"Row count differs: expected={len(left_norm)}, actual={len(right_norm)}")
    compared_rows = min(len(left_norm), len(right_norm))
    mismatch_count = 0
    mismatch_examples: list[dict[str, Any]] = []
    if common_columns and compared_rows:
        for row_index in range(compared_rows):
            row_mismatches: dict[str, dict[str, Any]] = {}
            for column in common_columns:
                left_value = left_norm.iloc[row_index][column]
                right_value = right_norm.iloc[row_index][column]
                if _values_equal(left_value, right_value, numeric_tolerance):
                    continue
                row_mismatches[column] = {
                    "expected": _json_scalar(left_value),
                    "actual": _json_scalar(right_value),
                }
            if row_mismatches:
                mismatch_count += 1
                if len(mismatch_examples) < limit:
                    mismatch_examples.append({"row_index": row_index, "columns": row_mismatches})
    if mismatch_count:
        issues.append(f"Content differs in {mismatch_count} aligned rows")
    if selected_keys:
        left_keys = set(_key_tuples(left_norm, selected_keys))
        right_keys = set(_key_tuples(right_norm, selected_keys))
        missing_keys = sorted(left_keys - right_keys)[:limit]
        extra_keys = sorted(right_keys - left_keys)[:limit]
        if missing_keys:
            issues.append(f"Missing keys: {missing_keys}")
        if extra_keys:
            issues.append(f"Extra keys: {extra_keys}")
    total_cells = max(compared_rows * max(len(common_columns), 1), 1)
    matched_cells = total_cells - sum(len(example["columns"]) for example in mismatch_examples)
    content_match_rate = 1.0 if not mismatch_count else max(0.0, matched_cells / total_cells)
    return {
        "mode": "file",
        "passed": not issues,
        "expected_path": str(expected),
        "actual_path": str(actual),
        "expected_format": left_format,
        "actual_format": right_format,
        "expected_rows": int(len(left)),
        "actual_rows": int(len(right)),
        "expected_columns": left_columns,
        "actual_columns": right_columns,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "key_columns": selected_keys,
        "row_count_equal": row_count_equal,
        "mismatch_count": int(mismatch_count),
        "content_match_rate": round(float(content_match_rate), 6),
        "mismatch_examples": mismatch_examples,
        "issues": issues,
    }


def _normalize_compare_frame(frame, *, normalize_empty: bool):
    import pandas as pd

    normalized = frame.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].where(pd.notna(normalized[column]), "")
        if normalize_empty:
            normalized[column] = normalized[column].replace({"nan": "", "None": "", "NaT": ""})
    return normalized


def _values_equal(left: Any, right: Any, numeric_tolerance: float) -> bool:
    import pandas as pd

    if pd.isna(left) and pd.isna(right):
        return True
    left_text = "" if pd.isna(left) else str(left)
    right_text = "" if pd.isna(right) else str(right)
    if left_text == right_text:
        return True
    left_num = pd.to_numeric(left_text, errors="coerce")
    right_num = pd.to_numeric(right_text, errors="coerce")
    if pd.notna(left_num) and pd.notna(right_num):
        return abs(float(left_num) - float(right_num)) <= numeric_tolerance
    return False


def _key_tuples(frame, columns: list[str]) -> list[tuple[str, ...]]:
    return [tuple(str(row[column]) for column in columns) for _, row in frame[columns].iterrows()]


def _json_scalar(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    if value is None:
        return None
    return str(value)


def _validate_reference_result_package(
    root: Path,
    *,
    key_column: str,
    expected_key_count: int,
    allow_empty_feature_tables: bool,
) -> dict[str, Any]:
    import pandas as pd

    issues: list[str] = []
    manifest_path = root / "package_manifest.json"
    reference_path = root / "reference.csv"
    checked_files: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("package_manifest.json must contain an object")

    declared_files = _declared_package_files(manifest, root) if manifest else []
    if not declared_files:
        declared_files = _scanned_package_files(root)
    if not declared_files:
        issues.append("result_package contains no structured cohort or feature artifacts")

    inferred_key = key_column
    key_coverages: list[int] = []
    for item in declared_files:
        path = item["path"]
        if not path.is_file():
            issues.append(f"Declared artifact does not exist: {path}")
            continue
        try:
            frame, _format_name = _load_structured_frame(path)
        except Exception as exc:
            issues.append(f"Cannot read declared artifact {path}: {exc}")
            continue
        row_count = int(len(frame))
        column_count = int(len(frame.columns))
        if not inferred_key:
            inferred_key = _infer_key_column_from_columns(frame.columns)
        key_unique_count = 0
        duplicate_count = 0
        if inferred_key and inferred_key in frame.columns:
            missing_key_count = int(frame[inferred_key].isna().sum() + (frame[inferred_key].astype(str) == "").sum())
            duplicate_count = int(frame[inferred_key].astype(str).duplicated().sum())
            key_unique_count = int(frame[inferred_key].dropna().astype(str).nunique())
            if key_unique_count:
                key_coverages.append(key_unique_count)
            if missing_key_count:
                issues.append(f"{path} has {missing_key_count} missing keys")
            if duplicate_count and str(item.get("kind") or "").lower() in {"cohort", "reference", "primary_reference"}:
                issues.append(f"{path} has {duplicate_count} duplicate keys")
        checked_files.append(
            {
                "kind": item.get("kind", ""),
                "path": str(path),
                "row_count": row_count,
                "column_count": column_count,
                "key_unique_count": key_unique_count,
                "duplicate_key_count": duplicate_count,
            }
        )
        is_feature_artifact = str(item.get("kind") or "").lower() in {"feature", "features", "feature_files"}
        if row_count == 0 and (not is_feature_artifact or not allow_empty_feature_tables):
            issues.append(f"Declared artifact is empty: {path}")
        manifest_rows = item.get("row_count")
        if manifest_rows not in (None, "") and int(manifest_rows) != row_count:
            issues.append(f"Manifest row_count mismatch for {path}: declared={manifest_rows}, actual={row_count}")
    feature_files = [
        item for item in checked_files
        if str(item.get("kind") or "").lower() in {"feature", "features", "feature_files"}
    ]
    if not feature_files:
        issues.append("No feature files were validated")
    if expected_key_count and inferred_key:
        best_key_count = max(key_coverages) if key_coverages else 0
        if best_key_count < max(1, int(expected_key_count * 0.5)):
            issues.append(f"No table covers expected keys: expected={expected_key_count}, best={best_key_count}")
    elif not inferred_key:
        issues.append("Could not infer result package key column")
    return {
        "valid": not issues,
        "package_root": str(root),
        "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
        "reference_path": str(reference_path) if reference_path.is_file() else "",
        "key_column": inferred_key,
        "reference_rows": 0,
        "reference_columns": [],
        "declared_file_count": len(declared_files),
        "checked_files": checked_files,
        "issues": issues,
    }


def _declared_package_files(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for section in ("cohort_files", "feature_files", "artifacts"):
        for item in manifest.get(section) or []:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path") or "")
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = root / raw_path
            if not path.exists():
                path = root / Path(raw_path).name
            kind = str(item.get("kind") or "")
            if not kind:
                kind = "features" if section == "feature_files" else "cohort"
            records.append({**item, "kind": kind, "path": path.resolve()})
    unique: dict[str, dict[str, Any]] = {}
    for item in records:
        unique[str(item["path"])] = item
    return list(unique.values())


def _scanned_package_files(root: Path) -> list[dict[str, Any]]:
    suffixes = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".xlsx", ".xls", ".parquet", ".jsonl", ".json")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.lower().endswith(suffixes):
            continue
        if path.name in {"package_manifest.json", "result_manifest.json", "target_field_mapping.json"}:
            continue
        parts = {part.casefold() for part in path.parts}
        if "cohort" in parts:
            kind = "cohort"
        elif "features" in parts:
            kind = "features"
        elif path.name == "reference.csv":
            kind = "reference"
        else:
            kind = "artifact"
        records.append({"kind": kind, "path": path.resolve()})
    return records


def _infer_key_column_from_columns(columns: Iterable[Any]) -> str:
    names = {str(column) for column in columns}
    for candidate in ("hadm_id", "stay_id", "subject_id", "case_id", "row_id"):
        if candidate in names:
            return candidate
    return ""


class EngineerTools:
    """Clawd-style atomic tools bound to one Engineer phase."""

    def __init__(self, context: EngineerToolContext) -> None:
        self.context = context

    def Read(self, file_path: str, offset: int = 1, limit: int = 400) -> ToolResponse:
        """Read a UTF-8 text file with one-based line pagination."""
        try:
            if offset < 1 or limit < 1 or limit > 2_000:
                raise ValueError("offset must be >= 1 and limit must be between 1 and 2000")
            path = self.context.resolve_read_path(file_path)
            if not path.exists() or not path.is_file():
                raise ValueError(f"File does not exist: {path}")
            raw = path.read_bytes()
            if b"\x00" in raw[:8192]:
                raise ValueError(f"Binary files are not supported by Read: {path}")
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            selected = lines[offset - 1 : offset - 1 + limit]
            content = "\n".join(
                f"{line_no}\t{line}"
                for line_no, line in enumerate(selected, start=offset)
            )
            content, content_truncated = _truncate(content, limit=12_000)
            self.context.mark_read(path)
            return _response(
                "SUCCESS",
                f"Read {len(selected)} lines from {path.name}.",
                {
                    "file_path": str(path),
                    "content": content,
                    "start_line": offset,
                    "num_lines": len(selected),
                    "total_lines": len(lines),
                    "truncated": content_truncated or offset - 1 + len(selected) < len(lines),
                    "content_truncated": content_truncated,
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Read failed.", issues=[str(exc)])

    def Glob(
        self,
        pattern: str,
        path: str = "",
        limit: int = 200,
    ) -> ToolResponse:
        """Find files under an authorized directory; directories are omitted. Use **/* recursively."""
        try:
            if not pattern or limit < 1 or limit > 10_000:
                raise ValueError("pattern is required and limit must be between 1 and 10000")
            base = self.context.resolve_read_path(path or str(self.context.workspace_dir))
            if not base.exists() or not base.is_dir():
                raise ValueError(f"Glob path is not a directory: {base}")
            all_files: list[Path] = []
            for candidate in base.glob(pattern):
                if not candidate.is_file():
                    continue
                try:
                    all_files.append(self.context.resolve_read_path(candidate))
                except Exception:
                    continue
            all_files.sort(key=lambda candidate: (-candidate.stat().st_mtime_ns, str(candidate)))
            selected = all_files[:limit]
            return _response(
                "SUCCESS",
                f"Glob matched {len(all_files)} files.",
                {
                    "filenames": [str(candidate) for candidate in selected],
                    "num_files": len(selected),
                    "total_matches": len(all_files),
                    "truncated": len(all_files) > limit,
                    "directories_omitted": True,
                    "recursive_hint": "Use pattern **/* to discover files at every nested depth.",
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Glob failed.", issues=[str(exc)])

    def Grep(
        self,
        pattern: str,
        path: str = "",
        glob: str = "",
        output_mode: str = "files_with_matches",
        ignore_case: bool = False,
        limit: int = 200,
    ) -> ToolResponse:
        """Search authorized text files using a regular expression."""
        try:
            if output_mode not in {"content", "files_with_matches", "count"}:
                raise ValueError("output_mode must be content, files_with_matches, or count")
            if limit < 1 or limit > 10_000:
                raise ValueError("limit must be between 1 and 10000")
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
            base = self.context.resolve_read_path(path or str(self.context.workspace_dir))
            if not base.exists():
                raise ValueError(f"Grep path does not exist: {base}")
            candidates = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
            if glob:
                candidates = [
                    candidate
                    for candidate in candidates
                    if fnmatch.fnmatch(candidate.name, glob)
                    or fnmatch.fnmatch(str(candidate), glob)
                ]

            matched_files: list[Path] = []
            matched_lines: list[str] = []
            total_matches = 0
            for candidate in candidates:
                try:
                    candidate = self.context.resolve_read_path(candidate)
                except Exception:
                    continue
                if any(part in {".git", ".svn", ".hg"} for part in candidate.parts):
                    continue
                try:
                    if candidate.stat().st_size > 10 * 1024 * 1024:
                        continue
                    raw = candidate.read_bytes()
                    if b"\x00" in raw[:8192]:
                        continue
                    text = raw.decode("utf-8", errors="replace")
                except OSError:
                    continue
                file_matches = list(regex.finditer(text))
                if not file_matches:
                    continue
                matched_files.append(candidate.resolve())
                total_matches += len(file_matches)
                if output_mode == "content":
                    for line_no, line in enumerate(text.splitlines(), start=1):
                        if regex.search(line):
                            matched_lines.append(f"{candidate.resolve()}:{line_no}:{line}")

            matched_files = matched_files[:limit]
            matched_lines = matched_lines[:limit]
            artifacts: dict[str, Any] = {
                "mode": output_mode,
                "filenames": [str(candidate) for candidate in matched_files],
                "num_files": len(matched_files),
                "num_matches": total_matches,
                "truncated": len(matched_lines) >= limit or len(matched_files) >= limit,
            }
            if output_mode == "content":
                artifacts["content"] = "\n".join(matched_lines)
            return _response("SUCCESS", f"Grep matched {len(matched_files)} files.", artifacts)
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Grep failed.", issues=[str(exc)])

    def InspectDataFile(
        self,
        file_path: str,
        sample_rows: int = 3,
        max_columns: int = 80,
    ) -> ToolResponse:
        """Inspect schema, samples, row count, and missing values of a structured data file."""
        try:
            if sample_rows < 0 or sample_rows > 50:
                raise ValueError("sample_rows must be between 0 and 50")
            if max_columns < 1 or max_columns > 500:
                raise ValueError("max_columns must be between 1 and 500")
            path = self.context.resolve_read_path(file_path)
            if not path.is_file():
                raise ValueError(f"Data file does not exist: {path}")
            inspection = _inspect_structured_file(path, sample_rows=sample_rows, max_columns=max_columns)
            sample_frame = inspection.pop("sample_frame")
            samples = sample_frame.head(sample_rows).where(sample_frame.head(sample_rows).notna(), None)
            sample_records = json.loads(
                json.dumps(samples.to_dict(orient="records"), ensure_ascii=False, default=str),
            )
            sample_records = [
                {
                    key: (
                        value[:500] + "... [truncated]"
                        if isinstance(value, str) and len(value) > 500
                        else value
                    )
                    for key, value in record.items()
                }
                for record in sample_records
            ]
            self.context.mark_read(path)
            return _response(
                "SUCCESS",
                f"Inspected {path.name}: {inspection['row_count']} rows x {inspection['column_count']} columns.",
                {
                    "file_path": str(path),
                    **inspection,
                    "sample_records": sample_records,
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Data-file inspection failed.", issues=[str(exc)])

    def CompareArtifact(
        self,
        expected_path: str,
        actual_path: str,
        key_columns_json: str = "[]",
        numeric_tolerance: float = 1e-8,
        ignore_row_order: bool = True,
        normalize_empty: bool = True,
        limit: int = 10,
    ) -> ToolResponse:
        """Compare two structured artifacts or package directories for train regression checks."""
        try:
            if numeric_tolerance < 0:
                raise ValueError("numeric_tolerance must be non-negative")
            if limit < 1 or limit > 100:
                raise ValueError("limit must be between 1 and 100")
            expected = self.context.resolve_read_path(expected_path)
            actual = self.context.resolve_read_path(actual_path)
            keys = _load_json_string_list(key_columns_json)
            if expected.is_dir() and actual.is_dir():
                result = _compare_artifact_directories(
                    expected,
                    actual,
                    key_columns=keys,
                    numeric_tolerance=numeric_tolerance,
                    ignore_row_order=ignore_row_order,
                    normalize_empty=normalize_empty,
                    limit=limit,
                )
            elif expected.is_file() and actual.is_file():
                result = _compare_artifact_files(
                    expected,
                    actual,
                    key_columns=keys,
                    numeric_tolerance=numeric_tolerance,
                    ignore_row_order=ignore_row_order,
                    normalize_empty=normalize_empty,
                    limit=limit,
                )
            else:
                raise ValueError("expected_path and actual_path must both be files or both be directories")
            self.context.mark_read(expected)
            self.context.mark_read(actual)
            status = "SUCCESS" if result["passed"] else "NEEDS_REPAIR"
            report_path = self.context.workspace_dir / "train_regression_report.json"
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            self.context.mark_read(report_path)
            record_step(
                "CompareArtifact",
                "Artifacts match." if result["passed"] else "Artifacts differ.",
                [expected, actual, report_path],
                metadata=result,
                status=status,
                issues=[] if result["passed"] else result.get("issues", []),
            )
            result = {**result, "report_path": str(report_path)}
            return _response(
                status,
                "Artifacts match." if result["passed"] else "Artifacts differ.",
                result,
                [] if result["passed"] else result.get("issues", []),
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Artifact comparison failed.", issues=[str(exc)])

    def ValidateResultPackage(
        self,
        package_root: str,
        key_column: str = "",
        expected_key_count: int = 0,
        allow_empty_feature_tables: bool = False,
    ) -> ToolResponse:
        """Validate a reference-guided result_package without reading hidden validation reference."""
        try:
            root = self.context.resolve_read_path(package_root)
            if not root.is_dir():
                raise ValueError(f"package_root is not a directory: {root}")
            result = _validate_reference_result_package(
                root,
                key_column=key_column,
                expected_key_count=expected_key_count,
                allow_empty_feature_tables=allow_empty_feature_tables,
            )
            if (root / "package_manifest.json").is_file():
                self.context.mark_read(root / "package_manifest.json")
            if (root / "reference.csv").is_file():
                self.context.mark_read(root / "reference.csv")
            status = "SUCCESS" if result["valid"] else "NEEDS_REPAIR"
            report_path = self.context.workspace_dir / "result_package_validation_report.json"
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            self.context.mark_read(report_path)
            artifacts = [report_path, *(Path(item["path"]) for item in result.get("checked_files", []))]
            if (root / "package_manifest.json").is_file():
                artifacts.insert(0, root / "package_manifest.json")
            record_step(
                "ValidateResultPackage",
                "Result package is valid." if result["valid"] else "Result package needs repair.",
                artifacts,
                metadata=result,
                status=status,
                issues=[] if result["valid"] else result.get("issues", []),
            )
            result = {**result, "report_path": str(report_path)}
            return _response(
                status,
                "Result package is valid." if result["valid"] else "Result package needs repair.",
                result,
                [] if result["valid"] else result.get("issues", []),
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Result package validation failed.", issues=[str(exc)])

    def RunSkill(self, skill_name: str, spec_json: str = "{}") -> ToolResponse:
        """Execute an allowed experiment Skill by name using its spec_json contract."""
        try:
            name = str(skill_name or "").strip()
            if not re.fullmatch(r"[a-z][a-z0-9_]{1,80}", name):
                raise ValueError("skill_name must be a local snake_case skill name")
            if name not in self.context.executable_skill_names:
                allowed = ", ".join(sorted(self.context.executable_skill_names)) or "(none)"
                raise ValueError(f"Skill is not executable in this agent: {name}. Allowed: {allowed}")
            spec = json.loads(spec_json or "{}")
            if not isinstance(spec, dict):
                raise ValueError("spec_json must be a JSON object")
            self._validate_skill_spec_paths(spec)

            module = importlib.import_module(f"skills.{name}.skill")
            function_name = f"{name}_tool"
            tool = getattr(module, function_name, None)
            if tool is None or not callable(tool):
                raise ValueError(f"Skill {name} does not expose callable {function_name}(spec_json)")

            response = tool(json.dumps(spec, ensure_ascii=False))
            if not isinstance(response, ToolResponse):
                raise ValueError(f"Skill {name} returned {type(response).__name__}, expected ToolResponse")
            payload = _tool_payload(response)
            status = str(payload.get("status") or "UNKNOWN")
            artifacts = _collect_phase_artifact_paths(payload, Path(self.context.engineer_phase_root))
            self.context.record_skill_call(name, function_name, status, artifacts=artifacts)
            return response
        except Exception as exc:
            self.context.record_skill_call(str(skill_name or ""), "RunSkill", "NEEDS_REPAIR")
            return _response("NEEDS_REPAIR", "RunSkill failed.", issues=[str(exc)])

    def PublishDirectoryArtifact(
        self,
        directory_path: str,
        alias: str = "result_package",
        target_name: str = "",
    ) -> ToolResponse:
        """Copy a phase-local output directory into workspace for package staging and validation."""
        try:
            if not alias or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", alias):
                raise ValueError("alias must be a lowercase snake_case identifier")
            if target_name and not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", target_name):
                raise ValueError("target_name must be a simple directory name")
            source = self.context.resolve_write_path(directory_path)
            if not source.exists() or not source.is_dir():
                raise ValueError(f"Directory artifact does not exist: {source}")
            target = (self.context.workspace_dir / (target_name or alias)).resolve()
            workspace = self.context.workspace_dir.resolve()
            if target != workspace and workspace not in target.parents:
                raise ValueError(f"Publish target is outside workspace: {target}")
            if source.resolve() != target:
                if source.resolve() in target.parents:
                    raise ValueError("Refusing to publish a directory into itself")
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            file_paths = sorted(path for path in target.rglob("*") if path.is_file())
            record = record_step(
                "PublishDirectoryArtifact",
                f"Published directory {source.name} as {alias}.",
                [target, *file_paths[:50]],
                metadata={
                    "alias": alias,
                    "source_path": str(source),
                    "target_path": str(target),
                    "file_count": len(file_paths),
                    "files_truncated": len(file_paths) > 50,
                },
            )
            return _response(
                "SUCCESS",
                f"Published directory {source.name} as {alias}.",
                {
                    "alias": alias,
                    "source_path": str(source),
                    "target_path": str(target),
                    "file_count": len(file_paths),
                    "sample_files": [str(path) for path in file_paths[:20]],
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Directory artifact publication failed.", issues=[str(exc)])

    def Write(self, file_path: str, content: str) -> ToolResponse:
        """Create or overwrite a UTF-8 file inside the Engineer phase."""
        try:
            path = self.context.resolve_write_path(file_path)
            self._guard_variant_metadata(path)
            if path.suffix.lower() == ".py":
                self.context.require_python_plan(path)
                self.context.validate_python_rule_ids(content)
            before = ""
            operation = "create"
            if path.exists():
                if not path.is_file():
                    raise ValueError(f"Write target is not a file: {path}")
                if not self.context.was_read_and_unchanged(path):
                    raise ValueError("Refusing to overwrite: call Read first and keep the file unchanged")
                before = path.read_text(encoding="utf-8", errors="replace")
                operation = "update"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self._invalidate_variant_after_source_change(path)
            self.context.forget_read(path)
            full_diff = _diff(before, content, path)
            response_diff, diff_truncated = _truncate(full_diff)
            record = record_step(
                "Write",
                f"{operation.title()}d {path.name}.",
                [path],
                metadata={"operation": operation, "diff": full_diff},
            )
            return _response(
                "SUCCESS",
                f"{operation.title()}d {path}.",
                {
                    "file_path": str(path),
                    "operation": operation,
                    "diff": response_diff,
                    "diff_truncated": diff_truncated,
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Write failed.", issues=[str(exc)])

    def Edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResponse:
        """Replace an exact string in a previously read Engineer file."""
        try:
            path = self.context.resolve_write_path(file_path)
            self._guard_variant_metadata(path)
            if not path.exists() or not path.is_file():
                raise ValueError(f"Edit target does not exist: {path}")
            if not self.context.was_read_and_unchanged(path):
                raise ValueError("Refusing to edit: call Read first and keep the file unchanged")
            before = path.read_text(encoding="utf-8", errors="replace")
            count = before.count(old_string)
            if count == 0:
                raise ValueError("old_string was not found")
            if count > 1 and not replace_all:
                raise ValueError("old_string is not unique; expand it or set replace_all=true")
            after = before.replace(old_string, new_string, -1 if replace_all else 1)
            if path.suffix.lower() == ".py":
                self.context.validate_python_rule_ids(after)
            path.write_text(after, encoding="utf-8")
            self._invalidate_variant_after_source_change(path)
            self.context.forget_read(path)
            replacements = count if replace_all else 1
            full_diff = _diff(before, after, path)
            response_diff, diff_truncated = _truncate(full_diff)
            record = record_step(
                "Edit",
                f"Edited {path.name} with {replacements} replacement(s).",
                [path],
                metadata={"replacements": replacements, "diff": full_diff},
            )
            return _response(
                "SUCCESS",
                f"Edited {path}.",
                {
                    "file_path": str(path),
                    "replacements": replacements,
                    "diff": response_diff,
                    "diff_truncated": diff_truncated,
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Edit failed.", issues=[str(exc)])

    def ExecutePython(
        self,
        file_path: str,
        args: list[str] | None = None,
        timeout_seconds: int = 300,
        task_id: str = "",
    ) -> ToolResponse:
        """Execute a workspace script; OUTPUT_DIR is the only writable destination. Data-producing standalone scripts require task_id."""
        step_ctx = None
        try:
            if timeout_seconds < 1 or timeout_seconds > 600:
                raise ValueError("timeout_seconds must be between 1 and 600")
            script = self.context.resolve_write_path(file_path)
            if not script.exists() or not script.is_file() or script.suffix.lower() != ".py":
                raise ValueError(f"ExecutePython requires an existing .py file: {script}")
            self.context.require_python_plan(script)
            script_content = script.read_text(encoding="utf-8", errors="replace")
            self.context.validate_python_rule_ids(script_content)
            variant_metadata = self._validate_variant_execution(script, args or [])
            if variant_metadata is None and task_id:
                self.context.validate_standalone_task(task_id, script_content)
            step_ctx = begin_step("ExecutePython")
            if step_ctx is None:
                output_dir = Path(self.context.engineer_phase_root) / "artifacts" / "execute_python"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = Path(step_ctx["step_dir"])
            runner = output_dir / "_execute_runner.py"
            runner.write_text(
                _runner_source(
                    output_dir,
                    read_roots=self.context.read_roots,
                    denied_read_roots=self.context.denied_read_roots,
                ),
                encoding="utf-8",
            )
            before_files = {p.resolve() for p in output_dir.rglob("*") if p.is_file()}

            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_DIR": str(output_dir),
                    "TMPDIR": str(output_dir),
                    "MPLCONFIGDIR": str(output_dir / ".mplconfig"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )
            command = [sys.executable, "-u", str(runner), str(script), *(args or [])]
            timed_out = False
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(output_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                returncode = completed.returncode
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = -1
                stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
                stderr += f"\nTimeoutError: execution exceeded {timeout_seconds} seconds."

            variant_result = None
            after_files = {p.resolve() for p in output_dir.rglob("*") if p.is_file()}
            output_files = sorted(
                str(path)
                for path in after_files - before_files
                if path != runner.resolve()
            )
            status = "SUCCESS" if returncode == 0 and not timed_out else "NEEDS_REPAIR"
            if (
                status == "SUCCESS"
                and variant_metadata is None
                and output_files
                and self.context.require_skill_plan
                and self.context.required_report_paths.get("extraction_task_plan")
                and not task_id
            ):
                status = "NEEDS_REPAIR"
                stderr += (
                    "\nStandalone Python produced data artifacts without task_id. "
                    "Bind the execution to a canonical capability-gap task or call the planned domain Skill."
                )
            if variant_metadata is not None and status == "SUCCESS":
                self._verify_variant_source_hashes(variant_metadata)
                try:
                    variant_result = parse_variant_result(stdout)
                except Exception as exc:
                    status = "NEEDS_REPAIR"
                    stderr += f"\nVariant protocol error: {exc}"
                else:
                    if variant_result["status"] != "SUCCESS":
                        status = "NEEDS_REPAIR"
                        stderr += "\nVariant reported NEEDS_REPAIR: " + "; ".join(variant_result["issues"])
                if status == "SUCCESS" and variant_result is not None:
                    try:
                        receipt_path = write_variant_execution_receipt(
                            script.parent,
                            variant_result,
                            output_files=[Path(path) for path in output_files],
                        )
                    except Exception as exc:
                        status = "NEEDS_REPAIR"
                        stderr += f"\nVariant receipt error: {exc}"
                    else:
                        output_files.append(str(receipt_path))
            stdout, stdout_truncated = _truncate(stdout)
            stderr, stderr_truncated = _truncate(stderr)
            issues = [] if status == "SUCCESS" else [stderr or f"Python exited with {returncode}"]
            record = record_step(
                "ExecutePython",
                "Python execution succeeded." if status == "SUCCESS" else "Python execution failed.",
                [script, *output_files],
                metadata={
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": timed_out,
                    "script_path": str(script),
                    "task_id": task_id,
                    "variant_name": variant_metadata.get("variant_name", "") if variant_metadata else "",
                },
                status=status,
                issues=issues,
                _step_ctx=step_ctx,
            )
            return _response(
                status,
                "Python execution succeeded." if status == "SUCCESS" else "Python execution failed.",
                {
                    "script_path": str(script),
                    "task_id": task_id,
                    "output_dir": str(output_dir),
                    "output_files": output_files,
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "timed_out": timed_out,
                    "variant_result": variant_result,
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
                issues,
            )
        except Exception as exc:
            if step_ctx is not None:
                record_step(
                    "ExecutePython",
                    "Python execution setup failed.",
                    [],
                    status="NEEDS_REPAIR",
                    issues=[str(exc)],
                    _step_ctx=step_ctx,
                )
            return _response("NEEDS_REPAIR", "ExecutePython failed.", issues=[str(exc)])

    def publish_artifact(self, file_path: str, alias: str) -> ToolResponse:
        """Publish an Engineer output through the existing phase handoff system."""
        try:
            if not alias or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", alias):
                raise ValueError("alias must be a lowercase snake_case identifier")
            path = self.context.resolve_write_path(file_path)
            if not path.exists() or not path.is_file():
                raise ValueError(f"Artifact does not exist: {path}")
            if self.context.require_skill_plan and not self.context.was_read_and_unchanged(path):
                raise ValueError("Read and inspect the final artifact before publish_artifact")
            record = record_step(
                "publish_artifact",
                f"Published {path.name} as {alias}.",
                [path],
                handoffs={alias: path},
            )
            handoffs = (record or {}).get("handoffs", {})
            return _response(
                "SUCCESS",
                f"Published {path.name} as {alias}.",
                {
                    "file_path": str(path),
                    "alias": alias,
                    "handoff": handoffs.get(alias, {}),
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Artifact publication failed.", issues=[str(exc)])

    def _validate_skill_spec_paths(self, spec: dict[str, Any]) -> None:
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for nested in value.values():
                    visit(nested)
                return
            if isinstance(value, list):
                for nested in value:
                    visit(nested)
                return
            if not isinstance(value, str) or "/" not in value:
                return
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                return
            resolved = candidate.resolve()
            if resolved.exists():
                self.context.resolve_read_path(resolved)
            elif not (
                resolved == self.context.engineer_phase_root
                or self.context.engineer_phase_root in resolved.parents
            ):
                raise ValueError(f"Skill spec references an unauthorized path: {resolved}")

        visit(spec)

    def _validate_variant_execution(
        self,
        script: Path,
        args: list[str],
    ) -> dict[str, Any] | None:
        variant_root = self.context.variant_root.resolve()
        if script != variant_root and variant_root not in script.parents:
            return None
        if script.name != "variant.py" or script.parent.parent != variant_root:
            raise ValueError("Derived Skill execution requires skill_variants/<name>/variant.py")
        metadata_path = script.parent / "variant.json"
        request_path = script.parent / "request.json"
        if not metadata_path.is_file() or not request_path.is_file():
            raise ValueError("Derived Skill is missing variant.json or request.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") not in {"ready", "validated"}:
            raise ValueError("Derived Skill must be ready before ExecutePython")
        if len(args) != 1 or Path(args[0]).expanduser().resolve() != request_path.resolve():
            raise ValueError("Derived Skill must run with args=[request.json]")
        metadata["variant_name"] = script.parent.name
        self._verify_variant_source_hashes(metadata)
        return metadata

    def _guard_variant_metadata(self, path: Path) -> None:
        variant_root = self.context.variant_root.resolve()
        if path.name == "variant.json" and path.parent.parent == variant_root:
            raise ValueError("variant.json is lifecycle-managed and cannot be changed with Write/Edit")

    def _invalidate_variant_after_source_change(self, path: Path) -> None:
        variant_root = self.context.variant_root.resolve()
        if path.name != "variant.py" or path.parent.parent != variant_root:
            return
        metadata_path = path.parent / "variant.json"
        if not metadata_path.is_file():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(metadata, dict):
            return
        metadata["status"] = "draft"
        metadata.pop("validated_at", None)
        metadata.pop("validated_artifact", None)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _verify_variant_source_hashes(metadata: dict[str, Any]) -> None:
        expected = metadata.get("base_source_hashes")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Derived Skill is missing base_source_hashes")
        current: dict[str, str] = {}
        for raw_path in expected:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise ValueError(f"Base Skill source disappeared: {path}")
            current[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != expected:
            raise ValueError("Base Skill source changed after derived Skill creation")


def register_engineer_atomic_tools(
    toolkit: Toolkit,
    context: EngineerToolContext,
) -> EngineerTools:
    """Register Engineer-only atomic tools directly on an AgentScope toolkit."""
    tools = EngineerTools(context)
    for tool in (
        tools.Read,
        tools.Glob,
        tools.Grep,
        tools.InspectDataFile,
        tools.CompareArtifact,
        tools.ValidateResultPackage,
        tools.RunSkill,
        tools.Write,
        tools.Edit,
        tools.ExecutePython,
        tools.PublishDirectoryArtifact,
        tools.publish_artifact,
    ):
        toolkit.register_tool_function(tool)
    return tools


def _runner_source(
    output_dir: Path,
    *,
    read_roots: Iterable[Path],
    denied_read_roots: Iterable[Path] = (),
) -> str:
    output_literal = repr(str(output_dir.resolve()))
    allowed_dirs = [
        str(path.resolve())
        for path in read_roots
        if path.exists() and path.is_dir()
    ]
    allowed_files = [
        str(path.resolve())
        for path in read_roots
        if path.exists() and path.is_file()
    ]
    for runtime_root in (Path(sys.prefix), Path(sys.base_prefix), output_dir):
        value = str(runtime_root.resolve())
        if value not in allowed_dirs:
            allowed_dirs.append(value)
    for runtime_file in (Path("/etc/localtime"), Path("/dev/null"), Path("/dev/urandom")):
        if runtime_file.exists():
            allowed_files.append(str(runtime_file.resolve()))
    allowed_dirs_literal = repr(allowed_dirs)
    allowed_files_literal = repr(allowed_files)
    denied_dirs_literal = repr(
        [str(path.resolve()) for path in denied_read_roots],
    )
    return f'''from __future__ import annotations

import os
import runpy
import socket
import sys
from pathlib import Path

WRITE_ROOT = Path({output_literal}).resolve()
READ_DIRS = tuple(Path(value).resolve() for value in {allowed_dirs_literal})
READ_FILES = frozenset(Path(value).resolve() for value in {allowed_files_literal})
DENIED_DIRS = tuple(Path(value).resolve() for value in {denied_dirs_literal})


def _resolve_path(value):
    if isinstance(value, int):
        return None
    if value is None:
        return Path.cwd().resolve()
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    path = Path(os.fspath(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _ensure_write_path(value):
    path = _resolve_path(value)
    if path is None:
        return
    if path != WRITE_ROOT and WRITE_ROOT not in path.parents:
        raise PermissionError(f"write outside the execution output directory is forbidden: {{path}}")


def _ensure_read_path(value):
    path = _resolve_path(value)
    if path is None:
        return
    if any(path == root or root in path.parents for root in DENIED_DIRS):
        raise PermissionError(f"read is within a denied read root: {{path}}")
    if path in READ_FILES or any(path == root or root in path.parents for root in READ_DIRS):
        return
    raise PermissionError(f"read outside authorized read roots is forbidden: {{path}}")


def _ensure_directory_listing_path(value):
    path = _resolve_path(value)
    if path is None:
        raise PermissionError("directory descriptor enumeration is forbidden")
    if any(
        path == root or root in path.parents or path in root.parents
        for root in DENIED_DIRS
    ):
        raise PermissionError(f"directory enumeration would expose a denied read root: {{path}}")
    _ensure_read_path(path)


def _audit(event, args):
    if event == "open":
        path, mode, flags = args
        write_mode = isinstance(mode, str) and any(char in mode for char in "wax+")
        write_flags = isinstance(flags, int) and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write_mode or write_flags:
            _ensure_write_path(path)
        else:
            _ensure_read_path(path)
    elif event in {{"os.remove", "os.unlink", "os.rmdir", "os.mkdir"}}:
        _ensure_write_path(args[0])
    elif event in {{"os.rename", "os.replace"}}:
        _ensure_write_path(args[0])
        _ensure_write_path(args[1])
    elif event in {{"os.listdir", "os.scandir"}}:
        _ensure_directory_listing_path(args[0])
    elif event in {{"subprocess.Popen", "os.system", "pty.spawn"}}:
        raise PermissionError(f"child processes and shell execution are forbidden: {{event}}")
    elif event.startswith("socket.") and event not in {{"socket.__new__"}}:
        raise PermissionError(f"network access is forbidden: {{event}}")


sys.addaudithook(_audit)
script_path = Path(sys.argv[1]).resolve()
script_args = sys.argv[2:]
sys.argv = [str(script_path), *script_args]
runpy.run_path(
    str(script_path),
    run_name="__main__",
    init_globals={{"OUTPUT_DIR": str(WRITE_ROOT)}},
)
'''
