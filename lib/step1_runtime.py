from __future__ import annotations

import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

STEP1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP1_DIR.parent
if str(STEP1_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TABULAR_SUFFIXES = {".csv", ".xls", ".xlsx", ".tsv"}
VISUAL_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}
SUPPORTED_SUFFIXES = TABULAR_SUFFIXES | VISUAL_SUFFIXES
EXCLUDED_DIR_NAMES = {"__pycache__", "reorganized_output"}
EXCLUDED_FILE_NAMES = {".DS_Store"}
PATIENT_ID_PATTERN = re.compile(r"patient[-_ ]?(\d+)", re.IGNORECASE)
PATIENT_DIR_PATTERN = re.compile(r"^p(\d+)$", re.IGNORECASE)
NUMERIC_ID_PATTERN = re.compile(r"(?<!\d)(\d{4,})(?:\(\d+\))?(?!\d)")
ProgressCallback = Callable[[str, int, int], None]
EmitCallback = Callable[[str], None]


@dataclass(frozen=True)
class FileTask:
    source_path: str
    relative_path: str
    suffix: str
    patient_id: str | None
    processable: bool


@dataclass(frozen=True)
class ModalityPatch:
    source_path: str
    modality: str
    evidence: str
    details: dict[str, Any] | None = None
    error: str = ""


@dataclass(frozen=True)
class TableSplitPatch:
    source_path: str
    should_split: bool
    id_column: str | None
    evidence: str
    error: str = ""


@dataclass(frozen=True)
class ValidatorResult:
    passed: bool
    issues: list[str]
    details: dict[str, Any]


class TerminalProgress:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._last: dict[str, int] = {}

    def emit(self, message: str) -> None:
        if self.enabled:
            print(message, file=sys.stderr, flush=True)

    def progress(self, label: str, current: int, total: int) -> None:
        if not self.enabled or total <= 0:
            return
        current = min(max(current, 0), total)
        step = max(1, total // 20)
        last = self._last.get(label, -1)
        if current not in {0, 1, total} and current - last < step:
            return
        self._last[label] = current
        percent = current * 100 / total
        print(f"[Step1][Progress] {label}: {current}/{total} ({percent:.1f}%)", file=sys.stderr, flush=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def get_step1_max_workers() -> int:
    raw = os.environ.get("STEP1_MAX_WORKERS", "4")
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def get_step1_reorganize_max_workers() -> int:
    raw = os.environ.get("STEP1_REORGANIZE_MAX_WORKERS")
    if raw is None:
        return get_step1_max_workers()
    try:
        return max(1, int(raw))
    except ValueError:
        return get_step1_max_workers()


def is_step1_parallel_enabled() -> bool:
    return _env_bool("STEP1_PARALLEL_ENABLED", True)


def is_step1_reorganize_parallel_enabled() -> bool:
    return _env_bool("STEP1_REORGANIZE_PARALLEL_ENABLED", True)


def _match_patient_id(text: str) -> str | None:
    match = PATIENT_ID_PATTERN.search(text or "")
    if not match:
        return None
    return f"patient-{match.group(1)}"


def extract_patient_id(source_path: str, content_text: str = "") -> str | None:
    path = Path(source_path or "")
    for part in path.parts:
        patient_dir_match = PATIENT_DIR_PATTERN.fullmatch(str(part))
        if patient_dir_match:
            return patient_dir_match.group(1)

    path_match = _match_patient_id(source_path)
    if path_match:
        return path_match

    numeric_name_match = NUMERIC_ID_PATTERN.search(path.name)
    if numeric_name_match:
        return numeric_name_match.group(1)

    numeric_path_match = NUMERIC_ID_PATTERN.search(source_path or "")
    if numeric_path_match:
        return numeric_path_match.group(1)

    content_match = _match_patient_id(content_text)
    if content_match:
        return content_match

    numeric_content_match = NUMERIC_ID_PATTERN.search(content_text or "")
    if numeric_content_match:
        return numeric_content_match.group(1)
    return None


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def _is_under_program_output(path: Path) -> bool:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "program" and index + 1 < len(parts) and parts[index + 1] == "output":
            return True
    return False


def _is_recordable_input_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in EXCLUDED_FILE_NAMES or path.name.startswith("."):
        return False
    if path.suffix.lower() in {".tmp", ".temp", ".bak", ".swp"}:
        return False
    if any(part.startswith(".") for part in path.parts):
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False
    return not _is_under_program_output(path)


def _is_supported_input_file(path: Path) -> bool:
    return _is_recordable_input_file(path) and path.suffix.lower() in SUPPORTED_SUFFIXES


def scan_source_files(input_path: str | Path) -> list[FileTask]:
    root = resolve_path(input_path)
    if root.is_file():
        files = [root] if _is_recordable_input_file(root) else []
        base = root.parent
    elif root.is_dir():
        files = [item for item in sorted(root.rglob("*")) if _is_recordable_input_file(item)]
        base = root
    else:
        return []

    tasks: list[FileTask] = []
    for file_path in sorted(files):
        relative_path = str(file_path.relative_to(base))
        tasks.append(
            FileTask(
                source_path=str(file_path),
                relative_path=relative_path,
                suffix=file_path.suffix.lower(),
                patient_id=extract_patient_id(str(file_path), ""),
                processable=_is_supported_input_file(file_path),
            )
        )
    return tasks


def build_base_records(
    tasks: list[FileTask],
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    records = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        path = Path(task.source_path)
        if task.suffix in TABULAR_SUFFIXES:
            modality = "table"
        elif task.processable:
            modality = "ocr"
        else:
            modality = "unsupported"
        records.append(
            {
                "source_path": task.source_path,
                "source_name": path.name,
                "patient_id": task.patient_id,
                "relative_path": task.relative_path,
                "file_info": {
                    "suffix": task.suffix,
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "processable": task.processable,
                },
                "observations": {
                    "modality": modality,
                    "should_split": False,
                    "id_column": None,
                    "status": "pending" if task.processable else "unsupported",
                },
            }
        )
        if progress:
            progress("SourceRecordWorker", index, total)
    return records


def infer_file_modality(task: FileTask) -> ModalityPatch:
    try:
        if not task.processable:
            return ModalityPatch(task.source_path, "unsupported", "unsupported suffix")
        if task.suffix in TABULAR_SUFFIXES:
            return ModalityPatch(task.source_path, "table", "tabular suffix")
        if task.suffix in VISUAL_SUFFIXES:
            return _fallback_visual_modality(task, evidence_prefix="doclayout fallback")
        return ModalityPatch(task.source_path, "ocr", "fallback")
    except Exception as exc:
        return ModalityPatch(task.source_path, "ocr", "fallback on error", error=str(exc))


def _fallback_visual_modality(task: FileTask, evidence_prefix: str = "fallback") -> ModalityPatch:
    path_parts = [part.lower() for part in Path(task.source_path).parts]
    if "images" in path_parts:
        return ModalityPatch(task.source_path, "figure", f"{evidence_prefix}: path contains images")
    return ModalityPatch(task.source_path, "ocr", f"{evidence_prefix}: visual suffix default")


def _doclayout_enabled() -> bool:
    raw = os.environ.get("STEP1_DOCLAYOUT_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _normalize_doclayout_modality(value: Any) -> str:
    raw = str(value or "ocr").strip().lower()
    if raw in {"figure", "ocr+figure"}:
        return "figure"
    return "ocr"


def _infer_doclayout_for_visual_tasks(
    input_path: str | Path,
    tasks: list[FileTask],
    progress: ProgressCallback | None = None,
    emit: EmitCallback | None = None,
) -> list[ModalityPatch]:
    if not tasks:
        return []
    if not _doclayout_enabled():
        patches = []
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            patches.append(_fallback_visual_modality(task, evidence_prefix="doclayout disabled"))
            if progress:
                progress("DocLayout disabled fallback", index, total)
        return patches
    try:
        from agent_1.layout_analysis_tool import infer_and_save_layout

        if emit:
            emit(f"[Step1][Tool] DocLayout-YOLO 开始识别图片/PDF文件: {len(tasks)} 个")
        response = infer_and_save_layout(
            str(input_path),
            save_outputs=False,
            progress_callback=progress,
        )
        payload = json.loads(getattr(response, "content", "{}") or "{}")
        if isinstance(payload, dict) and payload.get("error"):
            message = json.dumps(payload.get("error"), ensure_ascii=False)
            return [
                _fallback_visual_modality(task, evidence_prefix=f"doclayout error: {message}")
                for task in tasks
            ]
        classification = payload.get("classification") or []
        by_relative: dict[str, dict[str, Any]] = {}
        by_name: dict[str, dict[str, Any]] = {}
        for item in classification:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("relative_path") or "").strip()
            name = str(item.get("file_name") or "").strip()
            if rel:
                by_relative[rel] = item
            if name:
                by_name[name] = item

        patches: list[ModalityPatch] = []
        for task in tasks:
            source = Path(task.source_path)
            item = by_relative.get(task.relative_path) or by_name.get(source.name)
            if not item:
                patches.append(_fallback_visual_modality(task, evidence_prefix="doclayout missing result"))
                continue
            patches.append(
                ModalityPatch(
                    source_path=task.source_path,
                    modality=_normalize_doclayout_modality(item.get("modality")),
                    evidence="doclayout-yolo",
                    details={
                        "relative_path": item.get("relative_path"),
                        "file_name": item.get("file_name"),
                        "raw_modality": item.get("modality"),
                    },
                )
            )
        return patches
    except Exception as exc:
        patches = []
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            patches.append(_fallback_visual_modality(task, evidence_prefix=f"doclayout exception: {exc}"))
            if progress:
                progress("DocLayout exception fallback", index, total)
        return patches


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_excel(path, dtype=str)


def _detect_id_column(df: pd.DataFrame) -> str | None:
    normalized = {
        str(column).strip().lower().replace(" ", "").replace("_", ""): str(column)
        for column in df.columns
    }
    preferred = [
        "subjectid",
        "patientid",
        "hadmid",
        "stayid",
        "id",
        "病例号",
        "病人id",
        "患者id",
        "患者编号",
        "病历号",
    ]
    for item in preferred:
        if item in normalized:
            return normalized[item]
    return None


def infer_table_split_strategy(task: FileTask) -> TableSplitPatch:
    if task.suffix not in TABULAR_SUFFIXES:
        return TableSplitPatch(task.source_path, False, None, "non-tabular")
    try:
        df = _read_table(Path(task.source_path))
        id_column = _detect_id_column(df)
        return TableSplitPatch(
            source_path=task.source_path,
            should_split=id_column is not None,
            id_column=id_column,
            evidence="detected id column" if id_column else "no id column",
        )
    except Exception as exc:
        return TableSplitPatch(
            source_path=task.source_path,
            should_split=False,
            id_column=None,
            evidence="fallback on error",
            error=str(exc),
        )


def _run_pool(
    items,
    fn,
    max_workers: int,
    progress_label: str | None = None,
    progress: ProgressCallback | None = None,
):
    total = len(items)
    if max_workers <= 1:
        results = []
        for index, item in enumerate(items, start=1):
            results.append(fn(item))
            if progress and progress_label:
                progress(progress_label, index, total)
        return results
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(fn, item): item for item in items}
        for index, future in enumerate(as_completed(future_to_item), start=1):
            results.append(future.result())
            if progress and progress_label:
                progress(progress_label, index, total)
    return results


def source_scanner_handoff(input_path: str | Path) -> list[FileTask]:
    return scan_source_files(input_path)


def source_record_handoff(
    tasks: list[FileTask],
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    return build_base_records(tasks, progress=progress)


def modality_handoff(
    input_path: str | Path,
    tasks: list[FileTask],
    workers: int,
    progress: ProgressCallback | None = None,
    emit: EmitCallback | None = None,
) -> tuple[list[ModalityPatch], dict[str, int]]:
    processable_tasks = [task for task in tasks if task.processable]
    unsupported_tasks = [task for task in tasks if not task.processable]
    table_tasks = [task for task in processable_tasks if task.suffix in TABULAR_SUFFIXES]
    visual_tasks = [task for task in processable_tasks if task.suffix in VISUAL_SUFFIXES]
    other_tasks = [
        task
        for task in processable_tasks
        if task.suffix not in TABULAR_SUFFIXES and task.suffix not in VISUAL_SUFFIXES
    ]
    table_modality_patches = [ModalityPatch(task.source_path, "table", "tabular suffix") for task in table_tasks]
    if progress:
        progress("Modality table suffix", len(table_tasks), len(table_tasks))
    visual_modality_patches = _infer_doclayout_for_visual_tasks(
        input_path,
        visual_tasks,
        progress=progress,
        emit=emit,
    )
    other_modality_patches = _run_pool(
        other_tasks,
        infer_file_modality,
        workers,
        progress_label="Modality other files",
        progress=progress,
    )
    unsupported_modality_patches = []
    total_unsupported = len(unsupported_tasks)
    for index, task in enumerate(unsupported_tasks, start=1):
        unsupported_modality_patches.append(ModalityPatch(task.source_path, "unsupported", "unsupported suffix"))
        if progress:
            progress("Modality unsupported files", index, total_unsupported)
    return (
        table_modality_patches
        + visual_modality_patches
        + other_modality_patches
        + unsupported_modality_patches,
        {
            "processable_files": len(processable_tasks),
            "unsupported_files": len(unsupported_tasks),
        },
    )


def table_split_handoff(
    tasks: list[FileTask],
    workers: int,
    progress: ProgressCallback | None = None,
) -> list[TableSplitPatch]:
    table_tasks = [task for task in tasks if task.processable and task.suffix in TABULAR_SUFFIXES]
    return _run_pool(
        table_tasks,
        infer_table_split_strategy,
        workers,
        progress_label="TableSplitWorkerPool",
        progress=progress,
    )


def records_reducer_handoff(
    records: list[dict[str, Any]],
    modality_patches: list[ModalityPatch],
    table_patches: list[TableSplitPatch],
    records_path: str | Path,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    records_by_path = {record["source_path"]: record for record in records}
    worker_errors: list[str] = []
    total_modality = len(modality_patches)
    for index, patch in enumerate(sorted(modality_patches, key=lambda item: item.source_path), start=1):
        record = records_by_path.get(patch.source_path)
        if record:
            record["observations"]["modality"] = patch.modality
            if patch.modality == "unsupported":
                record["observations"]["status"] = "unsupported"
            elif record["observations"].get("status") != "unsupported":
                record["observations"]["status"] = "ready"
            record.setdefault("worker_evidence", {})["modality"] = patch.evidence
            if patch.details:
                record["observations"]["layout_analysis"] = dict(patch.details)
        if patch.error:
            worker_errors.append(f"modality:{patch.source_path}: {patch.error}")
        if progress:
            progress("RecordsReducer modality patches", index, total_modality)

    total_table = len(table_patches)
    for index, patch in enumerate(sorted(table_patches, key=lambda item: item.source_path), start=1):
        record = records_by_path.get(patch.source_path)
        if record:
            record["observations"]["should_split"] = bool(patch.should_split)
            record["observations"]["id_column"] = patch.id_column
            record.setdefault("worker_evidence", {})["table_split"] = patch.evidence
        if patch.error:
            worker_errors.append(f"table_split:{patch.source_path}: {patch.error}")
        if progress:
            progress("RecordsReducer table patches", index, total_table)

    output = Path(records_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered_records = sorted(records, key=lambda item: item["source_path"])
    tmp_path = output.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(ordered_records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output)
    return ordered_records, worker_errors


def build_records_parallel(
    input_path: str | Path,
    records_path: str | Path,
    max_workers: int | None = None,
    parallel_enabled: bool | None = None,
    progress: ProgressCallback | None = None,
    emit: EmitCallback | None = None,
) -> dict[str, Any]:
    if emit:
        emit(f"[Step1][Tool] SourceScanner 开始扫描: {input_path}")
    tasks = source_scanner_handoff(input_path)
    if emit:
        emit(f"[Step1][Tool] SourceScanner 完成: source_files={len(tasks)}")
    workers = max_workers if max_workers is not None else get_step1_max_workers()
    use_parallel = is_step1_parallel_enabled() if parallel_enabled is None else bool(parallel_enabled)
    if not use_parallel:
        workers = 1

    if emit:
        emit(f"[Step1][Tool] SourceRecordWorker 开始生成 records: {len(tasks)} 条")
    records = source_record_handoff(tasks, progress=progress)
    if emit:
        emit("[Step1][Tool] ModalityWorkerPool 开始模态识别")
    modality_patches, file_counts = modality_handoff(
        input_path,
        tasks,
        workers,
        progress=progress,
        emit=emit,
    )
    if emit:
        emit("[Step1][Tool] TableSplitWorkerPool 开始判断表格拆分")
    table_patches = table_split_handoff(tasks, workers, progress=progress)
    if emit:
        emit("[Step1][Tool] RecordsReducer 开始合并并写入 records.json")
    ordered_records, worker_errors = records_reducer_handoff(
        records=records,
        modality_patches=modality_patches,
        table_patches=table_patches,
        records_path=records_path,
        progress=progress,
    )
    if emit:
        emit(f"[Step1][Tool] RecordsReducer 完成: records_count={len(ordered_records)}")
    return {
        "records_path": str(Path(records_path)),
        "records_count": len(ordered_records),
        "source_files": len(tasks),
        "processable_files": file_counts["processable_files"],
        "unsupported_files": file_counts["unsupported_files"],
        "eligible_files": file_counts["processable_files"],
        "max_workers": workers,
        "parallel_enabled": use_parallel,
        "worker_errors": worker_errors,
        "records_modality_counts": count_modalities(ordered_records),
        "split_tables": sum(1 for item in ordered_records if item.get("observations", {}).get("should_split")),
        "handoffs": [
            "SourceScanner",
            "SourceRecordWorker",
            "ModalityWorkerPool",
            "TableSplitWorkerPool",
            "RecordsReducer",
        ],
    }


def count_modalities(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        modality = str(record.get("observations", {}).get("modality") or "")
        if modality:
            counts[modality] = counts.get(modality, 0) + 1
    return counts


def load_records(records_path: str | Path) -> list[dict[str, Any]]:
    parsed = json.loads(Path(records_path).read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("records.json 顶层必须是 list")
    return [item for item in parsed if isinstance(item, dict)]


def validate_records(
    records_path: str | Path,
    expected_count: int | None = None,
) -> ValidatorResult:
    path = Path(records_path)
    issues: list[str] = []
    try:
        records = load_records(path)
    except FileNotFoundError:
        records = []
        issues.append(f"records.json 不存在: {path}")
    except json.JSONDecodeError as exc:
        records = []
        issues.append(f"records.json 不是合法 JSON: {exc}")
    except ValueError as exc:
        records = []
        issues.append(str(exc))

    seen_source_paths: set[str] = set()
    modality_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    split_tables = 0

    if expected_count is not None and not issues and len(records) != expected_count:
        issues.append(f"records 数量不匹配: records={len(records)}, expected={expected_count}")

    for idx, record in enumerate(records):
        source_path = str(record.get("source_path") or "").strip()
        source_name = str(record.get("source_name") or "").strip()
        observations = record.get("observations")
        if not source_path:
            issues.append(f"record[{idx}] 缺少 source_path")
        if not source_name:
            issues.append(f"record[{idx}] 缺少 source_name")
        if source_path in seen_source_paths:
            issues.append(f"重复 source_path: {source_path}")
        seen_source_paths.add(source_path)
        if observations is None or not isinstance(observations, dict):
            issues.append(f"record[{idx}] 缺少 observations 对象")
            continue
        modality = str(observations.get("modality") or "").strip()
        if not modality:
            issues.append(f"record[{idx}] 缺少 observations.modality")
        else:
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        status = str(observations.get("status") or "ready").strip()
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        if observations.get("should_split") and not observations.get("id_column"):
            issues.append(f"record[{idx}] should_split=true 但缺少 id_column")
        if observations.get("should_split"):
            split_tables += 1

    return ValidatorResult(
        passed=not issues,
        issues=sorted(set(issues)),
        details={
            "records_path": str(path),
            "records_count": len(records),
            "expected_count": expected_count,
            "modality_counts": modality_counts,
            "status_counts": status_counts,
            "processable_records": len(records) - status_counts.get("unsupported", 0),
            "unsupported_records": status_counts.get("unsupported", 0),
            "split_tables": split_tables,
        },
    )


def validate_step1_output(output_root: str | Path, expected_min_files: int = 1) -> ValidatorResult:
    root = Path(output_root)
    issues: list[str] = []
    modality_counts: dict[str, int] = {}
    sample_modality_counts: dict[str, int] = {}
    sample_paths: list[str] = []

    if not root.is_dir():
        return ValidatorResult(
            passed=False,
            issues=[f"Step1 输出目录不存在: {root}"],
            details={"output_root": str(root), "output_file_count": 0},
        )

    files = [item for item in sorted(root.rglob("*")) if item.is_file() and "_meta" not in item.parts]
    if len(files) < expected_min_files:
        issues.append(f"Step1 输出文件数不足: output_files={len(files)}, expected_min={expected_min_files}")

    for index, file_path in enumerate(files):
        rel = file_path.relative_to(root)
        if index < 20:
            sample_paths.append(str(rel))
        if len(rel.parts) < 3:
            issues.append(f"输出路径层级不足: {rel}")
        elif len(rel.parts) >= 2:
            modality = rel.parts[1]
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            if index < 20:
                sample_modality_counts[modality] = sample_modality_counts.get(modality, 0) + 1

    return ValidatorResult(
        passed=not issues,
        issues=sorted(set(issues)),
        details={
            "output_root": str(root),
            "output_file_count": len(files),
            "expected_min_files": expected_min_files,
            "modality_counts": modality_counts,
            "sample_modality_counts": sample_modality_counts,
            "sample_paths": sample_paths,
        },
    )


def infer_input_root(records: list[dict[str, Any]]) -> Path:
    source_paths = [Path(str(item.get("source_path"))).resolve() for item in records if item.get("source_path")]
    if not source_paths:
        return PROJECT_ROOT
    common = Path(os.path.commonpath([str(path.parent) for path in source_paths]))
    if common.name in {"notes", "structured"} or common.name.startswith("s"):
        return common.parent
    return common


def safe_patient_id(value: Any, fallback: str = "unknown") -> str:
    extracted = extract_patient_id("", str(value or ""))
    return extracted or str(value or "").strip() or fallback


def _write_table(df: pd.DataFrame, target_path: Path, suffix: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".tsv":
        df.to_csv(target_path, sep="\t", index=False)
    elif suffix == ".csv":
        df.to_csv(target_path, index=False)
    else:
        df.to_excel(target_path, index=False)


def _copy_or_write_record_file(record: dict[str, Any], input_root: Path, output_root: Path) -> int:
    source = Path(str(record.get("source_path"))).resolve()
    observations = record.get("observations") or {}
    modality = str(observations.get("modality") or "ocr")
    patient_id = str(record.get("patient_id") or "") or safe_patient_id(source.stem)
    try:
        relative_parent = source.parent.relative_to(input_root)
    except ValueError:
        relative_parent = Path(str(record.get("relative_path") or source.name)).parent

    if modality == "table" and observations.get("should_split") and observations.get("id_column"):
        df = _read_table(source)
        id_column = str(observations["id_column"])
        if id_column not in df.columns:
            raise ValueError(f"表格缺少 id_column={id_column}: {source}")
        written = 0
        for raw_id, group in df.groupby(id_column, dropna=False):
            group_patient_id = safe_patient_id(raw_id, patient_id)
            target = output_root / group_patient_id / modality / relative_parent / source.name
            _write_table(group, target, source.suffix.lower())
            written += 1
        return written

    target = output_root / patient_id / modality / relative_parent / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return 1


def is_processable_record(record: dict[str, Any]) -> bool:
    observations = record.get("observations") or {}
    if str(observations.get("status") or "").lower() == "unsupported":
        return False
    if str(observations.get("modality") or "").lower() == "unsupported":
        return False
    source = Path(str(record.get("source_path") or ""))
    return source.suffix.lower() in SUPPORTED_SUFFIXES


def run_reorganize_from_records(
    records_path: str | Path,
    output_root: str | Path,
    input_root: str | Path | None = None,
    clean_output: bool = True,
    max_workers: int | None = None,
    parallel_enabled: bool | None = None,
    progress: ProgressCallback | None = None,
    emit: EmitCallback | None = None,
) -> dict[str, Any]:
    records = load_records(records_path)
    root = resolve_path(input_root) if input_root else infer_input_root(records)
    output = resolve_path(output_root)
    if emit:
        emit(f"[Step1][Tool] ReorganizeWorker 开始重组输出: records={len(records)}")
    if clean_output and output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True, exist_ok=True)

    workers = max_workers if max_workers is not None else get_step1_reorganize_max_workers()
    use_parallel = is_step1_reorganize_parallel_enabled() if parallel_enabled is None else bool(parallel_enabled)
    if not use_parallel:
        workers = 1

    written_files = 0
    errors: list[str] = []
    total = len(records)
    processable_records: list[dict[str, Any]] = []
    skipped_records = 0
    for record in records:
        if not is_processable_record(record):
            skipped_records += 1
            continue

        processable_records.append(record)

    processed_records = len(processable_records)
    if progress and skipped_records:
        progress("ReorganizeWorker records", skipped_records, total)

    if workers <= 1 or processed_records <= 1:
        completed = skipped_records
        for record in processable_records:
            completed += 1
            try:
                written_files += _copy_or_write_record_file(record, root, output)
            except Exception as exc:
                errors.append(f"{record.get('source_path')}: {exc}")
            if progress:
                progress("ReorganizeWorker records", completed, total)
    else:
        if emit:
            emit(f"[Step1][Tool] ReorganizeWorker 并行执行: workers={workers}, processable_records={processed_records}")
        completed = skipped_records
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_record = {
                executor.submit(_copy_or_write_record_file, record, root, output): record
                for record in processable_records
            }
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                completed += 1
                try:
                    written_files += future.result()
                except Exception as exc:
                    errors.append(f"{record.get('source_path')}: {exc}")
                if progress:
                    progress("ReorganizeWorker records", completed, total)

    errors = sorted(errors)
    if emit:
        emit(f"[Step1][Tool] ReorganizeWorker 完成: written_files={written_files}, skipped_records={skipped_records}")

    return {
        "status": "SUCCESS" if not errors else "FAILED",
        "records_path": str(records_path),
        "output_root": str(output),
        "input_root": str(root),
        "records_count": len(records),
        "processed_records": processed_records,
        "skipped_records": skipped_records,
        "written_files": written_files,
        "max_workers": workers,
        "parallel_enabled": use_parallel,
        "errors": errors,
    }


def write_generated_reorganizer(
    script_path: str | Path,
    records_path: str | Path,
    output_root: str | Path,
    input_root: str | Path | None = None,
) -> str:
    script = Path(script_path)
    script.parent.mkdir(parents=True, exist_ok=True)
    input_root_literal = str(input_root or "")
    content = f'''from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(r"{PROJECT_ROOT}")
STEP1_DIR = PROJECT_ROOT / "step-1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEP1_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1_DIR))

from step1_runtime import run_reorganize_from_records


if __name__ == "__main__":
    result = run_reorganize_from_records(
        records_path=Path(r"{records_path}"),
        output_root=Path(r"{output_root}"),
        input_root=Path(r"{input_root_literal}") if r"{input_root_literal}" else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("status") == "SUCCESS" else 1)
'''
    script.write_text(content, encoding="utf-8")
    return str(script)
