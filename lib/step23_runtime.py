from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

STEP23_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP23_DIR.parent
for path in (PROJECT_ROOT, STEP23_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools_v2.explore_tools import detect_input_type, scan_directory  # noqa: E402
from tools_v2.extract_tools import extract_from_text  # noqa: E402
from tools_v2.io_tools import save_result_json  # noqa: E402

STEP23_MODES = {"auto", "directory", "ocr_fill"}
TEXT_COLUMN_HINTS = ("text", "report_text", "notes", "finding", "findings", "impression", "description", "narrative")
DIRECTORY_HINTS = ("mimic", "目录处理", "不需要ocr", "不要ocr", "跳过ocr", "结构化", "notes", "structured", "report_text")
OCR_FILL_HINTS = ("rawdata", "ocr回填", "ocr 回填", "图片ocr", "图片 ocr", "填回表格", "回填", "ocr提取")


def get_default_output_root() -> str:
    return str(PROJECT_ROOT / "program" / "output" / "step2_3_results")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _emit_default(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _write_csv_rows(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _read_csv_header_and_count(path: str | Path) -> tuple[list[str], int]:
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with Path(path).open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                return header, sum(1 for _ in reader)
        except UnicodeDecodeError:
            continue
    return [], 0


def _clean_ocr_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _emit_progress(
    label: str,
    completed: int,
    total: int,
    emit: Callable[[str], None],
    last_bucket: int,
) -> int:
    if total <= 0:
        return last_bucket
    percent = completed / total * 100
    bucket = int(percent // 5)
    if completed == 1 or completed == total or bucket > last_bucket:
        emit(f"[Step2-3][Progress] {label}: {completed}/{total} ({percent:.1f}%)")
        return bucket
    return last_bucket


def _read_csv_rows(path: str | Path, limit: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with Path(path).open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                header = list(reader.fieldnames or [])
                rows: list[dict[str, str]] = []
                for idx, row in enumerate(reader):
                    if limit is not None and idx >= limit:
                        break
                    rows.append({str(k): "" if v is None else str(v) for k, v in row.items() if k is not None})
                return header, rows
        except UnicodeDecodeError:
            continue
    return [], []


def _read_csv_header(path: str | Path) -> list[str]:
    header, _ = _read_csv_rows(path, limit=0)
    return header


def _has_text_column(header: list[str]) -> bool:
    lowered = [str(col).strip().lower() for col in header]
    return any(any(hint in col for hint in TEXT_COLUMN_HINTS) for col in lowered)


def _safe_name(value: str) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip())
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    return text.strip("_") or "value"


def _join_unique(values: list[Any], limit: int = 50000) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    total = 0
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if total + len(text) > limit:
            parts.append("<truncated>")
            break
        parts.append(text)
        total += len(text)
    return " | ".join(parts)


def _iter_csv_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def _inspect_step23_directory(input_path: str) -> dict[str, Any]:
    csv_files = _iter_csv_files(input_path)
    notes_csv_count = 0
    structured_csv_count = 0
    text_csv_count = 0
    sample_text_csvs: list[str] = []
    for csv_path in csv_files[:1000]:
        parts = {part.lower() for part in csv_path.parts}
        if "notes" in parts:
            notes_csv_count += 1
        if "structured" in parts:
            structured_csv_count += 1
        header = _read_csv_header(csv_path)
        if _has_text_column(header):
            text_csv_count += 1
            if len(sample_text_csvs) < 5:
                sample_text_csvs.append(str(csv_path))
    return {
        "csv_file_count": len(csv_files),
        "notes_csv_count": notes_csv_count,
        "structured_csv_count": structured_csv_count,
        "text_csv_count": text_csv_count,
        "sample_text_csvs": sample_text_csvs,
    }


def _discover_patient_dirs(input_path: str) -> dict[str, dict[str, Any]]:
    root = Path(input_path)
    patients: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return patients
    for child in sorted(root.iterdir()):
        table_dir = child / "table"
        if child.is_dir() and table_dir.is_dir():
            first_csv = next(iter(sorted(table_dir.rglob("*.csv"))), None)
            patients[child.name] = {
                "files": [],
                "file_count": 0,
                "table_path": str(first_csv or ""),
                "table_sub_files": [str(path) for path in sorted(table_dir.rglob("*.csv"))],
            }
    return patients


def _mode_from_text(text: str) -> str | None:
    compact = re.sub(r"[\s_\-—–]+", "", str(text or "").lower())
    if any(re.sub(r"[\s_\-—–]+", "", token.lower()) in compact for token in DIRECTORY_HINTS):
        return "directory"
    if any(re.sub(r"[\s_\-—–]+", "", token.lower()) in compact for token in OCR_FILL_HINTS):
        return "ocr_fill"
    return None


def _do_ocr_task(item: tuple[str, str, str]) -> tuple[str, str, str]:
    pid, path, filename = item
    try:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR(
            det_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
            rec_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
            cls_ort_config={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
        )
        result, _ = engine(path)
        if not result:
            return pid, filename, ""
        text = "\n".join(row[1] for row in result if len(row) >= 2)
        return pid, filename, _clean_ocr_text(text)
    except Exception:
        return pid, filename, ""


def resolve_step23_input(input_path: str, mode: str = "auto") -> dict[str, Any]:
    path = Path(str(input_path or "")).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Step2-3 输入不存在: {path}")
    if not path.is_dir():
        raise ValueError(f"Step2-3 现在只接受目录输入，不再支持单文件: {path}")

    resolved_mode = str(mode or "auto")
    if resolved_mode not in STEP23_MODES:
        raise ValueError(f"不支持的 Step2-3 mode: {mode}")

    info = detect_input_type(str(path))
    return {
        "input_path": str(path),
        "mode": resolved_mode,
        "data_type": info.get("data_type", "unknown"),
        "is_dir": path.is_dir(),
        "suffix": path.suffix.lower(),
    }


def scan_step23_input(input_path: str) -> dict[str, Any]:
    info = scan_directory(input_path)
    if not info.get("success"):
        raise ValueError(str(info.get("error") or "Step2-3 输入扫描失败"))
    patients = dict(info.get("patients") or {})
    if not patients:
        patients = _discover_patient_dirs(input_path)
    image_count = sum(
        1
        for pdata in patients.values()
        for item in pdata.get("files", [])
        if item.get("file_type") == "image"
    )
    table_count = sum(
        1
        for pdata in patients.values()
        if pdata.get("table_path") or pdata.get("table_sub_files")
    )
    stats = dict(info.get("statistics") or {})
    directory_profile = _inspect_step23_directory(input_path)
    return {
        "input_path": input_path,
        "patient_count": len(patients),
        "image_count": image_count,
        "table_patient_count": table_count,
        "table_path": info.get("table_path"),
        "statistics": stats,
        **directory_profile,
        "summary": info.get("summary", ""),
    }


def select_step23_pipeline(
    input_path: str,
    scan_artifacts: dict[str, Any] | None = None,
    requested_mode: str = "auto",
    user_hint: str = "",
) -> dict[str, Any]:
    requested = str(requested_mode or "auto")
    if requested not in STEP23_MODES:
        raise ValueError(f"不支持的 Step2-3 mode: {requested_mode}")
    if requested in {"directory", "ocr_fill"}:
        return {
            "input_path": input_path,
            "requested_mode": requested,
            "resolved_mode": requested,
            "mode_reason": "用户或上游 TaskSpec 显式指定。",
        }

    hinted = _mode_from_text(user_hint)
    if hinted:
        return {
            "input_path": input_path,
            "requested_mode": requested,
            "resolved_mode": hinted,
            "mode_reason": f"根据用户提示选择 {hinted}。",
        }

    artifacts = dict(scan_artifacts or scan_step23_input(input_path))
    notes = int(artifacts.get("notes_csv_count") or 0)
    structured = int(artifacts.get("structured_csv_count") or 0)
    text_csv = int(artifacts.get("text_csv_count") or 0)
    images = int(artifacts.get("image_count") or 0)
    if notes or structured or text_csv:
        reason = (
            "检测到 notes/structured 或 text/report_text 字段，"
            f"notes_csv={notes}, structured_csv={structured}, text_csv={text_csv}。"
        )
        return {
            "input_path": input_path,
            "requested_mode": requested,
            "resolved_mode": "directory",
            "mode_reason": reason,
        }
    return {
        "input_path": input_path,
        "requested_mode": requested,
        "resolved_mode": "ocr_fill",
        "mode_reason": f"未检测到目录文本/结构化表，按图片 OCR 回填处理，images={images}。",
    }


def run_step23_ocr(
    input_path: str,
    output_root: str | None = None,
    ocr_workers: int = 8,
    cache_path: str | None = None,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = emit or _emit_default
    output_root = output_root or get_default_output_root()
    ts = _timestamp()
    results_dir = Path(output_root) / f"results_{ts}"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_path) if cache_path else results_dir / "ocr_cache.json"

    existing_cache: dict[str, dict[str, str]] = {}
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            existing_cache = json.load(f)
        emit(f"[Step2-3][Runtime] OCR 发现已有缓存: {cache_file}")

    info = scan_directory(input_path)
    patients = info.get("patients", {})
    image_tasks = [
        (pid, item["path"], item["filename"])
        for pid, pdata in patients.items()
        for item in pdata.get("files", [])
        if item.get("file_type") == "image"
    ]
    processed_keys = {
        (str(pid), str(filename))
        for pid, texts in existing_cache.items()
        for filename in texts
    }
    pending_tasks = [
        task for task in image_tasks if (str(task[0]), str(task[2])) not in processed_keys
    ]
    emit(
        "[Step2-3][Runtime] OCR 开始: "
        f"images={len(image_tasks)}, cached={len(processed_keys)}, pending={len(pending_tasks)}, workers={ocr_workers}"
    )

    ocr_cache: dict[str, dict[str, str]] = {str(pid): {} for pid in patients}
    for pid, texts in existing_cache.items():
        ocr_cache.setdefault(str(pid), {}).update({str(k): str(v) for k, v in dict(texts).items()})
    success_count = sum(len(v) for v in ocr_cache.values())
    fail_count = 0
    completed = 0
    last_bucket = -1
    if pending_tasks:
        save_every = max(1, min(50, len(pending_tasks) // 20 or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(ocr_workers))) as pool:
            futures = [pool.submit(_do_ocr_task, task) for task in pending_tasks]
            for future in concurrent.futures.as_completed(futures):
                pid, filename, text = future.result()
                completed += 1
                if text:
                    ocr_cache.setdefault(str(pid), {})[str(filename)] = text
                    success_count += 1
                else:
                    fail_count += 1
                last_bucket = _emit_progress("OCR", completed, len(pending_tasks), emit, last_bucket)
                if completed % save_every == 0:
                    _write_json_atomic(cache_file, ocr_cache)

    _write_json_atomic(cache_file, ocr_cache)
    emit(f"[Step2-3][Runtime] OCR 完成: success={success_count}, failed={fail_count}")
    return {
        "input_path": input_path,
        "output_root": output_root,
        "results_dir": str(results_dir),
        "ocr_cache_path": str(cache_file),
        "cache_used": bool(existing_cache),
        "ocr_patient_count": len(ocr_cache),
        "total_images": len(image_tasks),
        "successful_ocr": success_count,
        "failed_ocr": fail_count,
    }


async def run_step23_extraction(
    ocr_cache_path: str,
    results_dir: str,
    concurrency: int = 8,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = emit or _emit_default
    with Path(ocr_cache_path).open("r", encoding="utf-8") as f:
        ocr_cache: dict[str, dict[str, str]] = json.load(f)

    tasks: list[dict[str, str]] = []
    for pid, texts in ocr_cache.items():
        for filename, text in texts.items():
            if text and len(text.strip()) >= 50:
                tasks.append({"patient_id": pid, "filename": filename, "text": text})
    emit(f"[Step2-3][Runtime] LLM 抽取开始: texts={len(tasks)}, concurrency={concurrency}")

    sem = asyncio.Semaphore(max(1, int(concurrency)))
    lock = asyncio.Lock()
    entity_rows: list[dict[str, Any]] = []
    stats = {"success": 0, "failed": 0, "total_entities": 0}

    async def _extract_one(task: dict[str, str]) -> None:
        async with sem:
            result = await extract_from_text(task["text"])
        async with lock:
            if result.get("success") and result.get("entities"):
                for ent in result["entities"]:
                    row = {"patient_id": task["patient_id"], "source_file": task["filename"], "source_category": "ocr"}
                    row.update(ent)
                    entity_rows.append(row)
                stats["success"] += 1
                stats["total_entities"] += len(result["entities"])
            else:
                stats["failed"] += 1

    await asyncio.gather(*[_extract_one(task) for task in tasks])
    out_dir = Path(results_dir)
    ts = _timestamp()
    entities_csv = out_dir / f"entities_{ts}.csv"
    patients_csv = out_dir / f"patients_{ts}.csv"
    summary_json = out_dir / f"summary_{ts}.json"

    entity_columns = ["patient_id", "name", "category", "value", "unit", "source_file", "source_category"]
    for row in entity_rows:
        for key in row:
            if key not in entity_columns:
                entity_columns.append(key)
    _write_csv_rows(entities_csv, entity_rows, entity_columns)

    patient_rows = []
    for pid, texts in ocr_cache.items():
        patient_entities = [row for row in entity_rows if str(row.get("patient_id")) == str(pid)]
        patient_rows.append(
            {
                "patient_id": pid,
                "ocr_file_count": len(texts),
                "entity_count": len(patient_entities),
                "categories": "; ".join(sorted({str(row.get("category", "")) for row in patient_entities if row.get("category")})),
            }
        )
    _write_csv_rows(patients_csv, patient_rows, ["patient_id", "ocr_file_count", "entity_count", "categories"])

    summary = {
        "success": True,
        "processed_at": datetime.now().isoformat(),
        "statistics": {
            "total_patients": len(ocr_cache),
            "total_ocr_texts": sum(len(v) for v in ocr_cache.values()),
            "extracted_texts": stats["success"],
            "failed_texts": stats["failed"],
            "total_entities": stats["total_entities"],
        },
        "entities_csv": str(entities_csv),
        "patients_csv": str(patients_csv),
    }
    save_result_json(summary, str(summary_json))
    emit(f"[Step2-3][Runtime] LLM 抽取完成: entities={stats['total_entities']}")
    return {
        "results_dir": str(out_dir),
        "entities_csv": str(entities_csv),
        "patients_csv": str(patients_csv),
        "summary_json": str(summary_json),
        **summary["statistics"],
    }


def _table_dir_for_patient(input_path: str, patient_id: str, pdata: dict[str, Any]) -> Path | None:
    table_path = pdata.get("table_path")
    if table_path:
        parent = Path(str(table_path)).parent
        if parent.name.lower() in {"structured", "notes"}:
            parent = parent.parent
        if parent.is_dir():
            return parent
    candidate = Path(input_path) / str(patient_id) / "table"
    return candidate if candidate.is_dir() else None


def _load_patient_table_data(table_dir: Path) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    structured: dict[str, list[dict[str, str]]] = {}
    text_sources: list[dict[str, Any]] = []
    if not table_dir.is_dir():
        return structured, text_sources

    for csv_path in _iter_csv_files(table_dir):
        rel = csv_path.relative_to(table_dir)
        parts = [part.lower() for part in rel.parts]
        header, rows = _read_csv_rows(csv_path)
        if not header:
            continue
        if "structured" in parts:
            key = csv_path.stem
            structured.setdefault(key, []).extend(rows)
            continue
        text_cols = [
            col for col in header
            if any(hint in col.strip().lower() for hint in TEXT_COLUMN_HINTS)
        ]
        if text_cols:
            text_sources.append(
                {
                    "path": str(csv_path),
                    "relative_path": str(rel),
                    "source_category": "notes" if "notes" in parts else "table",
                    "rows": rows,
                    "text_columns": text_cols,
                }
            )
        else:
            key = csv_path.stem
            structured.setdefault(key, []).extend(rows)
    return structured, text_sources


def _row_hadm(row: dict[str, Any]) -> str:
    raw = str(row.get("hadm_id", "") or "").strip()
    return raw.split(".")[0] if raw else ""


def _rows_for_hadm(rows: list[dict[str, str]], hadm_id: str) -> list[dict[str, str]]:
    if not hadm_id:
        return rows
    matched = [row for row in rows if _row_hadm(row) == hadm_id]
    return matched or [row for row in rows if not _row_hadm(row)]


def _structured_value_columns(rows: list[dict[str, str]]) -> list[str]:
    excluded = {"subject_id", "hadm_id", "stay_id", "note_id", "dicom_id", "study_id"}
    columns: list[str] = []
    for row in rows:
        for col in row:
            if col not in columns and col.lower() not in excluded:
                columns.append(col)
    return columns


def _make_entity_column(entity: dict[str, Any]) -> str:
    category = _safe_name(str(entity.get("category") or "entity"))
    name = _safe_name(str(entity.get("name") or "value"))
    return f"entity__{category}__{name}"


async def run_step23_directory_pipeline(
    input_path: str,
    output_root: str | None = None,
    concurrency: int = 8,
    use_llm: bool = True,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = emit or _emit_default
    output_root = output_root or get_default_output_root()
    ts = _timestamp()
    results_dir = Path(output_root) / f"results_{ts}"
    results_dir.mkdir(parents=True, exist_ok=True)

    info = scan_directory(input_path)
    if not info.get("success"):
        raise ValueError(str(info.get("error") or "Step2-3 directory 扫描失败"))
    patients = dict(info.get("patients") or {})
    if not patients:
        patients = _discover_patient_dirs(input_path)
    emit(f"[Step2-3][Runtime] directory pipeline 开始: patients={len(patients)}, concurrency={concurrency}")

    patient_payloads: dict[str, dict[str, Any]] = {}
    text_tasks: list[dict[str, str]] = []
    for pid, pdata in patients.items():
        table_dir = _table_dir_for_patient(input_path, str(pid), dict(pdata or {}))
        structured, text_sources = _load_patient_table_data(table_dir) if table_dir else ({}, [])
        patient_payloads[str(pid)] = {
            "structured": structured,
            "text_sources": text_sources,
            "table_dir": str(table_dir or ""),
        }
        for source in text_sources:
            source_file = Path(source["path"]).name
            for row_idx, row in enumerate(source["rows"]):
                hadm_id = _row_hadm(row)
                row_id = str(row.get("note_id") or row.get("dicom_id") or row.get("study_id") or row_idx)
                for col in source["text_columns"]:
                    text = str(row.get(col, "") or "").strip()
                    if text:
                        text_tasks.append(
                            {
                                "patient_id": str(pid),
                                "hadm_id": hadm_id,
                                "row_id": row_id,
                                "source_file": source_file,
                                "source_category": str(source["source_category"]),
                                "column": str(col),
                                "text": text,
                            }
                        )

    entities_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    impressions_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    clinical_text_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    entity_rows: list[dict[str, Any]] = []

    for task in text_tasks:
        key = (task["patient_id"], task["hadm_id"])
        clinical_text_by_key[key].append(task["text"])

    if use_llm and text_tasks:
        emit(f"[Step2-3][Runtime] directory 文本抽取开始: texts={len(text_tasks)}, concurrency={concurrency}")
        sem = asyncio.Semaphore(max(1, int(concurrency)))
        lock = asyncio.Lock()
        completed = 0
        last_bucket = -1

        async def _extract_one(task: dict[str, str]) -> None:
            nonlocal completed, last_bucket
            try:
                async with sem:
                    result = await extract_from_text(task["text"])
            except Exception:
                result = {"success": False, "entities": []}
            async with lock:
                key = (task["patient_id"], task["hadm_id"])
                if result.get("success"):
                    for ent in result.get("entities") or []:
                        row = {
                            "patient_id": task["patient_id"],
                            "hadm_id": task["hadm_id"],
                            "source_file": task["source_file"],
                            "source_category": task["source_category"],
                            "row_id": task["row_id"],
                            "text_column": task["column"],
                        }
                        row.update(ent)
                        entity_rows.append(row)
                        entities_by_key[key].append(row)
                    if result.get("impression"):
                        impressions_by_key[key].append(str(result["impression"]))
                completed += 1
                last_bucket = _emit_progress("directory text extraction", completed, len(text_tasks), emit, last_bucket)

        await asyncio.gather(*[_extract_one(task) for task in text_tasks])
    else:
        emit(f"[Step2-3][Runtime] directory 文本抽取跳过: use_llm={use_llm}, texts={len(text_tasks)}")

    wide_rows: list[dict[str, Any]] = []
    for pid, payload in patient_payloads.items():
        structured = payload["structured"]
        patient_base: dict[str, Any] = {"subject_id": pid}
        patient_rows = structured.get("patients") or []
        if patient_rows:
            for col, value in patient_rows[0].items():
                if str(value).strip():
                    patient_base[col] = value

        admissions = structured.get("admissions") or [{}]
        for admission in admissions:
            hadm_id = _row_hadm(admission)
            row: dict[str, Any] = dict(patient_base)
            if hadm_id:
                row["hadm_id"] = hadm_id
            for col, value in admission.items():
                if str(value).strip():
                    row[col] = value

            for table_name, rows in structured.items():
                if table_name in {"patients", "admissions"}:
                    continue
                matched_rows = _rows_for_hadm(rows, hadm_id)
                prefix = f"structured__{_safe_name(table_name)}"
                row[f"{prefix}__row_count"] = len(matched_rows)
                for col in _structured_value_columns(matched_rows):
                    values = [item.get(col, "") for item in matched_rows]
                    row[f"{prefix}__{_safe_name(col)}"] = _join_unique(values)

            key = (pid, hadm_id)
            all_patient_text = list(clinical_text_by_key.get(key) or [])
            if not all_patient_text and not hadm_id:
                for (text_pid, _), values in clinical_text_by_key.items():
                    if text_pid == pid:
                        all_patient_text.extend(values)
            if all_patient_text:
                row["clinical_text"] = _join_unique(all_patient_text, limit=80000)
                row["clinical_text_count"] = len(all_patient_text)
            if impressions_by_key.get(key):
                row["impressions"] = _join_unique(impressions_by_key[key])
            for ent in entities_by_key.get(key, []):
                col = _make_entity_column(ent)
                value = ent.get("value") or ent.get("name") or ""
                row[col] = _join_unique([row.get(col, ""), value])
            wide_rows.append(row)

    if not wide_rows:
        wide_rows = [{"subject_id": pid} for pid in patient_payloads]

    wide_columns: list[str] = []
    seen_cols: set[str] = set()
    preferred = ["subject_id", "hadm_id"]
    for col in preferred:
        if any(col in row for row in wide_rows):
            wide_columns.append(col)
            seen_cols.add(col)
    for row in wide_rows:
        for col in row:
            if col not in seen_cols:
                seen_cols.add(col)
                wide_columns.append(col)

    admission_wide_csv = results_dir / f"admission_wide_{ts}.csv"
    entities_long_csv = results_dir / f"entities_long_{ts}.csv"
    summary_json = results_dir / f"summary_{ts}.json"
    _write_csv_rows(admission_wide_csv, wide_rows, wide_columns)
    entity_cols = ["patient_id", "hadm_id", "source_file", "source_category", "row_id", "text_column", "category", "name", "value", "unit"]
    for row in entity_rows:
        for col in row:
            if col not in entity_cols:
                entity_cols.append(col)
    _write_csv_rows(entities_long_csv, entity_rows, entity_cols)

    summary = {
        "success": True,
        "resolved_mode": "directory",
        "processed_at": datetime.now().isoformat(),
        "input_path": input_path,
        "results_dir": str(results_dir),
        "patient_count": len(patients),
        "text_task_count": len(text_tasks),
        "entity_count": len(entity_rows),
        "wide_row_count": len(wide_rows),
        "wide_column_count": len(wide_columns),
        "admission_wide_csv": str(admission_wide_csv),
        "entities_long_csv": str(entities_long_csv),
    }
    save_result_json(summary, str(summary_json))
    emit(
        "[Step2-3][Runtime] directory pipeline 完成: "
        f"rows={len(wide_rows)}, columns={len(wide_columns)}, entities={len(entity_rows)}"
    )
    return {
        "input_path": input_path,
        "output_root": output_root,
        "results_dir": str(results_dir),
        "resolved_mode": "directory",
        "mode_reason": "使用目录结构化 pipeline。",
        "admission_wide_csv": str(admission_wide_csv),
        "entities_long_csv": str(entities_long_csv),
        "summary_json": str(summary_json),
        **summary,
    }


async def run_step23_fill_table(
    input_path: str,
    results_dir: str,
    use_llm: bool = True,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = emit or _emit_default
    from fill_table import run as fill_run

    filled_dir = Path(results_dir) / "filled_tables"
    emit(f"[Step2-3][Runtime] 回填开始: {filled_dir}")
    await fill_run(
        reorg_dir=input_path,
        results_dir=results_dir,
        output_dir=str(filled_dir),
        use_llm=use_llm,
        dry_run=False,
        verbose=False,
    )
    merged_csv = filled_dir / "filled_merged.csv"
    merged_xlsx = filled_dir / "filled_merged_highlighted.xlsx"
    return {
        "results_dir": results_dir,
        "filled_tables_dir": str(filled_dir),
        "filled_merged_csv": str(merged_csv) if merged_csv.exists() else "",
        "filled_merged_xlsx": str(merged_xlsx) if merged_xlsx.exists() else "",
    }


def normalize_step23_outputs(
    output_root: str | None = None,
    filled_merged_csv: str | None = None,
    fallback_csv: str | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = output_root or get_default_output_root()
    source = Path(filled_merged_csv or "") if filled_merged_csv else None
    if source is None or not source.is_file():
        source = Path(fallback_csv or "") if fallback_csv else None
    if source is None or not source.is_file():
        raise FileNotFoundError("Step2-3 缺少可传给 Step4 的 CSV 产物。")

    next_dir = Path(output_root) / "next_input"
    next_dir.mkdir(parents=True, exist_ok=True)
    next_csv = next_dir / "input.csv"
    shutil.copy2(source, next_csv)
    summary_path = next_dir / "summary.json"
    payload = dict(summary or {})
    payload.update({"source_csv": str(source), "next_input_csv": str(next_csv)})
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    header, row_count = _read_csv_header_and_count(next_csv)
    return {
        "next_input_dir": str(next_dir),
        "next_input_csv": str(next_csv),
        "next_summary_json": str(summary_path),
        "source_csv": str(source),
        "output_column_count": len(header),
        "output_row_count": row_count,
    }


def validate_step23_output(next_input_csv: str, next_summary_json: str | None = None) -> dict[str, Any]:
    path = Path(next_input_csv)
    issues: list[str] = []
    if not path.is_file():
        issues.append(f"next_input/input.csv 不存在: {path}")
        return {"passed": False, "issues": issues, "details": {"next_input_csv": str(path)}}
    header, row_count = _read_csv_header_and_count(path)
    if not header:
        issues.append(f"next_input/input.csv 无表头: {path}")
    if row_count < 1:
        issues.append(f"next_input/input.csv 无数据行: {path}")
    if next_summary_json and not Path(next_summary_json).is_file():
        issues.append(f"summary.json 不存在: {next_summary_json}")
    return {
        "passed": not issues,
        "issues": issues,
        "details": {
            "next_input_csv": str(path),
            "next_summary_json": next_summary_json or "",
            "output_column_count": len(header),
            "output_row_count": row_count,
        },
    }
