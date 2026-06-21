from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DYNAMIC_OBJECT_PATHS = {"实验室检验"}

CSV_SUFFIXES = {".csv", ".tsv"}
SUPPORTED_GOLD_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".xlsx", ".xls"}
TABLE_ALIASES = {
    "admissions": ("structured/admissions.csv", "hosp/admissions.csv"),
    "patients": ("structured/patients.csv", "hosp/patients.csv"),
    "diagnoses_icd": ("structured/diagnoses_icd.csv", "hosp/diagnoses_icd.csv"),
    "d_icd_diagnoses": ("structured/d_icd_diagnoses.csv", "hosp/d_icd_diagnoses.csv"),
    "prescriptions": ("structured/prescriptions.csv", "hosp/prescriptions.csv"),
    "labevents": ("structured/labevents.csv", "hosp/labevents.csv"),
    "d_labitems": ("structured/d_labitems.csv", "hosp/d_labitems.csv"),
    "chartevents": ("icu/chartevents.csv",),
    "omr": ("hosp/omr.csv", "structured/omr.csv"),
    "discharge_notes": ("notes/discharge_notes.csv",),
    "radiology_notes": ("notes/radiology_notes.csv",),
}


def load_gold_examples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    gold_path = Path(path)
    suffix = gold_path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_examples(gold_path, limit=limit)
    if suffix == ".json":
        return load_json_examples(gold_path, limit=limit)
    if suffix in {".csv", ".tsv"}:
        return load_delimited_examples(gold_path, limit=limit)
    if suffix in {".xlsx", ".xls"}:
        return load_excel_examples(gold_path, limit=limit)
    raise ValueError(f"Unsupported gold examples format: {gold_path.suffix}")


def load_jsonl_examples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
            if limit is not None and len(records) >= limit:
                break
    return records


def load_json_examples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        records = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        if isinstance(data.get("records"), list):
            records = [item for item in data["records"] if isinstance(item, dict)]
        else:
            records = [data]
    else:
        records = []
    return records[:limit] if limit is not None else records


def load_delimited_examples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    table_path = Path(path)
    dialect = "excel-tab" if table_path.suffix.lower() == ".tsv" else "excel"
    records: list[dict[str, Any]] = []
    with table_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            records.append({str(key): _clean_table_value(value) for key, value in row.items() if key})
            if limit is not None and len(records) >= limit:
                break
    return records


def load_excel_examples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise ValueError("Reading Excel gold examples requires pandas.") from exc
    kwargs = {"nrows": limit} if limit is not None else {}
    frame = pd.read_excel(path, **kwargs)
    frame = frame.where(frame.notna(), None)
    return [
        {str(column): _clean_table_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def summarize_gold_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    field_stats: dict[str, dict[str, Any]] = {}

    def add_leaf(path: list[str], value: Any) -> None:
        if not path:
            return
        field_path = ".".join(path)
        stat = field_stats.setdefault(
            field_path,
            {
                "field_path": field_path,
                "present_count": 0,
                "value_types": set(),
                "sample_values": [],
                "dynamic_segments": [],
            },
        )
        stat["present_count"] += 1
        stat["value_types"].add(type(value).__name__)
        sample = _safe_sample_value(value)
        if sample not in stat["sample_values"] and len(stat["sample_values"]) < 5:
            stat["sample_values"].append(sample)
        dynamic_segments = _dynamic_segments_for_path(path)
        for segment in dynamic_segments:
            if segment not in stat["dynamic_segments"]:
                stat["dynamic_segments"].append(segment)

    def walk(value: Any, path: list[str]) -> None:
        if isinstance(value, dict):
            if not value:
                return
            if ".".join(path) in DYNAMIC_OBJECT_PATHS:
                for child in value.values():
                    walk(child, [*path, "*"])
                return
            for key, child in value.items():
                walk(child, [*path, str(key)])
            return
        if isinstance(value, list):
            list_path = _list_path(path)
            if not value:
                return
            for child in value:
                walk(child, list_path)
            return
        add_leaf(path, value)

    for record in records:
        walk(record, [])

    fields: list[dict[str, Any]] = []
    for field_path in sorted(field_stats):
        stat = field_stats[field_path]
        fields.append(
            {
                "field_path": stat["field_path"],
                "present_count": stat["present_count"],
                "value_types": sorted(stat["value_types"]),
                "sample_values": stat["sample_values"],
                "dynamic_segments": stat["dynamic_segments"],
            }
        )
    return {
        "record_count": len(records),
        "field_count": len(fields),
        "fields": fields,
    }


def build_gold_guidance(
    task_text: str,
    raw_data_root: str = "",
    gold_examples_path: str = "",
    prior_report_path: str = "",
    output_dir: str = "",
    learned_rules_dir: str = "",
) -> dict[str, Any]:
    resolved_raw_root, resolved_gold_path, resolved_prior_report = resolve_guidance_inputs(
        task_text=task_text,
        raw_data_root=raw_data_root,
        gold_examples_path=gold_examples_path,
        prior_report_path=prior_report_path,
    )
    if not resolved_raw_root:
        raise ValueError("raw_data_root is required or must be discoverable from task_text.")
    if not resolved_gold_path:
        raise ValueError("gold_examples_path is required or must be discoverable from task_text.")

    records = load_gold_examples(resolved_gold_path)
    schema_summary = summarize_gold_schema(records)
    source_schema = build_source_schema(resolved_raw_root)
    prior_recall = parse_gold_recall_report(resolved_prior_report) if resolved_prior_report else {}

    field_provenance = [
        infer_field_provenance(field["field_path"], source_schema, prior_recall)
        for field in schema_summary["fields"]
        if not _is_internal_source_metadata(field["field_path"])
    ]
    rules = [_rule_from_provenance(item) for item in field_provenance]
    rules = _dedupe_rules(rules)
    brief = build_planner_brief(
        task_text=task_text,
        raw_data_root=resolved_raw_root,
        gold_examples_path=resolved_gold_path,
        field_provenance=field_provenance,
        rules=rules,
    )

    report = {
        "task": task_text,
        "raw_data_root": resolved_raw_root,
        "gold_examples_path": resolved_gold_path,
        "prior_report_path": resolved_prior_report,
        "gold_schema": schema_summary,
        "source_schema": source_schema,
        "prior_recall_report": prior_recall,
        "field_provenance": field_provenance,
        "planner_extraction_brief": brief,
        "learned_rules": {
            "rule_count": len(rules),
            "rules": rules,
        },
        "feedback_template": {
            "field_level": [
                "target_field_path",
                "missing_count",
                "source_files",
                "source_columns",
                "suggested_derivation_logic",
            ],
            "sample_level": [
                "subject_id",
                "hadm_id",
                "target_field_path",
                "expected_source",
                "current_output_field",
            ],
        },
    }

    artifacts: dict[str, str] = {}
    if output_dir:
        artifacts.update(_write_guidance_artifacts(Path(output_dir), report, schema_summary, field_provenance, brief))
    if learned_rules_dir:
        rules_path = write_learned_rules(
            Path(learned_rules_dir),
            rules,
            filename=f"initial_gold_guidance_{_slug(task_text)}.json",
            status="draft",
        )
        artifacts["learned_rules_path"] = str(rules_path)
    report["artifacts"] = artifacts
    if "learned_rules_path" in artifacts:
        report["planner_extraction_brief"]["learned_rules_path"] = artifacts["learned_rules_path"]
    return report


def build_gold_guidance_from_analysis(
    task_text: str,
    analysis_findings: dict[str, Any],
    output_dir: str = "",
    learned_rules_dir: str = "",
) -> dict[str, Any]:
    """Write standard guidance artifacts from findings collected by atomic tools."""
    if not isinstance(analysis_findings, dict):
        raise ValueError("analysis_findings must be a JSON object")
    raw_provenance = analysis_findings.get("field_provenance")
    if not isinstance(raw_provenance, list) or not raw_provenance:
        raise ValueError("analysis_findings.field_provenance must be a non-empty list")

    field_provenance: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw_provenance):
        if not isinstance(candidate, dict):
            raise ValueError(f"field_provenance[{index}] must be an object")
        target = str(candidate.get("target_field_path") or "").strip()
        if not target:
            raise ValueError(f"field_provenance[{index}].target_field_path is required")
        source_files = _as_list(candidate.get("source_files"))
        source_columns = _as_list(candidate.get("source_columns"))
        source_status = str(candidate.get("source_status") or "").strip()
        if not source_status:
            source_status = "source_schema_available" if source_files else "source_schema_not_found"
        field_provenance.append(
            {
                "target_field_path": target,
                "source_files": source_files,
                "source_columns": source_columns,
                "join_keys": _as_list(candidate.get("join_keys")),
                "derivation_logic": str(
                    candidate.get("derivation_logic")
                    or "需要根据原子工具收集的证据补充抽取逻辑。"
                ),
                "source_status": source_status,
                "missing_source_columns": _as_list(candidate.get("missing_source_columns")),
                "confidence": str(candidate.get("confidence") or "medium"),
                "prior_recall": candidate.get("prior_recall") or {},
                "evidence": candidate.get("evidence") or [],
            }
        )

    schema_summary = analysis_findings.get("gold_schema")
    if not isinstance(schema_summary, dict):
        schema_summary = {
            "record_count": int(analysis_findings.get("record_count") or 0),
            "field_count": len(field_provenance),
            "fields": [
                {"field_path": item["target_field_path"]}
                for item in field_provenance
            ],
        }
    schema_summary = dict(schema_summary)
    schema_summary.setdefault("field_count", len(field_provenance))
    schema_summary.setdefault("fields", [])

    rules = _dedupe_rules([_rule_from_provenance(item) for item in field_provenance])
    raw_data_root = str(analysis_findings.get("raw_data_root") or "")
    gold_examples_path = str(analysis_findings.get("gold_examples_path") or "")
    brief = build_planner_brief(
        task_text=task_text,
        raw_data_root=raw_data_root,
        gold_examples_path=gold_examples_path,
        field_provenance=field_provenance,
        rules=rules,
    )
    report = {
        "task": task_text,
        "raw_data_root": raw_data_root,
        "gold_examples_path": gold_examples_path,
        "prior_report_path": str(analysis_findings.get("prior_report_path") or ""),
        "gold_schema": schema_summary,
        "source_schema": analysis_findings.get("source_schema") or {},
        "prior_recall_report": analysis_findings.get("prior_recall_report") or {},
        "field_provenance": field_provenance,
        "planner_extraction_brief": brief,
        "learned_rules": {"rule_count": len(rules), "rules": rules},
        "analysis_method": "atomic_tools",
        "feedback_template": {
            "field_level": [
                "target_field_path",
                "missing_count",
                "source_files",
                "source_columns",
                "suggested_derivation_logic",
            ],
            "sample_level": [
                "subject_id",
                "hadm_id",
                "target_field_path",
                "expected_source",
                "current_output_field",
            ],
        },
    }
    artifacts: dict[str, str] = {}
    if output_dir:
        artifacts.update(
            _write_guidance_artifacts(
                Path(output_dir),
                report,
                schema_summary,
                field_provenance,
                brief,
            )
        )
    if learned_rules_dir:
        rules_path = write_learned_rules(
            Path(learned_rules_dir),
            rules,
            filename=f"initial_gold_guidance_{_slug(task_text)}.json",
            status="draft",
        )
        artifacts["learned_rules_path"] = str(rules_path)
    report["artifacts"] = artifacts
    if "learned_rules_path" in artifacts:
        report["planner_extraction_brief"]["learned_rules_path"] = artifacts["learned_rules_path"]
    return report


def resolve_guidance_inputs(
    task_text: str,
    raw_data_root: str = "",
    gold_examples_path: str = "",
    prior_report_path: str = "",
) -> tuple[str, str, str]:
    paths = [Path(p) for p in re.findall(r"/[^\s，,;；]+", task_text or "")]
    existing = [p for p in paths if p.exists()]
    raw = Path(raw_data_root) if raw_data_root else None
    gold = Path(gold_examples_path) if gold_examples_path else None
    prior = Path(prior_report_path) if prior_report_path else None

    if raw is None:
        for path in existing:
            if path.is_dir() and _looks_like_raw_data_root(path):
                raw = path
                break
    if gold is None:
        for path in existing:
            if path.is_file() and path.suffix.lower() in SUPPORTED_GOLD_SUFFIXES:
                gold = path
                break
    if prior is None:
        for path in existing:
            if path.is_file() and path.suffix.lower() == ".md":
                prior = path
                break

    return (
        str(raw.resolve()) if raw and raw.exists() else "",
        str(gold.resolve()) if gold and gold.exists() else "",
        str(prior.resolve()) if prior and prior.exists() else "",
    )


def build_source_schema(raw_data_root: str | Path) -> dict[str, Any]:
    root = Path(raw_data_root)
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in CSV_SUFFIXES and suffix not in {".xlsx", ".xls"}:
            continue
        rel = path.relative_to(root).as_posix()
        columns = _read_columns(path)
        files.append(
            {
                "path": str(path.resolve()),
                "relative_path": rel,
                "columns": columns,
                "row_count": _count_rows(path) if suffix in CSV_SUFFIXES else None,
            }
        )
    return {
        "raw_data_root": str(root.resolve()),
        "file_count": len(files),
        "files": files,
    }


def infer_field_provenance(
    field_path: str,
    source_schema: dict[str, Any],
    prior_recall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _mapping_spec_for_field(field_path)
    prior = _prior_entry_for_field(field_path, prior_recall or {})
    if prior:
        spec = _merge_prior_spec(spec, prior)

    candidate_files = spec.get("source_files", [])
    expected_columns = spec.get("source_columns", [])
    available_files = [_resolve_table_alias(source_schema, file_name) for file_name in candidate_files]
    available_files = [item for item in available_files if item is not None]

    source_files = _dedupe_strings([item["relative_path"] for item in available_files]) or _dedupe_strings(candidate_files)
    available_columns = sorted({column for item in available_files for column in item.get("columns", [])})
    missing_columns = [column for column in expected_columns if column not in available_columns]

    if not available_files:
        status = "source_schema_not_found"
    elif missing_columns:
        status = "source_column_missing"
    else:
        status = "source_schema_available"

    return {
        "target_field_path": field_path,
        "source_files": source_files,
        "source_columns": expected_columns,
        "join_keys": spec.get("join_keys", []),
        "derivation_logic": spec.get("derivation_logic", "需要根据字段语义和源表样本补充抽取逻辑。"),
        "source_status": status,
        "missing_source_columns": missing_columns,
        "confidence": spec.get("confidence", "medium" if candidate_files else "low"),
        "prior_recall": prior,
    }


def build_planner_brief(
    task_text: str,
    raw_data_root: str,
    gold_examples_path: str,
    field_provenance: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> dict[str, Any]:
    available = [item for item in field_provenance if item["source_status"] == "source_schema_available"]
    missing = [item for item in field_provenance if item["source_status"] != "source_schema_available"]
    return {
        "task": task_text,
        "raw_data_root": raw_data_root,
        "gold_examples_path": gold_examples_path,
        "target_field_count": len(field_provenance),
        "available_field_count": len(available),
        "unavailable_field_count": len(missing),
        "high_priority_sources": _source_counts(field_provenance),
        "extraction_targets": rules,
        "unavailable_targets": [
            {
                "target_field_path": item["target_field_path"],
                "source_files": item["source_files"],
                "missing_source_columns": item["missing_source_columns"],
                "source_status": item["source_status"],
            }
            for item in missing
        ],
    }


def write_learned_rules(
    rules_dir: str | Path,
    rules: list[dict[str, Any]],
    filename: str = "",
    status: str | None = None,
) -> Path:
    target_dir = Path(rules_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    final_rules = []
    for rule in rules:
        item = dict(rule)
        if status is not None:
            item["status"] = status
        item.setdefault("rule_id", _rule_id(item))
        item.setdefault("created_at", _now())
        final_rules.append(item)
    if not filename:
        filename = f"learned_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = target_dir / filename
    _write_json(path, {"rules": _dedupe_rules(final_rules), "created_at": _now()})
    return path


def load_learned_rules(
    rules_dir: str | Path,
    include_statuses: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    target_dir = Path(rules_dir)
    statuses = set(include_statuses or {"active", "frozen"})
    rules: list[dict[str, Any]] = []
    if target_dir.is_file():
        candidate_paths = [target_dir]
    elif target_dir.is_dir():
        candidate_paths = sorted(target_dir.glob("*.json"))
    else:
        candidate_paths = []
    for path in candidate_paths:
        rules.extend(_read_rules_file(path))
    filtered = [rule for rule in rules if rule.get("status", "draft") in statuses]
    filtered = _dedupe_rules(filtered)
    return {
        "rules_dir": str(target_dir.resolve()),
        "include_statuses": sorted(statuses),
        "rule_count": len(filtered),
        "rules": filtered,
    }


def promote_feedback_rules(
    feedback_report_path: str | Path,
    rules_dir: str | Path,
    min_metric_delta: float = 0.0,
) -> dict[str, Any]:
    feedback = _read_json(Path(feedback_report_path), {})
    candidates = feedback.get("rules")
    if not isinstance(candidates, list):
        candidates = feedback.get("field_gaps") if isinstance(feedback.get("field_gaps"), list) else []

    promoted: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        delta = float(candidate.get("validation_metric_delta") or 0)
        if delta < min_metric_delta:
            continue
        rule = _normalize_rule(candidate)
        rule["status"] = "active"
        rule["created_from_run"] = rule.get("created_from_run") or str(feedback_report_path)
        rule["rule_id"] = _rule_id(rule)
        rule["created_at"] = _now()
        promoted.append(rule)

    path = None
    if promoted:
        path = write_learned_rules(
            rules_dir,
            promoted,
            filename=f"promoted_validation_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    return {
        "feedback_report_path": str(Path(feedback_report_path).resolve()),
        "rules_dir": str(Path(rules_dir).resolve()),
        "promoted_count": len(promoted),
        "rules_path": str(path) if path else "",
        "rules": promoted,
    }


def freeze_rules(rules_dir: str | Path) -> dict[str, Any]:
    loaded = load_learned_rules(rules_dir, include_statuses={"draft", "active", "frozen"})
    frozen = []
    for rule in loaded["rules"]:
        item = dict(rule)
        item["status"] = "frozen"
        item["frozen_at"] = _now()
        item["rule_id"] = _rule_id(item)
        frozen.append(item)
    path = write_learned_rules(rules_dir, frozen, filename="frozen_rules.json")
    return {
        "rules_dir": str(Path(rules_dir).resolve()),
        "frozen_count": len(frozen),
        "rules_path": str(path),
        "rules": frozen,
    }


def parse_gold_recall_report(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    report_path = Path(path)
    if not report_path.exists():
        return {}
    text = report_path.read_text(encoding="utf-8")
    summary: dict[str, Any] = {
        "report_path": str(report_path.resolve()),
        "matched_fields": [],
        "missing_fields": [],
    }
    for key in ("visible cases", "recall", "matched / expected"):
        match = re.search(rf"-\s*{re.escape(key)}：?`?([^`\n]+)`?", text)
        if match:
            summary[key.replace(" ", "_").replace("/", "_")] = match.group(1).strip()
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## 命中字段"):
            current = "matched_fields"
            continue
        if line.startswith("## 缺失字段"):
            current = "missing_fields"
            continue
        if not current or not line.startswith("|") or "---" in line or "Gold 字段" in line:
            continue
        cells = [_strip_md_cell(cell) for cell in line.strip().strip("|").split("|")]
        if current == "matched_fields" and len(cells) >= 6:
            summary[current].append(
                {
                    "gold_field": cells[0],
                    "matched_output_field": cells[1],
                    "source_files": cells[2],
                    "source_columns": cells[3],
                    "derivation_logic": cells[4],
                    "source_status": cells[5],
                }
            )
        elif current == "missing_fields" and len(cells) >= 5:
            summary[current].append(
                {
                    "gold_field": cells[0],
                    "source_files": cells[1],
                    "source_columns": cells[2],
                    "derivation_logic": cells[3],
                    "source_status": cells[4],
                }
            )
    return summary


def _mapping_spec_for_field(field_path: str) -> dict[str, Any]:
    clean = field_path.replace("[]", "").replace("*.", "")
    if clean.startswith("病历.人口学.年龄_入院时"):
        return _spec(["patients", "admissions"], ["subject_id", "anchor_age", "anchor_year", "admittime"], "用 patients anchor_age/anchor_year 结合 admissions.admittime 估计入院年龄。")
    if clean.startswith("病历.人口学.性别"):
        return _spec(["patients"], ["subject_id", "gender"], "保留 patients.gender。")
    if clean.startswith("病历.人口学.种族"):
        return _spec(["admissions"], ["subject_id", "hadm_id", "race"], "保留 admissions.race。")
    if clean.startswith("病历.入院时间"):
        return _spec(["admissions"], ["subject_id", "hadm_id", "admittime"], "保留 admissions.admittime。")
    if clean.startswith("病历.出院时间"):
        return _spec(["admissions"], ["subject_id", "hadm_id", "dischtime"], "保留 admissions.dischtime。")
    if clean.startswith("病历.入院科室"):
        return _spec(["admissions"], ["subject_id", "hadm_id", "admission_location"], "使用 admissions.admission_location 作为入院来源/科室近似字段。")
    if clean.startswith("病历.出院去向"):
        return _spec(["admissions"], ["subject_id", "hadm_id", "discharge_location"], "保留 admissions.discharge_location。")
    if clean.startswith("病历.入院类型"):
        return _spec(["admissions"], ["subject_id", "hadm_id", "admission_type"], "保留 admissions.admission_type。")
    if clean.startswith("病历.死亡时间"):
        return _spec(["admissions", "patients"], ["subject_id", "hadm_id", "deathtime", "dod", "hospital_expire_flag"], "优先保留 admissions.deathtime；缺失时参考 patients.dod 或 hospital_expire_flag。")
    if clean.startswith("病历.住院天数"):
        return _spec(["admissions"], ["subject_id", "hadm_id", "admittime", "dischtime"], "由 admissions.dischtime - admissions.admittime 计算住院天数。")
    if clean.startswith("诊断列表.icd_title"):
        return _spec(["diagnoses_icd", "d_icd_diagnoses"], ["subject_id", "hadm_id", "icd_code", "icd_version", "long_title"], "用 diagnoses_icd.icd_code/icd_version 关联 ICD 标题映射。")
    if clean.startswith("诊断列表.icd_code"):
        return _spec(["diagnoses_icd"], ["subject_id", "hadm_id", "icd_code"], "保留 diagnoses_icd.icd_code。")
    if clean.startswith("诊断列表.icd_version"):
        return _spec(["diagnoses_icd"], ["subject_id", "hadm_id", "icd_version"], "保留 diagnoses_icd.icd_version。")
    if clean.startswith("诊断列表.是否主诊断"):
        return _spec(["diagnoses_icd"], ["subject_id", "hadm_id", "seq_num"], "当 diagnoses_icd.seq_num=1 时标记为主诊断。")
    if clean.startswith("住院用药."):
        med_columns = {
            "药名": "drug",
            "剂量值": "dose_val_rx",
            "剂量单位": "dose_unit_rx",
            "给药途径": "route",
            "开始时间": "starttime",
            "结束时间": "stoptime",
        }
        source_column = next((value for key, value in med_columns.items() if clean.endswith(key)), "drug")
        return _spec(["prescriptions"], ["subject_id", "hadm_id", source_column], f"从 prescriptions.{source_column} 抽取住院用药字段。")
    if clean.startswith("就诊文本.notes."):
        note_columns = {
            "text": "text",
            "charttime": "charttime",
            "note_id": "note_id",
            "type": "note_type",
        }
        source_column = next((value for key, value in note_columns.items() if clean.endswith(key)), "note_id")
        return _spec(["discharge_notes", "radiology_notes"], ["subject_id", "hadm_id", source_column], f"从 notes 表保留或聚合 {source_column}。")
    if clean.startswith("就诊文本.num_notes"):
        return _spec(["discharge_notes", "radiology_notes"], ["subject_id", "hadm_id", "note_id"], "按 hadm_id 统计 note_id 数量。")
    if (
        clean.startswith("实验室检验.项目明细.名称")
        or clean.startswith("实验室检验.名称")
        or clean.startswith("实验室检验.检验套餐")
    ):
        return _spec(["labevents", "d_labitems"], ["itemid", "label"], "用 labevents.itemid 关联 d_labitems.label 获取检验名称。")
    if clean.startswith("实验室检验."):
        lab_columns = {
            "检验套餐": "itemid",
            "检查时间": "charttime",
            "itemid": "itemid",
            "结果数值": "valuenum",
            "结果文本": "value",
            "单位": "valueuom",
            "参考下限": "ref_range_lower",
            "参考上限": "ref_range_upper",
            "异常标记": "flag",
            "项目时间": "charttime",
        }
        source_column = next((value for key, value in lab_columns.items() if clean.endswith(key)), "itemid")
        return _spec(["labevents"], ["subject_id", "hadm_id", "itemid", source_column], f"从 labevents.{source_column} 抽取检验字段，并按 subject_id/hadm_id 对齐。")
    if clean.startswith("生命体征.肝病相关指标."):
        return _spec(["labevents"], ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"], "从 labevents 筛选肝病相关检验 itemid 并聚合。")
    if clean.startswith("生命体征."):
        return _spec(["chartevents"], ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum", "valueuom"], "从 ICU chartevents 筛选生命体征 itemid。")
    if clean.startswith("体格测量."):
        return _spec(["omr", "chartevents"], ["subject_id", "hadm_id", "charttime", "result_name", "result_value", "valuenum"], "从 OMR 或 chartevents 抽取身高、体重、BMI。")
    return {
        "source_files": [],
        "source_columns": [],
        "join_keys": ["subject_id", "hadm_id"],
        "derivation_logic": "未找到内置映射，需要由 DataExplorer 根据源表样本补充。",
        "confidence": "low",
    }


def _spec(table_names: list[str], columns: list[str], logic: str) -> dict[str, Any]:
    source_files = []
    for table in table_names:
        source_files.extend(TABLE_ALIASES.get(table, (table,)))
    return {
        "source_files": source_files,
        "source_columns": _dedupe_strings(columns),
        "join_keys": [column for column in ("subject_id", "hadm_id", "stay_id", "itemid") if column in columns],
        "derivation_logic": logic,
        "confidence": "high",
    }


def _merge_prior_spec(spec: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    merged = dict(spec)
    if prior.get("source_files"):
        merged["source_files"] = _dedupe_strings([*spec.get("source_files", []), *_split_prior_list(prior["source_files"])])
    if prior.get("source_columns"):
        merged["source_columns"] = _dedupe_strings([*spec.get("source_columns", []), *_split_prior_list(prior["source_columns"])])
    if prior.get("derivation_logic"):
        merged["derivation_logic"] = prior["derivation_logic"]
    return merged


def _prior_entry_for_field(field_path: str, prior_recall: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_field_path(field_path)
    for section in ("matched_fields", "missing_fields"):
        for item in prior_recall.get(section, []):
            if _normalize_field_path(item.get("gold_field", "")) == normalized:
                return item
    return {}


def _rule_from_provenance(item: dict[str, Any]) -> dict[str, Any]:
    rule = {
        "target_field_path": item["target_field_path"],
        "source_files": item["source_files"],
        "source_columns": item["source_columns"],
        "join_keys": item["join_keys"],
        "derivation_logic": item["derivation_logic"],
        "examples_or_failure_cases": [],
        "validation_metric_delta": 0.0,
        "status": "draft",
        "source_status": item["source_status"],
        "confidence": item["confidence"],
    }
    rule["rule_id"] = _rule_id(rule)
    return rule


def _normalize_rule(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_field_path": str(candidate.get("target_field_path") or candidate.get("gold_field") or ""),
        "source_files": _as_list(candidate.get("source_files")),
        "source_columns": _as_list(candidate.get("source_columns")),
        "join_keys": _as_list(candidate.get("join_keys")),
        "derivation_logic": str(candidate.get("derivation_logic") or candidate.get("suggested_derivation_logic") or ""),
        "examples_or_failure_cases": candidate.get("examples_or_failure_cases") or [],
        "validation_metric_delta": float(candidate.get("validation_metric_delta") or 0),
        "status": str(candidate.get("status") or "draft"),
        "created_from_run": str(candidate.get("created_from_run") or ""),
    }


def _write_guidance_artifacts(
    output_dir: Path,
    report: dict[str, Any],
    schema_summary: dict[str, Any],
    field_provenance: list[dict[str, Any]],
    brief: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "gold_schema_summary.json"
    provenance_path = output_dir / "gold_field_provenance_report.json"
    md_path = output_dir / "gold_field_provenance_report.md"
    brief_path = output_dir / "planner_extraction_brief.json"
    _write_json(schema_path, schema_summary)
    _write_json(provenance_path, report)
    _write_json(brief_path, brief)
    md_path.write_text(_render_provenance_markdown(report, field_provenance), encoding="utf-8")
    return {
        "gold_schema_summary": str(schema_path.resolve()),
        "gold_field_provenance_report": str(provenance_path.resolve()),
        "gold_field_provenance_markdown": str(md_path.resolve()),
        "planner_extraction_brief": str(brief_path.resolve()),
    }


def _render_provenance_markdown(report: dict[str, Any], field_provenance: list[dict[str, Any]]) -> str:
    lines = [
        "# Gold 字段来源分析报告",
        "",
        f"- task: `{report.get('task', '')}`",
        f"- raw_data_root: `{report.get('raw_data_root', '')}`",
        f"- gold_examples_path: `{report.get('gold_examples_path', '')}`",
        f"- target fields: {len(field_provenance)}",
        "",
        "| Gold 字段 | 来源文件 | 来源列 | 派生逻辑 | 来源状态 |",
        "|---|---|---|---|---|",
    ]
    for item in field_provenance:
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {} |".format(
                item["target_field_path"],
                ", ".join(item["source_files"]),
                ", ".join(item["source_columns"]),
                item["derivation_logic"],
                item["source_status"],
            )
        )
    return "\n".join(lines) + "\n"


def _read_columns(path: Path) -> list[str]:
    if path.suffix.lower() in CSV_SUFFIXES:
        dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, dialect=dialect)
            try:
                return [str(column).strip() for column in next(reader)]
            except StopIteration:
                return []
    try:
        import pandas as pd

        return [str(column).strip() for column in pd.read_excel(path, nrows=0).columns]
    except Exception:
        return []


def _count_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return -1


def _resolve_table_alias(source_schema: dict[str, Any], file_name: str) -> dict[str, Any] | None:
    wanted = Path(file_name).as_posix().lower()
    base = Path(wanted).name
    for item in source_schema.get("files", []):
        rel = str(item.get("relative_path", "")).lower()
        if rel == wanted or rel.endswith("/" + wanted) or Path(rel).name == base:
            return item
    return None


def _read_rules_file(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path, {})
    raw_rules = data if isinstance(data, list) else data.get("rules", [])
    rules = []
    if isinstance(raw_rules, list):
        for item in raw_rules:
            if isinstance(item, dict):
                rule = _normalize_rule(item)
                rule.update({key: item[key] for key in item if key not in rule})
                rule.setdefault("rule_id", _rule_id(rule))
                rules.append(rule)
    return rules


def _dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rule in rules:
        item = dict(rule)
        item.setdefault("rule_id", _rule_id(item))
        by_id[item["rule_id"]] = item
    return [by_id[key] for key in sorted(by_id)]


def _rule_id(rule: dict[str, Any]) -> str:
    material = {
        "target_field_path": rule.get("target_field_path", ""),
        "source_files": _as_list(rule.get("source_files")),
        "source_columns": _as_list(rule.get("source_columns")),
        "derivation_logic": rule.get("derivation_logic", ""),
    }
    text = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _source_counts(field_provenance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in field_provenance:
        for source_file in item.get("source_files", []):
            counts[source_file] = counts.get(source_file, 0) + 1
    return [
        {"source_file": source_file, "target_field_count": count}
        for source_file, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _list_path(path: list[str]) -> list[str]:
    if not path:
        return ["[]"]
    return [*path[:-1], f"{path[-1]}[]"]


def _dynamic_segments_for_path(path: list[str]) -> list[str]:
    segments = []
    for index, segment in enumerate(path):
        if segment == "*" and index > 0:
            segments.append(".".join(path[: index + 1]))
    return segments


def _safe_sample_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value) if isinstance(value, str) else value
        if isinstance(text, str) and len(text) > 120:
            return text[:117] + "..."
        return text
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text[:117] + "..." if len(text) > 120 else text


def _clean_table_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _is_internal_source_metadata(field_path: str) -> bool:
    return field_path.endswith(".source.table") or field_path.endswith(".source.row_id")


def _looks_like_raw_data_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.name in {"structured", "hosp", "notes", "icu"} for child in path.iterdir())


def _split_prior_list(value: str) -> list[str]:
    parts = re.split(r"\s*/\s*|,\s*", str(value))
    return [part.strip(" `") for part in parts if part.strip(" `")]


def _strip_md_cell(value: str) -> str:
    return value.strip().strip("`").strip()


def _normalize_field_path(value: str) -> str:
    return str(value).strip().strip("`").replace("[]", "").replace("*.", "")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        return [item for item in _split_prior_list(value) if item]
    return [value]


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z一-鿿]+", "_", str(value or "")).strip("_")
    return (text or "task")[:48]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
