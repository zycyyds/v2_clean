from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lib.gold_guidance import _mapping_spec_for_field


_MIMIC_SIGNATURE = {"admissions.csv", "patients.csv", "diagnoses_icd.csv", "labevents.csv"}
_JOIN_KEYS = {"case_id", "subject_id", "hadm_id", "stay_id", "itemid"}


def build_dataset_profile_suggestions(
    rules_path: str | Path,
    source_root: str | Path,
) -> dict[str, Any]:
    """Build schema-validated optional profile suggestions for Explorer."""
    rules_file = Path(rules_path).expanduser().resolve()
    root = Path(source_root).expanduser().resolve()
    payload = json.loads(rules_file.read_text(encoding="utf-8"))
    rules = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(rules, list):
        raise ValueError("field_extraction_rules.json must contain a rules list")
    files = [path for path in root.rglob("*") if path.is_file()]
    profile = "mimic" if _MIMIC_SIGNATURE.issubset({path.name for path in files}) else "generic"
    if profile != "mimic":
        return {"dataset_profile": profile, "updates": []}

    schema_cache = {path.resolve(): _columns(path) for path in files}
    updates = [
        _mimic_suggestion(rule, root, schema_cache)
        for rule in rules
        if isinstance(rule, dict) and rule.get("target_field_id")
    ]
    return {"dataset_profile": profile, "updates": updates}


def _mimic_suggestion(
    rule: dict[str, Any],
    root: Path,
    schemas: dict[Path, set[str]],
) -> dict[str, Any]:
    field_path = str(rule.get("target_field_path") or "")
    base = {
        "target_field_id": str(rule["target_field_id"]),
        "target_field_path": field_path,
    }
    if rule.get("target_type") in {"object", "array"}:
        return {
            **base,
            "status": "unsupported",
            "unsupported_reason": "training examples contain only an empty container, so no inner schema is observable",
            "source_files": [],
            "source_columns": [],
            "join_keys": [],
            "evidence": {"dataset_profile": "mimic", "schema_validated": False},
        }

    spec = _mapping_spec_for_field(field_path)
    candidate_names = [str(value) for value in spec.get("source_files") or []]
    matched = _match_files(root, candidate_names)
    if not candidate_names:
        return {
            **base,
            "status": "unsupported",
            "unsupported_reason": "MIMIC profile has no candidate mapping for this Gold field",
            "source_files": [],
            "source_columns": [],
            "join_keys": [],
            "evidence": {"dataset_profile": "mimic", "schema_validated": False},
        }
    if not matched:
        return {
            **base,
            "status": "unsupported",
            "unsupported_reason": f"candidate source is absent from training raw data: {', '.join(candidate_names)}",
            "source_files": [],
            "source_columns": [],
            "join_keys": list(spec.get("join_keys") or []),
            "evidence": {"dataset_profile": "mimic", "schema_validated": False},
        }

    available = set().union(*(schemas.get(path, set()) for path in matched))
    relative_files = sorted({_relative_suffix(root, path) for path in matched})
    if field_path.endswith(".source.table"):
        table_name = Path(relative_files[0]).stem
        return {
            **base,
            "status": "supported",
            "source_files": relative_files,
            "source_columns": [],
            "join_keys": list(spec.get("join_keys") or []),
            "derivation_logic": {
                "operation": "constant",
                "value": f"hosp/{table_name}",
                "description": "来源表名由已验证的 MIMIC 文件路径确定。",
            },
            "capability_type": "constant",
            "confidence": 0.95,
            "evidence": {
                "dataset_profile": "mimic",
                "schema_validated": True,
                "observed_source_files": relative_files,
                "constant_value": f"hosp/{table_name}",
            },
        }
    if field_path.endswith(".source.row_id"):
        if "labevents.csv" in {path.name for path in matched} and "labevent_id" in available:
            return {
                **base,
                "status": "supported",
                "source_files": relative_files,
                "source_columns": ["labevent_id"],
                "join_keys": [key for key in ("subject_id", "hadm_id", "itemid") if key in available],
                "derivation_logic": {"operation": "direct", "description": "保留 labevents.labevent_id。"},
                "capability_type": "aggregate",
                "confidence": 0.95,
                "evidence": {
                    "dataset_profile": "mimic",
                    "schema_validated": True,
                    "observed_source_files": relative_files,
                },
            }
        return {
            **base,
            "status": "unsupported",
            "unsupported_reason": "Gold row_id is null and the verified source table has no stable row identifier",
            "source_files": relative_files,
            "source_columns": [],
            "join_keys": list(spec.get("join_keys") or []),
            "evidence": {
                "dataset_profile": "mimic",
                "schema_validated": True,
                "observed_source_files": relative_files,
            },
        }
    if field_path.startswith("生命体征.肝病相关指标."):
        candidate_columns = [column for column in ("itemid", "charttime", "valuenum") if column in available]
        return {
            **base,
            "status": "ambiguous",
            "source_files": relative_files,
            "source_columns": candidate_columns,
            "join_keys": [key for key in ("subject_id", "hadm_id", "itemid") if key in available],
            "filters": [{"status": "unresolved", "reason": "No verified lab item dictionary or itemid rule is available"}],
            "derivation_logic": {
                "operation": "field_derivation",
                "description": str(spec.get("derivation_logic") or ""),
            },
            "capability_type": "field_derivation",
            "confidence": 0.5,
            "evidence": {
                "dataset_profile": "mimic",
                "schema_validated": True,
                "observed_source_files": relative_files,
                "unresolved_requirement": "verified lab itemid mapping",
            },
        }
    declared = [str(value) for value in spec.get("source_columns") or []]
    join_keys = [value for value in spec.get("join_keys") or [] if value in available]
    value_columns = _target_value_columns(field_path, declared)
    missing = [value for value in value_columns if value not in available]
    evidence = {
        "dataset_profile": "mimic",
        "schema_validated": not missing,
        "candidate_source_files": candidate_names,
        "observed_source_files": relative_files,
        "missing_columns": missing,
    }
    if missing or not value_columns:
        reason = (
            f"candidate source columns are absent: {', '.join(missing)}"
            if missing
            else "candidate mapping does not identify a value column"
        )
        return {
            **base,
            "status": "unsupported",
            "unsupported_reason": reason,
            "source_files": relative_files,
            "source_columns": value_columns,
            "join_keys": join_keys,
            "evidence": evidence,
        }

    capability = _capability(rule, value_columns, str(spec.get("derivation_logic") or ""))
    operation, operation_args = _target_operation(field_path, capability)
    return {
        **base,
        "status": "supported",
        "source_files": relative_files,
        "source_columns": value_columns,
        "join_keys": join_keys,
        "derivation_logic": {
            "operation": operation,
            "description": str(spec.get("derivation_logic") or ""),
            **operation_args,
        },
        "capability_type": capability,
        "confidence": 0.95,
        "evidence": evidence,
    }


def _target_value_columns(field_path: str, declared: list[str]) -> list[str]:
    clean = field_path.replace("[]", "")
    exact_targets = {
        "实验室检验.项目明细.itemid": "itemid",
        "诊断列表.icd_code": "icd_code",
        "诊断列表.icd_version": "icd_version",
        "诊断列表.是否主诊断": "seq_num",
    }
    target = next((column for path, column in exact_targets.items() if clean.startswith(path)), "")
    if target:
        return [target] if target in declared else []
    return [value for value in declared if value not in _JOIN_KEYS]


def _target_operation(field_path: str, capability: str) -> tuple[str, dict[str, Any]]:
    clean = field_path.replace("[]", "")
    if clean.startswith("诊断列表.是否主诊断"):
        return "equals", {"value": 1}
    return ("direct" if capability == "structured_extract" else capability), {}


def _capability(rule: dict[str, Any], value_columns: list[str], logic: str) -> str:
    field_path = str(rule.get("target_field_path") or "")
    if field_path.startswith("就诊文本") and field_path.endswith("text"):
        return "text_extract"
    if str(rule.get("cardinality")) == "many" or "[]" in field_path:
        return "aggregate"
    if len(value_columns) > 1 or any(marker in logic for marker in ("计算", "估计", "关联", "派生")):
        return "field_derivation"
    return "structured_extract"


def _match_files(root: Path, candidates: list[str]) -> list[Path]:
    matched: set[Path] = set()
    for candidate in candidates:
        normalized = Path(candidate).as_posix().lstrip("./")
        for path in root.rglob(Path(normalized).name):
            if path.is_file() and path.as_posix().endswith(normalized):
                matched.add(path.resolve())
    return sorted(matched)


def _relative_suffix(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    for marker in ("raw_mimic", "raw", "source", "source_records"):
        if marker in parts:
            return Path(*parts[parts.index(marker) + 1 :]).as_posix()
    return path.name


def _columns(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return {str(value) for value in pd.read_csv(path, nrows=0).columns}
        if suffix == ".tsv":
            return {str(value) for value in pd.read_csv(path, sep="\t", nrows=0).columns}
        if suffix in {".xlsx", ".xls"}:
            return {str(value) for value in pd.read_excel(path, nrows=0).columns}
        if suffix == ".parquet":
            return {str(value) for value in pd.read_parquet(path).columns}
        if suffix == ".jsonl":
            return {str(value) for value in pd.read_json(path, lines=True).columns}
        if suffix == ".json":
            return {str(value) for value in pd.read_json(path).columns}
    except Exception:
        return set()
    return set()
