from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from workflow.task_compiler import compile_extraction_task_plan


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
GOLD_TOKENS = ("gold", "target", "expected", "clean", "reference", "label")
RAW_TOKENS = ("raw", "source", "input", "original")


@dataclass(frozen=True)
class ExamplePair:
    case_id: str
    case_root: Path
    raw_root: Path
    gold_paths: tuple[Path, ...]


def discover_example_pairs(examples_root: str | Path) -> list[ExamplePair]:
    root = Path(examples_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Training example directory does not exist: {root}")
    pairs: list[ExamplePair] = []
    for case_root in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        raw_dirs = [
            path
            for path in case_root.iterdir()
            if path.is_dir() and any(token in path.name.lower() for token in RAW_TOKENS)
        ]
        if not raw_dirs:
            raw_dirs = [path for path in case_root.iterdir() if path.is_dir()]
        if not raw_dirs:
            continue
        raw_root = sorted(raw_dirs, key=lambda path: path.name)[0]
        candidates = [
            path
            for path in case_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and raw_root not in path.parents
        ]
        named = [path for path in candidates if any(token in path.name.lower() for token in GOLD_TOKENS)]
        gold_paths = tuple(sorted(named or candidates))
        if not gold_paths:
            continue
        pairs.append(
            ExamplePair(
                case_id=case_root.name,
                case_root=case_root,
                raw_root=raw_root,
                gold_paths=gold_paths,
            )
        )
    if not pairs:
        raise ValueError(f"No raw/gold example pairs found under: {root}")
    return pairs


def analyze_training_examples(examples_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    pairs = discover_example_pairs(examples_root)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    target_values: dict[str, list[Any]] = defaultdict(list)
    target_cases: dict[str, set[str]] = defaultdict(set)
    source_fields: dict[str, dict[str, Any]] = {}
    input_files: list[Path] = []
    gold_records: list[tuple[str, dict[str, Any]]] = []
    target_categories: set[str] = set()
    for pair in pairs:
        for gold_path in pair.gold_paths:
            input_files.append(gold_path)
            for record in _read_records(gold_path):
                gold_records.append((pair.case_id, record))
                target_categories.update(str(key) for key in record)
        for raw_path in _data_files(pair.raw_root):
            input_files.append(raw_path)
            source_key = raw_path.relative_to(pair.raw_root).as_posix()
            entry = source_fields.setdefault(source_key, {"fields": defaultdict(list), "examples": []})
            entry["examples"].append(str(raw_path))
            for record in _read_records(raw_path, sample_limit=100):
                for field_path, value in _flatten_leaves(record):
                    entry["fields"][field_path].append(value)

    dynamic_keys = _discover_dynamic_keys(record for _, record in gold_records)
    for case_id, record in gold_records:
        for field_path, value in _flatten_leaves(
            record,
            dynamic_keys=dynamic_keys,
            include_empty=True,
        ):
            target_values[field_path].append(value)
            target_cases[field_path].add(case_id)

    record_grain = _infer_record_grain(target_values, len(pairs))
    rules = [
        _build_rule(
            field_path=field_path,
            values=values,
            case_count=len(target_cases[field_path]),
            example_count=len(pairs),
            source_fields=source_fields,
            record_grain=record_grain,
        )
        for field_path, values in sorted(target_values.items())
    ]
    rules_payload = {
        "schema_version": 1,
        "example_root": str(Path(examples_root).expanduser().resolve()),
        "example_count": len(pairs),
        "record_grain": record_grain,
        "target_categories": sorted(target_categories),
        "field_count": len(rules),
        "rules": rules,
    }
    supported = sum(1 for rule in rules if rule["status"] == "supported")
    ambiguous = sum(1 for rule in rules if rule["status"] == "ambiguous")
    pending = sum(1 for rule in rules if rule["status"] == "pending")
    report = {
        "schema_version": 1,
        "example_count": len(pairs),
        "field_count": len(rules),
        "record_grain": record_grain,
        "target_categories": sorted(target_categories),
        "status_counts": {
            "supported": supported,
            "ambiguous": ambiguous,
            "pending": pending,
            "unsupported": len(rules) - supported - ambiguous - pending,
        },
        "source_file_count": len(source_fields),
        "source_files": sorted(source_fields),
        "capability_counts": _count_values(rule["capability_type"] for rule in rules),
        "field_analysis": [_field_analysis_row(rule) for rule in rules],
    }
    task_plan = compile_extraction_task_plan(rules, record_grain=record_grain)
    task_plan["target_categories"] = sorted(target_categories)
    report["task_count"] = task_plan["task_count"]
    report["executable_rule_count"] = task_plan["executable_rule_count"]
    rules_path = output / "field_extraction_rules.json"
    task_plan_path = output / "extraction_task_plan.json"
    report_path = output / "data_analysis_report.json"
    run_record_path = output / "explorer_run_record.json"
    _write_json(rules_path, rules_payload)
    _write_json(task_plan_path, task_plan)
    _write_json(report_path, report)
    run_record = {
        "status": "SUCCESS",
        "example_count": len(pairs),
        "field_count": len(rules),
        "input_fingerprints": {str(path): _sha256(path) for path in sorted(set(input_files))},
        "artifacts": {
            "field_extraction_rules": str(rules_path),
            "extraction_task_plan": str(task_plan_path),
            "data_analysis_report": str(report_path),
            "explorer_run_record": str(run_record_path),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_record_path, run_record)
    return run_record["artifacts"]


def _build_rule(
    *,
    field_path: str,
    values: list[Any],
    case_count: int,
    example_count: int,
    source_fields: dict[str, dict[str, Any]],
    record_grain: str,
) -> dict[str, Any]:
    leaf = _normalize_name(field_path.rsplit(".", 1)[-1].replace("[]", ""))
    candidates: list[tuple[str, str, float]] = []
    for source_file, source in source_fields.items():
        for source_column, source_values in source["fields"].items():
            source_leaf = _normalize_name(source_column.rsplit(".", 1)[-1].replace("[]", ""))
            if source_leaf != leaf:
                continue
            candidates.append((source_file, source_column, _value_overlap(values, source_values)))
    if candidates:
        best_score = max(score for _, _, score in candidates)
        best = [item for item in candidates if abs(item[2] - best_score) < 1e-9]
    else:
        best = []
    source_files = sorted({item[0] for item in best})
    source_columns = sorted({item[1] for item in best})
    status = "pending"
    if len(source_files) == 1:
        status = "supported"
    elif len(source_files) > 1:
        status = "ambiguous"
    cardinality = "many" if "[]" in field_path or len(values) > case_count else "one"
    target_type = _infer_type(values)
    evaluation_policy = _evaluation_policy(target_type, cardinality, values)
    capability = "text_extract" if evaluation_policy == "semantic_text" else "structured_extract"
    if cardinality == "many" and capability == "structured_extract":
        capability = "aggregate"
    return {
        "target_field_id": "field_" + hashlib.sha256(field_path.encode("utf-8")).hexdigest()[:16],
        "target_field_path": field_path,
        "target_type": target_type,
        "cardinality": cardinality,
        "record_grain": record_grain,
        "source_files": source_files,
        "source_columns": source_columns,
        "join_keys": _infer_join_keys(source_fields, source_files, record_grain),
        "filters": [],
        "derivation_logic": {"operation": "direct" if status == "supported" else "requires_analysis"},
        "capability_type": capability,
        "evaluation_policy": evaluation_policy,
        "evidence": {
            "observed_value_count": len(values),
            "example_coverage": round(case_count / max(example_count, 1), 4),
            "candidate_count": len(candidates),
        },
        "confidence": round(best[0][2], 4) if len(best) == 1 else 0.0,
        "status": status,
    }


def _read_records(path: Path, sample_limit: int | None = None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        records = value if isinstance(value, list) else [value]
    elif suffix in {".xlsx", ".xls"}:
        records = pd.read_excel(path, dtype=object).to_dict(orient="records")
    elif suffix == ".parquet":
        records = pd.read_parquet(path).to_dict(orient="records")
    else:
        records = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=object).to_dict(orient="records")
    normalized = [record for record in records if isinstance(record, dict)]
    return normalized[:sample_limit] if sample_limit is not None else normalized


def _flatten_leaves(
    value: Any,
    prefix: str = "",
    *,
    schema_prefix: str = "",
    dynamic_keys: dict[str, set[str]] | None = None,
    include_empty: bool = False,
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        if include_empty and prefix and not value:
            yield prefix, {}
        for key, nested in value.items():
            key_text = str(key)
            is_dynamic = key_text in (dynamic_keys or {}).get(schema_prefix, set())
            path = f"{prefix}[]" if is_dynamic else f"{prefix}.{key_text}" if prefix else key_text
            schema_key = "{dynamic}" if _looks_dynamic_key(key_text) else key_text
            nested_schema = f"{schema_prefix}.{schema_key}" if schema_prefix else schema_key
            yield from _flatten_leaves(
                nested,
                path,
                schema_prefix=nested_schema,
                dynamic_keys=dynamic_keys,
                include_empty=include_empty,
            )
    elif isinstance(value, (list, tuple)):
        path = f"{prefix}[]"
        nested_schema = f"{schema_prefix}[]"
        if include_empty and not value:
            yield path, []
        for nested in value:
            yield from _flatten_leaves(
                nested,
                path,
                schema_prefix=nested_schema,
                dynamic_keys=dynamic_keys,
                include_empty=include_empty,
            )
    elif prefix and (include_empty or not _is_missing(value)):
        yield prefix, _json_scalar(value)


def _discover_dynamic_keys(records: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    observations: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        _collect_key_shapes(record, "", observations)

    dynamic: dict[str, set[str]] = defaultdict(set)
    for container, signatures in observations.items():
        for keys in signatures.values():
            if len(keys) >= 2 and all(_looks_dynamic_key(key) for key in keys):
                dynamic[container].update(keys)
    return dict(dynamic)


def _collect_key_shapes(
    value: Any,
    schema_prefix: str,
    observations: dict[str, dict[str, set[str]]],
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            observations[schema_prefix][_structural_signature(nested)].add(key_text)
            schema_key = "{dynamic}" if _looks_dynamic_key(key_text) else key_text
            nested_schema = f"{schema_prefix}.{schema_key}" if schema_prefix else schema_key
            _collect_key_shapes(nested, nested_schema, observations)
    elif isinstance(value, (list, tuple)):
        nested_schema = f"{schema_prefix}[]"
        for nested in value:
            _collect_key_shapes(nested, nested_schema, observations)


def _structural_signature(value: Any) -> str:
    if isinstance(value, dict):
        fields = sorted(
            (
                "{dynamic}" if _looks_dynamic_key(str(key)) else str(key),
                _structural_signature(nested),
            )
            for key, nested in value.items()
        )
        return "dict:" + json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        signatures = sorted({_structural_signature(item) for item in value})
        return "list:" + json.dumps(signatures, ensure_ascii=False, separators=(",", ":"))
    return "scalar"


def _looks_dynamic_key(key: str) -> bool:
    text = key.strip()
    if re.fullmatch(r"\d{4,}", text):
        return True
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}", text):
        return True
    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return True
    if re.search(r"\d{4,}", text) and re.search(r"[_:/-]", text):
        return True
    return bool(len(text) >= 12 and re.search(r"\d", text) and re.search(r"[_:/-]", text))


def _data_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def _infer_record_grain(fields: dict[str, list[Any]], example_count: int) -> str:
    candidates = []
    for path, values in fields.items():
        leaf = path.rsplit(".", 1)[-1].lower()
        if "[]" not in path and (leaf == "id" or leaf.endswith("_id") or leaf.endswith("key")):
            non_missing = [str(value) for value in values if not _is_missing(value)]
            if len(set(non_missing)) >= min(len(non_missing), example_count):
                candidates.append(path)
    return sorted(candidates, key=lambda value: (value.count("."), len(value), value))[0] if candidates else "case"


def _infer_join_keys(source_fields: dict[str, dict[str, Any]], source_files: list[str], record_grain: str) -> list[str]:
    grain_leaf = record_grain.rsplit(".", 1)[-1]
    keys = []
    for source_file in source_files:
        for field in source_fields[source_file]["fields"]:
            if field.rsplit(".", 1)[-1] == grain_leaf:
                keys.append(field)
    return sorted(set(keys))


def _infer_type(values: list[Any]) -> str:
    present = [value for value in values if not _is_missing(value)]
    if not present:
        return "unknown"
    if all(isinstance(value, dict) for value in present):
        return "object"
    if all(isinstance(value, (list, tuple)) for value in present):
        return "array"
    if all(isinstance(value, bool) for value in present):
        return "boolean"
    if all(_to_number(value) is not None for value in present):
        return "number"
    if all(_to_datetime(value) is not None for value in present):
        return "datetime"
    return "string"


def _evaluation_policy(target_type: str, cardinality: str, values: list[Any]) -> str:
    if cardinality == "many":
        return "collection_contains"
    if target_type == "number":
        return "numeric_exact"
    if target_type == "datetime":
        return "datetime_normalized"
    average_length = sum(len(str(value)) for value in values) / max(len(values), 1)
    return "semantic_text" if average_length >= 80 else "canonical"


def _value_overlap(left: list[Any], right: list[Any]) -> float:
    left_values = {_canonical_value(value) for value in left if not _is_missing(value)}
    right_values = {_canonical_value(value) for value in right if not _is_missing(value)}
    if not left_values:
        return 0.0
    return len(left_values & right_values) / len(left_values)


def _canonical_value(value: Any) -> str:
    number = _to_number(value)
    if number is not None:
        return f"number:{number:g}"
    dt = _to_datetime(value)
    if dt is not None:
        return f"datetime:{dt}"
    return "text:" + re.sub(r"\s+", " ", str(value)).strip()


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        text = str(value).strip().replace(",", "")
        if not re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_datetime(value: Any) -> str | None:
    if not isinstance(value, str) or not re.search(r"[-/:]", value):
        return None
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except Exception:
        return None
    return parsed.isoformat()


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if not isinstance(result, (list, tuple)) else False
    except Exception:
        return value is None


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_field_rule_completion(
    rules_path: str | Path,
    source_root: str | Path,
) -> dict[str, Any]:
    """Reject incomplete or unverifiable Explorer field rules."""
    path = Path(rules_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(rules, list):
        raise ValueError("field_extraction_rules.json must contain a rules list")
    if int(payload.get("field_count", -1)) != len(rules):
        raise ValueError("field_count does not match the locked rule count")
    root = Path(source_root).expanduser().resolve()
    issues: list[str] = []
    status_counts: dict[str, int] = defaultdict(int)
    for rule in rules:
        field_id = str(rule.get("target_field_id") or "")
        field_path = str(rule.get("target_field_path") or field_id)
        status = str(rule.get("status") or "pending")
        status_counts[status] += 1
        if status == "pending":
            issues.append(f"{field_path}: pending")
            continue
        if status == "unsupported":
            if not str(rule.get("unsupported_reason") or "").strip():
                issues.append(f"{field_path}: unsupported rule is missing unsupported_reason")
            continue
        if status == "ambiguous":
            if not rule.get("source_files") or not rule.get("source_columns"):
                issues.append(f"{field_path}: ambiguous rule is missing candidates")
            continue
        if status not in {"supported", "active", "frozen"}:
            issues.append(f"{field_path}: invalid status {status}")
            continue
        source_files = [str(value) for value in rule.get("source_files") or []]
        source_columns = [str(value) for value in rule.get("source_columns") or []]
        derivation = rule.get("derivation_logic") or {}
        evidence = rule.get("evidence") or {}
        is_constant = (
            derivation.get("operation") == "constant"
            and bool(derivation.get("constant_value") or evidence.get("constant_value"))
        )
        if not source_files:
            issues.append(f"{field_path}: supported rule requires source files")
            continue
        if not source_columns and not is_constant:
            issues.append(f"{field_path}: supported rule requires source columns")
            continue
        matched_paths = [candidate for name in source_files for candidate in _find_source_files(root, name)]
        if not matched_paths:
            issues.append(f"{field_path}: source file not found: {', '.join(source_files)}")
            continue
        available_columns = {
            str(column)
            for source_path in matched_paths
            for column in _source_columns(source_path)
        }
        missing = [column for column in source_columns if column not in available_columns]
        if missing:
            issues.append(f"{field_path}: source columns not found: {', '.join(missing)}")
    if issues:
        preview = "; ".join(issues[:12])
        suffix = f"; and {len(issues) - 12} more" if len(issues) > 12 else ""
        raise ValueError(f"Field rule completion failed: {preview}{suffix}")
    return {"field_count": len(rules), "status_counts": dict(status_counts), "complete": True}


def write_data_analysis_report(
    rules_path: str | Path,
    output_path: str | Path,
    *,
    base_report_path: str | Path | None = None,
) -> str:
    """Write the canonical field-level report from the latest locked rules."""
    rules_payload = json.loads(Path(rules_path).expanduser().resolve().read_text(encoding="utf-8"))
    rules = rules_payload.get("rules") if isinstance(rules_payload, dict) else None
    if not isinstance(rules, list):
        raise ValueError("field_extraction_rules.json must contain a rules list")
    report: dict[str, Any] = {}
    if base_report_path:
        base = Path(base_report_path).expanduser().resolve()
        if base.is_file():
            value = json.loads(base.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                report.update(value)
    counts = _count_values(str(rule.get("status") or "pending") for rule in rules)
    report.update(
        {
            "schema_version": 2,
            "analysis_schema_version": 2,
            "example_count": int(rules_payload.get("example_count") or report.get("example_count") or 0),
            "field_count": len(rules),
            "record_grain": str(rules_payload.get("record_grain") or report.get("record_grain") or "case"),
            "target_categories": list(rules_payload.get("target_categories") or []),
            "status_counts": counts,
            "capability_counts": _count_values(str(rule.get("capability_type") or "") for rule in rules),
            "field_analysis": [_field_analysis_row(rule) for rule in rules],
        }
    )
    target = Path(output_path).expanduser().resolve()
    _write_json(target, report)
    return str(target)


def _field_analysis_row(rule: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "target_field_id",
        "target_field_path",
        "target_type",
        "cardinality",
        "record_grain",
        "source_files",
        "source_columns",
        "join_keys",
        "filters",
        "derivation_logic",
        "capability_type",
        "evaluation_policy",
        "evidence",
        "confidence",
        "status",
        "unsupported_reason",
    )
    return {key: rule.get(key) for key in keys}


def _find_source_files(root: Path, relative_name: str) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.name == Path(relative_name).name else []
    normalized = Path(relative_name).as_posix().lstrip("./")
    return sorted(
        {
            candidate.resolve()
            for candidate in root.rglob(Path(normalized).name)
            if candidate.is_file() and candidate.as_posix().endswith(normalized)
        }
    )


def _source_columns(path: Path) -> list[str]:
    try:
        return [str(column) for column in _read_records(path, sample_limit=1)[0]]
    except (IndexError, ValueError, TypeError):
        if path.suffix.lower() in {".csv", ".tsv"}:
            return [str(column) for column in pd.read_csv(path, nrows=0, sep="\t" if path.suffix.lower() == ".tsv" else ",").columns]
        return []
