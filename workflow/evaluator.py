from __future__ import annotations

import json
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from workflow.gold_analysis import (
    SUPPORTED_SUFFIXES,
    _data_files,
    _discover_dynamic_keys,
    _flatten_leaves,
    _read_records,
)


SemanticScorer = Callable[[str, str], float | None]


def create_embedding_semantic_scorer(
    *,
    client: Any | None = None,
    model: str = "",
    config_path: str | Path | None = None,
) -> SemanticScorer | None:
    if client is None:
        config = _embedding_config(config_path)
        api_key = str(os.environ.get("EMBEDDING_API_KEY") or config.get("api_key") or "")
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            return None
        base_url = str(os.environ.get("EMBEDDING_API_BASE") or config.get("base_url") or "https://api.openai.com/v1")
        model = str(os.environ.get("EMBEDDING_MODEL") or model or config.get("model") or "text-embedding-3-small")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception:
            return None
    selected_model = model or "text-embedding-3-small"

    def score(gold_text: str, prediction_text: str) -> float | None:
        try:
            response = client.embeddings.create(
                model=selected_model,
                input=[str(gold_text)[:3000], str(prediction_text)[:3000]],
            )
            left = list(response.data[0].embedding)
            right = list(response.data[1].embedding)
            denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
                sum(value * value for value in right)
            )
            if not denominator:
                return 0.0
            return round(sum(a * b for a, b in zip(left, right)) / denominator, 4)
        except Exception:
            return None

    return score


def evaluate_result_package(
    gold_path: str | Path,
    rules_path: str | Path,
    result_manifest_path: str | Path,
    target_mapping_path: str | Path,
    output_dir: str | Path,
    *,
    semantic_scorer: SemanticScorer | None = None,
    source_data_path: str | Path | None = None,
) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rules_payload = _load_json(Path(rules_path))
    rules = {
        str(item["target_field_id"]): item
        for item in rules_payload.get("rules", [])
        if isinstance(item, dict) and item.get("target_field_id")
    }
    record_grain = str(rules_payload.get("record_grain") or "case_id")
    known_paths = {str(rule.get("target_field_path") or "") for rule in rules.values()}
    gold_records = _load_gold_records(Path(gold_path), record_grain, known_paths=known_paths)
    gold_paths = sorted({field_path for fields in gold_records.values() for field_path in fields})
    for field_path in gold_paths:
        if field_path in known_paths:
            continue
        field_id = "field_" + hashlib.sha256(field_path.encode("utf-8")).hexdigest()[:16]
        rules[field_id] = {
            "target_field_id": field_id,
            "target_field_path": field_path,
            "evaluation_policy": "canonical",
            "source_files": [],
            "source_columns": [],
            "join_keys": [],
            "derivation_logic": {"operation": "requires_validation_analysis"},
            "status": "missing_rule",
        }
    manifest_path = Path(result_manifest_path).expanduser().resolve()
    manifest = _load_json(manifest_path)
    mapping_payload = _load_json(Path(target_mapping_path))
    artifacts, artifact_issues = _load_artifacts(manifest, manifest_path.parent)
    mappings, mapping_issues = _validate_mappings(mapping_payload, rules, artifacts)

    predictions: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    mapped_fields: set[str] = set()
    for mapping in mappings:
        field_id = mapping["target_field_id"]
        artifact = artifacts[mapping["artifact"]]
        source_column = mapping["source_column"]
        case_column = str(mapping.get("case_id_column") or artifact["case_id_column"])
        for record in artifact["records"]:
            case_id = _clean_id(record.get(case_column))
            value = record.get(source_column)
            if case_id and not _is_missing(value):
                predictions[case_id][field_id].append(_json_scalar(value))
                mapped_fields.add(field_id)

    comparisons: list[dict[str, Any]] = []
    field_results: dict[str, list[bool]] = defaultdict(list)
    for case_id, gold_fields in sorted(gold_records.items()):
        for field_id, rule in rules.items():
            field_path = str(rule.get("target_field_path") or "")
            gold_values = gold_fields.get(field_path, [])
            if not gold_values:
                continue
            pred_values = predictions.get(case_id, {}).get(field_id, [])
            if rule.get("status") == "missing_rule":
                passed, error_type, semantic_score = False, "missing_rule", None
            else:
                passed, error_type, semantic_score = _compare_values(
                    gold_values,
                    pred_values,
                    policy=str(rule.get("evaluation_policy") or "canonical"),
                    semantic_scorer=semantic_scorer,
                )
            field_results[field_id].append(passed)
            comparisons.append(
                {
                    "case_id": case_id,
                    "target_field_id": field_id,
                    "target_field_path": field_path,
                    "gold_value": _collapse(gold_values),
                    "pred_value": _collapse(pred_values),
                    "passed": passed,
                    "error_type": error_type,
                    "semantic_score": semantic_score,
                }
            )

    expected_fields = {field_id for field_id, values in field_results.items() if values}
    valid_mapped_fields = expected_fields & mapped_fields
    field_coverage = len(valid_mapped_fields) / max(len(expected_fields), 1)
    field_value_scores = [sum(results) / len(results) for results in field_results.values() if results]
    value_correctness = sum(field_value_scores) / max(len(field_value_scores), 1)
    value_micro = sum(1 for item in comparisons if item["passed"]) / max(len(comparisons), 1)
    semantic_rows = [
        item
        for item in comparisons
        if rules[item["target_field_id"]].get("evaluation_policy") == "semantic_text"
    ]
    semantic_score = sum(1 for item in semantic_rows if item["passed"]) / max(len(semantic_rows), 1) if semantic_rows else 0.0
    per_case: dict[str, list[bool]] = defaultdict(list)
    for item in comparisons:
        per_case[item["case_id"]].append(bool(item["passed"]))
    per_case_scores = [
        {
            "case_id": case_id,
            "passed": sum(values),
            "total": len(values),
            "score": round(sum(values) / max(len(values), 1), 4),
        }
        for case_id, values in sorted(per_case.items())
    ]
    gold_cases = set(gold_records)
    predicted_cases = set(predictions)
    sample_coverage = len(gold_cases & predicted_cases) / max(len(gold_cases), 1)
    source_schema = _load_source_schema(Path(source_data_path)) if source_data_path is not None else None
    explainable = [rule for rule in rules.values() if _is_explainable(rule, source_schema)]
    source_explainability = len(explainable) / max(len(rules), 1)
    composite = (
        0.4 * field_coverage
        + 0.4 * value_correctness
        + 0.1 * sample_coverage
        + 0.1 * source_explainability
    )
    metrics = {
        "field_coverage": round(field_coverage, 4),
        "value_correctness": round(value_correctness, 4),
        "sample_coverage": round(sample_coverage, 4),
        "source_explainability": round(source_explainability, 4),
        "composite_score": round(composite, 4),
    }
    issues = [*artifact_issues, *mapping_issues]
    artifact_validation = {
        "valid": not issues,
        "artifact_count": len(artifacts),
        "mapping_count": len(mappings),
        "issues": issues,
    }
    detail_metrics = {
        "field_value_macro_score": round(value_correctness, 4),
        "field_value_micro_score": round(value_micro, 4),
        "semantic_text_match_score": round(semantic_score, 4),
        "per_case_scores": per_case_scores,
    }
    report = {
        "status": "SUCCESS" if not issues else "NEEDS_REPAIR",
        "metrics": metrics,
        "artifact_validation": artifact_validation,
        "detail_metrics": detail_metrics,
        "comparison_count": len(comparisons),
        "field_count": len(expected_fields),
        "case_count": len(gold_cases),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    private_report = {**report, "comparisons": comparisons}
    public_feedback = {
        "status": report["status"],
        "metrics": metrics,
        "detail_metrics": detail_metrics,
        "artifact_issues": issues,
        "field_feedback": _public_field_feedback(comparisons, rules),
        "privacy": {"contains_gold_values": False, "contains_prediction_values": False},
    }
    report_path = output / "evaluation_report.json"
    private_path = output / "private_report.json"
    public_path = output / "public_feedback.json"
    _write_json(report_path, report)
    _write_json(private_path, private_report)
    _write_json(public_path, public_feedback)
    return {
        "evaluation_report": str(report_path),
        "private_report": str(private_path),
        "public_feedback": str(public_path),
    }


def _load_gold_records(
    path: Path,
    record_grain: str,
    *,
    known_paths: set[str] | None = None,
) -> dict[str, dict[str, list[Any]]]:
    root = path.expanduser().resolve()
    files = [root] if root.is_file() else sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    )
    loaded: list[tuple[Path, int, dict[str, Any]]] = []
    for file_path in files:
        loaded.extend(
            (file_path, index, record)
            for index, record in enumerate(_read_records(file_path), start=1)
        )
    dynamic_keys = _discover_dynamic_keys(record for _path, _index, record in loaded)
    result: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for file_path, index, record in loaded:
        case_id = _clean_id(_get_path(record, record_grain))
        if not case_id:
            case_id = _fallback_case_id(record, file_path, index)
        for field_path, value in _flatten_leaves(record, dynamic_keys=dynamic_keys):
            canonical_path = _canonicalize_gold_path(field_path, known_paths or set())
            result[case_id][canonical_path].append(value)
    return {case_id: dict(fields) for case_id, fields in result.items()}


def _canonicalize_gold_path(field_path: str, known_paths: set[str]) -> str:
    if field_path in known_paths:
        return field_path
    matches = []
    for candidate in known_paths:
        pattern = re.escape(candidate).replace(re.escape("[]"), r"(?:\[\]|\.[^.]+)")
        if re.fullmatch(pattern, field_path):
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else field_path


def _load_artifacts(manifest: dict[str, Any], base_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    artifacts: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for index, item in enumerate(manifest.get("artifacts", [])):
        if not isinstance(item, dict):
            issues.append(f"artifacts[{index}] must be an object")
            continue
        alias = str(item.get("alias") or f"artifact_{index}")
        if alias in artifacts:
            issues.append(f"Duplicate artifact alias: {alias}")
            continue
        path = Path(str(item.get("path") or "")).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            issues.append(f"Artifact does not exist: {path}")
            continue
        sheet_name = str(item.get("sheet_name") or "")
        try:
            records = _read_artifact_records(path, sheet_name)
        except ValueError as exc:
            issues.append(f"Artifact {alias}: {exc}")
            continue
        columns = sorted({str(key) for record in records for key in record})
        case_id_column = str(item.get("case_id_column") or "")
        if case_id_column and case_id_column not in columns:
            issues.append(f"Artifact {alias} is missing case id column: {case_id_column}")
            continue
        artifacts[alias] = {
            "path": str(path),
            "records": records,
            "columns": columns,
            "case_id_column": case_id_column,
            "sheet_name": sheet_name,
        }
    return artifacts, issues


def _read_artifact_records(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        return _read_records(path)
    if not sheet_name:
        raise ValueError("Excel artifact requires sheet_name")
    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:
        raise ValueError(f"cannot open Excel workbook: {exc}") from exc
    if sheet_name not in workbook.sheet_names:
        raise ValueError(f"Excel sheet does not exist: {sheet_name}")
    try:
        frame = pd.read_excel(path, sheet_name=sheet_name)
    except Exception as exc:
        raise ValueError(f"cannot read Excel sheet {sheet_name}: {exc}") from exc
    return [
        {str(key): _json_scalar(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _validate_mappings(
    payload: dict[str, Any],
    rules: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[str]]:
    mappings: list[dict[str, str]] = []
    issues: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload.get("mappings", [])):
        if not isinstance(item, dict):
            issues.append(f"mappings[{index}] must be an object")
            continue
        field_id = str(item.get("target_field_id") or "")
        alias = str(item.get("artifact") or "")
        column = str(item.get("source_column") or "")
        if field_id not in rules:
            issues.append(f"Unknown target field id: {field_id}")
            continue
        if field_id in seen:
            issues.append(f"Duplicate target field mapping: {field_id}")
            continue
        if alias not in artifacts:
            issues.append(f"Unknown artifact alias for {field_id}: {alias}")
            continue
        artifact_sheet = str(artifacts[alias].get("sheet_name") or "")
        mapping_sheet = str(item.get("sheet_name") or "")
        if artifact_sheet and mapping_sheet != artifact_sheet:
            issues.append(
                f"Mapping sheet_name for {field_id} does not match artifact {alias}: "
                f"{mapping_sheet or '<missing>'} != {artifact_sheet}"
            )
            continue
        if column not in artifacts[alias]["columns"]:
            issues.append(f"Artifact {alias} is missing mapped column {column} for {field_id}")
            continue
        case_column = str(item.get("case_id_column") or artifacts[alias]["case_id_column"])
        if case_column not in artifacts[alias]["columns"]:
            issues.append(f"Artifact {alias} is missing mapping case id column: {case_column}")
            continue
        if not any(not _is_missing(record.get(column)) for record in artifacts[alias]["records"]):
            issues.append(f"Artifact {alias} mapped column {column} has no non-empty values for {field_id}")
            continue
        seen.add(field_id)
        mappings.append(
            {
                "target_field_id": field_id,
                "artifact": alias,
                "source_column": column,
                "case_id_column": case_column,
                "sheet_name": artifact_sheet,
            }
        )
    return mappings, issues


def _compare_values(
    gold_values: list[Any],
    pred_values: list[Any],
    *,
    policy: str,
    semantic_scorer: SemanticScorer | None,
) -> tuple[bool, str, float | None]:
    if not pred_values:
        return False, "missing_value", None
    if policy == "collection_contains" or len(gold_values) > 1 or len(pred_values) > 1:
        gold_set = {_collection_key(value) for value in gold_values if not _is_missing(value)}
        pred_set = {_collection_key(value) for value in pred_values if not _is_missing(value)}
        if gold_set and gold_set.issubset(pred_set):
            return True, "collection_exact_or_contains", None
        if gold_set & pred_set:
            return False, "collection_partial_match", None
        return False, "canonical_mismatch", None
    gold = gold_values[0]
    pred = pred_values[0]
    if policy == "semantic_text":
        if _normalize(gold) == _normalize(pred):
            return True, "exact_match", 1.0
        if semantic_scorer is None:
            return False, "semantic_unavailable", None
        score = semantic_scorer(str(gold), str(pred))
        if score is not None and score >= 0.86:
            return True, "semantic_match_high", round(float(score), 4)
        if score is not None and score <= 0.72:
            return False, "semantic_match_low", round(float(score), 4)
        return False, "semantic_uncertain", None if score is None else round(float(score), 4)
    if policy == "numeric_exact":
        left, right = _number(gold), _number(pred)
        return (True, "numeric_equal", None) if left is not None and right is not None and abs(left - right) <= 1e-9 else (False, "numeric_mismatch", None)
    if policy == "datetime_normalized":
        left, right = _datetime(gold), _datetime(pred)
        return (True, "datetime_equal", None) if left and left == right else (False, "datetime_mismatch", None)
    return (True, "exact_match", None) if _normalize(gold) == _normalize(pred) else (False, "canonical_mismatch", None)


def _public_field_feedback(comparisons: list[dict[str, Any]], rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        if not item["passed"]:
            grouped[item["target_field_id"]].append(item)
    feedback = []
    for field_id, items in sorted(grouped.items()):
        rule = rules[field_id]
        error_types = Counter(str(item["error_type"]) for item in items)
        feedback.append(
            {
                "target_field_id": field_id,
                "target_field_path": rule.get("target_field_path"),
                "failed_case_count": len({item["case_id"] for item in items}),
                "error_types": dict(sorted(error_types.items())),
                "source_files": rule.get("source_files", []),
                "source_columns": rule.get("source_columns", []),
                "join_keys": rule.get("join_keys", []),
                "derivation_logic": rule.get("derivation_logic", {}),
                "recommended_action": _recommended_action(error_types),
            }
        )
    return feedback


def _recommended_action(errors: Counter[str]) -> str:
    if "missing_value" in errors:
        return "检查字段 mapping、来源筛选、join 和缺失值处理。"
    if "missing_rule" in errors:
        return "验证原始数据中定位该字段来源，并补充新的字段抽取规则。"
    if any(key.startswith("collection_") for key in errors):
        return "检查一对多记录收集、去重和聚合范围。"
    if any(key.startswith("semantic_") for key in errors):
        return "检查文本来源范围、抽取规则和文本规范化。"
    return "检查字段派生逻辑、类型转换和记录归属。"


def _is_explainable(
    rule: dict[str, Any],
    source_schema: dict[str, set[str]] | None = None,
) -> bool:
    files = [str(value) for value in rule.get("source_files") or []]
    columns = [str(value) for value in rule.get("source_columns") or []]
    if not (files and columns and rule.get("derivation_logic")):
        return False
    if source_schema is None:
        return True
    required = {column.rsplit(".", 1)[-1].replace("[]", "") for column in columns}
    for source_file in files:
        matches = [
            available
            for key, available in source_schema.items()
            if key == source_file or Path(key).name == Path(source_file).name
        ]
        if any(required <= {column.rsplit(".", 1)[-1].replace("[]", "") for column in available} for available in matches):
            return True
    return False


def _load_source_schema(path: Path) -> dict[str, set[str]]:
    root = path.expanduser().resolve()
    files = [root] if root.is_file() else _data_files(root)
    schema: dict[str, set[str]] = {}
    for source in files:
        try:
            records = _read_records(source, sample_limit=50)
        except Exception:
            continue
        columns = {field_path for record in records for field_path, _value in _flatten_leaves(record)}
        key = source.name if root.is_file() else source.relative_to(root).as_posix()
        schema[key] = columns
    return schema


def _get_path(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part.replace("[]", ""))
    return current


def _fallback_case_id(record: dict[str, Any], path: Path, index: int) -> str:
    for key in ("case_id", "record_id", "id"):
        if not _is_missing(record.get(key)):
            return _clean_id(record[key])
    id_keys = sorted(key for key in record if str(key).lower().endswith(("_id", "key")))
    if id_keys:
        return _clean_id(record[id_keys[0]])
    return path.parent.name if path.parent.name else f"case_{index:04d}"


def _clean_id(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _collapse(values: list[Any]) -> Any:
    return values[0] if len(values) == 1 else values


def _collection_key(value: Any) -> str:
    number = _number(value)
    if number is not None:
        return f"num:{number:g}"
    dt = _datetime(value)
    if dt:
        return f"dt:{dt}"
    return f"text:{_normalize(value)}"


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    text = str(value).strip().replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _datetime(value: Any) -> str | None:
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except Exception:
        return None
    return parsed.isoformat()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na", "n/a"}


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _embedding_config(config_path: str | Path | None) -> dict[str, Any]:
    candidates = []
    if config_path:
        candidates.append(Path(config_path).expanduser())
    root = Path(__file__).resolve().parents[1]
    candidates.extend([root / "model_config.local.yaml", root / "model_config.yaml"])
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        embedding = payload.get("embedding") if isinstance(payload, dict) else None
        if isinstance(embedding, dict):
            return embedding
    return {}
