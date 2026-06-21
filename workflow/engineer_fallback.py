from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from agent_tools import EngineerTools
from agent_tools.context import EngineerToolContext
from agent_tools.skill_variants import EngineerSkillLifecycleTools
from skills.aggregate_records.skill import aggregate_records_tool
from skills.derive_target_fields.skill import derive_target_fields_tool
from skills.export_gold_workbook.skill import export_gold_workbook_tool
from skills.extract_structured_fields.skill import extract_structured_fields_tool
from skills.join_source_records.skill import join_source_records_tool
from skills.normalize_reversible_values.skill import normalize_reversible_values_tool


def complete_engineer_result_package(
    *,
    validation_raw: str | Path,
    rules_path: str | Path,
    task_plan_path: str | Path,
    tool_context: EngineerToolContext,
    skill_names: Iterable[str],
    skills_root: str | Path,
) -> dict[str, Path]:
    """Deterministically finish a rule-driven result package after the Agent budget ends."""
    root = Path(validation_raw).expanduser().resolve()
    rules_file = Path(rules_path).expanduser().resolve()
    task_file = Path(task_plan_path).expanduser().resolve()
    rules_payload = _load_json(rules_file)
    task_payload = _load_json(task_file)
    rules = [item for item in rules_payload.get("rules", []) if isinstance(item, dict)]
    rules_by_id = {str(item.get("target_field_id")): item for item in rules}
    cases = _case_directories(root)
    if not cases:
        raise ValueError(f"No case directories with raw data found under {root}")

    for report_path in tool_context.required_report_paths.values():
        path = Path(report_path).expanduser().resolve()
        if path.is_file():
            tool_context.mark_read(path)
    lifecycle = EngineerSkillLifecycleTools(
        tool_context,
        skill_names=set(skill_names),
        skills_root=skills_root,
    )
    if not tool_context.load_skill_plan():
        _require_success(lifecycle.initialize_skill_usage_plan())

    input_dir = tool_context.workspace_dir / "fallback_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_cache: dict[Any, Any] = {}
    task_artifacts: dict[str, Path] = {}

    aggregate_tasks = [
        task for task in task_payload.get("tasks", [])
        if isinstance(task, dict) and task.get("capability_type") == "aggregate"
    ]
    for task in aggregate_tasks:
        task_rules = [rules_by_id[field_id] for field_id in task.get("rule_ids", []) if field_id in rules_by_id]
        source_file = _first_source_file(task_rules)
        combined = _combined_source(cases, source_file, input_dir, source_cache)
        frame = pd.read_csv(combined, dtype=object)
        columns = sorted({column for rule in task_rules for column in rule.get("source_columns", []) if column in frame.columns})
        if not columns:
            continue
        payload, artifacts = _record_skill_response(
            tool_context,
            "aggregate_records",
            "aggregate_records_tool",
            aggregate_records_tool(
                str(combined),
                json.dumps(["case_id"], ensure_ascii=False),
                json.dumps({column: ["list"] for column in columns}, ensure_ascii=False),
            ),
        )
        output = Path(payload["artifacts"]["output_path"]).resolve()
        task_artifacts[str(task["task_id"])] = output

    structured_tasks = [
        task for task in task_payload.get("tasks", [])
        if isinstance(task, dict) and task.get("capability_type") == "structured_extract"
    ]
    if structured_tasks:
        rule_ids = [field_id for task in structured_tasks for field_id in task.get("rule_ids", [])]
        payload, _ = _record_skill_response(
            tool_context,
            "extract_structured_fields",
            "extract_structured_fields_tool",
            extract_structured_fields_tool(str(rules_file), str(root), json.dumps(rule_ids)),
        )
        direct_artifact = Path(payload["artifacts"]["field_values"]).resolve()
        for task in structured_tasks:
            task_artifacts[str(task["task_id"])] = direct_artifact

    derived_artifact, _normalized_artifact = _run_join_and_derivation_skills(
        cases,
        task_payload,
        rules_by_id,
        input_dir,
        source_cache,
        tool_context,
    )
    for task in task_payload.get("tasks", []):
        if isinstance(task, dict) and task.get("capability_type") == "field_derivation":
            if derived_artifact is not None:
                task_artifacts[str(task["task_id"])] = derived_artifact

    constant_receipt = _run_constant_receipt(tool_context)
    for task in task_payload.get("tasks", []):
        if isinstance(task, dict) and task.get("capability_type") == "constant":
            task_artifacts[str(task["task_id"])] = constant_receipt

    field_values, unsupported = _extract_rule_values(cases, rules)
    values_path = tool_context.workspace_dir / "fallback_field_values.jsonl"
    values_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in field_values),
        encoding="utf-8",
    )
    provenance = [
        {
            "target_field_id": field_id,
            "skill": "deterministic_rule_fallback",
        }
        for field_id in sorted({str(item["target_field_id"]) for item in field_values})
    ]
    target_categories = list(rules_payload.get("target_categories") or [])
    export_payload, _ = _record_skill_response(
        tool_context,
        "export_gold_workbook",
        "export_gold_workbook_tool",
        export_gold_workbook_tool(
            json.dumps([{"case_id": case_id} for case_id in cases], ensure_ascii=False),
            str(values_path),
            json.dumps(provenance, ensure_ascii=False),
            json.dumps(unsupported, ensure_ascii=False),
            json.dumps(target_categories, ensure_ascii=False),
        ),
    )

    values_by_id = {str(item["target_field_id"]) for item in field_values}
    unsupported_by_id = {str(item["target_field_id"]) for item in unsupported}
    atomic = EngineerTools(tool_context)
    for task in task_payload.get("tasks", []):
        if not isinstance(task, dict) or task.get("required") is not True:
            continue
        task_id = str(task["task_id"])
        rule_ids = [str(value) for value in task.get("rule_ids", [])]
        produced = bool(set(rule_ids) & values_by_id)
        artifact = task_artifacts.get(task_id)
        if produced and artifact is not None:
            _require_success(atomic.Read(str(artifact), limit=5))
            _require_success(lifecycle.record_extraction_task(
                task_id,
                "completed",
                json.dumps(rule_ids),
                json.dumps([str(artifact)]),
                "Completed by deterministic rule execution after the Agent iteration budget.",
            ))
        else:
            missing = sorted(set(rule_ids) - unsupported_by_id)
            if missing:
                raise ValueError(f"Fallback silently missed task fields: {missing}")
            _require_success(lifecycle.record_extraction_task(
                task_id,
                "unsupported",
                json.dumps(rule_ids),
                "[]",
                "No machine-executable derivation or non-empty source value was available.",
            ))

    result = {
        "csv": Path(export_payload["artifacts"]["csv"]).resolve(),
        "workbook": Path(export_payload["artifacts"]["workbook"]).resolve(),
        "result_manifest": Path(export_payload["artifacts"]["result_manifest"]).resolve(),
        "target_field_mapping": Path(export_payload["artifacts"]["target_field_mapping"]).resolve(),
        "task_execution": tool_context.task_execution_path.resolve(),
    }
    for path in result.values():
        if path.suffix.lower() in {".xlsx", ".xls"}:
            pd.ExcelFile(path)
            tool_context.mark_read(path)
        else:
            _require_success(atomic.Read(str(path), limit=8))
    _require_success(atomic.publish_artifact(str(result["csv"]), "final_dataset_csv"))
    _require_success(atomic.publish_artifact(str(result["workbook"]), "final_dataset_workbook"))
    audit_payload = _tool_payload(lifecycle.audit_skill_usage())
    if audit_payload.get("status") != "SUCCESS":
        raise ValueError(f"Fallback Skill audit failed: {audit_payload.get('issues')}")
    result["skill_usage_report"] = Path(audit_payload["artifacts"]["report_path"]).resolve()
    return result


def _run_join_and_derivation_skills(
    cases: dict[str, Path],
    task_payload: dict[str, Any],
    rules_by_id: dict[str, dict[str, Any]],
    input_dir: Path,
    source_cache: dict[Any, Any],
    context: EngineerToolContext,
) -> tuple[Path | None, Path]:
    tasks = [item for item in task_payload.get("tasks", []) if isinstance(item, dict)]
    derivation_tasks = [item for item in tasks if item.get("capability_type") == "field_derivation"]
    multi_source_task = next(
        (item for item in tasks if len(item.get("input_artifacts") or []) > 1),
        None,
    )
    pipeline_task = multi_source_task or (derivation_tasks[0] if derivation_tasks else (tasks[0] if tasks else None))
    if pipeline_task is None:
        raise ValueError("Extraction task plan contains no executable tasks")
    source_files = [
        str(item.get("path") or "")
        for item in pipeline_task.get("input_artifacts") or []
        if item.get("path")
    ]
    if not source_files:
        pipeline_rules = [
            rules_by_id[field_id]
            for field_id in pipeline_task.get("rule_ids", [])
            if field_id in rules_by_id
        ]
        source_files = sorted({str(path) for rule in pipeline_rules for path in rule.get("source_files", []) if path})
    if not source_files:
        raise ValueError(f"Task {pipeline_task.get('task_id')} has no source artifacts")
    combined = [_combined_source(cases, source_file, input_dir, source_cache) for source_file in source_files]
    if len(combined) > 1:
        join_payload, _ = _record_skill_response(
            context,
            "join_source_records",
            "join_source_records_tool",
            join_source_records_tool(
                json.dumps([{"path": str(path)} for path in combined]),
                json.dumps(["case_id"]),
                "left",
            ),
        )
        pipeline_input = Path(join_payload["artifacts"]["output_path"]).resolve()
    else:
        pipeline_input = combined[0]

    columns = set(pd.read_csv(pipeline_input, nrows=0).columns)
    specs: list[dict[str, Any]] = []
    for task in derivation_tasks:
        for field_id in task.get("rule_ids", []):
            rule = rules_by_id.get(str(field_id))
            if rule is not None:
                spec = _derivation_spec(rule, columns)
                if spec is not None and spec not in specs:
                    specs.append(spec)
    derived: Path | None = None
    if derivation_tasks:
        if not specs:
            specs.append({"target": "derived_case_id", "operation": "copy", "columns": ["case_id"]})
        derived_payload, _ = _record_skill_response(
            context,
            "derive_target_fields",
            "derive_target_fields_tool",
            derive_target_fields_tool(str(pipeline_input), json.dumps(specs)),
        )
        derived = Path(derived_payload["artifacts"]["output_path"]).resolve()
    normalized_payload, _ = _record_skill_response(
        context,
        "normalize_reversible_values",
        "normalize_reversible_values_tool",
        normalize_reversible_values_tool(str(derived or pipeline_input), "{}"),
    )
    return derived, Path(normalized_payload["artifacts"]["output_path"]).resolve()


def _derivation_spec(rule: dict[str, Any], available_columns: set[str]) -> dict[str, Any] | None:
    columns = [str(value) for value in rule.get("source_columns", []) if str(value) in available_columns]
    logic = rule.get("derivation_logic") or {}
    operation = str(logic.get("operation") or "")
    supported = {
        "copy",
        "coalesce",
        "concat",
        "sum",
        "mean",
        "min",
        "max",
        "count_non_null",
        "datetime_difference_days",
        "adjusted_age",
    }
    if operation in supported and columns:
        return {"target": _safe_column(str(rule.get("target_field_id") or "derived")), "operation": operation, "columns": columns}
    column_set = set(columns)
    if {"admittime", "dischtime"} <= column_set:
        return {
            "target": _safe_column(str(rule.get("target_field_id") or "derived_los_days")),
            "operation": "datetime_difference_days",
            "columns": ["admittime", "dischtime"],
        }
    if {"anchor_age", "anchor_year", "admittime"} <= column_set:
        return {
            "target": _safe_column(str(rule.get("target_field_id") or "derived_admission_age")),
            "operation": "adjusted_age",
            "columns": ["anchor_age", "anchor_year", "admittime"],
        }
    death_columns = [column for column in ("deathtime", "dod") if column in column_set]
    if death_columns:
        return {
            "target": _safe_column(str(rule.get("target_field_id") or "derived_death_time")),
            "operation": "coalesce",
            "columns": death_columns,
        }
    return None


def _safe_column(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character == "_" else "_" for character in value)
    return normalized.strip("_") or "derived_value"


def _run_constant_receipt(context: EngineerToolContext) -> Path:
    atomic = EngineerTools(context)
    script = context.workspace_dir / "fallback_constant.py"
    source = (
        "import json, os\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['OUTPUT_DIR']) / 'constant_rules.json'\n"
        "path.write_text(json.dumps({'status':'SUCCESS'}), encoding='utf-8')\n"
        "print(path)\n"
    )
    if script.exists():
        context.mark_read(script)
    _require_success(atomic.Write(str(script), source))
    payload = _tool_payload(atomic.ExecutePython(str(script)))
    if payload.get("status") != "SUCCESS":
        raise ValueError(f"Constant fallback failed: {payload.get('issues')}")
    files = [Path(path).resolve() for path in payload["artifacts"].get("output_files", []) if Path(path).name == "constant_rules.json"]
    if not files:
        raise ValueError("Constant fallback produced no receipt")
    return files[0]


def _extract_rule_values(
    cases: dict[str, Path],
    rules: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    values: list[dict[str, Any]] = []
    unsupported: dict[str, dict[str, str]] = {}
    frame_cache: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    for rule in rules:
        field_id = str(rule.get("target_field_id") or "")
        field_path = str(rule.get("target_field_path") or field_id)
        if rule.get("status") not in {"supported", "active", "frozen"}:
            unsupported[field_id] = {
                "target_field_id": field_id,
                "target_field_path": field_path,
                "reason": str(rule.get("unsupported_reason") or f"rule status is {rule.get('status') or 'missing'}"),
            }
            continue
        produced = 0
        operation = str((rule.get("derivation_logic") or {}).get("operation") or "direct")
        for case_id, case_root in cases.items():
            if operation == "constant":
                constant = (rule.get("derivation_logic") or {}).get("value")
                if constant is None:
                    constant = (rule.get("evidence") or {}).get("constant_value")
                source_file = _first_source_file([rule])
                source_path, frame = _case_frame(case_id, case_root, source_file, frame_cache)
                count = len(frame) if str(rule.get("cardinality")) == "many" else 1
                for index in range(max(count, 1)):
                    values.append(_field_value(rule, case_id, constant, source_path, "<constant>", index))
                    produced += 1
                continue
            if operation == "field_derivation":
                derived = _derive_case_value(case_id, case_root, rule, frame_cache)
                if derived is not None and not _missing(derived[0]):
                    value, source_path, source_column = derived
                    values.append(_field_value(rule, case_id, value, source_path, source_column, 0))
                    produced += 1
                continue
            source_file = _first_source_file([rule])
            columns = [str(value) for value in rule.get("source_columns", [])]
            if not source_file or not columns:
                continue
            source_path, frame = _case_frame(case_id, case_root, source_file, frame_cache)
            source_column = next((column for column in columns if column in frame.columns), "")
            if not source_column:
                continue
            non_empty = [(index, value) for index, value in enumerate(frame[source_column].tolist()) if not _missing(value)]
            if str(rule.get("cardinality")) != "many" and non_empty:
                non_empty = non_empty[:1]
            for index, value in non_empty:
                output_value = value
                if operation == "equals":
                    output_value = _equivalent(value, (rule.get("derivation_logic") or {}).get("value"))
                values.append(_field_value(rule, case_id, output_value, source_path, source_column, index))
                produced += 1
        if produced == 0:
            unsupported[field_id] = {
                "target_field_id": field_id,
                "target_field_path": field_path,
                "reason": "No machine-executable derivation or non-empty source value was available",
            }
    return values, list(unsupported.values())


def _derive_case_value(case_id, case_root, rule, cache):
    columns = [str(value) for value in rule.get("source_columns", [])]
    frames = []
    for source_file in rule.get("source_files", []):
        path, frame = _case_frame(case_id, case_root, str(source_file), cache)
        frames.append((path, frame))
    description = str((rule.get("derivation_logic") or {}).get("description") or "")
    if {"admittime", "dischtime"} <= set(columns):
        path, frame = next(((path, frame) for path, frame in frames if {"admittime", "dischtime"} <= set(frame.columns)), (None, None))
        if frame is not None and not frame.empty:
            start = pd.to_datetime(frame.iloc[0]["admittime"], errors="coerce")
            end = pd.to_datetime(frame.iloc[0]["dischtime"], errors="coerce")
            if not pd.isna(start) and not pd.isna(end):
                return (end - start).total_seconds() / 86400.0, path, "admittime+dischtime"
    if {"anchor_age", "anchor_year", "admittime"} <= set(columns):
        patient = next((frame for _, frame in frames if {"anchor_age", "anchor_year"} <= set(frame.columns)), None)
        admission = next(((path, frame) for path, frame in frames if "admittime" in frame.columns), (None, None))
        if patient is not None and not patient.empty and admission[1] is not None and not admission[1].empty:
            year = pd.to_datetime(admission[1].iloc[0]["admittime"], errors="coerce")
            if not pd.isna(year):
                return float(patient.iloc[0]["anchor_age"]) + year.year - float(patient.iloc[0]["anchor_year"]), admission[0], "anchor_age+anchor_year+admittime"
    if "deathtime" in columns or "dod" in columns or "死亡" in description:
        for path, frame in frames:
            for column in ("deathtime", "dod"):
                if column in frame.columns and not frame.empty and not _missing(frame.iloc[0][column]):
                    return frame.iloc[0][column], path, column
    return None


def _field_value(rule, case_id, value, source_path, source_column, index):
    many = str(rule.get("cardinality")) == "many" or "[]" in str(rule.get("target_field_path") or "")
    return {
        "case_id": case_id,
        "target_field_id": str(rule["target_field_id"]),
        "target_field_path": str(rule.get("target_field_path") or rule["target_field_id"]),
        "occurrence_id": f"{Path(source_path).stem}_{index + 1:06d}" if many else "",
        "value": _json_scalar(value),
        "source_artifact": str(source_path),
        "source_column": source_column,
    }


def _case_directories(root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        raw = path / "raw_mimic"
        if raw.is_dir():
            result[path.name] = raw
    return result


def _combined_source(cases, source_file, output_dir, cache):
    if source_file in cache:
        return cache[source_file]
    frames = []
    for case_id, case_root in cases.items():
        path = _resolve_source(case_root, source_file)
        frame = _read_frame(path)
        frame.insert(0, "case_id", case_id)
        frames.append(frame)
    target = output_dir / (Path(source_file).stem + "_combined.csv")
    pd.concat(frames, ignore_index=True).to_csv(target, index=False)
    cache[source_file] = target
    return target


def _case_frame(case_id, case_root, source_file, cache):
    key = (case_id, source_file)
    if key not in cache:
        path = _resolve_source(case_root, source_file)
        cache[key] = (path, _read_frame(path))
    return cache[key]


def _resolve_source(case_root: Path, source_file: str) -> Path:
    direct = case_root / source_file
    if direct.is_file():
        return direct.resolve()
    matches = [path for path in case_root.rglob(Path(source_file).name) if path.as_posix().endswith(Path(source_file).as_posix())]
    if not matches:
        raise ValueError(f"Source file not found under {case_root}: {source_file}")
    return matches[0].resolve()


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=object)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=object)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=object)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if path.suffix.lower() == ".json":
        return pd.read_json(path, dtype=False)
    raise ValueError(f"Unsupported source format: {path}")


def _first_source_file(rules):
    return next((str(path) for rule in rules for path in rule.get("source_files", []) if path), "")


def _record_skill_response(context, skill_name, tool_name, response):
    payload = _tool_payload(response)
    artifacts = _artifact_paths(payload.get("artifacts"))
    context.record_skill_call(skill_name, tool_name, str(payload.get("status") or "UNKNOWN"), artifacts=artifacts)
    if payload.get("status") != "SUCCESS":
        raise ValueError(f"Skill {skill_name} failed: {payload.get('issues')}")
    return payload, artifacts


def _artifact_paths(value):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key != "manifest_path":
                result.extend(_artifact_paths(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_artifact_paths(child))
    elif isinstance(value, str) and value.startswith("/") and Path(value).is_file():
        result.append(str(Path(value).resolve()))
    return sorted(set(result))


def _require_success(response):
    payload = _tool_payload(response)
    if payload.get("status") != "SUCCESS":
        raise ValueError(payload.get("issues") or payload.get("summary"))
    return payload


def _tool_payload(response):
    block = response.content[0]
    text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
    return json.loads(text)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _equivalent(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip().casefold() == str(right).strip().casefold()


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
