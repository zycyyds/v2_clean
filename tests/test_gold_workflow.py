from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context
from skills.analyze_gold_examples.skill import analyze_gold_examples_tool
from workflow.dataset_profiles import build_dataset_profile_suggestions
from workflow.gold_analysis import (
    analyze_training_examples,
    discover_example_pairs,
    validate_field_rule_completion,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_discovers_flexible_example_pairs_and_all_nested_leaf_fields(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    case_a = examples / "case-a"
    raw_a = case_a / "source_records"
    raw_a.mkdir(parents=True)
    pd.DataFrame(
        [{"visit_key": "v1", "code": "A10", "value": 3.5}]
    ).to_csv(raw_a / "events.csv", index=False)
    _write_json(
        case_a / "expected.json",
        {
            "visit_key": "v1",
            "summary": {"age": 42, "comment": "stable"},
            "events": [{"code": "A10", "value": 3.5}],
        },
    )

    case_b = examples / "case-b"
    raw_b = case_b / "raw"
    raw_b.mkdir(parents=True)
    pd.DataFrame(
        [{"visit_key": "v2", "code": "B20", "value": 7.0}]
    ).to_csv(raw_b / "events.csv", index=False)
    pd.DataFrame(
        [{"visit_key": "v2", "age": 55, "comment": "follow up"}]
    ).to_csv(case_b / "gold_table.csv", index=False)

    pairs = discover_example_pairs(examples)
    assert [pair.case_id for pair in pairs] == ["case-a", "case-b"]
    assert pairs[0].gold_paths[0].name == "expected.json"
    assert pairs[0].raw_root.name == "source_records"

    output = tmp_path / "explorer"
    result = analyze_training_examples(examples, output)

    rules = json.loads(Path(result["field_extraction_rules"]).read_text(encoding="utf-8"))
    paths = {item["target_field_path"] for item in rules["rules"]}
    assert {
        "visit_key",
        "summary.age",
        "summary.comment",
        "events[].code",
        "events[].value",
        "age",
        "comment",
    }.issubset(paths)
    assert len({item["target_field_id"] for item in rules["rules"]}) == len(rules["rules"])
    assert all(item["evaluation_policy"] for item in rules["rules"])
    assert all(item["capability_type"] for item in rules["rules"])

    assert Path(result["data_analysis_report"]).is_file()
    assert Path(result["extraction_task_plan"]).is_file()
    run_record = json.loads(Path(result["explorer_run_record"]).read_text(encoding="utf-8"))
    assert run_record["status"] == "SUCCESS"
    assert run_record["example_count"] == 2
    assert run_record["field_count"] == len(rules["rules"])


def test_normalizes_dynamic_dictionary_keys_to_stable_field_ids(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    dynamic_keys_by_case = {
        "case-a": ["ALT_2153-10-29-22:00", "AST_2153-10-29-22:00"],
        "case-b": ["ALT_2180-03-10-08:15", "AST_2180-03-10-08:15"],
    }
    for index, (case_id, dynamic_keys) in enumerate(dynamic_keys_by_case.items(), start=1):
        case = examples / case_id
        raw = case / "raw"
        raw.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "case_id": case_id,
                    "检查时间": f"20{index:02d}-01-01 08:00",
                    "itemid": 50000 + index,
                    "结果数值": float(index),
                }
            ]
        ).to_csv(raw / "labs.csv", index=False)
        _write_json(
            case / "gold.json",
            {
                "case_id": case_id,
                "实验室检验": {
                    key: {
                        "检查时间": f"20{index:02d}-01-01 08:00",
                        "项目明细": {
                            str(50000 + index): {
                                "itemid": 50000 + index,
                                "结果数值": float(index),
                            }
                        },
                    }
                    for key in dynamic_keys
                },
            },
        )

    result = analyze_training_examples(examples, tmp_path / "out")
    payload = json.loads(Path(result["field_extraction_rules"]).read_text(encoding="utf-8"))
    rules_by_path = {rule["target_field_path"]: rule for rule in payload["rules"]}

    expected_paths = {
        "实验室检验[].检查时间",
        "实验室检验[].项目明细[].itemid",
        "实验室检验[].项目明细[].结果数值",
    }
    assert expected_paths.issubset(rules_by_path)
    assert not any("2153" in path or "2180" in path or "5000" in path for path in rules_by_path)
    assert all(rules_by_path[path]["evidence"]["example_coverage"] == 1.0 for path in expected_paths)


def test_extraction_task_plan_covers_each_executable_rule_exactly_once(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    for case_id, age, values in (("case-a", 42, [1.0, 2.0]), ("case-b", 55, [3.0])):
        case = examples / case_id
        raw = case / "raw"
        raw.mkdir(parents=True)
        pd.DataFrame([{"case_id": case_id, "age": age}]).to_csv(raw / "people.csv", index=False)
        pd.DataFrame([{"value": value} for value in values]).to_csv(
            raw / "labs.csv",
            index=False,
        )
        _write_json(
            case / "gold.json",
            {
                "case_id": case_id,
                "demographics": {"age": age},
                "labs": [{"value": value} for value in values],
                "notes": {"summary": "not available from the structured source"},
            },
        )

    result = analyze_training_examples(examples, tmp_path / "out")
    rules_payload = json.loads(Path(result["field_extraction_rules"]).read_text(encoding="utf-8"))
    task_payload = json.loads(Path(result["extraction_task_plan"]).read_text(encoding="utf-8"))

    executable_ids = {
        rule["target_field_id"] for rule in rules_payload["rules"] if rule["status"] == "supported"
    }
    covered_ids = [rule_id for task in task_payload["tasks"] for rule_id in task["rule_ids"]]

    assert task_payload["executable_rule_count"] == len(executable_ids)
    assert set(covered_ids) == executable_ids
    assert len(covered_ids) == len(set(covered_ids))
    assert all(task["required"] is True for task in task_payload["tasks"])
    assert all(task["capability_type"] and task["target_category"] for task in task_payload["tasks"])
    assert {(task["target_category"], task["capability_type"]) for task in task_payload["tasks"]} == {
        ("case_id", "structured_extract"),
        ("demographics", "structured_extract"),
        ("labs", "aggregate"),
    }
    assert set(task_payload["unassigned_rule_ids"]) == {
        rule["target_field_id"] for rule in rules_payload["rules"] if rule["status"] != "supported"
    }


def test_preserves_empty_gold_top_level_categories_for_workbook_layout(tmp_path: Path) -> None:
    case = tmp_path / "examples" / "case-a"
    raw = case / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame([{"case_id": "case-a", "value": 1}]).to_csv(raw / "source.csv", index=False)
    _write_json(
        case / "gold.json",
        {
            "profile": {"value": 1},
            "procedures": [],
            "free_text": {},
        },
    )

    result = analyze_training_examples(case.parent, tmp_path / "out")
    rules = json.loads(Path(result["field_extraction_rules"]).read_text(encoding="utf-8"))
    tasks = json.loads(Path(result["extraction_task_plan"]).read_text(encoding="utf-8"))

    assert rules["target_categories"] == ["free_text", "procedures", "profile"]
    assert tasks["target_categories"] == rules["target_categories"]


def test_marks_multiple_equally_plausible_sources_as_ambiguous(tmp_path: Path) -> None:
    case = tmp_path / "examples" / "id-1"
    raw = case / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame([{"case_id": "id-1", "score": 8}]).to_csv(raw / "left.csv", index=False)
    pd.DataFrame([{"case_id": "id-1", "score": 8}]).to_csv(raw / "right.csv", index=False)
    _write_json(case / "gold.json", {"case_id": "id-1", "score": 8})

    result = analyze_training_examples(tmp_path / "examples", tmp_path / "out")
    payload = json.loads(Path(result["field_extraction_rules"]).read_text(encoding="utf-8"))
    score_rule = next(item for item in payload["rules"] if item["target_field_path"] == "score")

    assert score_rule["status"] == "ambiguous"
    assert sorted(Path(path).name for path in score_rule["source_files"]) == ["left.csv", "right.csv"]


def test_gold_skill_publishes_the_four_dataset_agnostic_artifacts(tmp_path: Path) -> None:
    case = tmp_path / "examples" / "case-1"
    raw = case / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame([{"record_id": "case-1", "value": 2}]).to_csv(raw / "source.csv", index=False)
    _write_json(case / "gold.json", {"record_id": "case-1", "value": 2})
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    try:
        response = analyze_gold_examples_tool(
            task_text="generic",
            gold_examples_path=str(tmp_path / "examples"),
        )
    finally:
        clear_phase_context()
    payload = json.loads(response.content[0]["text"])

    assert payload["status"] == "SUCCESS"
    assert Path(payload["artifacts"]["field_extraction_rules"]).is_file()
    assert Path(payload["artifacts"]["extraction_task_plan"]).is_file()
    assert Path(payload["artifacts"]["data_analysis_report"]).is_file()
    assert Path(payload["artifacts"]["explorer_run_record"]).is_file()
    assert payload["artifacts"]["field_count"] == 2


def test_real_mimic_gold_preserves_all_58_logical_terminals(tmp_path: Path) -> None:
    examples = Path(__file__).parents[1] / "mimic_4case_train_validation_split" / "train_examples"

    result = analyze_training_examples(examples, tmp_path / "explorer")
    payload = json.loads(Path(result["field_extraction_rules"]).read_text(encoding="utf-8"))
    rules_by_path = {rule["target_field_path"]: rule for rule in payload["rules"]}

    assert payload["field_count"] == 58
    assert {
        "住院用药[].source.row_id",
        "实验室检验[].项目明细[].参考上限",
        "实验室检验[].项目明细[].参考下限",
        "就诊文本",
        "手术与操作[]",
        "生命体征.肝病相关指标.总蛋白",
        "生命体征.肝病相关指标.白蛋白",
        "病历.人口学.种族",
        "诊断列表[].source.row_id",
    }.issubset(rules_by_path)
    assert rules_by_path["就诊文本"]["target_type"] == "object"
    assert rules_by_path["手术与操作[]"]["target_type"] == "array"
    report = json.loads(Path(result["data_analysis_report"]).read_text(encoding="utf-8"))
    assert len(report["field_analysis"]) == 58
    assert {
        "target_field_id",
        "target_field_path",
        "source_files",
        "source_columns",
        "join_keys",
        "derivation_logic",
        "evidence",
        "confidence",
        "status",
    } <= set(report["field_analysis"][0])


def test_mimic_profile_suggestions_are_validated_against_real_schema(tmp_path: Path) -> None:
    examples = Path(__file__).parents[1] / "mimic_4case_train_validation_split" / "train_examples"
    result = analyze_training_examples(examples, tmp_path / "explorer")

    suggestions = build_dataset_profile_suggestions(result["field_extraction_rules"], examples)
    by_path = {item["target_field_path"]: item for item in suggestions["updates"]}

    assert suggestions["dataset_profile"] == "mimic"
    assert by_path["病历.入院时间"]["source_files"] == ["structured/admissions.csv"]
    assert "admittime" in by_path["病历.入院时间"]["source_columns"]
    assert by_path["病历.入院时间"]["evidence"]["schema_validated"] is True
    assert by_path["住院用药[].source.table"]["derivation_logic"] == {
        "operation": "constant",
        "value": "hosp/prescriptions",
        "description": "来源表名由已验证的 MIMIC 文件路径确定。",
    }
    assert by_path["住院用药[].source.row_id"]["status"] == "unsupported"
    assert by_path["实验室检验[].项目明细[].source.row_id"]["source_columns"] == ["labevent_id"]
    assert by_path["实验室检验[].项目明细[].itemid"]["status"] == "supported"
    assert by_path["实验室检验[].项目明细[].itemid"]["source_columns"] == ["itemid"]
    assert by_path["诊断列表[].icd_code"]["source_columns"] == ["icd_code"]
    assert by_path["诊断列表[].icd_version"]["source_columns"] == ["icd_version"]
    assert by_path["诊断列表[].是否主诊断"]["source_columns"] == ["seq_num"]
    assert by_path["诊断列表[].是否主诊断"]["derivation_logic"]["operation"] == "equals"
    assert by_path["诊断列表[].是否主诊断"]["derivation_logic"]["value"] == 1
    assert by_path["生命体征.肝病相关指标.ALT"]["status"] == "ambiguous"
    assert by_path["生命体征.体温_C"]["status"] == "unsupported"
    assert "chartevents" in by_path["生命体征.体温_C"]["unsupported_reason"]


def test_rule_completion_rejects_pending_and_nonexistent_supported_sources(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "field_count": 1,
                "rules": [
                    {
                        "target_field_id": "field_a",
                        "target_field_path": "profile.value",
                        "status": "pending",
                        "source_files": [],
                        "source_columns": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_field_rule_completion(rules_path, tmp_path / "raw")
    except ValueError as exc:
        assert "pending" in str(exc)
    else:
        raise AssertionError("pending rule must fail completion validation")

    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    payload["rules"][0].update(
        {"status": "supported", "source_files": ["missing.csv"], "source_columns": ["value"]}
    )
    rules_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "raw").mkdir()
    try:
        validate_field_rule_completion(rules_path, tmp_path / "raw")
    except ValueError as exc:
        assert "missing.csv" in str(exc)
    else:
        raise AssertionError("supported rule with missing source must fail completion validation")


def test_rule_completion_accepts_constant_derivation_without_source_column(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([{"case_id": "a", "value": 1}]).to_csv(raw / "events.csv", index=False)
    rules_path = tmp_path / "rules.json"
    _write_json(
        rules_path,
        {
            "field_count": 1,
            "rules": [{
                "target_field_id": "field_source_table",
                "target_field_path": "events[].source.table",
                "status": "supported",
                "source_files": ["events.csv"],
                "source_columns": [],
                "derivation_logic": {
                    "operation": "constant",
                    "constant_value": "events",
                },
            }],
        },
    )

    result = validate_field_rule_completion(rules_path, raw)

    assert result["complete"] is True
