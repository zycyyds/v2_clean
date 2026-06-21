from __future__ import annotations

import json
from pathlib import Path

from agent.two_phase_agents import create_explorer_toolkit
from agent_tools.report_tools import ExplorerReportTools
from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context
from skills.analyze_gold_examples import skill as gold_skill
from skills.list_data_files.skill import list_data_files_tool
from skills.run_python_code.skill import run_python_code_tool
from skills.sample_table.skill import sample_table_tool
from workflow.dataset_profiles import build_dataset_profile_suggestions
from workflow.gold_analysis import analyze_training_examples, validate_field_rule_completion


def _payload(response) -> dict:
    return json.loads(response.content[0]["text"])


def test_gold_analysis_forces_rules_into_current_step(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    captured: dict[str, Path] = {}

    def fake_build_gold_guidance(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        rules_dir = Path(kwargs["learned_rules_dir"])
        captured["output_dir"] = output_dir
        captured["rules_dir"] = rules_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        rules_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "gold_field_provenance_report": str(output_dir / "gold_field_provenance_report.json"),
            "planner_extraction_brief": str(output_dir / "planner_extraction_brief.json"),
            "learned_rules_path": str(rules_dir / "learned_rules.json"),
        }
        for path in artifacts.values():
            Path(path).write_text("{}", encoding="utf-8")
        return {"artifacts": artifacts, "field_provenance": [], "learned_rules": {"rule_count": 0}}

    monkeypatch.setattr(gold_skill, "build_gold_guidance", fake_build_gold_guidance)
    try:
        result = _payload(gold_skill.analyze_gold_examples_tool(
            task_text="demo",
            learned_rules_dir=str(Path(phase["phase_root"])),
        ))
    finally:
        clear_phase_context()

    assert result["status"] == "SUCCESS"
    assert captured["rules_dir"] == captured["output_dir"] / "learned_rules"


def test_gold_analysis_can_write_reports_from_tool_collected_findings(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    findings = {
        "raw_data_root": "/datasets/raw",
        "gold_examples_path": "/datasets/examples",
        "gold_schema": {
            "record_count": 2,
            "field_count": 1,
            "fields": [{"field_path": "病历.入院时间", "value_types": ["str"]}],
        },
        "field_provenance": [
            {
                "target_field_path": "病历.入院时间",
                "source_files": ["admissions.csv"],
                "source_columns": ["admittime"],
                "join_keys": ["subject_id", "hadm_id"],
                "derivation_logic": "按 hadm_id 读取 admittime。",
                "source_status": "source_schema_available",
                "confidence": "high",
            }
        ],
    }
    try:
        result = _payload(
            gold_skill.analyze_gold_examples_tool(
                task_text="根据原子工具读取结果生成报告",
                analysis_findings=json.dumps(findings, ensure_ascii=False),
            )
        )
    finally:
        clear_phase_context()

    assert result["status"] == "SUCCESS", result["issues"]
    assert result["artifacts"]["field_count"] == 1
    assert Path(result["artifacts"]["gold_schema_summary"]).exists()
    assert Path(result["artifacts"]["gold_field_provenance_report"]).exists()
    assert Path(result["artifacts"]["planner_extraction_brief"]).exists()
    assert Path(result["artifacts"]["learned_rules_path"]).exists()


def test_sample_table_compact_mode_bounds_tool_output(tmp_path: Path) -> None:
    table = tmp_path / "table.csv"
    table.write_text(
        "id,category,value,text\n"
        "1,a,10," + "long clinical text " * 8 + "\n"
        "2,b,20," + "another report text " * 8 + "\n"
        "3,c,30," + "third narrative text " * 8 + "\n"
        "4,a,40," + "fourth clinical text " * 8 + "\n"
        "5,b,50," + "fifth report text " * 8 + "\n"
        "6,c,60," + "sixth narrative text " * 8 + "\n",
        encoding="utf-8",
    )

    compact_response = sample_table_tool(str(table), max_unique=20, compact=True)
    detailed_response = sample_table_tool(str(table), max_unique=20, compact=False)
    compact = _payload(compact_response)

    assert compact["status"] == "SUCCESS"
    assert compact["artifacts"]["detail_level"] == "compact"
    assert len(compact_response.content[0]["text"]) < len(detailed_response.content[0]["text"])
    columns = {item["column"]: item for item in compact["artifacts"]["columns"]}
    assert "sample_values" not in columns["value"]
    assert len(columns["category"]["top_values"]) <= 5
    assert len(columns["text"].get("sample_values", [])) <= 1


def test_explorer_toolkit_does_not_register_legacy_reading_skills() -> None:
    toolkit, _manifest = create_explorer_toolkit()

    assert {
        "list_data_files_tool",
        "sample_table_tool",
        "scan_source_files_tool",
        "scan_patients_tool",
        "run_python_code_tool",
    }.isdisjoint(toolkit.tools)


def test_list_data_files_compact_mode_limits_repeated_schema_output(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for index in range(3):
        path = data_dir / f"case_{index}" / "wide.csv"
        path.parent.mkdir(parents=True)
        columns = ",".join(f"column_{number}" for number in range(30))
        path.write_text(columns + "\n" + ",".join("1" for _ in range(30)) + "\n", encoding="utf-8")

    result = _payload(list_data_files_tool(str(data_dir), compact=True, max_columns=12))

    assert result["status"] == "SUCCESS"
    assert result["artifacts"]["detail_level"] == "compact"
    assert all(len(item["columns"]) == 12 for item in result["artifacts"]["files"])
    assert all(item["columns_truncated"] is True for item in result["artifacts"]["files"])


def test_explorer_python_truncates_large_stdout(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    try:
        result = _payload(run_python_code_tool("print('x' * 50000)"))
    finally:
        clear_phase_context()

    assert result["status"] == "SUCCESS"
    assert result["artifacts"]["stdout_truncated"] is True
    assert len(result["artifacts"]["stdout"]) < 22000


def test_explorer_can_enrich_but_not_replace_locked_field_rules(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    rules = Path(phase["artifacts_dir"]) / "locked" / "field_extraction_rules.json"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        json.dumps(
            {
                "field_count": 1,
                "rules": [
                    {
                        "target_field_id": "field_a",
                        "target_field_path": "nested.value",
                        "source_files": [],
                        "source_columns": [],
                        "join_keys": [],
                        "derivation_logic": {"operation": "requires_analysis"},
                        "status": "unsupported",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tools = ExplorerReportTools()
    try:
        updated = _payload(
            tools.publish_field_rule_updates(
                str(rules),
                json.dumps(
                    {
                        "updates": [
                            {
                                "target_field_id": "field_a",
                                "source_files": ["events.csv"],
                                "source_columns": ["value"],
                                "join_keys": ["record_id"],
                                "derivation_logic": {"operation": "direct"},
                                "status": "supported",
                            }
                        ]
                    }
                ),
            )
        )
        rejected = _payload(
            tools.publish_field_rule_updates(
                str(rules),
                json.dumps({"updates": [{"target_field_id": "invented", "status": "supported"}]}),
            )
        )
    finally:
        clear_phase_context()

    assert updated["status"] == "SUCCESS"
    payload = json.loads(Path(updated["artifacts"]["file_path"]).read_text())
    assert payload["field_count"] == 1
    assert payload["rules"][0]["target_field_path"] == "nested.value"
    assert payload["rules"][0]["source_files"] == ["events.csv"]
    assert rejected["status"] == "NEEDS_REPAIR"
    assert "invented" in rejected["issues"][0]


def test_profile_updates_can_complete_all_locked_mimic_rules(tmp_path: Path) -> None:
    examples = Path(__file__).parents[1] / "mimic_4case_train_validation_split" / "train_examples"
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    try:
        artifacts = analyze_training_examples(examples, Path(phase["artifacts_dir"]) / "locked")
        suggestions = build_dataset_profile_suggestions(artifacts["field_extraction_rules"], examples)
        updates = [
            {key: value for key, value in item.items() if key != "target_field_path"}
            for item in suggestions["updates"]
        ]
        result = _payload(
            ExplorerReportTools().publish_field_rule_updates(
                artifacts["field_extraction_rules"],
                json.dumps({"updates": updates}, ensure_ascii=False),
            )
        )
    finally:
        clear_phase_context()

    assert result["status"] == "SUCCESS", result["issues"]
    completed = validate_field_rule_completion(result["artifacts"]["file_path"], examples)
    assert completed["complete"] is True
    assert completed["field_count"] == 58
    payload = json.loads(Path(result["artifacts"]["file_path"]).read_text(encoding="utf-8"))
    assert not any(rule["status"] == "pending" for rule in payload["rules"])
    unsupported = [rule for rule in payload["rules"] if rule["status"] == "unsupported"]
    assert unsupported and all(rule.get("unsupported_reason") for rule in unsupported)
