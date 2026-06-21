from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentscope.tool import Toolkit

from agent.two_phase_agents import create_explorer_toolkit
from agent.system_prompt import build_explorer_prompt
from agent_tools import (
    ExplorerToolContext,
    ExplorerTools,
    register_explorer_atomic_tools,
)
from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context


def _payload(response) -> dict:
    return json.loads(response.content[0]["text"])


@pytest.fixture()
def explorer_env(tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "gold.jsonl").write_text(
        '{"patient_id": 1, "label": "alpha"}\n'
        '{"patient_id": 2, "label": "beta"}\n',
        encoding="utf-8",
    )
    (inputs / "table.csv").write_text(
        "patient_id,value,note\n1,10,first\n2,,second\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    context = ExplorerToolContext.from_task(
        task_text=f"分析 {inputs}",
        explorer_phase_root=phase["phase_root"],
    )
    try:
        yield run_root, inputs, context, ExplorerTools(context)
    finally:
        clear_phase_context()


def test_explorer_registers_read_and_analysis_tools_without_generic_writes(explorer_env) -> None:
    _run_root, _inputs, context, _tools = explorer_env
    toolkit = Toolkit()

    register_explorer_atomic_tools(toolkit, context)

    assert {"Read", "Glob", "Grep", "InspectDataFile", "ExecuteAnalysisPython"} <= set(toolkit.tools)
    assert {"Write", "Edit", "ExecutePython"}.isdisjoint(toolkit.tools)


def test_explorer_read_glob_and_grep_use_task_authorized_roots(explorer_env) -> None:
    run_root, inputs, _context, tools = explorer_env

    globbed = _payload(tools.Glob("**/*.jsonl", str(inputs)))
    read = _payload(tools.Read(str(inputs / "gold.jsonl"), limit=1))
    grepped = _payload(tools.Grep("patient_id", str(inputs), glob="*.jsonl", output_mode="content"))
    forbidden = run_root.parent / "secret.txt"
    forbidden.write_text("secret", encoding="utf-8")
    rejected = _payload(tools.Read(str(forbidden)))

    assert globbed["status"] == "SUCCESS"
    assert globbed["artifacts"]["num_files"] == 1
    assert read["status"] == "SUCCESS"
    assert read["artifacts"]["num_lines"] == 1
    assert grepped["status"] == "SUCCESS"
    assert grepped["artifacts"]["num_files"] == 1
    assert rejected["status"] == "NEEDS_REPAIR"


def test_inspect_data_file_profiles_csv_and_jsonl(explorer_env) -> None:
    _run_root, inputs, _context, tools = explorer_env

    csv_result = _payload(tools.InspectDataFile(str(inputs / "table.csv"), sample_rows=1))
    jsonl_result = _payload(tools.InspectDataFile(str(inputs / "gold.jsonl"), sample_rows=1))

    assert csv_result["status"] == "SUCCESS"
    assert csv_result["artifacts"]["format"] == "csv"
    assert csv_result["artifacts"]["row_count"] == 2
    assert csv_result["artifacts"]["columns"][1]["name"] == "value"
    assert csv_result["artifacts"]["columns"][1]["missing_count"] == 1
    assert len(csv_result["artifacts"]["sample_records"]) == 1
    assert jsonl_result["status"] == "SUCCESS"
    assert jsonl_result["artifacts"]["format"] == "jsonl"
    assert jsonl_result["artifacts"]["row_count"] == 2


def test_execute_analysis_python_reads_authorized_data_and_writes_only_step(explorer_env) -> None:
    _run_root, inputs, _context, tools = explorer_env
    code = (
        "from pathlib import Path\n"
        "import pandas as pd\n"
        f"df = pd.read_csv({str(inputs / 'table.csv')!r})\n"
        "df.describe(include='all').to_json(Path(OUTPUT_DIR) / 'profile.json')\n"
        "print({'rows': len(df), 'columns': list(df.columns)})\n"
    )

    result = _payload(tools.ExecuteAnalysisPython(code, timeout_seconds=20))

    assert result["status"] == "SUCCESS", result["artifacts"].get("stderr")
    assert "'rows': 2" in result["artifacts"]["stdout"]
    assert any(path.endswith("profile.json") for path in result["artifacts"]["output_files"])
    assert Path(result["artifacts"]["manifest_path"]).exists()


def test_explorer_toolkit_combines_atomic_tools_and_report_skills(explorer_env) -> None:
    _run_root, _inputs, context, _tools = explorer_env

    toolkit, manifest = create_explorer_toolkit(context)

    assert {"Read", "Glob", "Grep", "InspectDataFile", "ExecuteAnalysisPython"} <= set(toolkit.tools)
    assert "analyze_gold_examples_tool" in toolkit.tools
    assert "publish_analysis_report" in toolkit.tools
    assert {"Write", "Edit", "ExecutePython"}.isdisjoint(toolkit.tools)
    assert {
        "list_data_files_tool",
        "sample_table_tool",
        "scan_source_files_tool",
        "scan_patients_tool",
        "run_python_code_tool",
    }.isdisjoint(toolkit.tools)
    assert {item["name"] for item in manifest} == {"analyze_gold_examples"}


def test_explorer_prompt_locks_deterministic_leaf_field_inventory() -> None:
    prompt = build_explorer_prompt("manifest")

    assert "先调用 `analyze_gold_examples`" in prompt
    assert "不得删除、合并或替换" in prompt
    assert "field_extraction_rules.json" in prompt
    assert "publish_field_rule_updates" in prompt
