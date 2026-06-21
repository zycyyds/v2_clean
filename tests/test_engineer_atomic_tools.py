from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest
from agentscope.message import Msg
from agentscope.tool import Toolkit

from agent_tools import EngineerToolContext, EngineerTools, register_engineer_atomic_tools
from agent.system_prompt import (
    ENGINEER_SYSTEM_PROMPT_TEMPLATE,
    EXPLORER_SYSTEM_PROMPT_TEMPLATE,
    build_engineer_prompt,
)
from agent.two_phase_agents import (
    ENGINEER_CODE_READ_ROOTS,
    ENGINEER_SKILLS,
    create_engineer_toolkit,
    create_explorer_toolkit,
    infer_split_mode,
)
from agent.bounded_memory import BoundedInMemoryMemory
from lib.agent_artifacts import (
    clear_phase_context,
    init_phase_session,
    resolve_phase_handoff,
    set_phase_context,
)
from skills.extract_text_features import skill as text_skill
from skills.run_ocr import skill as ocr_skill


def _payload(response) -> dict:
    return json.loads(response.content[0]["text"])


@pytest.fixture()
def tool_env(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "input.jsonl").write_text(
        '{"patient_id": 1, "value": "alpha"}\n'
        '{"patient_id": 2, "value": "beta"}\n',
        encoding="utf-8",
    )

    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "engineer")
    set_phase_context(run_root.name, run_root, "engineer", phase["phase_root"])
    context = EngineerToolContext(
        engineer_phase_root=phase["phase_root"],
        read_roots=[external, phase["phase_root"]],
    )
    try:
        yield run_root, external, context, EngineerTools(context)
    finally:
        clear_phase_context()


def test_read_then_write_allows_safe_overwrite(tool_env):
    _run_root, _external, context, tools = tool_env
    target = context.workspace_dir / "scripts" / "task.py"

    created = _payload(tools.Write(str(target), "print('v1')\n"))
    assert created["status"] == "SUCCESS"
    assert target.read_text(encoding="utf-8") == "print('v1')\n"

    rejected = _payload(tools.Write(str(target), "print('v2')\n"))
    assert rejected["status"] == "NEEDS_REPAIR"
    assert "Read" in rejected["issues"][0]

    read = _payload(tools.Read(str(target)))
    assert read["status"] == "SUCCESS"
    assert "print('v1')" in read["artifacts"]["content"]

    updated = _payload(tools.Write(str(target), "print('v2')\n"))
    assert updated["status"] == "SUCCESS"
    assert target.read_text(encoding="utf-8") == "print('v2')\n"


def test_read_truncates_a_single_very_long_line(tool_env):
    _run_root, external, context, tools = tool_env
    target = external / "large.jsonl"
    target.write_text('{"payload":"' + ("x" * 50_000) + '"}\n', encoding="utf-8")

    result = _payload(tools.Read(str(target)))

    assert result["status"] == "SUCCESS"
    assert len(result["artifacts"]["content"]) <= 13_000
    assert result["artifacts"]["content_truncated"] is True
    assert context.was_read_and_unchanged(target)


def test_bounded_memory_keeps_task_and_recent_context():
    memory = BoundedInMemoryMemory(max_chars=2_000, keep_recent_messages=4)

    async def exercise():
        await memory.add(Msg("user", "original task", "user"))
        for index in range(12):
            await memory.add(Msg("assistant", f"old-{index}-" + ("x" * 500), "assistant"))
        return await memory.get_memory()

    messages = asyncio.run(exercise())
    rendered = "\n".join(str(message.content) for message in messages)
    assert "original task" in rendered
    assert "old-11" in rendered
    assert "old-0" not in rendered
    assert "早期工具对话已裁剪" in rendered


def test_write_and_edit_reject_paths_outside_engineer_phase(tool_env):
    _run_root, external, _context, tools = tool_env
    outside = external / "do_not_touch.py"

    written = _payload(tools.Write(str(outside), "bad = True\n"))
    assert written["status"] == "NEEDS_REPAIR"
    assert not outside.exists()

    outside.write_text("value = 1\n", encoding="utf-8")
    _payload(tools.Read(str(outside)))
    edited = _payload(tools.Edit(str(outside), "1", "2"))
    assert edited["status"] == "NEEDS_REPAIR"
    assert outside.read_text(encoding="utf-8") == "value = 1\n"


def test_edit_requires_unique_match_and_supports_replace_all(tool_env):
    _run_root, _external, context, tools = tool_env
    target = context.workspace_dir / "scripts" / "edit.py"
    assert _payload(tools.Write(str(target), "value = 1\nother = 1\n"))["status"] == "SUCCESS"
    _payload(tools.Read(str(target)))

    ambiguous = _payload(tools.Edit(str(target), "1", "2"))
    assert ambiguous["status"] == "NEEDS_REPAIR"

    _payload(tools.Read(str(target)))
    edited = _payload(tools.Edit(str(target), "1", "2", replace_all=True))
    assert edited["status"] == "SUCCESS"
    assert edited["artifacts"]["replacements"] == 2
    assert target.read_text(encoding="utf-8") == "value = 2\nother = 2\n"


def test_glob_and_grep_search_authorized_inputs(tool_env):
    _run_root, external, _context, tools = tool_env

    globbed = _payload(tools.Glob("**/*.jsonl", str(external)))
    assert globbed["status"] == "SUCCESS"
    assert globbed["artifacts"]["num_files"] == 1

    grepped = _payload(
        tools.Grep(
            "patient_id",
            str(external),
            glob="*.jsonl",
            output_mode="content",
        ),
    )
    assert grepped["status"] == "SUCCESS"
    assert grepped["artifacts"]["num_files"] == 1
    assert ":1:" in grepped["artifacts"]["content"]


def test_grep_rejects_missing_paths_and_skips_symlink_escape(tool_env):
    run_root, external, _context, tools = tool_env
    missing = _payload(tools.Grep("anything", str(external / "missing")))
    assert missing["status"] == "NEEDS_REPAIR"

    secret = run_root.parent / "outside-secret.txt"
    secret.write_text("sensitive-marker", encoding="utf-8")
    link = external / "linked-secret.txt"
    link.symlink_to(secret)

    result = _payload(
        tools.Grep(
            "sensitive-marker",
            str(external),
            glob="*.txt",
            output_mode="content",
        ),
    )
    assert result["status"] == "SUCCESS"
    assert result["artifacts"]["num_files"] == 0
    globbed = _payload(tools.Glob("**/*.txt", str(external)))
    assert str(secret.resolve()) not in globbed["artifacts"]["filenames"]


def test_execute_python_uses_step_output_dir_and_records_outputs(tool_env):
    _run_root, external, context, tools = tool_env
    script = context.workspace_dir / "scripts" / "extract.py"
    code = (
        "from pathlib import Path\n"
        f"source = Path({str(external / 'input.jsonl')!r})\n"
        "text = source.read_text(encoding='utf-8')\n"
        "Path(OUTPUT_DIR, 'result.txt').write_text(text, encoding='utf-8')\n"
        "print('rows', len(text.splitlines()))\n"
    )
    assert _payload(tools.Write(str(script), code))["status"] == "SUCCESS"

    result = _payload(tools.ExecutePython(str(script), timeout_seconds=10))

    assert result["status"] == "SUCCESS"
    assert result["artifacts"]["returncode"] == 0
    assert "rows 2" in result["artifacts"]["stdout"]
    output_files = [Path(path) for path in result["artifacts"]["output_files"]]
    assert any(path.name == "result.txt" and path.exists() for path in output_files)
    assert Path(result["artifacts"]["manifest_path"]).exists()


def test_execute_python_supports_pandas_data_processing(tool_env):
    _run_root, external, context, tools = tool_env
    source = external / "input.csv"
    source.write_text("patient_id,value\n1,10\n2,20\n", encoding="utf-8")
    script = context.workspace_dir / "scripts" / "pandas_task.py"
    code = (
        "from pathlib import Path\n"
        "import pandas as pd\n"
        f"df = pd.read_csv({str(source)!r})\n"
        "df['double'] = df['value'] * 2\n"
        "df.to_csv(Path(OUTPUT_DIR) / 'features.csv', index=False)\n"
        "print(df.shape)\n"
    )
    assert _payload(tools.Write(str(script), code))["status"] == "SUCCESS"

    result = _payload(tools.ExecutePython(str(script), timeout_seconds=20))

    assert result["status"] == "SUCCESS", result["artifacts"]["stderr"]
    output = next(Path(path) for path in result["artifacts"]["output_files"] if path.endswith("features.csv"))
    assert output.read_text(encoding="utf-8") == "patient_id,value,double\n1,10,20\n2,20,40\n"


def test_execute_python_blocks_writes_outside_step(tool_env):
    _run_root, external, context, tools = tool_env
    script = context.workspace_dir / "scripts" / "escape.py"
    outside = external / "escaped.txt"
    code = (
        "from pathlib import Path\n"
        f"Path({str(outside)!r}).write_text('forbidden', encoding='utf-8')\n"
    )
    assert _payload(tools.Write(str(script), code))["status"] == "SUCCESS"

    result = _payload(tools.ExecutePython(str(script), timeout_seconds=10))

    assert result["status"] == "NEEDS_REPAIR"
    assert result["artifacts"]["returncode"] != 0
    assert "outside the execution output directory" in result["artifacts"]["stderr"]
    assert not outside.exists()


def test_execute_python_blocks_reads_outside_authorized_roots(tool_env):
    run_root, _external, context, tools = tool_env
    secret = run_root.parent / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    script = context.workspace_dir / "scripts" / "read_secret.py"
    code = (
        "from pathlib import Path\n"
        f"print(Path({str(secret)!r}).read_text(encoding='utf-8'))\n"
    )
    assert _payload(tools.Write(str(script), code))["status"] == "SUCCESS"

    result = _payload(tools.ExecutePython(str(script), timeout_seconds=10))

    assert result["status"] == "NEEDS_REPAIR"
    assert "outside authorized read roots" in result["artifacts"]["stderr"]


def test_execute_python_passes_args_and_blocks_child_processes(tool_env):
    _run_root, _external, context, tools = tool_env
    args_script = context.workspace_dir / "scripts" / "args.py"
    args_code = "import sys\nprint('|'.join(sys.argv[1:]))\n"
    assert _payload(tools.Write(str(args_script), args_code))["status"] == "SUCCESS"

    args_result = _payload(tools.ExecutePython(str(args_script), args=["a", "b"], timeout_seconds=10))
    assert args_result["status"] == "SUCCESS"
    assert "a|b" in args_result["artifacts"]["stdout"]

    child_script = context.workspace_dir / "scripts" / "child.py"
    child_code = "import subprocess\nsubprocess.run(['echo', 'forbidden'], check=True)\n"
    assert _payload(tools.Write(str(child_script), child_code))["status"] == "SUCCESS"

    child_result = _payload(tools.ExecutePython(str(child_script), timeout_seconds=10))
    assert child_result["status"] == "NEEDS_REPAIR"
    assert "child processes and shell execution are forbidden" in child_result["artifacts"]["stderr"]


def test_execute_python_times_out(tool_env):
    _run_root, _external, context, tools = tool_env
    script = context.workspace_dir / "scripts" / "sleep.py"
    assert _payload(tools.Write(str(script), "import time\ntime.sleep(2)\n"))["status"] == "SUCCESS"

    result = _payload(tools.ExecutePython(str(script), timeout_seconds=1))

    assert result["status"] == "NEEDS_REPAIR"
    assert result["artifacts"]["timed_out"] is True
    assert "TimeoutError" in result["artifacts"]["stderr"]


def test_publish_artifact_creates_standard_handoff(tool_env):
    run_root, _external, context, tools = tool_env
    artifact = context.workspace_dir / "final" / "dataset.csv"
    assert _payload(tools.Write(str(artifact), "patient_id,label\n1,1\n"))["status"] == "SUCCESS"

    published = _payload(tools.publish_artifact(str(artifact), "ml_dataset_csv"))

    assert published["status"] == "SUCCESS"
    resolved = resolve_phase_handoff(run_root, "engineer", "ml_dataset_csv")
    assert resolved is not None
    assert Path(resolved).read_text(encoding="utf-8") == "patient_id,label\n1,1\n"


def test_registers_atomic_tools_directly_on_agentscope_toolkit(tool_env):
    _run_root, _external, context, _tools = tool_env
    toolkit = Toolkit()

    register_engineer_atomic_tools(toolkit, context)

    assert {"Read", "Glob", "Grep", "Write", "Edit", "ExecutePython", "publish_artifact"} <= set(
        toolkit.tools,
    )
    schema_names = {
        schema["function"]["name"]
        for schema in toolkit.get_json_schemas()
    }
    assert {"Read", "Glob", "Grep", "Write", "Edit", "ExecutePython", "publish_artifact"} <= schema_names


def test_feature_engineer_toolkit_combines_atomic_and_selected_domain_tools(tool_env):
    _run_root, _external, context, _tools = tool_env

    toolkit, manifest = create_engineer_toolkit(context)

    tool_names = set(toolkit.tools)
    skill_names = {item["name"] for item in manifest}
    assert {
        "Read", "Glob", "Grep", "Write", "Edit", "ExecutePython", "publish_artifact",
        "initialize_skill_usage_plan", "revise_skill_usage", "plan_skill_usage",
        "inspect_skill", "create_skill_variant",
        "validate_skill_variant", "execute_skill_variant", "audit_skill_usage",
    } <= tool_names
    assert "sample_table_tool" not in tool_names
    assert "run_step23_ocr_tool" in tool_names
    assert "run_step23_directory_pipeline_tool" in tool_names
    assert "run_python_code_tool" not in tool_names
    assert {
        "run_ocr", "extract_text_features", "extract_structured_fields",
        "join_source_records", "derive_target_fields", "aggregate_records",
        "normalize_reversible_values", "export_gold_workbook",
    } <= skill_names
    assert all(item["layer"] in {"process", "label", "util"} for item in manifest)


def test_bundle_managed_toolkit_excludes_global_learned_rule_tools(tool_env):
    _run_root, _external, context, _tools = tool_env

    toolkit, manifest = create_engineer_toolkit(context, bundle_managed_rules=True)

    assert "load_learned_rules_tool" not in toolkit.tools
    assert "promote_validation_feedback_tool" not in toolkit.tools
    assert "freeze_learned_rules_tool" not in toolkit.tools
    assert "learned_extraction_rules" not in {item["name"] for item in manifest}
    assert "build_mimic_liver_dataset" not in {item["name"] for item in manifest}
    assert "clean_data" not in {item["name"] for item in manifest}


def test_label_skills_are_only_exposed_for_explicit_label_tasks(tool_env):
    _run_root, _external, context, _tools = tool_env

    _, extraction_manifest = create_engineer_toolkit(context, task_text="抽取并清洗字段")
    _, label_manifest = create_engineer_toolkit(context, task_text="构建分类目标标签")

    assert not ({item["name"] for item in extraction_manifest} & {
        "build_label_from_icd", "build_label_from_event", "build_label_from_column",
    })
    assert "build_label_from_column" in {item["name"] for item in label_manifest}


def test_feature_engineer_allowlist_excludes_legacy_stateful_skills():
    assert "run_python_code" not in ENGINEER_SKILLS
    assert "check_patient_consistency" not in ENGINEER_SKILLS
    assert "load_task_context" not in ENGINEER_SKILLS
    assert "sample_table" not in ENGINEER_SKILLS
    assert {"run_ocr", "extract_text_features", "extract_structured_fields"} <= ENGINEER_SKILLS


def test_feature_engineer_can_read_skill_runtime_sources():
    roots = {path.name for path in ENGINEER_CODE_READ_ROOTS}

    assert {"skills", "lib", "workflow"} <= roots


def test_feature_engineer_prompt_uses_atomic_execution_loop():
    assert "Write" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "ExecutePython" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "Glob/Grep" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "run_python_code" not in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "initialize_skill_usage_plan" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "inspect_skill" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "create_skill_variant" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "audit_skill_usage" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "参数" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "field_extraction_rules.json" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "result_manifest.json" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "target_field_mapping.json" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    rendered = build_engineer_prompt("test manifest")
    assert '{"value":["count","mean","list"]}' in rendered
    assert "record_extraction_task" in ENGINEER_SYSTEM_PROMPT_TEMPLATE
    assert "不得访问验证 gold" in ENGINEER_SYSTEM_PROMPT_TEMPLATE


def test_explorer_toolkit_has_dedicated_analysis_report_publisher():
    toolkit, _manifest = create_explorer_toolkit()

    assert "publish_analysis_report" in toolkit.tools


def test_explorer_prompt_forbids_engineer_result_artifacts():
    assert "严禁创建 `final_dataset.csv`" in EXPLORER_SYSTEM_PROMPT_TEMPLATE
    assert "publish_analysis_report` 成功后立即返回" in EXPLORER_SYSTEM_PROMPT_TEMPLATE


def test_test_split_removes_rule_mutation_tools(tool_env):
    _run_root, _external, context, _tools = tool_env
    context.split_mode = "test"

    toolkit, _manifest = create_engineer_toolkit(context)

    assert "load_learned_rules_tool" in toolkit.tools
    loader = toolkit.tools["load_learned_rules_tool"]
    assert loader.preset_kwargs["include_statuses"] == "frozen"
    assert "include_statuses" not in loader.json_schema["function"]["parameters"]["properties"]
    assert "promote_validation_feedback_tool" not in toolkit.tools
    assert "freeze_learned_rules_tool" not in toolkit.tools


def test_split_mode_detection_defaults_to_training():
    assert infer_split_mode("处理训练集并构建特征") == "training"
    assert infer_split_mode("在 test set 上只读取冻结规则") == "test"
    assert infer_split_mode("开始运行测试集") == "test"


def test_ocr_skill_defaults_output_to_current_phase_step(tool_env, monkeypatch):
    _run_root, _external, _context, _tools = tool_env
    captured: dict[str, str] = {}

    def fake_ocr(input_path, output_root=None, ocr_workers=8, cache_path=None):
        captured["output_root"] = str(output_root)
        cache = Path(output_root) / "ocr_cache.json"
        cache.write_text("{}", encoding="utf-8")
        return _tool_response(
            {
                "status": "SUCCESS",
                "summary": "OCR done",
                "artifacts": {"ocr_cache_path": str(cache), "successful_ocr": 1},
                "issues": [],
            },
        )

    monkeypatch.setattr(ocr_skill, "_legacy_run_step23_ocr_tool", fake_ocr)

    result = _payload(ocr_skill.run_step23_ocr_tool("/input/images"))

    assert result["status"] == "SUCCESS"
    assert "/artifacts/step" in captured["output_root"]
    assert captured["output_root"].endswith("_run_ocr")
    assert Path(result["artifacts"]["manifest_path"]).exists()


def test_text_skill_defaults_output_to_current_phase_step(tool_env, monkeypatch):
    _run_root, _external, _context, _tools = tool_env
    captured: dict[str, str] = {}

    async def fake_pipeline(input_path, output_root=None, concurrency=8, use_llm=True):
        captured["output_root"] = str(output_root)
        output = Path(output_root) / "text_features.csv"
        output.write_text("patient_id,feature\n1,1\n", encoding="utf-8")
        return _tool_response(
            {
                "status": "SUCCESS",
                "summary": "Text extraction done",
                "artifacts": {"wide_csv_path": str(output), "wide_row_count": 1},
                "issues": [],
            },
        )

    monkeypatch.setattr(
        text_skill,
        "_legacy_run_step23_directory_pipeline_tool",
        fake_pipeline,
    )

    response = asyncio.run(
        text_skill.run_step23_directory_pipeline_tool("/input/notes", use_llm=False),
    )
    result = _payload(response)

    assert result["status"] == "SUCCESS"
    assert "/artifacts/step" in captured["output_root"]
    assert captured["output_root"].endswith("_extract_text_features")
    assert Path(result["artifacts"]["manifest_path"]).exists()


def _tool_response(payload: dict):
    from agentscope.message import TextBlock
    from agentscope.tool import ToolResponse

    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False))],
    )
