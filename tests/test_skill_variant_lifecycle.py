from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_tools import EngineerToolContext, EngineerTools
from agent_tools.report_tools import ExplorerReportTools
from agent_tools.skill_variants import EngineerSkillLifecycleTools
from lib.agent_artifacts import (
    clear_phase_context,
    init_phase_session,
    resolve_phase_handoff,
    resolve_canonical_phase_handoff,
    set_phase_context,
)
from skills.run_python_code.skill import run_python_code_tool


def _payload(response) -> dict:
    return json.loads(response.content[0]["text"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publish_analysis_report_preserves_manifest_and_creates_handoff(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    report = Path(phase["artifacts_dir"]) / "step01_analysis" / "data_analysis_report.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"task": "demo", "tables": []}', encoding="utf-8")

    try:
        result = _payload(ExplorerReportTools().publish_analysis_report(str(report)))
    finally:
        clear_phase_context()

    assert result["status"] == "SUCCESS"
    manifest = json.loads(Path(phase["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["phase"] == "explorer"
    assert manifest["steps"][-1]["skill_name"] == "publish_analysis_report"
    canonical = resolve_phase_handoff(run_root, "explorer", "analysis_report")
    assert canonical == str(Path(phase["next_input_dir"]) / "data_analysis_report.json")
    assert json.loads(Path(canonical).read_text(encoding="utf-8"))["task"] == "demo"
    assert resolve_canonical_phase_handoff(run_root, "explorer", "analysis_report") == canonical


def test_publish_analysis_report_accepts_structured_report_content(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    report_content = json.dumps(
        {"task": "demo", "tables": [], "feature_plan": [], "gold_guided_extraction": {}},
        ensure_ascii=False,
    )
    try:
        result = _payload(
            ExplorerReportTools().publish_analysis_report(
                file_path="",
                report_json=report_content,
            )
        )
    finally:
        clear_phase_context()

    assert result["status"] == "SUCCESS", result["issues"]
    report_path = Path(result["artifacts"]["file_path"])
    assert report_path.name == "data_analysis_report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["task"] == "demo"
    assert resolve_canonical_phase_handoff(run_root, "explorer", "analysis_report") is not None


def test_canonical_handoff_does_not_fallback_to_unpublished_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    report = Path(phase["artifacts_dir"]) / "step01" / "data_analysis_report.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"task": "not-published"}', encoding="utf-8")

    assert resolve_phase_handoff(run_root, "explorer", "analysis_report") == str(report.resolve())
    assert resolve_canonical_phase_handoff(run_root, "explorer", "analysis_report") is None


def test_explorer_python_cannot_overwrite_phase_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    manifest_path = Path(phase["manifest_path"])
    set_phase_context(run_root.name, run_root, "explorer", phase["phase_root"])
    code = (
        "from pathlib import Path\n"
        f"Path({str(manifest_path)!r}).write_text('{{}}', encoding='utf-8')\n"
    )
    try:
        result = _payload(run_python_code_tool(code))
    finally:
        clear_phase_context()

    assert result["status"] == "NEEDS_REPAIR"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase"] == "explorer"
    assert manifest["steps"][-1]["skill_name"] == "run_python_code"
    assert any("manifest.json" in issue for issue in result["issues"])


@pytest.fixture()
def lifecycle_env(tmp_path: Path):
    skills_root = tmp_path / "readonly_skills"
    base = skills_root / "base_skill"
    base.mkdir(parents=True)
    (base / "skill.py").write_text(
        "SKILL = {'name': 'base_skill', 'description': 'base'}\n",
        encoding="utf-8",
    )
    (base / "SKILL.md").write_text("# Base Skill\n", encoding="utf-8")

    reports = tmp_path / "reports"
    reports.mkdir()
    analysis = reports / "data_analysis_report.json"
    analysis.write_text('{"tables": []}', encoding="utf-8")

    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "engineer")
    set_phase_context(run_root.name, run_root, "engineer", phase["phase_root"])
    context = EngineerToolContext(
        engineer_phase_root=phase["phase_root"],
        read_roots=[reports, skills_root],
        require_skill_plan=True,
    )
    atomic = EngineerTools(context)
    lifecycle = EngineerSkillLifecycleTools(
        context,
        skill_names={"base_skill"},
        skills_root=skills_root,
    )
    try:
        yield run_root, context, atomic, lifecycle, analysis, base
    finally:
        clear_phase_context()


def _plan(atomic, lifecycle, analysis: Path, *, include_standalone: bool = False) -> dict:
    assert _payload(atomic.Read(str(analysis)))["status"] == "SUCCESS"
    decisions = [
        {
            "skill_name": "base_skill",
            "decision": "adapt",
            "reason": "输入格式需要适配",
            "parameters": {},
            "capability_gap": "基础工具只接受单表",
            "why_parameters_insufficient": "现有参数不能表达多表 join",
        },
    ]
    if include_standalone:
        decisions.append(
            {
                "skill_name": "standalone_python",
                "decision": "use",
                "reason": "没有相近领域 Skill",
                "parameters": {},
                "capability_gap": "需要一次性格式转换",
                "why_parameters_insufficient": "没有可调整的基础 Skill",
            },
        )
    result = _payload(
        lifecycle.plan_skill_usage(
            json.dumps({"analysis_report": str(analysis)}),
            json.dumps(decisions, ensure_ascii=False),
        ),
    )
    if result["status"] == "SUCCESS":
        assert _payload(lifecycle.inspect_skill("base_skill"))["status"] == "SUCCESS"
    return result


def _implement_csv_variant(atomic, variant_dir: Path) -> None:
    script = variant_dir / "variant.py"
    source = '''from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output_path = Path(os.environ["OUTPUT_DIR"]) / "result.csv"
rows = request.get("rows") or [{"patient_id": "example", "value": 1}]
with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["patient_id", "value"])
    writer.writeheader()
    writer.writerows(rows)
print("VARIANT_RESULT_JSON=" + json.dumps({
    "status": "SUCCESS",
    "artifacts": [{"path": str(output_path)}],
    "field_mappings": [],
    "metrics": {},
    "issues": [],
}))
'''
    assert _payload(atomic.Read(str(script)))["status"] == "SUCCESS"
    assert _payload(atomic.Write(str(script), source))["status"] == "SUCCESS"


def test_python_write_and_execution_require_skill_plan(lifecycle_env) -> None:
    _run_root, context, atomic, lifecycle, analysis, _base = lifecycle_env
    script = context.workspace_dir / "task.py"

    rejected = _payload(atomic.Write(str(script), "print('x')\n"))
    assert rejected["status"] == "NEEDS_REPAIR"
    assert "plan_skill_usage" in rejected["issues"][0]

    planned = _plan(atomic, lifecycle, analysis, include_standalone=True)
    assert planned["status"] == "SUCCESS"
    assert Path(planned["artifacts"]["plan_path"]).exists()

    assert _payload(atomic.Write(str(script), "print('x')\n"))["status"] == "SUCCESS"
    assert _payload(atomic.ExecutePython(str(script), timeout_seconds=10))["status"] == "SUCCESS"


def test_automatic_skill_plan_avoids_one_large_decisions_payload(lifecycle_env) -> None:
    _run_root, context, atomic, lifecycle, analysis, _base = lifecycle_env
    assert _payload(atomic.Read(str(analysis)))["status"] == "SUCCESS"

    initialized = _payload(lifecycle.initialize_skill_usage_plan())

    assert initialized["status"] == "SUCCESS", initialized["issues"]
    plan = json.loads(Path(initialized["artifacts"]["plan_path"]).read_text(encoding="utf-8"))
    assert plan["decisions"] == [
        {
            "skill_name": "base_skill",
            "decision": "skip",
            "reason": "No required extraction capability matches this Skill.",
            "parameters": {},
            "capability_gap": "",
            "why_parameters_insufficient": "",
        }
    ]
    script = context.workspace_dir / "automatic.py"
    assert _payload(atomic.Write(str(script), "print('x')\n"))["status"] == "NEEDS_REPAIR"

    revised = _payload(
        lifecycle.revise_skill_usage(
            "standalone_python",
            "use",
            "No registered Skill covers the required custom transform.",
            "custom_transform",
            "The available Skill parameters cannot express this transform.",
        )
    )
    assert revised["status"] == "SUCCESS"
    assert _payload(atomic.Write(str(script), "print('x')\n"))["status"] == "SUCCESS"


def test_automatic_skill_plan_includes_execution_and_output_capabilities(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    for name, capabilities in {
        "joiner": ["structured_join"],
        "cleaner": ["reversible_cleaning"],
        "exporter": ["result_package_build"],
        "extractor": ["structured_extract"],
    }.items():
        skill_dir = skills_root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "skill.py").write_text(
            f"SKILL = {{'name': {name!r}, 'capability_types': {capabilities!r}}}\n",
            encoding="utf-8",
        )

    reports = tmp_path / "reports"
    reports.mkdir()
    analysis = reports / "data_analysis_report.json"
    analysis.write_text('{"tables": []}', encoding="utf-8")
    task_plan = reports / "extraction_task_plan.json"
    task_plan.write_text(
        json.dumps({
            "tasks": [{
                "task_id": "task_1",
                "required": True,
                "capability_type": "structured_extract",
                "input_artifacts": ["a.csv", "b.csv"],
            }],
        }),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "engineer")
    set_phase_context(run_root.name, run_root, "engineer", phase["phase_root"])
    context = EngineerToolContext(
        engineer_phase_root=phase["phase_root"],
        read_roots=[reports, skills_root],
        require_skill_plan=True,
        required_report_paths={
            "analysis_report": str(analysis),
            "extraction_task_plan": str(task_plan),
        },
    )
    atomic = EngineerTools(context)
    lifecycle = EngineerSkillLifecycleTools(
        context,
        skill_names={"joiner", "cleaner", "exporter", "extractor"},
        skills_root=skills_root,
    )
    try:
        assert _payload(atomic.Read(str(analysis)))["status"] == "SUCCESS"
        assert _payload(atomic.Read(str(task_plan)))["status"] == "SUCCESS"
        initialized = _payload(lifecycle.initialize_skill_usage_plan())
    finally:
        clear_phase_context()

    assert initialized["status"] == "SUCCESS", initialized["issues"]
    plan = json.loads(context.skill_usage_plan_path.read_text(encoding="utf-8"))
    assert set(plan["required_capabilities"]) == {
        "structured_extract",
        "structured_join",
        "reversible_cleaning",
        "result_package_build",
    }


def test_plan_requires_every_domain_skill_and_explains_adapt(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    assert _payload(atomic.Read(str(analysis)))["status"] == "SUCCESS"

    missing = _payload(lifecycle.plan_skill_usage(
        json.dumps({"analysis_report": str(analysis)}),
        "[]",
    ))
    assert missing["status"] == "NEEDS_REPAIR"
    assert "base_skill" in missing["issues"][0]

    unexplained = _payload(lifecycle.plan_skill_usage(
        json.dumps({"analysis_report": str(analysis)}),
        json.dumps([{
            "skill_name": "base_skill",
            "decision": "adapt",
            "reason": "需要调整",
            "parameters": {},
            "capability_gap": "",
            "why_parameters_insufficient": "",
        }]),
    ))
    assert unexplained["status"] == "NEEDS_REPAIR"
    assert "why_parameters_insufficient" in unexplained["issues"][0]


def test_variant_is_run_local_and_base_skill_remains_unchanged(lifecycle_env) -> None:
    _run_root, context, atomic, lifecycle, analysis, base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    before = {_path.name: _sha256(_path) for _path in base.iterdir() if _path.is_file()}

    inspected = _payload(lifecycle.inspect_skill("base_skill"))
    assert inspected["status"] == "SUCCESS"
    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "multi_table_adapter",
        "adapter",
        "把多表输入转换成基础 Skill 的单表请求",
        json.dumps({
            "input_contract": {"required": ["source"]},
            "output_contract": {
                "filename": "result.csv",
                "format": "csv",
                "required_columns": ["patient_id", "value"],
                "primary_key": "patient_id",
                "allow_empty": False,
            },
        }),
    ))

    assert created["status"] == "SUCCESS"
    variant_dir = Path(created["artifacts"]["variant_dir"])
    assert variant_dir.parent == context.workspace_dir / "skill_variants"
    assert {"SKILL.md", "variant.json", "variant.py", "request.json"} <= {
        path.name for path in variant_dir.iterdir()
    }
    assert json.loads((variant_dir / "request.json").read_text(encoding="utf-8")) == {
        "rule_ids": [],
        "input_artifacts": [],
        "parameters": {},
        "output_contract": {
            "filename": "result.csv",
            "format": "csv",
            "required_columns": ["patient_id", "value"],
            "primary_key": "patient_id",
            "allow_empty": False,
        },
    }
    assert json.loads((variant_dir / "variant.json").read_text(encoding="utf-8"))["status"] == "draft"
    after = {_path.name: _sha256(_path) for _path in base.iterdir() if _path.is_file()}
    assert after == before


def test_variant_must_be_ready_then_validated_against_artifact(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "csv_adapter",
        "adapter",
        "生成标准 CSV",
        json.dumps({
            "input_contract": {"required": []},
            "output_contract": {
                "filename": "result.csv",
                "format": "csv",
                "required_columns": ["patient_id", "value"],
                "primary_key": "patient_id",
                "allow_empty": False,
            },
        }),
    ))
    variant_dir = Path(created["artifacts"]["variant_dir"])
    script = variant_dir / "variant.py"

    blocked = _payload(atomic.ExecutePython(
        str(script),
        args=[str(variant_dir / "request.json")],
        timeout_seconds=10,
    ))
    assert blocked["status"] == "NEEDS_REPAIR"
    assert "ready" in blocked["issues"][0]

    scaffold = _payload(lifecycle.validate_skill_variant("csv_adapter"))
    assert scaffold["status"] == "NEEDS_REPAIR"
    assert "IMPLEMENTATION_REQUIRED" in scaffold["issues"][0]
    _implement_csv_variant(atomic, variant_dir)
    ready = _payload(lifecycle.validate_skill_variant("csv_adapter"))
    assert ready["status"] == "SUCCESS"
    assert ready["artifacts"]["variant_status"] == "ready"

    executed = _payload(atomic.ExecutePython(
        str(script),
        args=[str(variant_dir / "request.json")],
        timeout_seconds=10,
    ))
    assert executed["status"] == "SUCCESS", executed
    assert "VARIANT_RESULT_JSON" in executed["artifacts"]["stdout"]
    artifact = next(
        Path(path) for path in executed["artifacts"]["output_files"]
        if path.endswith("result.csv")
    )

    validated = _payload(lifecycle.validate_skill_variant("csv_adapter", str(artifact)))
    assert validated["status"] == "SUCCESS"
    assert validated["artifacts"]["variant_status"] == "validated"
    receipt = variant_dir / "execution_receipt.json"
    assert receipt.is_file()
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_value["status"] == "SUCCESS"
    assert receipt_value["request_sha256"] == _sha256(variant_dir / "request.json")


def test_execute_skill_variant_supplies_request_protocol_automatically(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "automatic_request_adapter",
        "adapter",
        "自动使用统一请求协议",
        json.dumps({"input_contract": {}, "output_contract": {"format": "csv"}}),
    ))
    variant_dir = Path(created["artifacts"]["variant_dir"])
    _implement_csv_variant(atomic, variant_dir)
    assert _payload(lifecycle.validate_skill_variant("automatic_request_adapter"))["status"] == "SUCCESS"

    executed = _payload(lifecycle.execute_skill_variant("automatic_request_adapter", timeout_seconds=10))

    assert executed["status"] == "SUCCESS", executed
    assert executed["artifacts"]["variant_result"]["status"] == "SUCCESS"
    assert (variant_dir / "execution_receipt.json").is_file()


def test_required_extraction_tasks_must_be_recorded_before_audit(lifecycle_env) -> None:
    _run_root, context, atomic, lifecycle, analysis, _base = lifecycle_env
    task_plan = analysis.parent / "extraction_task_plan.json"
    task_plan.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task_structured",
                        "required": True,
                        "rule_ids": ["field_a"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    context.required_report_paths = {
        "analysis_report": str(analysis),
        "extraction_task_plan": str(task_plan),
    }
    assert _payload(atomic.Read(str(analysis)))["status"] == "SUCCESS"
    assert _payload(atomic.Read(str(task_plan)))["status"] == "SUCCESS"
    planned = _payload(
        lifecycle.plan_skill_usage(
            json.dumps({"analysis_report": str(analysis), "extraction_task_plan": str(task_plan)}),
            json.dumps(
                [
                    {
                        "skill_name": "base_skill",
                        "decision": "skip",
                        "reason": "任务由结构化规则直接完成",
                        "parameters": {},
                        "capability_gap": "",
                        "why_parameters_insufficient": "",
                    }
                ]
            ),
        )
    )
    assert planned["status"] == "SUCCESS"

    missing = _payload(lifecycle.audit_skill_usage())
    assert missing["status"] == "NEEDS_REPAIR"
    assert any("task_structured" in issue for issue in missing["issues"])

    artifact = context.workspace_dir / "field_values.jsonl"
    artifact.write_text('{"case_id":"a","target_field_id":"field_a","value":1}\n', encoding="utf-8")
    assert _payload(atomic.Read(str(artifact)))["status"] == "SUCCESS"
    recorded = _payload(
        lifecycle.record_extraction_task(
            "task_structured",
            "completed",
            json.dumps(["field_a"]),
            json.dumps([str(artifact)]),
            "",
        )
    )
    assert recorded["status"] == "SUCCESS"


def test_completed_task_requires_matching_skill_output(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    aggregate = skills_root / "aggregate_records"
    aggregate.mkdir(parents=True)
    (aggregate / "skill.py").write_text(
        "SKILL = {'name': 'aggregate_records', 'capability_types': ['aggregate']}\n",
        encoding="utf-8",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    analysis = reports / "data_analysis_report.json"
    analysis.write_text('{"tables": []}', encoding="utf-8")
    task_plan = reports / "extraction_task_plan.json"
    task_plan.write_text(
        json.dumps({
            "tasks": [{
                "task_id": "task_aggregate",
                "required": True,
                "capability_type": "aggregate",
                "rule_ids": ["field_a"],
            }],
        }),
        encoding="utf-8",
    )
    phase = init_phase_session(tmp_path / "run", "engineer")
    set_phase_context("run", tmp_path / "run", "engineer", phase["phase_root"])
    context = EngineerToolContext(
        engineer_phase_root=phase["phase_root"],
        read_roots=[reports, skills_root],
        require_skill_plan=True,
        required_report_paths={
            "analysis_report": str(analysis),
            "extraction_task_plan": str(task_plan),
        },
    )
    atomic = EngineerTools(context)
    lifecycle = EngineerSkillLifecycleTools(
        context,
        skill_names={"aggregate_records"},
        skills_root=skills_root,
    )
    try:
        assert _payload(atomic.Read(str(analysis)))["status"] == "SUCCESS"
        assert _payload(atomic.Read(str(task_plan)))["status"] == "SUCCESS"
        assert _payload(lifecycle.initialize_skill_usage_plan())["status"] == "SUCCESS"
        artifact = context.workspace_dir / "aggregated.csv"
        artifact.write_text("case_id,value\na,1\n", encoding="utf-8")
        assert _payload(atomic.Read(str(artifact)))["status"] == "SUCCESS"

        missing_call = _payload(lifecycle.record_extraction_task(
            "task_aggregate",
            "completed",
            json.dumps(["field_a"]),
            json.dumps([str(artifact)]),
        ))
        assert missing_call["status"] == "NEEDS_REPAIR"
        assert "aggregate_records" in missing_call["issues"][0]

        other_artifact = context.workspace_dir / "other.csv"
        other_artifact.write_text("case_id,value\na,2\n", encoding="utf-8")
        context.record_skill_call(
            "aggregate_records",
            "aggregate_records_tool",
            "SUCCESS",
            artifacts=[str(other_artifact)],
        )
        wrong_output = _payload(lifecycle.record_extraction_task(
            "task_aggregate",
            "completed",
            json.dumps(["field_a"]),
            json.dumps([str(artifact)]),
        ))
        assert wrong_output["status"] == "NEEDS_REPAIR"
        assert "产物" in wrong_output["issues"][0]

        context.record_skill_call(
            "aggregate_records",
            "aggregate_records_tool",
            "SUCCESS",
            artifacts=[str(artifact)],
        )
        recorded = _payload(lifecycle.record_extraction_task(
            "task_aggregate",
            "completed",
            json.dumps(["field_a"]),
            json.dumps([str(artifact)]),
        ))
        assert recorded["status"] == "SUCCESS"
    finally:
        clear_phase_context()


def test_variant_execution_rejects_incomplete_result_protocol(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "bad_protocol_adapter",
        "adapter",
        "验证统一结果协议",
        json.dumps({"input_contract": {}, "output_contract": {"format": "json"}}),
    ))
    variant_dir = Path(created["artifacts"]["variant_dir"])
    script = variant_dir / "variant.py"
    source = '''import json
import os
import sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
output_dir = Path(os.environ["OUTPUT_DIR"])
print('VARIANT_RESULT_JSON={"status":"SUCCESS","artifacts":[]}')
'''
    assert _payload(atomic.Read(str(script)))["status"] == "SUCCESS"
    assert _payload(atomic.Write(str(script), source))["status"] == "SUCCESS"
    assert _payload(lifecycle.validate_skill_variant("bad_protocol_adapter"))["status"] == "SUCCESS"

    executed = _payload(atomic.ExecutePython(
        str(script),
        args=[str(variant_dir / "request.json")],
        timeout_seconds=10,
    ))

    assert executed["status"] == "NEEDS_REPAIR"
    assert "field_mappings" in executed["issues"][0]


def test_audit_rejects_unvalidated_variant_and_writes_report(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    _payload(lifecycle.create_skill_variant(
        "base_skill",
        "draft_adapter",
        "adapter",
        "测试审计",
        json.dumps({"input_contract": {}, "output_contract": {"filename": "x.csv"}}),
    ))

    audit = _payload(lifecycle.audit_skill_usage())

    assert audit["status"] == "NEEDS_REPAIR"
    assert any("validated" in issue for issue in audit["issues"])
    assert Path(audit["artifacts"]["report_path"]).name == "skill_usage_report.json"


def test_use_decision_cannot_create_unnecessary_variant(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    assert _payload(atomic.Read(str(analysis)))["status"] == "SUCCESS"
    planned = _payload(lifecycle.plan_skill_usage(
        json.dumps({"analysis_report": str(analysis)}),
        json.dumps([{
            "skill_name": "base_skill",
            "decision": "use",
            "reason": "现有参数足够",
            "parameters": {"mode": "wide"},
            "capability_gap": "",
            "why_parameters_insufficient": "",
        }], ensure_ascii=False),
    ))
    assert planned["status"] == "SUCCESS"

    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "unnecessary_adapter",
        "adapter",
        "不应创建",
        json.dumps({"input_contract": {}, "output_contract": {}}),
    ))

    assert created["status"] == "NEEDS_REPAIR"
    assert "adapt" in created["issues"][0]


def test_variant_safety_and_contract_failures_do_not_validate(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "unsafe_adapter",
        "fork",
        "测试危险调用",
        json.dumps({
            "input_contract": {},
            "output_contract": {
                "filename": "result.csv",
                "format": "csv",
                "required_columns": ["patient_id"],
                "allow_empty": False,
            },
        }),
    ))
    variant_dir = Path(created["artifacts"]["variant_dir"])
    _implement_csv_variant(atomic, variant_dir)
    script = variant_dir / "variant.py"
    assert _payload(atomic.Read(str(script)))["status"] == "SUCCESS"
    unsafe_source = script.read_text(encoding="utf-8").replace(
        "import sys",
        "import sys\nimport subprocess",
    )
    assert _payload(atomic.Write(str(script), unsafe_source))["status"] == "SUCCESS"

    unsafe = _payload(lifecycle.validate_skill_variant("unsafe_adapter"))
    assert unsafe["status"] == "NEEDS_REPAIR"
    assert "Forbidden import" in unsafe["issues"][0]

    _implement_csv_variant(atomic, variant_dir)
    assert _payload(lifecycle.validate_skill_variant("unsafe_adapter"))["status"] == "SUCCESS"
    bad_artifact = Path(created["artifacts"]["variant_dir"]) / "result.csv"
    bad_artifact.write_text("wrong\n1\n", encoding="utf-8")
    mismatch = _payload(lifecycle.validate_skill_variant("unsafe_adapter", str(bad_artifact)))
    assert mismatch["status"] == "NEEDS_REPAIR"
    assert "required columns" in mismatch["issues"][0]


def test_base_hash_change_prevents_variant_validation(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "hash_adapter",
        "adapter",
        "测试基础源码哈希",
        json.dumps({"input_contract": {}, "output_contract": {"filename": "result.csv"}}),
    ))
    (base / "skill.py").write_text("SKILL = {'name': 'changed'}\n", encoding="utf-8")

    validated = _payload(lifecycle.validate_skill_variant("hash_adapter"))

    assert created["status"] == "SUCCESS"
    assert validated["status"] == "NEEDS_REPAIR"
    assert "changed" in validated["issues"][0]


def test_complete_variant_publish_and_audit_flow(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "complete_adapter",
        "adapter",
        "完整运行内派生流程",
        json.dumps({
            "input_contract": {},
            "output_contract": {
                "filename": "result.csv",
                "format": "csv",
                "required_columns": ["patient_id", "value"],
                "primary_key": "patient_id",
                "allow_empty": False,
            },
        }),
    ))
    variant_dir = Path(created["artifacts"]["variant_dir"])
    _implement_csv_variant(atomic, variant_dir)
    assert _payload(lifecycle.validate_skill_variant("complete_adapter"))["status"] == "SUCCESS"
    executed = _payload(atomic.ExecutePython(
        str(variant_dir / "variant.py"),
        args=[str(variant_dir / "request.json")],
        timeout_seconds=10,
    ))
    artifact = next(Path(path) for path in executed["artifacts"]["output_files"] if path.endswith("result.csv"))
    assert _payload(lifecycle.validate_skill_variant("complete_adapter", str(artifact)))["status"] == "SUCCESS"
    assert _payload(atomic.Read(str(artifact)))["status"] == "SUCCESS"
    assert _payload(atomic.publish_artifact(str(artifact), "ml_dataset_csv"))["status"] == "SUCCESS"

    audit = _payload(lifecycle.audit_skill_usage())

    assert audit["status"] == "SUCCESS", audit
    report = json.loads(Path(audit["artifacts"]["report_path"]).read_text(encoding="utf-8"))
    assert report["unexplained_deviations"] == []


def test_editing_ready_variant_resets_status_and_metadata_is_protected(lifecycle_env) -> None:
    _run_root, _context, atomic, lifecycle, analysis, _base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    created = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "mutable_adapter",
        "adapter",
        "验证编辑后状态失效",
        json.dumps({"input_contract": {}, "output_contract": {"filename": "result.csv"}}),
    ))
    variant_dir = Path(created["artifacts"]["variant_dir"])
    _implement_csv_variant(atomic, variant_dir)
    assert _payload(lifecycle.validate_skill_variant("mutable_adapter"))["status"] == "SUCCESS"

    metadata = variant_dir / "variant.json"
    assert _payload(atomic.Read(str(metadata)))["status"] == "SUCCESS"
    protected = _payload(atomic.Edit(str(metadata), '"ready"', '"validated"'))
    assert protected["status"] == "NEEDS_REPAIR"

    script = variant_dir / "variant.py"
    assert _payload(atomic.Read(str(script)))["status"] == "SUCCESS"
    edited = _payload(atomic.Edit(str(script), '"value": 1', '"value": 2'))
    assert edited["status"] == "SUCCESS"
    assert json.loads(metadata.read_text(encoding="utf-8"))["status"] == "draft"

    blocked = _payload(atomic.ExecutePython(
        str(script),
        args=[str(variant_dir / "request.json")],
        timeout_seconds=10,
    ))
    assert blocked["status"] == "NEEDS_REPAIR"
    assert "ready" in blocked["issues"][0]


def test_new_run_does_not_discover_previous_run_variants(lifecycle_env, tmp_path: Path) -> None:
    _run_root, _context, atomic, lifecycle, analysis, base = lifecycle_env
    _plan(atomic, lifecycle, analysis)
    first = _payload(lifecycle.create_skill_variant(
        "base_skill",
        "run_one_adapter",
        "adapter",
        "仅本次运行",
        json.dumps({"input_contract": {}, "output_contract": {"filename": "result.csv"}}),
    ))
    assert first["status"] == "SUCCESS"

    second_phase = init_phase_session(tmp_path / "second_run", "engineer")
    second_context = EngineerToolContext(
        engineer_phase_root=second_phase["phase_root"],
        read_roots=[base.parent],
    )
    second_lifecycle = EngineerSkillLifecycleTools(
        second_context,
        skill_names={"base_skill"},
        skills_root=base.parent,
    )

    missing = _payload(second_lifecycle.validate_skill_variant("run_one_adapter"))

    assert missing["status"] == "NEEDS_REPAIR"
    assert not second_context.variant_root.exists()


def test_audit_ignores_unused_restored_variant(lifecycle_env) -> None:
    _run_root, context, atomic, lifecycle, analysis, base = lifecycle_env
    assert _payload(atomic.Read(str(analysis)))["status"] == "SUCCESS"
    planned = _payload(
        lifecycle.plan_skill_usage(
            json.dumps({"analysis_report": str(analysis)}),
            json.dumps(
                [
                    {
                        "skill_name": "base_skill",
                        "decision": "skip",
                        "reason": "本轮字段与该能力无关",
                        "parameters": {},
                        "capability_gap": "",
                        "why_parameters_insufficient": "",
                    }
                ]
            ),
        )
    )
    assert planned["status"] == "SUCCESS"
    restored = context.variant_root / "restored_adapter"
    restored.mkdir(parents=True)
    source_hashes = {str(path.resolve()): _sha256(path) for path in base.iterdir() if path.is_file()}
    (restored / "variant.json").write_text(
        json.dumps(
            {
                "base_skill": "base_skill",
                "mode": "adapter",
                "status": "validated",
                "restored_from_bundle": True,
                "base_source_hashes": source_hashes,
            }
        ),
        encoding="utf-8",
    )

    audit = _payload(lifecycle.audit_skill_usage())

    assert not any("Skipped Skill" in issue for issue in audit["issues"])
    assert not any("restored_adapter" in issue and "not executed" in issue for issue in audit["issues"])
