from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pytest
from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk


class _FakeNativeCodeGraphTool:
    def __init__(self, text: str, *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[dict] = []

    async def call(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("transport closed")
        return ToolChunk(
            content=[TextBlock(text=self.text)],
            state=ToolResultState.RUNNING,
        )


def test_restricted_codegraph_schema_hides_project_path_and_injects_fixed_root(
    tmp_path: Path,
) -> None:
    from agent_tools.codegraph_mcp import RestrictedCodeGraphExplore

    project = tmp_path / "project"
    allowed = project / "workflow"
    allowed.mkdir(parents=True)
    native = _FakeNativeCodeGraphTool(
        "## workflow/example.py\n1\tdef clean(): pass\n",
    )
    tool = RestrictedCodeGraphExplore(
        native_tool=native,
        project_root=project,
        allowed_roots=[allowed],
        usage_path=tmp_path / "usage.jsonl",
        attempt="attempt_0001",
    )

    result = asyncio.run(tool.call(query="clean flow", max_files=6))

    assert set(tool.input_schema["properties"]) == {"query", "max_files"}
    assert "projectPath" not in json.dumps(tool.input_schema)
    assert native.calls == [
        {
            "query": "clean flow",
            "maxFiles": 6,
            "projectPath": str(project.resolve()),
        },
    ]
    assert result.state is ToolResultState.RUNNING


def test_restricted_codegraph_rejects_unknown_arguments_and_out_of_range_limits(
    tmp_path: Path,
) -> None:
    from agent_tools.codegraph_mcp import RestrictedCodeGraphExplore

    project = tmp_path / "project"
    allowed = project / "workflow"
    allowed.mkdir(parents=True)
    native = _FakeNativeCodeGraphTool("## workflow/example.py\n")
    tool = RestrictedCodeGraphExplore(
        native_tool=native,
        project_root=project,
        allowed_roots=[allowed],
        usage_path=tmp_path / "usage.jsonl",
        attempt="attempt_0001",
    )

    injected = asyncio.run(
        tool.call(
            query="clean flow",
            projectPath="/tmp/another-project",
        ),
    )
    oversized = asyncio.run(tool.call(query="clean flow", max_files=9))

    assert injected.state is ToolResultState.DENIED
    assert oversized.state is ToolResultState.DENIED
    assert native.calls == []
    assert "/tmp/another-project" not in injected.content[0].text


def test_restricted_codegraph_denies_entire_response_with_unauthorized_source(
    tmp_path: Path,
) -> None:
    from agent_tools.codegraph_mcp import RestrictedCodeGraphExplore

    project = tmp_path / "project"
    allowed = project / "workflow"
    allowed.mkdir(parents=True)
    denied = project / "tests"
    denied.mkdir()
    (allowed / "example.py").write_text("def clean(): pass\n", encoding="utf-8")
    (denied / "private_rule.py").write_text("SECRET = 'hidden'\n", encoding="utf-8")
    native = _FakeNativeCodeGraphTool(
        "## workflow/example.py\n1\tdef clean(): pass\n"
        "## tests/private_rule.py\n1\tSECRET = 'hidden'\n",
    )
    usage = tmp_path / "usage.jsonl"
    tool = RestrictedCodeGraphExplore(
        native_tool=native,
        project_root=project,
        allowed_roots=[allowed],
        usage_path=usage,
        attempt="attempt_0002",
    )

    result = asyncio.run(tool.call(query="clean flow"))

    assert result.state is ToolResultState.DENIED
    assert "SECRET" not in result.content[0].text
    assert "private_rule.py" not in result.content[0].text
    event = json.loads(usage.read_text(encoding="utf-8").splitlines()[-1])
    assert event["status"] == "denied"
    assert event["returned_files"] == []


def test_restricted_codegraph_records_runtime_failure_for_agent_fallback(
    tmp_path: Path,
) -> None:
    from agent_tools.codegraph_mcp import RestrictedCodeGraphExplore

    project = tmp_path / "project"
    allowed = project / "workflow"
    allowed.mkdir(parents=True)
    usage = tmp_path / "usage.jsonl"
    tool = RestrictedCodeGraphExplore(
        native_tool=_FakeNativeCodeGraphTool("", fail=True),
        project_root=project,
        allowed_roots=[allowed],
        usage_path=usage,
        attempt="attempt_0003",
    )

    result = asyncio.run(tool.call(query="clean flow"))

    assert result.state is ToolResultState.ERROR
    assert "Read/Grep" in result.content[0].text
    event = json.loads(usage.read_text(encoding="utf-8").splitlines()[-1])
    assert event["status"] == "error"
    assert event["fallback"] is True


def test_codegraph_source_hash_only_covers_authorized_roots(tmp_path: Path) -> None:
    from agent_tools.codegraph_mcp import codegraph_source_bundle_sha256

    project = tmp_path / "project"
    allowed = project / "workflow"
    denied = project / "tests"
    allowed.mkdir(parents=True)
    denied.mkdir()
    source = allowed / "clean.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    metadata = allowed / ".DS_Store"
    metadata.write_bytes(b"finder metadata")
    (denied / "private.py").write_text("SECRET = 1\n", encoding="utf-8")

    original = codegraph_source_bundle_sha256(project, [allowed])
    (denied / "private.py").write_text("SECRET = 2\n", encoding="utf-8")
    metadata.write_bytes(b"changed finder metadata")
    unchanged = codegraph_source_bundle_sha256(project, [allowed])
    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = codegraph_source_bundle_sha256(project, [allowed])

    assert len(original) == 64
    assert unchanged == original
    assert changed != original


def test_restricted_codegraph_uses_shared_tool_audit_middleware(tmp_path: Path) -> None:
    from agent_tools.codegraph_mcp import RestrictedCodeGraphExplore

    project = tmp_path / "project"
    allowed = project / "workflow"
    allowed.mkdir(parents=True)
    (allowed / "clean.py").write_text("def clean(): pass\n", encoding="utf-8")
    audit = tmp_path / "tool_audit.jsonl"
    tool = RestrictedCodeGraphExplore(
        native_tool=_FakeNativeCodeGraphTool("## workflow/clean.py\n"),
        project_root=project,
        allowed_roots=[allowed],
        usage_path=tmp_path / "codegraph_usage.jsonl",
        audit_path=audit,
        attempt="attempt_0004",
    )

    async def invoke():
        stream = await tool(query="clean flow")
        return [chunk async for chunk in stream]

    chunks = asyncio.run(invoke())

    assert chunks[-1].state is ToolResultState.RUNNING
    event = json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])
    assert event["tool"] == "CodeGraphExplore"
    assert event["status"] == "SUCCESS"


def test_open_codegraph_runtime_uses_native_mcp_and_closes_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent_tools.codegraph_mcp as codegraph

    project = tmp_path / "project"
    allowed = project / "workflow"
    allowed.mkdir(parents=True)
    (allowed / "clean.py").write_text("def clean(): pass\n", encoding="utf-8")
    index = project / ".codegraph"
    index.mkdir()
    (index / "codegraph.db").touch()
    binary = tmp_path / "codegraph"
    binary.touch()
    events: list[str] = []
    captured: dict = {}
    native = _FakeNativeCodeGraphTool("## workflow/clean.py\n")

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.is_connected = False

        async def connect(self) -> None:
            events.append("connect")
            self.is_connected = True

        async def get_tool(self, name: str):
            events.append(f"get:{name}")
            return native

        async def close(self) -> None:
            events.append("close")
            self.is_connected = False

    monkeypatch.setattr(codegraph, "MCPClient", FakeClient)
    monkeypatch.setattr(codegraph, "_resolve_codegraph_binary", lambda: binary)
    monkeypatch.setattr(codegraph, "_installed_codegraph_version", lambda _binary: "1.5.0")

    runtime = asyncio.run(
        codegraph.open_codegraph_runtime(
            project_root=project,
            allowed_roots=[allowed],
            usage_path=tmp_path / "usage.jsonl",
            attempt="attempt_0004",
        ),
    )

    assert events == ["connect", "get:codegraph_explore"]
    assert runtime.tool.name == "CodeGraphExplore"
    config = captured["mcp_config"]
    assert config.command == str(binary.resolve())
    assert config.args == ["serve", "--mcp", "--path", str(project.resolve())]
    assert config.cwd == project.resolve()
    assert config.env["CODEGRAPH_TELEMETRY"] == "0"
    assert config.env["CODEGRAPH_MCP_TOOLS"] == "explore"
    assert "OPENAI_API_KEY" not in config.env
    assert captured["enable_tools"] == ["codegraph_explore"]

    asyncio.run(runtime.close())
    assert events[-1] == "close"


def test_codegraph_preflight_requires_existing_index_and_exact_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent_tools.codegraph_mcp as codegraph

    project = tmp_path / "project"
    project.mkdir()
    binary = tmp_path / "codegraph"
    binary.touch()
    monkeypatch.setattr(codegraph, "_resolve_codegraph_binary", lambda: binary)
    monkeypatch.setattr(codegraph, "_installed_codegraph_version", lambda _binary: "1.5.0")

    with pytest.raises(RuntimeError, match="codegraph init"):
        codegraph.validate_codegraph_installation(project)

    index = project / ".codegraph"
    index.mkdir()
    (index / "codegraph.db").touch()
    checked = codegraph.validate_codegraph_installation(project)
    assert checked == {
        "project_root": str(project.resolve()),
        "binary": str(binary.resolve()),
        "version": "1.5.0",
    }

    monkeypatch.setattr(codegraph, "_installed_codegraph_version", lambda _binary: "1.6.0")
    with pytest.raises(RuntimeError, match="requires CodeGraph 1.5.0"):
        codegraph.validate_codegraph_installation(project)


def test_reference_toolkit_only_registers_explicit_codegraph_tool(tmp_path: Path) -> None:
    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.codegraph_mcp import RestrictedCodeGraphExplore
    from agent_tools.context import EngineerToolContext

    context = EngineerToolContext.from_task(
        task_text="clean data",
        engineer_phase_root=tmp_path / "phase",
        require_skill_plan=False,
    )
    fake = RestrictedCodeGraphExplore(
        native_tool=_FakeNativeCodeGraphTool("no matches"),
        project_root=tmp_path,
        allowed_roots=[tmp_path],
        usage_path=tmp_path / "usage.jsonl",
        attempt="attempt_0005",
    )

    baseline, _ = create_reference_toolkit(context)
    enabled, _ = create_reference_toolkit(context, codegraph_tool=fake)

    assert asyncio.run(baseline.get_tool("CodeGraphExplore")) is None
    assert asyncio.run(enabled.get_tool("CodeGraphExplore")) is fake


def test_codegraph_cli_is_opt_in() -> None:
    from main import parse_args

    baseline = parse_args(
        [
            "--workflow",
            "reference-guided-train-validate",
            "--dataset-split",
            "/tmp/dataset",
            "--experiment-dir",
            "/tmp/experiment",
            "task",
        ],
    )
    enabled = parse_args(
        [
            "--workflow",
            "reference-guided-correct",
            "--dataset-split",
            "/tmp/dataset",
            "--experiment-dir",
            "/tmp/experiment",
            "--enable-codegraph",
            "task",
        ],
    )

    assert baseline.enable_codegraph is False
    assert enabled.enable_codegraph is True


def test_codegraph_prompt_guidance_only_appears_when_enabled(tmp_path: Path) -> None:
    from agent_tools.context import EngineerToolContext
    from workflow.reference_correction import _correction_system_prompt
    from workflow.reference_guided import _data_cleaning_agent_system_prompt

    context = EngineerToolContext.from_task(
        task_text="clean data",
        engineer_phase_root=tmp_path / "phase",
        require_skill_plan=False,
    )
    validation_baseline = _data_cleaning_agent_system_prompt(
        "",
        context,
        codegraph_enabled=False,
    )
    validation_enabled = _data_cleaning_agent_system_prompt(
        "",
        context,
        codegraph_enabled=True,
    )
    correction_baseline = _correction_system_prompt(context, codegraph_enabled=False)
    correction_enabled = _correction_system_prompt(context, codegraph_enabled=True)

    assert "CodeGraphExplore" not in validation_baseline
    assert "CodeGraphExplore" in validation_enabled
    assert "CodeGraphExplore" not in correction_baseline
    assert "CodeGraphExplore" in correction_enabled


def test_codegraph_usage_summary_includes_tool_and_token_counts(tmp_path: Path) -> None:
    from agent_tools.codegraph_mcp import summarize_codegraph_usage

    phase = tmp_path / "attempt_0001" / "agent_runs" / "phase_0001"
    phase.mkdir(parents=True)
    events = [
        {
            "status": "success",
            "returned_files": ["workflow/a.py", "lib/b.py"],
            "output_characters": 120,
            "duration_seconds": 0.2,
            "stale": False,
            "fallback": False,
        },
        {
            "status": "denied",
            "returned_files": [],
            "output_characters": 0,
            "duration_seconds": 0.1,
            "stale": False,
            "fallback": True,
        },
    ]
    (phase / "codegraph_usage.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    (phase / "tool_audit.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"tool": "Read", "status": "SUCCESS"}),
                json.dumps({"tool": "Read", "status": "SUCCESS"}),
                json.dumps({"tool": "Grep", "status": "SUCCESS"}),
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (phase / "skill_usage_report.json").write_text(
        json.dumps({"agent_usage": {"total_tokens": 321}}),
        encoding="utf-8",
    )

    summary = summarize_codegraph_usage(tmp_path)

    assert summary["call_count"] == 2
    assert summary["status_counts"] == {"denied": 1, "success": 1}
    assert summary["returned_file_count"] == 2
    assert summary["returned_files"] == ["lib/b.py", "workflow/a.py"]
    assert summary["output_characters"] == 120
    assert summary["stale_calls"] == 0
    assert summary["fallback_calls"] == 1
    assert summary["read_calls"] == 2
    assert summary["grep_calls"] == 1
    assert summary["model_tokens"] == 321


def test_checkpoint_test_agent_has_no_codegraph_integration() -> None:
    from workflow.reference_test_stage import ReferenceCheckpointTestRuntime

    source = inspect.getsource(ReferenceCheckpointTestRuntime._run_agent_session)

    assert "open_codegraph_runtime" not in source
    assert "codegraph_tool" not in source
    assert "CodeGraphExplore" not in source
