from __future__ import annotations

import asyncio
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_reference_runtime_uses_native_agentscope2_context() -> None:
    from agent.reference_runtime import (
        AGENTSCOPE_VERSION,
        create_reference_context_config,
        create_reference_react_config,
    )

    context = create_reference_context_config("MiniMax-M3")
    react = create_reference_react_config(37)

    assert AGENTSCOPE_VERSION == "2.0.4.post1"
    assert context.trigger_ratio == pytest.approx(0.8999)
    assert context.reserve_ratio == pytest.approx(0.1)
    assert context.tool_result_limit == 4_000
    assert react.max_iters == 37


def test_openai_model_forwards_context_size_and_seed(monkeypatch) -> None:
    import lib.agent_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "effective_agent_config",
        lambda _key: {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "model": "MiniMax-M3",
            "temperature": 0.0,
            "seed": 666,
        },
    )

    model, _ = runtime.create_openai_model_and_formatter("react_planner", "fallback")

    assert model.context_size == 204_800
    assert model.parameters.temperature == 0.0
    assert model.parameters.parallel_tool_calls is False
    assert model.extra_body == {"seed": 666}
    assert model.credential.base_url == "https://example.invalid/v1"


def test_openai_model_rotates_to_next_key_after_rate_limit(monkeypatch) -> None:
    from agentscope.message import TextBlock
    from agentscope.model import ChatResponse

    import lib.agent_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "effective_agent_config",
        lambda _key: {
            "api_keys": ["first-key", "second-key"],
            "base_url": "https://example.invalid/v1",
            "model": "MiniMax-M3",
        },
    )
    model, _ = runtime.create_openai_model_and_formatter("react_planner", "fallback")
    seen_keys: list[str] = []

    class FakeRateLimitError(Exception):
        status_code = 429

    async def fake_call_api(self, *_args, **_kwargs):
        seen_keys.append(self.credential.api_key.get_secret_value())
        if len(seen_keys) == 1:
            raise FakeRateLimitError("quota exhausted")
        return ChatResponse(content=[TextBlock(text="recovered")], is_last=True)

    monkeypatch.setattr(type(model), "_call_api", fake_call_api)

    response = asyncio.run(model([]))

    assert response.content[0].text == "recovered"
    assert seen_keys == ["first-key", "second-key"]


def test_openai_model_accepts_worker_key_bundle_and_env_parameters(monkeypatch) -> None:
    import lib.agent_runtime as runtime

    monkeypatch.setattr(runtime, "effective_agent_config", lambda _key: {})
    monkeypatch.setenv("OPENAI_API_KEYS_JSON", '["worker-first", "worker-second"]')
    monkeypatch.setenv("OPENAI_API_BASE", "https://worker.invalid/v1")
    monkeypatch.setenv("AGENT_TEMPERATURE", "0.25")
    monkeypatch.setenv("AGENT_SEED", "777")

    model, _ = runtime.create_openai_model_and_formatter("react_planner", "MiniMax-M3")

    assert model._api_keys == ("worker-first", "worker-second")
    assert model.credential.base_url == "https://worker.invalid/v1"
    assert model.parameters.temperature == pytest.approx(0.25)
    assert model.extra_body == {"seed": 777}


def test_build_worker_model_environment_contains_no_unrelated_host_values(monkeypatch) -> None:
    import lib.agent_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "effective_agent_config",
        lambda _key: {
            "api_keys": ["first", "second"],
            "base_url": "https://example.invalid/v1",
            "model": "MiniMax-M3",
            "temperature": 0.0,
            "seed": 666,
        },
    )
    environment = runtime.build_worker_model_environment("react_planner", "MiniMax-M3")

    assert json.loads(environment["OPENAI_API_KEYS_JSON"]) == ["first", "second"]
    assert environment["OPENAI_API_BASE"] == "https://example.invalid/v1"
    assert environment["MODEL_NAME"] == "MiniMax-M3"
    assert environment["AGENT_TEMPERATURE"] == "0.0"
    assert environment["AGENT_SEED"] == "666"
    assert environment["V2_SKIP_LOCAL_MODEL_CONFIG"] == "1"


def test_openai_model_raises_single_error_when_all_keys_are_rate_limited(monkeypatch) -> None:
    import lib.agent_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "effective_agent_config",
        lambda _key: {
            "api_keys": ["first-key", "second-key"],
            "base_url": "https://example.invalid/v1",
            "model": "MiniMax-M3",
        },
    )
    model, _ = runtime.create_openai_model_and_formatter("react_planner", "fallback")

    class FakeRateLimitError(Exception):
        status_code = 429

    async def fake_call_api(self, *_args, **_kwargs):
        raise FakeRateLimitError("quota exhausted")

    monkeypatch.setattr(type(model), "_call_api", fake_call_api)

    with pytest.raises(runtime.AllApiKeysRateLimitedError):
        asyncio.run(model([]))


def test_rate_limit_block_state_preserves_attempt_history() -> None:
    from lib.agent_runtime import AllApiKeysRateLimitedError
    from workflow.reference_guided import _rate_limit_block_state

    original = {
        "status": "active",
        "attempts": [{"attempt": 1, "status": "invalid"}],
        "invalid_attempt_count": 1,
    }

    blocked = _rate_limit_block_state(
        original,
        AllApiKeysRateLimitedError("all 2 configured API keys are rate limited"),
    )

    assert blocked["status"] == "blocked_rate_limited"
    assert blocked["termination_reason"] == "all_api_keys_rate_limited"
    assert blocked["attempts"] == original["attempts"]
    assert blocked["invalid_attempt_count"] == 1


def test_managed_openai_model_closes_owned_http_client(monkeypatch) -> None:
    import lib.agent_runtime as runtime

    closed = False

    class FakeHttpClient:
        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(runtime.httpx, "AsyncClient", lambda **_kwargs: FakeHttpClient())
    monkeypatch.setattr(runtime, "effective_agent_config", lambda _key: {"api_key": "test-key"})

    model, _ = runtime.create_openai_model_and_formatter("react_planner", "MiniMax-M3")
    assert isinstance(model, runtime.ManagedOpenAIChatModel)
    asyncio.run(model.aclose())
    assert closed


def test_close_reference_agent_closes_workspace_and_model() -> None:
    from agent.reference_runtime import close_reference_agent

    events: list[str] = []

    class FakeWorkspace:
        async def close(self) -> None:
            events.append("workspace")

    class FakeModel:
        async def aclose(self) -> None:
            events.append("model")

    asyncio.run(
        close_reference_agent(
            SimpleNamespace(model=FakeModel()),
            FakeWorkspace(),
        ),
    )

    assert events == ["workspace", "model"]


def test_openai_request_body_contains_seed_and_generation_parameters(monkeypatch) -> None:
    import openai
    from agentscope.message import UserMsg

    import lib.agent_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "effective_agent_config",
        lambda _key: {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "model": "MiniMax-M3",
            "temperature": 0.0,
            "seed": 666,
        },
    )
    model, _ = runtime.create_openai_model_and_formatter("react_planner", "fallback")
    captured: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        type(model),
        "_parse_completion_response",
        lambda self, *args: "parsed",
    )

    result = asyncio.run(
        model._call_api(
            model.model,
            [UserMsg(name="user", content="test")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        ),
    )

    assert result == "parsed"
    assert captured["model"] == "MiniMax-M3"
    assert captured["temperature"] == 0.0
    assert captured["extra_body"] == {"seed": 666}
    assert captured["parallel_tool_calls"] is False
    assert captured["client"]["base_url"] == "https://example.invalid/v1"


def test_restricted_backend_enforces_roots_and_symlink_escape(tmp_path: Path) -> None:
    from agent_tools.restricted_backend import RestrictedLocalBackend

    readable = tmp_path / "readable"
    writable = tmp_path / "workspace"
    outside = tmp_path / "outside"
    readable.mkdir()
    writable.mkdir()
    outside.mkdir()
    (readable / "input.txt").write_text("input", encoding="utf-8")
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (writable / "escape").symlink_to(outside, target_is_directory=True)

    backend = RestrictedLocalBackend(
        read_roots=[readable, writable],
        write_roots=[writable],
        cwd=writable,
    )

    assert asyncio.run(backend.read_file(str(readable / "input.txt"))) == b"input"
    asyncio.run(backend.write_file(str(writable / "result.txt"), b"result"))
    assert (writable / "result.txt").read_text(encoding="utf-8") == "result"

    with pytest.raises(PermissionError):
        asyncio.run(backend.read_file(str(outside / "secret.txt")))
    with pytest.raises(PermissionError):
        asyncio.run(backend.read_file(str(writable / "escape/secret.txt")))
    with pytest.raises(PermissionError):
        asyncio.run(backend.write_file(str(readable / "changed.txt"), b"no"))


def test_restricted_backend_denied_read_roots_override_parent_access(tmp_path: Path) -> None:
    from agent_tools.restricted_backend import RestrictedLocalBackend

    workspace = tmp_path / "workspace"
    denied = tmp_path / "skills/error_view"
    workspace.mkdir()
    denied.mkdir(parents=True)
    (denied / "SKILL.md").write_text("private prior", encoding="utf-8")

    backend = RestrictedLocalBackend(
        read_roots=[tmp_path],
        denied_read_roots=[denied],
        write_roots=[workspace],
        cwd=workspace,
    )

    with pytest.raises(PermissionError, match="denied read root"):
        asyncio.run(backend.read_file(str(denied / "SKILL.md")))


def test_reference_toolkit_uses_native_file_tools(tmp_path: Path) -> None:
    from agentscope.tool import Edit, Glob, Grep, Read, Write

    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.context import EngineerToolContext

    task = json.dumps(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "engineer_phase_root": str(tmp_path / "phase"),
            "read_roots": [str(tmp_path / "read")],
            "write_roots": [str(tmp_path / "workspace")],
        },
    )
    context = EngineerToolContext.from_task(
        task_text=task,
        engineer_phase_root=tmp_path / "phase",
        explorer_phase_root=tmp_path,
        additional_read_roots=[tmp_path / "read"],
        require_skill_plan=False,
        split_mode="validation",
    )
    toolkit, _ = create_reference_toolkit(context)
    names = [
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Bash",
        "ExecutePython",
        "RunSkill",
        "PublishDirectoryArtifact",
    ]
    tools = {name: asyncio.run(toolkit.get_tool(name)) for name in names}

    assert isinstance(tools["Read"], Read)
    assert isinstance(tools["Write"], Write)
    assert isinstance(tools["Edit"], Edit)
    assert isinstance(tools["Glob"], Glob)
    assert isinstance(tools["Grep"], Grep)
    assert tools["Bash"] is None
    assert tools["ExecutePython"] is not None
    assert tools["RunSkill"] is not None
    assert tools["PublishDirectoryArtifact"] is not None


def test_native_file_tools_execute_through_restricted_backend(tmp_path: Path) -> None:
    from agentscope.message import ToolResultState

    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.context import EngineerToolContext

    phase = tmp_path / "phase"
    context = EngineerToolContext.from_task(
        task_text=json.dumps(
            {
                "workspace_dir": str(phase / "workspace"),
                "engineer_phase_root": str(phase),
                "read_roots": [str(tmp_path / "read")],
                "write_roots": [str(phase)],
            },
        ),
        engineer_phase_root=phase,
        explorer_phase_root=tmp_path,
        additional_read_roots=[tmp_path / "read"],
        require_skill_plan=False,
        split_mode="validation",
    )
    toolkit, _ = create_reference_toolkit(context)
    target = context.workspace_dir / "nested" / "result.py"

    async def exercise():
        write = await toolkit.get_tool("Write")
        read = await toolkit.get_tool("Read")
        glob = await toolkit.get_tool("Glob")
        grep = await toolkit.get_tool("Grep")
        edit = await toolkit.get_tool("Edit")
        written = await write.call(file_path=str(target), content="print('ok')\n")
        observed = await read.call(file_path=str(target))
        matched_paths = await glob.call(pattern="**/*.py", path=str(context.workspace_dir))
        matched_text = await grep.call(
            pattern="print",
            path=str(context.workspace_dir),
            output_mode="content",
        )
        edited = await edit.call(
            file_path=str(target),
            old_string="print('ok')",
            new_string="print('done')",
        )
        return written, observed, matched_paths, matched_text, edited

    results = asyncio.run(exercise())

    names = ("Write", "Read", "Glob", "Grep", "Edit")
    errors = [
        {
            "tool": name,
            "state": str(result.state),
            "content": "\n".join(
                str(getattr(block, "text", block))
                for block in result.content
            ),
        }
        for name, result in zip(names, results)
        if result.state is ToolResultState.ERROR
    ]
    assert not errors, errors
    assert target.read_text(encoding="utf-8") == "print('done')\n"
    assert str(target) in results[2].content[0].text
    assert "print('ok')" in results[3].content[0].text


def test_skill_viewer_reads_registered_skill_body(tmp_path: Path) -> None:
    from agentscope.state import AgentState

    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.context import EngineerToolContext

    context = EngineerToolContext.from_task(
        task_text="MIMIC ICU mortality",
        engineer_phase_root=tmp_path / "phase",
        require_skill_plan=False,
    )
    toolkit, manifest = create_reference_toolkit(context)

    async def exercise():
        instructions = await toolkit.get_skill_instructions()
        viewer = await toolkit.get_tool("Skill")
        result = await viewer.call(
            skill="pipeline_build_cohort",
            _agent_state=AgentState(),
        )
        schemas = await toolkit.get_tool_schemas()
        return instructions, result, schemas

    instructions, result, schemas = asyncio.run(exercise())
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    tool_names = {schema["function"]["name"] for schema in schemas}

    assert len(manifest) == 9
    assert instructions is not None and "pipeline_build_cohort" in instructions
    assert "correction_intra_table_errors" not in instructions
    assert "pipeline_build_cohort" in text
    assert "Skill" in tool_names


def test_reference_toolkit_can_disable_pipeline_skills(tmp_path: Path) -> None:
    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.context import EngineerToolContext

    context = EngineerToolContext.from_task(
        task_text="MIMIC ICU mortality",
        engineer_phase_root=tmp_path / "phase",
        require_skill_plan=False,
    )
    toolkit, manifest = create_reference_toolkit(context, include_pipeline_skills=False)

    async def exercise():
        return await toolkit.get_skill_instructions(), await toolkit.get_tool("Skill")

    instructions, skill_viewer = asyncio.run(exercise())
    assert manifest == []
    assert instructions is None
    assert skill_viewer is None
    assert context.executable_skill_names == set()


def test_error_view_skills_are_native_knowledge_only_skills() -> None:
    from agent.reference_runtime import DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS, V2_DIR

    forbidden = {
        "990045",
        "220045",
        "995158",
        "225158",
        "999001",
        "e87_alias",
        "repair_hint",
        "ground_truth",
        "reference_private",
    }
    assert len(DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS) == 4
    for name in DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS:
        skill_dir = V2_DIR / "skills" / name
        skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        lowered = skill_text.casefold()
        assert not (skill_dir / "skill.py").exists()
        assert f"name: {name}" in skill_text
        assert all(value not in lowered for value in forbidden)
        for heading in (
            "## 适用条件",
            "## 证据要求",
            "## 推荐工具",
            "## 执行步骤",
            "## 修复与停止条件",
            "## 禁止事项",
        ):
            assert heading in skill_text
        assert all(
            tool_name in skill_text
            for tool_name in ("InspectDataFile", "CompareArtifact", "ExecutePython")
        )


def test_error_view_skills_are_loaded_on_demand_but_not_executable(tmp_path: Path) -> None:
    from agentscope.message import ToolResultState
    from agentscope.state import AgentState

    from agent.reference_runtime import (
        DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS,
        DATA_CLEANING_AGENT_PIPELINE_SKILLS,
        create_reference_toolkit,
    )
    from agent_tools.context import EngineerToolContext

    context = EngineerToolContext.from_task(
        task_text="reference-guided correction",
        engineer_phase_root=tmp_path / "phase",
        require_skill_plan=False,
    )
    toolkit, manifest = create_reference_toolkit(
        context,
        include_pipeline_skills=True,
        include_error_view_skills=True,
    )

    async def exercise():
        instructions = await toolkit.get_skill_instructions()
        viewer = await toolkit.get_tool("Skill")
        viewed = await viewer.call(
            skill="correction_intra_table_errors",
            _agent_state=AgentState(),
        )
        run_skill = await toolkit.get_tool("RunSkill")
        executed = await run_skill.call(
            skill_name="correction_intra_table_errors",
            spec_json="{}",
        )
        return instructions, viewed, executed

    instructions, viewed, executed = asyncio.run(exercise())
    viewed_text = "\n".join(
        block.text for block in viewed.content if hasattr(block, "text")
    )
    manifest_names = {item["name"] for item in manifest}

    assert manifest_names == (
        DATA_CLEANING_AGENT_PIPELINE_SKILLS | DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS
    )
    assert all(name in instructions for name in DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS)
    assert "# 单表错误纠正" in viewed_text
    assert context.executable_skill_names == DATA_CLEANING_AGENT_PIPELINE_SKILLS
    assert executed.state is ToolResultState.ERROR


def test_custom_tool_schemas_match_wrapped_method_signatures(tmp_path: Path) -> None:
    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.agentscope2_tools import MethodTool
    from agent_tools.context import EngineerToolContext

    context = EngineerToolContext.from_task(
        task_text="MIMIC ICU mortality",
        engineer_phase_root=tmp_path / "phase",
        require_skill_plan=False,
    )
    toolkit, _ = create_reference_toolkit(context)

    async def exercise():
        schemas = await toolkit.get_tool_schemas()
        tools = {
            schema["function"]["name"]: await toolkit.get_tool(schema["function"]["name"])
            for schema in schemas
        }
        return tools

    tools = asyncio.run(exercise())
    custom_tools = [tool for tool in tools.values() if isinstance(tool, MethodTool)]

    assert custom_tools
    for tool in custom_tools:
        parameters = inspect.signature(tool.method).parameters
        properties = tool.input_schema["properties"]
        required = set(tool.input_schema.get("required") or [])
        assert set(properties) <= set(parameters), tool.name
        assert required <= set(properties), tool.name
        for name, parameter in parameters.items():
            if parameter.default is inspect.Parameter.empty:
                assert name in required, f"{tool.name}.{name} must be required"


def test_run_skill_and_execute_python_keep_restricted_runtime(tmp_path: Path) -> None:
    from agentscope.message import ToolResultState

    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.context import EngineerToolContext
    from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context

    agent_runs = tmp_path / "agent_runs"
    phase = init_phase_session(agent_runs, "data_cleaning_agent")
    phase_root = Path(phase["phase_root"])
    set_phase_context("agentscope2-tool-test", agent_runs, "data_cleaning_agent", phase_root)
    context = EngineerToolContext.from_task(
        task_text="MIMIC ICU mortality",
        engineer_phase_root=phase_root,
        explorer_phase_root=tmp_path,
        additional_read_roots=[tmp_path],
        require_skill_plan=False,
        split_mode="validation",
    )
    source = context.workspace_dir / "source.csv"
    allowed_script = context.workspace_dir / "allowed.py"
    blocked_script = context.workspace_dir / "blocked.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("stay_id,value\n1,ok\n", encoding="utf-8")
    allowed_script.write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['OUTPUT_DIR'], 'result.txt').write_text('ok')\n",
        encoding="utf-8",
    )
    blocked_script.write_text(
        "import socket\nsocket.create_connection(('example.com', 80), timeout=1)\n",
        encoding="utf-8",
    )

    async def exercise():
        toolkit, _ = create_reference_toolkit(context)
        run_skill = await toolkit.get_tool("RunSkill")
        execute_python = await toolkit.get_tool("ExecutePython")
        assembled = await run_skill.call(
            skill_name="pipeline_assemble_reference_package",
            spec_json=json.dumps(
                {
                    "files": [
                        {
                            "source": str(source),
                            "target": "cohort/cohort.csv",
                        },
                    ],
                },
            ),
        )
        allowed = await execute_python.call(file_path=str(allowed_script))
        blocked = await execute_python.call(file_path=str(blocked_script))
        return assembled, allowed, blocked

    try:
        assembled, allowed, blocked = asyncio.run(exercise())
    finally:
        clear_phase_context()

    assert assembled.state is ToolResultState.SUCCESS
    assert allowed.state is ToolResultState.SUCCESS
    assert blocked.state is ToolResultState.ERROR
    assert "socket" in blocked.content[0].text.lower()


def test_execute_python_cannot_read_or_enumerate_denied_skill_roots(tmp_path: Path) -> None:
    from agentscope.message import ToolResultState

    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.context import EngineerToolContext
    from lib.agent_artifacts import clear_phase_context, init_phase_session, set_phase_context

    agent_runs = tmp_path / "agent_runs"
    phase = init_phase_session(agent_runs, "data_cleaning_agent")
    phase_root = Path(phase["phase_root"])
    set_phase_context("agentscope2-denied-root-test", agent_runs, "data_cleaning_agent", phase_root)
    skills_root = tmp_path / "skills"
    denied = skills_root / "correction_intra_table_errors"
    denied.mkdir(parents=True)
    (denied / "SKILL.md").write_text("private prior", encoding="utf-8")
    context = EngineerToolContext.from_task(
        task_text="correction ablation",
        engineer_phase_root=phase_root,
        additional_read_roots=[tmp_path],
        denied_read_roots=[denied],
        require_skill_plan=False,
        split_mode="correction",
    )
    read_script = context.workspace_dir / "read_denied.py"
    enumerate_script = context.workspace_dir / "enumerate_denied.py"
    read_script.write_text(
        f"from pathlib import Path\nprint(Path({str(denied / 'SKILL.md')!r}).read_text())\n",
        encoding="utf-8",
    )
    enumerate_script.write_text(
        f"import os\nprint(os.listdir({str(skills_root)!r}))\n",
        encoding="utf-8",
    )

    async def exercise():
        toolkit, _ = create_reference_toolkit(
            context,
            include_pipeline_skills=False,
            include_error_view_skills=False,
        )
        execute_python = await toolkit.get_tool("ExecutePython")
        denied_read = await execute_python.call(file_path=str(read_script))
        denied_enumeration = await execute_python.call(file_path=str(enumerate_script))
        return denied_read, denied_enumeration

    try:
        denied_read, denied_enumeration = asyncio.run(exercise())
    finally:
        clear_phase_context()

    assert denied_read.state is ToolResultState.ERROR
    assert "denied read root" in denied_read.content[0].text.lower()
    assert denied_enumeration.state is ToolResultState.ERROR
    assert "denied read root" in denied_enumeration.content[0].text.lower()


def test_tool_middleware_audits_error_result_state(tmp_path: Path) -> None:
    from agentscope.message import ToolCallBlock
    from agentscope.state import AgentState

    from agent.reference_runtime import create_reference_toolkit
    from agent_tools.context import EngineerToolContext

    phase = tmp_path / "phase"
    context = EngineerToolContext.from_task(
        task_text="MIMIC ICU mortality",
        engineer_phase_root=phase,
        require_skill_plan=False,
    )
    toolkit, _ = create_reference_toolkit(context)

    async def exercise() -> None:
        call = ToolCallBlock(
            id="missing-package",
            name="ValidateResultPackage",
            input=json.dumps({"package_root": str(phase / "missing")}),
        )
        async for _ in toolkit.call_tool(call, AgentState()):
            pass

    asyncio.run(exercise())
    records = [
        json.loads(line)
        for line in (context.workspace_dir / "tool_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert records[-1]["tool"] == "ValidateResultPackage"
    assert records[-1]["status"] == "ERROR"


def test_reference_agent_factory_resets_state_and_iteration_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agentscope.credential import OpenAICredential
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.model import OpenAIChatModel
    from agentscope.tool import Toolkit

    import agent.reference_runtime as runtime

    def model_factory():
        model = OpenAIChatModel(
            credential=OpenAICredential(
                api_key="test-key",
                base_url="https://example.invalid/v1",
            ),
            model="MiniMax-M3",
            context_size=204_800,
            formatter=OpenAIChatFormatter(),
            stream=False,
        )
        return model, model.formatter

    monkeypatch.setattr(runtime, "make_reference_model", model_factory)

    async def exercise():
        first, first_workspace = await runtime.create_reference_agent(
            name="Data Cleaning Agent",
            system_prompt="test",
            toolkit=Toolkit(),
            workspace_dir=tmp_path / "attempt_1",
            max_iters=73,
        )
        second, second_workspace = await runtime.create_reference_agent(
            name="Data Cleaning Agent",
            system_prompt="test",
            toolkit=Toolkit(),
            workspace_dir=tmp_path / "attempt_2",
            max_iters=73,
        )
        try:
            return first, second
        finally:
            await first_workspace.close()
            await second_workspace.close()

    first, second = asyncio.run(exercise())

    first.state.cur_iter = 72
    assert first.state.session_id != second.state.session_id
    assert second.state.cur_iter == 0
    assert first.react_config.max_iters == second.react_config.max_iters == 73


def test_native_agent_completes_react_tool_loop_without_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agentscope.credential import OpenAICredential
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.message import TextBlock, ToolCallBlock, UserMsg
    from agentscope.model import ChatResponse, OpenAIChatModel

    import agent.reference_runtime as runtime
    from agent_tools.context import EngineerToolContext

    phase = tmp_path / "phase"
    context = EngineerToolContext.from_task(
        task_text="synthetic parity smoke",
        engineer_phase_root=phase,
        require_skill_plan=False,
    )
    toolkit, _ = runtime.create_reference_toolkit(context)
    target = context.workspace_dir / "react_result.py"
    calls = 0

    def model_factory():
        model = OpenAIChatModel(
            credential=OpenAICredential(
                api_key="test-key",
                base_url="https://example.invalid/v1",
            ),
            model="MiniMax-M3",
            context_size=204_800,
            formatter=OpenAIChatFormatter(),
            stream=False,
        )
        return model, model.formatter

    async def fake_call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="write-1",
                        name="Write",
                        input=json.dumps(
                            {
                                "file_path": str(target),
                                "content": "print('react-ok')\n",
                            },
                        ),
                    ),
                ],
                is_last=True,
            )
        return ChatResponse(content=[TextBlock(text="completed")], is_last=True)

    monkeypatch.setattr(runtime, "make_reference_model", model_factory)
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)
    report_path = tmp_path / "skill_usage_report.json"

    async def exercise():
        agent, workspace = await runtime.create_reference_agent(
            name="Data Cleaning Agent",
            system_prompt="Use tools and finish.",
            toolkit=toolkit,
            workspace_dir=context.workspace_dir,
            max_iters=5,
        )
        try:
            terminal = io.StringIO()
            response = await runtime.stream_reference_agent_reply(
                agent,
                UserMsg(name="user", content="write the file"),
                output=terminal,
                skill_usage_report_path=report_path,
                exposed_pipeline_skills=runtime.DATA_CLEANING_AGENT_PIPELINE_SKILLS,
                exposed_error_view_skills=runtime.DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS,
            )
            return agent, response, terminal.getvalue()
        finally:
            await workspace.close()

    agent, response, terminal = asyncio.run(exercise())

    assert calls == 2
    assert target.read_text(encoding="utf-8") == "print('react-ok')\n"
    assert response.content[0].text == "completed"
    assert "[Model] MiniMax-M3" in terminal
    assert "[Tool Call] Write" in terminal
    assert "[Tool Result] Write" in terminal
    assert "completed" in terminal
    assert "[Reply End] completed" in terminal
    assert agent.state.session_id
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "SUCCESS"
    assert report["loaded"] == {"pipeline": [], "error_view": []}
    assert report["not_loaded"]["pipeline"] == sorted(
        runtime.DATA_CLEANING_AGENT_PIPELINE_SKILLS
    )
    assert report["not_loaded"]["error_view"] == sorted(
        runtime.DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS
    )
    assert report["calls"] == []
    assert report["call_counts"]["pipeline"] == {
        name: 0 for name in sorted(runtime.DATA_CLEANING_AGENT_PIPELINE_SKILLS)
    }
    assert report["call_counts"]["error_view"] == {
        name: 0 for name in sorted(runtime.DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS)
    }
    assert report["agent_usage"]["model_calls"] == 2
    assert report["agent_usage"]["react_iterations"] == agent.state.cur_iter
    assert report["agent_usage"]["duration_seconds"] >= 0
    assert any(
        getattr(block, "name", "") == "Write"
        for message in agent.state.context
        for block in message.get_content_blocks()
    )


def test_stream_audits_native_error_view_skill_calls(tmp_path: Path, monkeypatch) -> None:
    from agentscope.credential import OpenAICredential
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.message import TextBlock, ToolCallBlock, UserMsg
    from agentscope.model import ChatResponse, OpenAIChatModel

    import agent.reference_runtime as runtime
    from agent_tools.context import EngineerToolContext

    context = EngineerToolContext.from_task(
        task_text="audit native Skill viewer",
        engineer_phase_root=tmp_path / "phase",
        require_skill_plan=False,
    )
    toolkit, _ = runtime.create_reference_toolkit(
        context,
        include_pipeline_skills=True,
        include_error_view_skills=True,
    )
    calls = 0

    def model_factory():
        model = OpenAIChatModel(
            credential=OpenAICredential(
                api_key="test-key",
                base_url="https://example.invalid/v1",
            ),
            model="MiniMax-M3",
            context_size=204_800,
            formatter=OpenAIChatFormatter(),
            stream=False,
        )
        return model, model.formatter

    async def fake_call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="skill-1",
                        name="Skill",
                        input=json.dumps({"skill": "correction_intra_table_errors"}),
                    ),
                ],
                is_last=True,
            )
        return ChatResponse(content=[TextBlock(text="completed")], is_last=True)

    monkeypatch.setattr(runtime, "make_reference_model", model_factory)
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)
    report_path = tmp_path / "skill_usage_report.json"

    async def exercise():
        agent, workspace = await runtime.create_reference_agent(
            name="Data Cleaning Agent",
            system_prompt="Read the relevant Skill and finish.",
            toolkit=toolkit,
            workspace_dir=context.workspace_dir,
            max_iters=5,
        )
        try:
            return await runtime.stream_reference_agent_reply(
                agent,
                UserMsg(name="user", content="inspect one error view"),
                output=io.StringIO(),
                skill_usage_report_path=report_path,
                exposed_pipeline_skills=runtime.DATA_CLEANING_AGENT_PIPELINE_SKILLS,
                exposed_error_view_skills=runtime.DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS,
            )
        finally:
            await workspace.close()

    asyncio.run(exercise())
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "SUCCESS"
    assert report["exposed"]["error_view"] == sorted(
        runtime.DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS
    )
    assert report["loaded"]["error_view"] == ["correction_intra_table_errors"]
    assert "correction_entity_alignment_errors" in report["not_loaded"]["error_view"]
    assert report["calls"] == [
        {
            "tool_call_id": "skill-1",
            "skill_name": "correction_intra_table_errors",
            "status": "SUCCESS",
            "started_at": report["calls"][0]["started_at"],
            "finished_at": report["calls"][0]["finished_at"],
        }
    ]
    assert report["call_counts"]["error_view"] == {
        name: int(name == "correction_intra_table_errors")
        for name in sorted(runtime.DATA_CLEANING_AGENT_ERROR_VIEW_SKILLS)
    }
    assert report["agent_usage"]["model_calls"] == 2
    assert report["agent_usage"]["react_iterations"] >= 1
    assert report["agent_usage"]["duration_seconds"] >= 0


def test_collect_tool_results_reads_agentscope2_blocks() -> None:
    from agentscope.message import AssistantMsg, TextBlock, ToolResultBlock, ToolResultState

    from lib.agent_runtime import collect_tool_results

    agent = SimpleNamespace(
        state=SimpleNamespace(
            context=[
                AssistantMsg(
                    name="Data Cleaning Agent",
                    content=[
                        ToolResultBlock(
                            id="tool-1",
                            name="ValidateResultPackage",
                            output=[TextBlock(text='{"status": "SUCCESS", "valid": true}')],
                            state=ToolResultState.SUCCESS,
                        ),
                    ],
                ),
            ],
        ),
    )

    observed = asyncio.run(collect_tool_results(agent))

    assert observed == {
        "ValidateResultPackage": [{"status": "SUCCESS", "valid": True}],
    }


def test_environment_locks_python311_and_agentscope2() -> None:
    import yaml

    root = Path(__file__).resolve().parent.parent
    environment = yaml.safe_load((root / "environment.yml").read_text(encoding="utf-8"))
    dependencies = environment["dependencies"]
    pip_dependencies = next(item["pip"] for item in dependencies if isinstance(item, dict))

    assert environment["name"] == "py3102"
    assert "python=3.11" in dependencies
    assert "ripgrep" in dependencies
    assert "agentscope==2.0.4.post1" in pip_dependencies
    assert not (root / "agent" / "bounded_memory.py").exists()
