from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.agent import Agent, ContextConfig, ModelConfig, ReActConfig
from agentscope.message import TextBlock, ToolCallBlock
from agentscope.model import ChatResponse, FinishedReason, OpenAIChatModel
from agentscope.state import AgentState
from agentscope.event import ModelCallStartEvent, TextBlockDeltaEvent
from agentscope.tool import ExecResult

from agent.pi_runtime import (
    M3_CONTEXT_SIZE,
    PI_KEEP_RECENT_TOKENS,
    PI_RESERVE_TOKENS,
    AgentTurnResult,
    PiAgentConfig,
    PiCompactionAgent,
    PiAgentRuntime,
    PiCompressionPolicy,
    PiReasoningRetryMiddleware,
    build_pi_toolkit,
    create_pi_context_config,
    is_context_overflow,
)


def test_full_toolkit_bash_defaults_to_180_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, float | None] = {}

    async def fake_exec_shell(self, command, *, cwd=None, timeout=None):
        observed["timeout"] = timeout
        return ExecResult(exit_code=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(
        "lib.reliable_local_backend.ReliableLocalBackend.exec_shell",
        fake_exec_shell,
    )

    async def exercise() -> int:
        toolkit = build_pi_toolkit(tmp_path)
        schemas = await toolkit.get_tool_schemas()
        bash_schema = next(
            schema for schema in schemas if schema["function"]["name"] == "Bash"
        )
        bash = await toolkit.get_tool("Bash")
        chunks = [chunk async for chunk in bash.call(command="true")]
        assert chunks[-1].content[0].text == "ok"
        return bash_schema["function"]["parameters"]["properties"]["timeout"][
            "default"
        ]

    assert asyncio.run(exercise()) == 180_000
    assert observed["timeout"] == 180.0


def _test_model(*, stream: bool = False) -> OpenAIChatModel:
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key="test-key",
            base_url="https://example.invalid/v1",
        ),
        model="MiniMax-M3",
        context_size=M3_CONTEXT_SIZE,
        formatter=OpenAIChatFormatter(),
        stream=stream,
    )


def test_pi_agent_config_normalizes_paths(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    skill_root.mkdir()

    config = PiAgentConfig(
        workdir=tmp_path,
        max_iters=123,
        skill_dirs=(skill_root,),
    ).normalized()

    assert config.workdir == tmp_path.resolve()
    assert config.max_iters == 123
    assert config.skill_dirs == (skill_root.resolve(),)


def test_pi_agent_config_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_iters"):
        PiAgentConfig(workdir=tmp_path, max_iters=0).normalized()

    with pytest.raises(ValueError, match="skill directory"):
        PiAgentConfig(
            workdir=tmp_path,
            skill_dirs=(tmp_path / "missing",),
        ).normalized()


def test_pi_runtime_has_no_test_declaration_configuration(tmp_path: Path) -> None:
    config = PiAgentConfig(workdir=tmp_path)

    assert not hasattr(config, "tool_profile")
    assert not hasattr(config, "runner_spec_path")
    with pytest.raises(TypeError):
        PiAgentConfig(workdir=tmp_path, runner_spec_path=tmp_path / "runner_spec.json")


def test_pi_compression_policy_matches_m3_pi_defaults() -> None:
    policy = PiCompressionPolicy()

    assert M3_CONTEXT_SIZE == 1_000_000
    assert PI_RESERVE_TOKENS == 16_384
    assert PI_KEEP_RECENT_TOKENS == 20_000
    assert policy.trigger_tokens == 983_616
    assert not policy.should_compress(983_615)
    assert policy.should_compress(983_616)
    assert policy.should_compress(999_999)
    assert policy.keep_recent_ratio == pytest.approx(0.02)


def test_context_config_keeps_pi_recent_window_and_structured_summary() -> None:
    config = create_pi_context_config()

    # AgentScope requires trigger_ratio < 0.9. PiCompactionAgent owns the real
    # 983,616-token trigger and calls this config only after that trigger fires.
    assert config.trigger_ratio == 0.8999
    assert config.reserve_ratio == pytest.approx(0.02)
    assert config.tool_result_limit == 4_000
    properties = config.summary_schema["properties"]
    assert "verified_findings_and_evidence" in properties
    assert "failed_attempts_and_counterexamples" in properties
    assert "files_read" in properties
    assert "files_created_or_modified" in properties
    assert "key_commands" in properties
    assert "next_steps" in properties


def test_make_pi_model_forces_m3_streaming_and_parallel_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_runtime as runtime_module

    captured: dict = {}

    def fake_factory(agent_key, default_model, **kwargs):
        captured.update(
            {
                "agent_key": agent_key,
                "default_model": default_model,
                **kwargs,
            },
        )
        return object(), object()

    monkeypatch.setattr(runtime_module, "create_openai_model_and_formatter", fake_factory)

    assert runtime_module.make_pi_model() is not None
    assert captured == {
        "agent_key": "react_planner",
        "default_model": "MiniMax-M3",
        "stream": True,
        "context_size_override": 1_000_000,
        "parallel_tool_calls": True,
        "model_name_override": "MiniMax-M3",
    }


def test_context_overflow_detection_uses_status_and_error_code() -> None:
    class ErrorPayload:
        code = "context_length_exceeded"

    class ContextError(Exception):
        status_code = 400
        error = ErrorPayload()

    assert is_context_overflow(ContextError("request rejected"))
    assert is_context_overflow(RuntimeError("maximum context length exceeded"))
    assert not is_context_overflow(RuntimeError("ordinary bad request"))


def test_reasoning_middleware_compacts_then_retries_overflow_once() -> None:
    calls = 0
    compressed = 0

    class ContextError(Exception):
        status_code = 400

    class FakeAgent:
        async def force_compress_context(self) -> None:
            nonlocal compressed
            compressed += 1

    async def next_handler(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ContextError("context window exceeded")
        yield "recovered"

    async def exercise() -> list[str]:
        middleware = PiReasoningRetryMiddleware(sleep=lambda _delay: asyncio.sleep(0))
        return [
            item
            async for item in middleware.on_reasoning(
                FakeAgent(),
                {"tool_choice": None},
                next_handler,
            )
        ]

    assert asyncio.run(exercise()) == ["recovered"]
    assert calls == 2
    assert compressed == 1


def test_reasoning_middleware_does_not_retry_overflow_twice() -> None:
    calls = 0

    class ContextError(Exception):
        status_code = 413

    class FakeAgent:
        async def force_compress_context(self) -> None:
            return None

    async def next_handler(**_kwargs):
        nonlocal calls
        calls += 1
        raise ContextError("too many tokens")
        yield  # pragma: no cover

    async def exercise() -> None:
        middleware = PiReasoningRetryMiddleware(sleep=lambda _delay: asyncio.sleep(0))
        async for _ in middleware.on_reasoning(
            FakeAgent(),
            {},
            next_handler,
        ):
            pass

    with pytest.raises(ContextError):
        asyncio.run(exercise())
    assert calls == 2


def test_reasoning_middleware_retries_transient_errors_with_backoff() -> None:
    calls = 0
    delays: list[float] = []

    class FakeAgent:
        pass

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def next_handler(**_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("temporary disconnect")
        yield "done"

    async def exercise() -> list[str]:
        middleware = PiReasoningRetryMiddleware(
            transient_retries=2,
            base_delay=0.25,
            sleep=fake_sleep,
        )
        return [
            item
            async for item in middleware.on_reasoning(
                FakeAgent(),
                {},
                next_handler,
            )
        ]

    assert asyncio.run(exercise()) == ["done"]
    assert calls == 3
    assert delays == [0.25, 0.5]


def test_reasoning_middleware_never_replays_after_visible_stream_output() -> None:
    calls = 0

    class FakeAgent:
        async def force_compress_context(self) -> None:
            raise AssertionError("must not compact after visible output")

    async def next_handler(**_kwargs):
        nonlocal calls
        calls += 1
        yield TextBlockDeltaEvent(
            reply_id="reply",
            block_id="text",
            delta="partial",
        )
        raise ConnectionError("stream dropped")

    async def exercise() -> None:
        middleware = PiReasoningRetryMiddleware(sleep=lambda _delay: asyncio.sleep(0))
        async for _ in middleware.on_reasoning(FakeAgent(), {}, next_handler):
            pass

    with pytest.raises(ConnectionError):
        asyncio.run(exercise())
    assert calls == 1


def test_pi_compaction_agent_uses_fixed_outer_trigger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressed: list[ContextConfig] = []
    count = 983_615

    async def fake_count_tokens(*_args, **_kwargs) -> int:
        return count

    async def fake_base_compress(
        self,
        context_config=None,
        instructions=None,
    ) -> None:
        compressed.append(context_config)

    model = _test_model()
    monkeypatch.setattr(model, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(Agent, "compress_context", fake_base_compress)
    agent = PiCompactionAgent(
        name="test",
        system_prompt="test",
        model=model,
        toolkit=build_pi_toolkit(tmp_path),
        context_config=create_pi_context_config(),
        compression_policy=PiCompressionPolicy(),
    )

    asyncio.run(agent.compress_context())
    assert compressed == []

    count = 983_616
    asyncio.run(agent.compress_context())
    assert len(compressed) == 1
    assert compressed[0].reserve_ratio == pytest.approx(0.02)


def test_force_compaction_bypasses_local_pi_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressed: list[ContextConfig] = []

    async def fake_count_tokens(*_args, **_kwargs) -> int:
        return 100_000

    async def fake_base_compress(
        self,
        context_config=None,
        instructions=None,
    ) -> None:
        compressed.append(context_config)

    model = _test_model()
    monkeypatch.setattr(model, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(Agent, "compress_context", fake_base_compress)
    agent = PiCompactionAgent(
        name="test",
        system_prompt="test",
        model=model,
        toolkit=build_pi_toolkit(tmp_path),
        context_config=create_pi_context_config(),
        compression_policy=PiCompressionPolicy(),
    )

    asyncio.run(agent.force_compress_context())
    assert len(compressed) == 1
    assert compressed[0].trigger_ratio < 0.1
    assert compressed[0].reserve_ratio == pytest.approx(0.02)


def test_toolkit_contains_six_native_tools_without_skill_viewer(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        toolkit = build_pi_toolkit(tmp_path)
        schemas = await toolkit.get_tool_schemas()
        names = {item["function"]["name"] for item in schemas}

        assert names == {"Read", "Write", "Edit", "Glob", "Grep", "Bash"}
        expected = {
            "Read": (True, True),
            "Glob": (True, True),
            "Grep": (True, True),
            "Write": (False, False),
            "Edit": (False, False),
            "Bash": (False, False),
        }
        for name, flags in expected.items():
            tool = await toolkit.get_tool(name)
            assert tool is not None
            assert (tool.is_read_only, tool.is_concurrency_safe) == flags

    asyncio.run(exercise())


def test_registered_skill_is_exposed_on_demand(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "demo_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: demo_skill\n"
        "description: A test skill for runtime registration.\n"
        "---\n\n"
        "Use Read before editing.\n",
        encoding="utf-8",
    )

    async def exercise() -> None:
        toolkit = build_pi_toolkit(tmp_path, skill_dirs=(skill_root,))
        schemas = await toolkit.get_tool_schemas()
        names = {item["function"]["name"] for item in schemas}
        assert "Skill" in names

        viewer = await toolkit.get_tool("Skill")
        assert viewer is not None
        response = await viewer.call(
            skill="demo_skill",
            _agent_state=AgentState(),
        )
        assert "Use Read before editing" in str(response)

    asyncio.run(exercise())


def test_native_tools_are_batched_by_concurrency_safety(tmp_path: Path) -> None:
    async def exercise() -> list[tuple[str, list[str]]]:
        agent = PiCompactionAgent(
            name="test",
            system_prompt="test",
            model=_test_model(),
            toolkit=build_pi_toolkit(tmp_path),
            context_config=create_pi_context_config(),
            compression_policy=PiCompressionPolicy(),
        )
        agent._save_to_context(
            [
                ToolCallBlock(id="1", name="Read", input="{}"),
                ToolCallBlock(id="2", name="Grep", input="{}"),
                ToolCallBlock(id="3", name="Bash", input="{}"),
                ToolCallBlock(id="4", name="Glob", input="{}"),
                ToolCallBlock(id="5", name="Write", input="{}"),
                ToolCallBlock(id="6", name="Edit", input="{}"),
            ],
        )
        batches = await agent._batch_tool_calls()
        return [
            (batch.type, [tool.name for tool in batch.tool_calls])
            for batch in batches
        ]

    assert asyncio.run(exercise()) == [
        ("concurrent", ["Read", "Grep"]),
        ("sequential", ["Bash"]),
        ("concurrent", ["Glob"]),
        ("sequential", ["Write", "Edit"]),
    ]


def test_runtime_uses_one_agent_for_tool_loop_and_followup_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_runtime as runtime_module

    target = tmp_path / "generated.py"
    calls = 0

    async def fake_call_api(
        self,
        model_name,
        messages,
        tools=None,
        tool_choice=None,
        **kwargs,
    ):
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
                                "content": "print('pi-style')\n",
                            },
                        ),
                    ),
                ],
                is_last=True,
            )
        if calls == 2:
            return ChatResponse(content=[TextBlock(text="first complete")], is_last=True)
        return ChatResponse(content=[TextBlock(text="feedback handled")], is_last=True)

    monkeypatch.setattr(runtime_module, "make_pi_model", lambda: _test_model())
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)

    async def exercise() -> tuple[AgentTurnResult, AgentTurnResult, str, str]:
        terminal = io.StringIO()
        runtime = PiAgentRuntime(
            PiAgentConfig(workdir=tmp_path, max_iters=10),
            output=terminal,
        )
        await runtime.initialize()
        try:
            assert runtime.agent is not None
            session_id = runtime.agent.state.session_id
            first = await runtime.run_turn("create and verify a script")
            first_iter = runtime.agent.state.cur_iter
            second = await runtime.run_turn("validation feedback: improve nothing")
            assert runtime.agent.state.session_id == session_id
            assert first_iter >= 1
            assert runtime.agent.state.cur_iter == 0
            return first, second, terminal.getvalue(), session_id
        finally:
            await runtime.close()

    first, second, terminal, _ = asyncio.run(exercise())

    assert calls == 3
    assert target.read_text(encoding="utf-8") == "print('pi-style')\n"
    assert first.status == "SUCCESS"
    assert first.text == "first complete"
    assert second.text == "feedback handled"
    assert "[Tool Call] Write" in terminal
    assert "[Tool Result] Write" in terminal
    assert "[Reply End] completed" in terminal


def test_runtime_writes_non_blocking_audit_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_runtime as runtime_module

    async def fake_call_api(
        self,
        model_name,
        messages,
        tools=None,
        tool_choice=None,
        **kwargs,
    ):
        return ChatResponse(content=[TextBlock(text="done")], is_last=True)

    monkeypatch.setattr(runtime_module, "make_pi_model", lambda: _test_model())
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)

    async def exercise() -> Path:
        runtime = PiAgentRuntime(PiAgentConfig(workdir=tmp_path), output=io.StringIO())
        await runtime.initialize()
        try:
            result = await runtime.run_turn("finish")
            assert result.status == "SUCCESS"
            assert runtime.run_dir is not None
            return runtime.run_dir
        finally:
            await runtime.close()

    run_dir = asyncio.run(exercise())
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "transcript.log").is_file()
    assert (run_dir / "tool_audit.jsonl").is_file()
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "SUCCESS"
    assert report["model"]["context_size"] == 1_000_000
    assert report["compression"]["trigger_tokens"] == 983_616
    serialized = json.dumps(report)
    assert "test-key" not in serialized


def test_runtime_streams_incremental_model_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_runtime as runtime_module

    async def fake_call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        async def chunks():
            yield ChatResponse(
                content=[TextBlock(id="stream-text", text="hel")],
                is_last=False,
            )
            yield ChatResponse(
                content=[TextBlock(id="stream-text", text="lo")],
                is_last=False,
            )

        return chunks()

    monkeypatch.setattr(runtime_module, "make_pi_model", lambda: _test_model(stream=True))
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)

    async def exercise() -> tuple[AgentTurnResult, str]:
        terminal = io.StringIO()
        runtime = PiAgentRuntime(PiAgentConfig(workdir=tmp_path), output=terminal)
        await runtime.initialize()
        try:
            return await runtime.run_turn("stream"), terminal.getvalue()
        finally:
            await runtime.close()

    result, terminal = asyncio.run(exercise())
    assert result.text == "hello"
    assert "hello" in terminal


def test_invalid_tool_arguments_return_to_same_agent_for_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_runtime as runtime_module

    calls = 0

    async def fake_call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ChatResponse(
                content=[ToolCallBlock(id="bad", name="Write", input="{")],
                is_last=True,
            )
        return ChatResponse(content=[TextBlock(text="corrected")], is_last=True)

    monkeypatch.setattr(runtime_module, "make_pi_model", lambda: _test_model())
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)

    async def exercise() -> tuple[AgentTurnResult, str]:
        terminal = io.StringIO()
        runtime = PiAgentRuntime(PiAgentConfig(workdir=tmp_path), output=terminal)
        await runtime.initialize()
        try:
            return await runtime.run_turn("repair malformed call"), terminal.getvalue()
        finally:
            await runtime.close()

    result, terminal = asyncio.run(exercise())
    assert calls == 2
    assert result.status == "SUCCESS"
    assert result.text == "corrected"
    assert result.tool_calls == 1
    assert result.tool_errors == 1
    assert result.api_failovers == 0
    assert result.compactions == 0
    assert result.stream_event_counts["ToolResultEndEvent"] == 1
    assert len(result.text_block_ids) == 1
    assert "[Tool Result End] error" in terminal


def test_failed_bash_returns_to_same_agent_for_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_runtime as runtime_module

    calls = 0

    async def fake_call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="bash-failure",
                        name="Bash",
                        input=json.dumps({"command": "exit 9"}),
                    ),
                ],
                is_last=True,
            )
        return ChatResponse(content=[TextBlock(text="recovered")], is_last=True)

    monkeypatch.setattr(runtime_module, "make_pi_model", lambda: _test_model())
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)

    async def exercise() -> tuple[AgentTurnResult, str]:
        terminal = io.StringIO()
        runtime = PiAgentRuntime(PiAgentConfig(workdir=tmp_path), output=terminal)
        await runtime.initialize()
        try:
            return await runtime.run_turn("run and recover"), terminal.getvalue()
        finally:
            await runtime.close()

    result, terminal = asyncio.run(exercise())

    assert calls == 2
    assert result.status == "SUCCESS"
    assert result.text == "recovered"
    assert result.tool_calls == 1
    assert result.tool_errors == 1
    assert result.stream_event_counts["ToolResultEndEvent"] == 1
    assert "[Tool Result End] error" in terminal


def test_truncated_tool_call_is_not_executed_and_is_regenerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_runtime as runtime_module

    target = tmp_path / "must-not-exist.txt"
    calls = 0

    async def fake_call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="truncated",
                        name="Write",
                        input='{"file_path": "' + str(target),
                    ),
                ],
                is_last=True,
                finished_reason=FinishedReason.INTERRUPTED,
            )
        assert any("truncated" in str(message.content) for message in messages)
        return ChatResponse(content=[TextBlock(text="regenerated safely")], is_last=True)

    monkeypatch.setattr(runtime_module, "make_pi_model", lambda: _test_model())
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)

    async def exercise() -> AgentTurnResult:
        runtime = PiAgentRuntime(PiAgentConfig(workdir=tmp_path), output=io.StringIO())
        await runtime.initialize()
        try:
            return await runtime.run_turn("write a file")
        finally:
            await runtime.close()

    result = asyncio.run(exercise())
    assert calls == 2
    assert result.status == "SUCCESS"
    assert result.text == "regenerated safely"
    assert not target.exists()


def test_max_iters_preserves_files_without_old_gate_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.pi_runtime as runtime_module

    calls = 0

    async def fake_call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        nonlocal calls
        calls += 1
        return ChatResponse(
            content=[ToolCallBlock(id=f"missing-{calls}", name="MissingTool", input="{}")],
            is_last=True,
        )

    monkeypatch.setattr(runtime_module, "make_pi_model", lambda: _test_model())
    monkeypatch.setattr(OpenAIChatModel, "_call_api", fake_call_api)

    async def exercise() -> tuple[AgentTurnResult, Path]:
        runtime = PiAgentRuntime(
            PiAgentConfig(workdir=tmp_path, max_iters=2),
            output=io.StringIO(),
        )
        await runtime.initialize()
        try:
            result = await runtime.run_turn("keep trying")
            assert runtime.run_dir is not None
            return result, runtime.run_dir
        finally:
            await runtime.close()

    result, run_dir = asyncio.run(exercise())
    assert calls == 2
    assert result.status == "MAX_ITERS"
    assert result.finished_reason == "exceed_max_iters"
    names = {item.name for item in run_dir.iterdir()}
    assert names == {
        "run_manifest.json",
        "transcript.log",
        "tool_audit.jsonl",
        "run_report.json",
        "workspace",
    }
    assert not any(tmp_path.glob("**/candidate*"))
    assert not any(tmp_path.glob("**/attempt*"))


def test_pi_cli_returns_130_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import pi_cli

    def interrupt(coroutine) -> None:
        coroutine.close()
        raise KeyboardInterrupt()

    monkeypatch.setattr(pi_cli.asyncio, "run", interrupt)
    assert pi_cli.main(["--workdir", str(tmp_path), "task"]) == 130
