from __future__ import annotations

import asyncio
from pathlib import Path

from agentscope.message import TextBlock, ToolResultBlock, ToolResultState, UserMsg
from agentscope.state import AgentState
from agentscope.workspace import LocalWorkspace

from agent.reference_runtime import (
    REFERENCE_COMPRESSION_PROMPT,
    ReferenceCompressionSummary,
    create_reference_context_config,
)


def test_reference_context_config_uses_m3_ninety_percent_trigger() -> None:
    config = create_reference_context_config("MiniMax-M3")

    assert config.trigger_ratio == 0.8999
    assert config.reserve_ratio == 0.1
    assert config.tool_result_limit == 4_000
    assert config.summary_schema == ReferenceCompressionSummary.model_json_schema()
    assert "已验证规则" in REFERENCE_COMPRESSION_PROMPT
    assert "不得在压缩时发明新规则" in REFERENCE_COMPRESSION_PROMPT


def test_local_workspace_offloads_context_and_large_tool_results(tmp_path: Path) -> None:
    async def exercise() -> None:
        workspace = LocalWorkspace(workdir=str(tmp_path / "workspace"))
        await workspace.initialize()
        try:
            context_path = await workspace.offload_context(
                "session-1",
                [UserMsg(name="user", content="保留完整历史")],
            )
            result_path = await workspace.offload_tool_result(
                "session-1",
                ToolResultBlock(
                    id="tool-1",
                    name="InspectDataFile",
                    output=[TextBlock(text="x" * 6_000)],
                    state=ToolResultState.SUCCESS,
                ),
            )
        finally:
            await workspace.close()

        context_file = tmp_path / "workspace" / context_path
        result_file = tmp_path / "workspace" / result_path
        assert context_file.is_file()
        assert "保留完整历史" in context_file.read_text(encoding="utf-8")
        assert result_file.is_file()
        assert result_file.stat().st_size == 6_000

    asyncio.run(exercise())


def test_each_attempt_starts_with_a_fresh_agent_state() -> None:
    first = AgentState()
    first.context.append(UserMsg(name="user", content="attempt one"))
    first.cur_iter = 23

    second = AgentState()

    assert second.session_id != first.session_id
    assert second.context == []
    assert second.cur_iter == 0
    assert second.summary == ""
