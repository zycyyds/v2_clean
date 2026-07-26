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
    load_agent_state,
    save_agent_state,
)


def test_reference_context_config_uses_semantic_eighty_percent_trigger() -> None:
    config = create_reference_context_config()

    assert config.trigger_ratio == 0.8
    assert config.reserve_ratio == 0.1
    assert config.tool_result_limit == 50_000
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
                    output=[TextBlock(text="x" * 60_000)],
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
        assert result_file.stat().st_size == 60_000

    asyncio.run(exercise())


def test_agent_state_round_trips_summary_context_and_host_state_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_agent" / "agent_state.json"
    state = AgentState(
        summary="已验证规则 R001",
        context=[UserMsg(name="user", content="继续当前实验")],
        middle_context={
            "validation_host": {
                "best_round": 3,
                "best_pipeline_sha256": "abc123",
                "rule_ledger": "workspace/rule_ledger.json",
            },
        },
    )

    save_agent_state(path, state)
    loaded = load_agent_state(path)

    assert loaded is not None
    assert loaded.session_id == state.session_id
    assert loaded.summary == "已验证规则 R001"
    assert loaded.context[0].content[0].text == "继续当前实验"
    assert loaded.middle_context["validation_host"]["best_round"] == 3
    assert not list(path.parent.glob(f".{path.name}.tmp"))
