from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import ChatResponse
from agentscope.token import TokenCounterBase

from agent.bounded_memory import BoundedInMemoryMemory
from agent.reference_runtime import (
    REFERENCE_COMPRESSION_PROMPT,
    REFERENCE_COMPRESSION_TEMPLATE,
    ReferenceCompressionSummary,
    create_reference_compression_config,
)


class _AlwaysOverThreshold(TokenCounterBase):
    async def count(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> int:
        return 10_000


class _CompressionModel:
    stream = False

    async def __call__(self, messages, **kwargs) -> ChatResponse:
        return ChatResponse(
            content=[],
            metadata={
                "task_overview": "修复当前数据包。",
                "current_state": "已完成chart分析。",
                "verified_rules_and_evidence": "R001已验证，证据位于step_0042。",
                "rejected_hypotheses_and_counterexamples": "H002被反例否定。",
                "unresolved_work": "medication amount仍待验证。",
                "artifacts_and_paths": "/tmp/workspace/rules.json",
                "next_steps": "继续medication分析并执行train replay。",
                "constraints_to_preserve": "不得读取隐藏答案。",
            },
        )


def test_character_limit_no_longer_triggers_destructive_compaction(tmp_path: Path) -> None:
    async def exercise() -> None:
        memory = BoundedInMemoryMemory(
            max_chars=10,
            context_window_tokens=100_000,
            compact_threshold=0.8,
            report_dir=tmp_path / "context",
        )
        messages = [
            Msg("user", "x" * 100, "user"),
            Msg("assistant", "y" * 100, "assistant"),
        ]
        await memory.add(messages)

        observed = await memory.get_memory()

        assert [message.id for message in observed] == [message.id for message in messages]
        assert memory.compaction_count == 0
        assert not (tmp_path / "context/latest_compaction_summary.md").exists()

    asyncio.run(exercise())


def test_reference_compression_config_uses_model_context_at_eighty_percent() -> None:
    config = create_reference_compression_config(SimpleNamespace(model_name="MiniMax-M3"))

    assert config.enable is True
    assert config.trigger_threshold == 163_840
    assert config.keep_recent == 14
    assert config.summary_schema is ReferenceCompressionSummary
    assert "已验证规则" in config.compression_prompt
    assert "不得在压缩时发明新规则" in config.compression_prompt


def test_compaction_archive_numbering_continues_after_memory_restart(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        report_dir = tmp_path / "context"
        first = BoundedInMemoryMemory(report_dir=report_dir)
        await first.update_compressed_summary("first")

        resumed = BoundedInMemoryMemory(report_dir=report_dir)
        assert resumed.compaction_count == 1
        await resumed.update_compressed_summary("second")

        assert (report_dir / "compaction_summary_0001.md").read_text() == "first\n"
        assert (report_dir / "compaction_summary_0002.md").read_text() == "second\n"

    asyncio.run(exercise())


def test_native_semantic_compression_keeps_verified_rules_and_recent_context(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        memory = BoundedInMemoryMemory(report_dir=tmp_path / "context")
        messages = [
            Msg("user", "原始任务", "user"),
            Msg("assistant", "早期分析", "assistant"),
            Msg("user", "工具结果", "user"),
            Msg("assistant", "最近进展", "assistant"),
        ]
        await memory.add(messages)
        config = ReActAgent.CompressionConfig(
            enable=True,
            agent_token_counter=_AlwaysOverThreshold(),
            trigger_threshold=1,
            keep_recent=1,
            compression_prompt=REFERENCE_COMPRESSION_PROMPT,
            summary_template=REFERENCE_COMPRESSION_TEMPLATE,
            summary_schema=ReferenceCompressionSummary,
        )
        agent = ReActAgent(
            name="Data Cleaning Agent",
            sys_prompt="只进行测试。",
            model=_CompressionModel(),
            formatter=OpenAIChatFormatter(),
            memory=memory,
            compression_config=config,
            print_hint_msg=False,
        )

        await agent._compress_memory_if_needed()
        active = await memory.get_memory(exclude_mark="compressed")

        assert "R001已验证" in str(active[0].content)
        assert active[-1].id == messages[-1].id
        assert memory.compaction_count == 1
        assert (tmp_path / "context/compaction_summary_0001.md").is_file()
        assert "H002被反例否定" in (
            tmp_path / "context/latest_compaction_summary.md"
        ).read_text(encoding="utf-8")

    asyncio.run(exercise())
