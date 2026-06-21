from __future__ import annotations

from typing import Any

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg


class BoundedInMemoryMemory(InMemoryMemory):
    """Keep the original task and a bounded set of recent complete tool turns."""

    def __init__(self, *, max_chars: int = 120_000, keep_recent_messages: int = 14) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.keep_recent_messages = keep_recent_messages

    async def get_memory(
        self,
        mark: str | None = None,
        exclude_mark: str | None = None,
        prepend_summary: bool = True,
        **kwargs: Any,
    ) -> list[Msg]:
        messages = await super().get_memory(
            mark=mark,
            exclude_mark=exclude_mark,
            prepend_summary=prepend_summary,
            **kwargs,
        )
        if self._char_count(messages) <= self.max_chars:
            return messages

        original_task = next(
            (message for message in messages if getattr(message, "role", "") == "user"),
            messages[0] if messages else None,
        )
        recent: list[Msg] = []
        recent_chars = 0
        unmatched_tool_results: set[str] = set()
        for message in reversed(messages):
            if original_task is not None and message.id == original_task.id:
                continue
            recent.append(message)
            recent_chars += len(str(message.content))
            for block in message.get_content_blocks("tool_result"):
                unmatched_tool_results.add(str(block.get("id") or ""))
            for block in message.get_content_blocks("tool_use"):
                unmatched_tool_results.discard(str(block.get("id") or ""))
            enough_messages = len(recent) >= self.keep_recent_messages
            enough_chars = recent_chars >= self.max_chars
            if (enough_messages or enough_chars) and not unmatched_tool_results:
                break
        recent.reverse()

        summary = Msg(
            "user",
            (
                "<system-info>早期工具对话已裁剪。原始任务仍然有效；"
                "已完成状态以当前 workspace、manifest、Skill 使用计划和 extraction task "
                "执行记录为准。需要旧细节时使用 Read/Glob 重新检查文件。</system-info>"
            ),
            "user",
        )
        result = [summary, *recent]
        if original_task is not None and all(message.id != original_task.id for message in recent):
            result.insert(0, original_task)
        return result

    @staticmethod
    def _char_count(messages: list[Msg]) -> int:
        return sum(len(str(message.content)) for message in messages)
