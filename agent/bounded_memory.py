from __future__ import annotations

import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from agentscope.memory import InMemoryMemory
from agentscope.message import Msg


MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "claude": 200_000,
    "glm": 128_000,
    "minimax-m2.7": 204_800,
    "m2.7": 204_800,
    "minimax": 204_800,
    "abab": 128_000,
}
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000


class BoundedInMemoryMemory(InMemoryMemory):
    """Track context usage while AgentScope owns semantic compression."""

    def __init__(
        self,
        *,
        max_chars: int = 120_000,
        keep_recent_messages: int = 14,
        context_window_tokens: int | None = None,
        compact_threshold: float | None = None,
        model_name: str | None = None,
        report_dir: str | Path | None = None,
        workflow_hint: str | None = None,
    ) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.keep_recent_messages = keep_recent_messages
        self.model_name = str(
            model_name or os.environ.get("MODEL_NAME") or "",
        ).strip()
        self.context_window_tokens = context_window_tokens or _context_window_tokens(
            self.model_name,
        )
        self.compact_threshold = _compact_threshold(compact_threshold)
        self.report_dir = Path(report_dir).expanduser().resolve() if report_dir else None
        self.workflow_hint = str(workflow_hint or "").strip()
        self.compaction_count = self._existing_compaction_count()

    async def add(
        self,
        memories: list[Msg] | Msg | None,
        allow_duplicates: bool = False,
        marks: str | list[str] | None = None,
    ) -> None:
        if memories is None:
            return
        if isinstance(memories, Msg):
            messages = [memories]
        elif isinstance(memories, list):
            messages = memories
        else:
            raise TypeError(
                "The memories should be a list of Msg or a single Msg, "
                f"but got {type(memories)}.",
            )
        if any(not isinstance(message, Msg) for message in messages):
            raise TypeError("The memories should contain Msg instances only.")

        mark_list = _normalize_marks(marks)
        use_marked_storage = bool(mark_list) or all(
            _is_marked_memory_item(item)
            for item in getattr(self, "content", [])
        )
        if use_marked_storage:
            self._normalize_storage_shape()
            existing_ids = [
                item[0].id
                for item in self.content
                if _is_marked_memory_item(item)
            ]
            for message in messages:
                if allow_duplicates or message.id not in existing_ids:
                    self.content.append((message, list(mark_list)))
            return
        await super().add(messages, allow_duplicates=allow_duplicates)

    async def get_memory(
        self,
        mark: str | None = None,
        exclude_mark: str | None = None,
        prepend_summary: bool = True,
        **kwargs: Any,
    ) -> list[Msg]:
        self._normalize_storage_shape()
        parent_get_memory = super().get_memory
        if "mark" in inspect.signature(parent_get_memory).parameters:
            messages = await parent_get_memory(
                mark=mark,
                exclude_mark=exclude_mark,
                prepend_summary=prepend_summary,
                **kwargs,
            )
        else:
            messages = await parent_get_memory()
        messages = _coerce_messages(messages)
        stats = self._stats(messages)
        self._write_context_report(stats, stats, compacted=False, reason="below_threshold")
        return messages

    async def update_compressed_summary(self, summary: str) -> None:
        """Persist the model-generated continuation summary for auditing."""
        await super().update_compressed_summary(summary)
        self.compaction_count += 1
        messages = _coerce_messages(
            [
                item[0] if _is_marked_memory_item(item) else item
                for item in getattr(self, "content", [])
            ],
        )
        stats = self._stats(messages)
        self._write_compaction_summary(summary)
        self._write_context_report(
            stats,
            stats,
            compacted=True,
            reason="semantic_token_threshold",
        )

    def _write_compaction_summary(self, summary: str) -> None:
        if self.report_dir is None:
            return
        self.report_dir.mkdir(parents=True, exist_ok=True)
        archive = self.report_dir / f"compaction_summary_{self.compaction_count:04d}.md"
        archive.write_text(summary.rstrip() + "\n", encoding="utf-8")
        latest = self.report_dir / "latest_compaction_summary.md"
        temporary = latest.with_name(f".{latest.name}.tmp")
        temporary.write_text(summary.rstrip() + "\n", encoding="utf-8")
        os.replace(temporary, latest)

    def _existing_compaction_count(self) -> int:
        if self.report_dir is None or not self.report_dir.is_dir():
            return 0
        indexes: list[int] = []
        for path in self.report_dir.glob("compaction_summary_*.md"):
            try:
                indexes.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return max(indexes, default=0)

    def _stats(self, messages: list[Msg]) -> dict[str, Any]:
        return {
            "messages": len(messages),
            "chars": _char_count(messages),
            "tokens": _message_tokens(messages),
            "context_window_tokens": self.context_window_tokens,
            "compact_threshold": self.compact_threshold,
            "threshold_tokens": int(self.context_window_tokens * self.compact_threshold),
            "max_chars": self.max_chars,
            "max_chars_trigger_enabled": False,
            "trigger_mode": "agentscope_semantic_token_threshold",
            "model_name": self.model_name,
            "compaction_count": self.compaction_count,
        }

    def _write_context_report(
        self,
        pre_stats: dict[str, Any],
        post_stats: dict[str, Any],
        *,
        compacted: bool,
        reason: str,
    ) -> None:
        if self.report_dir is None:
            return
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "COMPACTED" if compacted else "OK",
            "reason": reason,
            "pre": pre_stats,
            "post": post_stats,
            "usage_ratio": (
                round(pre_stats["tokens"] / pre_stats["context_window_tokens"], 6)
                if pre_stats["context_window_tokens"]
                else 0
            ),
        }
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            latest = self.report_dir / "context_usage_report.json"
            latest.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with (self.report_dir / "context_events.jsonl").open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _normalize_storage_shape(self) -> None:
        """Repair accidental old-style Msg-only storage before AgentScope reads it."""
        content = getattr(self, "content", [])
        if not content:
            return
        if all(_is_marked_memory_item(item) for item in content):
            return
        if all(isinstance(item, Msg) for item in content):
            self.content = [(item, []) for item in content]


def _compact_threshold(value: float | None) -> float:
    if value is None:
        raw = os.environ.get("AGENT_AUTO_COMPACT_THRESHOLD", "")
        try:
            value = float(raw) if raw else 0.80
        except ValueError:
            value = 0.80
    return min(0.95, max(0.10, float(value)))


def _context_window_tokens(model_name: str) -> int:
    for env_key in ("AGENT_CONTEXT_WINDOW_TOKENS", "MODEL_CONTEXT_TOKENS"):
        raw = os.environ.get(env_key)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    normalized = str(model_name or "").lower()
    for marker, tokens in MODEL_CONTEXT_WINDOWS.items():
        if marker in normalized:
            return tokens
    return DEFAULT_CONTEXT_WINDOW_TOKENS


def _coerce_messages(items: list[Any]) -> list[Msg]:
    messages: list[Msg] = []
    for item in items:
        if _is_marked_memory_item(item):
            messages.append(item[0])
        elif isinstance(item, Msg):
            messages.append(item)
    return messages


def _normalize_marks(marks: str | list[str] | None) -> list[str]:
    if marks is None:
        return []
    if isinstance(marks, str):
        return [marks]
    return [str(mark) for mark in marks if str(mark)]


def _char_count(messages: list[Msg]) -> int:
    return sum(len(str(message.content)) for message in messages)


def _message_tokens(messages: list[Msg]) -> int:
    return sum(
        _text_tokens(str(getattr(message, "role", "")))
        + _text_tokens(str(message.content))
        + 4
        for message in messages
    )


def _text_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _is_marked_memory_item(item: Any) -> bool:
    return (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and isinstance(item[0], Msg)
        and isinstance(item[1], list)
    )
