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
    """Keep the original task and compact old tool-heavy turns automatically."""

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
        self.model_name = str(model_name or os.environ.get("MODEL_NAME") or "").strip()
        self.context_window_tokens = context_window_tokens or _context_window_tokens(self.model_name)
        self.compact_threshold = _compact_threshold(compact_threshold)
        self.report_dir = Path(report_dir).expanduser().resolve() if report_dir else None
        self.workflow_hint = str(workflow_hint or "").strip()
        self.compaction_count = 0

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
            raise TypeError(f"The memories should be a list of Msg or a single Msg, but got {type(memories)}.")
        if any(not isinstance(message, Msg) for message in messages):
            raise TypeError("The memories should contain Msg instances only.")

        mark_list = _normalize_marks(marks)
        use_marked_storage = bool(mark_list) or all(_is_marked_memory_item(item) for item in getattr(self, "content", []))
        if use_marked_storage:
            self._normalize_storage_shape()
            existing_ids = [item[0].id for item in self.content if _is_marked_memory_item(item)]
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
        pre_stats = self._stats(messages)
        should_compact, reason = self._should_compact(pre_stats)
        if not should_compact:
            self._write_context_report(pre_stats, pre_stats, compacted=False, reason=reason)
            return messages

        result = self._compact_messages(messages, pre_stats, reason)
        self._replace_content(result)
        post_stats = self._stats(result)
        self._write_context_report(pre_stats, post_stats, compacted=True, reason=reason)
        return result

    def _should_compact(self, stats: dict[str, Any]) -> tuple[bool, str]:
        token_threshold = int(self.context_window_tokens * self.compact_threshold)
        if stats["tokens"] >= token_threshold:
            return True, "token_threshold"
        if stats["chars"] > self.max_chars:
            return True, "char_threshold"
        return False, "below_threshold"

    def _compact_messages(self, messages: list[Msg], pre_stats: dict[str, Any], reason: str) -> list[Msg]:
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
            for block in _content_blocks(message, "tool_result"):
                unmatched_tool_results.add(_tool_block_id(block))
            for block in _content_blocks(message, "tool_use"):
                unmatched_tool_results.discard(_tool_block_id(block))
            enough_messages = len(recent) >= self.keep_recent_messages
            enough_chars = recent_chars >= self.max_chars
            if (enough_messages or enough_chars) and not unmatched_tool_results:
                break
        recent.reverse()

        self.compaction_count += 1
        dropped_count = max(0, len(messages) - len(recent) - (1 if original_task else 0))
        summary = Msg("user", self._compaction_summary(pre_stats, dropped_count, reason), "user")
        result = [summary, *recent]
        if original_task is not None and all(message.id != original_task.id for message in recent):
            result.insert(0, original_task)
        return result

    def _compaction_summary(self, pre_stats: dict[str, Any], dropped_count: int, reason: str) -> str:
        base = (
            "<system-info>"
            f"早期工具对话已裁剪；上下文自动压缩已触发（reason={reason}, "
            f"pre_tokens≈{pre_stats['tokens']}, pre_messages={pre_stats['messages']}, "
            f"dropped_messages≈{dropped_count}）。原始用户任务仍然有效；不要重新开始。"
        )
        if self._is_reference_code_agent_memory():
            return base + self._reference_code_agent_resume_text() + "</system-info>"
        return base + self._legacy_extraction_resume_text() + "</system-info>"

    def _is_reference_code_agent_memory(self) -> bool:
        hint = self.workflow_hint.lower()
        if "reference" in hint and "code" in hint:
            return True
        if not self.report_dir:
            return False
        path_text = str(self.report_dir).lower()
        return "reference_code_agent" in path_text or "reference-code" in path_text

    def _reference_code_agent_resume_text(self) -> str:
        phase_root = self.report_dir.parent if self.report_dir else None
        paths = _reference_resume_paths(phase_root)
        lines = [
            "这是 ReferenceCodeAgent 运行；旧工具结果已从对话上下文移除，但真实状态保存在实验目录中。",
            "下一步先恢复工程状态，而不是重新扫描全部数据：",
        ]
        for label, path in paths.items():
            lines.append(f"- {label}: {path}")
        lines.extend(
            [
                "恢复顺序：先 Read context_summary.md，再读取 runtime_trace.jsonl 末尾和最新验证/比较报告；",
                "检查 result_package 下 cohort/ 与 features/ 是否和 train/reference 同构；",
                "如果 ValidateResultPackage 或 CompareArtifact 失败，继续修复最近失败的脚本/adapter/manifest，而不是重新写一套流程；",
                "不要调用 get_extraction_progress；不要重新读取整份大 CSV；只针对当前差异文件做小范围 InspectDataFile、Read 或脚本验证。",
            ]
        )
        return "".join(lines)

    @staticmethod
    def _legacy_extraction_resume_text() -> str:
        return (
            "旧工具结果已从对话上下文移除，但真实状态保存在当前 workspace、manifest、"
            "Skill 使用计划、extraction_task_execution.json 和产物文件中。"
            "下一步必须先调用 get_extraction_progress 恢复当前 next_task；"
            "不要重新读取整份 field_extraction_rules.json 或 extraction_task_plan.json，"
            "只在当前 task 需要时读取小范围文件。"
        )

    def _stats(self, messages: list[Msg]) -> dict[str, Any]:
        return {
            "messages": len(messages),
            "chars": _char_count(messages),
            "tokens": _message_tokens(messages),
            "context_window_tokens": self.context_window_tokens,
            "compact_threshold": self.compact_threshold,
            "threshold_tokens": int(self.context_window_tokens * self.compact_threshold),
            "max_chars": self.max_chars,
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
            latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with (self.report_dir / "context_events.jsonl").open("a", encoding="utf-8") as handle:
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

    def _replace_content(self, messages: list[Msg]) -> None:
        current = getattr(self, "content", [])
        if all(_is_marked_memory_item(item) for item in current):
            marks_by_id = {item[0].id: list(item[1]) for item in current}
            self.content = [(message, marks_by_id.get(message.id, [])) for message in messages]
            return
        self.content = list(messages)


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
    return sum(_text_tokens(str(getattr(message, "role", ""))) + _text_tokens(str(message.content)) + 4 for message in messages)


def _text_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _content_blocks(message: Msg, block_type: str) -> list[dict[str, Any]]:
    try:
        return [block for block in message.get_content_blocks(block_type) if isinstance(block, dict)]
    except Exception:
        return []


def _tool_block_id(block: dict[str, Any]) -> str:
    return str(block.get("id") or block.get("tool_use_id") or block.get("tool_call_id") or "")


def _is_marked_memory_item(item: Any) -> bool:
    return (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and isinstance(item[0], Msg)
        and isinstance(item[1], list)
    )


def _reference_resume_paths(phase_root: Path | None) -> dict[str, str]:
    if phase_root is None:
        return {}
    candidates = {
        "context_summary": phase_root / "context_summary.md",
        "runtime_trace": phase_root / "runtime_trace.jsonl",
        "manifest": phase_root / "manifest.json",
        "workspace": phase_root / "workspace",
        "artifacts": phase_root / "artifacts",
        "result_package_validation_report": phase_root / "workspace" / "result_package_validation_report.json",
    }
    result_package = _latest_result_package_dir(phase_root)
    if result_package is not None:
        candidates["latest_result_package"] = result_package
        candidates["latest_package_manifest"] = result_package / "package_manifest.json"
    return {key: str(value) for key, value in candidates.items()}


def _latest_result_package_dir(root: Path) -> Path | None:
    try:
        manifests = [
            path
            for path in root.rglob("package_manifest.json")
            if path.is_file() and "result_package" in path.parts
        ]
    except Exception:
        return None
    if not manifests:
        return None
    return max(manifests, key=lambda path: path.stat().st_mtime).parent
