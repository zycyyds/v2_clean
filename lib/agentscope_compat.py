"""Small compatibility shims for legacy runtimes on AgentScope 1.x."""

from __future__ import annotations

import json
from typing import Any

try:
    from agentscope.token import CharTokenCounter as CharTokenCounter
except ImportError:
    from agentscope.token import TokenCounterBase

    class CharTokenCounter(TokenCounterBase):
        """Approximate the removed legacy character token counter."""

        async def count(self, messages: list[dict], **kwargs: Any) -> int:
            text = json.dumps(messages, ensure_ascii=False, default=str)
            return max(1, (len(text) + 3) // 4)
