from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from lib.agent_artifacts import begin_step, record_step


def response(status: str, summary: str, artifacts: dict[str, Any] | None = None, issues=()) -> ToolResponse:
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "status": status,
                        "summary": summary,
                        "artifacts": artifacts or {},
                        "issues": list(issues),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        ]
    )


def step_output(skill_name: str) -> tuple[dict[str, Any] | None, Path]:
    step = begin_step(skill_name)
    if step is not None:
        return step, Path(step["step_dir"])
    output = Path(__file__).resolve().parents[1] / "output" / "code_outputs" / skill_name
    output.mkdir(parents=True, exist_ok=True)
    return None, output


def record(
    skill_name: str,
    summary: str,
    paths: list[str | Path],
    *,
    step: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
    status: str = "SUCCESS",
    issues=(),
):
    return record_step(
        skill_name,
        summary,
        paths,
        metadata=metadata,
        status=status,
        issues=issues,
        _step_ctx=step,
    )
