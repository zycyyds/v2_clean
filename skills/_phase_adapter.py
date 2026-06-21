from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from lib.agent_artifacts import begin_step, record_step


def begin_phase_skill(
    skill_name: str,
    requested_output_root: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Allocate a phase step when running under the v2 artifact context."""
    step_ctx = begin_step(skill_name)
    if step_ctx is None:
        return None, requested_output_root
    return step_ctx, str(Path(step_ctx["step_dir"]).resolve())


def finalize_phase_skill(
    skill_name: str,
    response: ToolResponse,
    step_ctx: dict[str, Any] | None,
) -> ToolResponse:
    """Record a legacy ToolResponse in the active phase manifest."""
    if step_ctx is None:
        return response
    payload = _payload(response)
    status = str(payload.get("status") or "NEEDS_REPAIR")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
    paths = _existing_paths(artifacts)
    record = record_step(
        skill_name,
        str(payload.get("summary") or f"{skill_name} completed."),
        paths,
        metadata={"legacy_artifacts": artifacts},
        status=status,
        issues=[str(issue) for issue in issues],
        _step_ctx=step_ctx,
    )
    artifacts = dict(artifacts)
    artifacts.update(
        {
            "manifest_path": (record or {}).get("manifest_path", ""),
            "step_artifact_dir": (record or {}).get("step_artifact_dir", step_ctx["step_dir"]),
        },
    )
    payload["artifacts"] = artifacts
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, indent=2),
            ),
        ],
    )


def _payload(response: ToolResponse) -> dict[str, Any]:
    if not response.content:
        return {
            "status": "NEEDS_REPAIR",
            "summary": "Legacy skill returned an empty response.",
            "artifacts": {},
            "issues": ["empty ToolResponse"],
        }
    block = response.content[0]
    text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
    try:
        data = json.loads(text)
    except Exception:
        return {
            "status": "NEEDS_REPAIR",
            "summary": "Legacy skill returned a non-JSON response.",
            "artifacts": {"raw_response": str(text)},
            "issues": ["invalid ToolResponse payload"],
        }
    return data if isinstance(data, dict) else {
        "status": "NEEDS_REPAIR",
        "summary": "Legacy skill returned a non-object payload.",
        "artifacts": {},
        "issues": ["invalid ToolResponse payload"],
    }


def _existing_paths(value: Any) -> list[Path]:
    paths: list[Path] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            candidate = Path(item).expanduser()
            if candidate.is_absolute() and candidate.exists() and candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in paths:
                    paths.append(resolved)

    visit(value)
    return paths
