from __future__ import annotations

import json
from pathlib import Path

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from lib.agent_artifacts import begin_step, get_phase_context, record_step
from workflow.task_compiler import compile_extraction_task_plan


def _response(status: str, summary: str, artifacts=None, issues=()) -> ToolResponse:
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
            ),
        ],
    )


class ExplorerReportTools:
    """Lifecycle tools for publishing Explorer-owned reports."""

    def publish_analysis_report(self, file_path: str = "", report_json: str = "") -> ToolResponse:
        """Create or publish an Explorer JSON report as the canonical handoff."""
        try:
            context = get_phase_context()
            if context is None or context["phase_name"] != "explorer":
                raise ValueError("publish_analysis_report requires an active Explorer phase")
            phase_root = Path(context["phase_root"]).resolve()
            artifacts_root = Path(context["artifacts_dir"]).resolve()
            step_ctx = None
            if report_json:
                if file_path:
                    raise ValueError("Provide either file_path or report_json, not both")
                payload = json.loads(report_json)
                if not isinstance(payload, dict):
                    raise ValueError("Analysis report JSON must contain an object")
                step_ctx = begin_step("publish_analysis_report")
                if step_ctx is None:
                    target_dir = artifacts_root / "analysis_report"
                    target_dir.mkdir(parents=True, exist_ok=True)
                else:
                    target_dir = Path(step_ctx["step_dir"])
                path = target_dir / "data_analysis_report.json"
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                if not file_path:
                    raise ValueError("file_path or report_json is required")
                path = Path(file_path).expanduser().resolve()
            if path == Path(context["manifest_path"]).resolve():
                raise ValueError("Explorer must not publish or overwrite manifest.json")
            if path != artifacts_root and artifacts_root not in path.parents:
                raise ValueError("Analysis report must be inside the Explorer artifacts directory")
            if not path.is_file() or path.suffix.lower() != ".json":
                raise ValueError(f"Analysis report must be an existing JSON file: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Analysis report JSON must contain an object")
            if path == phase_root / "manifest.json":
                raise ValueError("manifest.json cannot be used as an analysis report")
            record = record_step(
                "publish_analysis_report",
                f"Published {path.name} as the canonical analysis report.",
                [path],
                handoffs={"analysis_report": path},
                _step_ctx=step_ctx,
            )
            handoff = ((record or {}).get("handoffs") or {}).get("analysis_report", {})
            return _response(
                "SUCCESS",
                "Canonical Explorer analysis report published.",
                {
                    "file_path": str(path),
                    "handoff": handoff,
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Analysis report publication failed.", issues=[str(exc)])

    def publish_field_rule_updates(self, rules_path: str, updates_json: str) -> ToolResponse:
        """Enrich locked field rules without changing their IDs, paths, or count."""
        try:
            context = get_phase_context()
            if context is None or context["phase_name"] != "explorer":
                raise ValueError("publish_field_rule_updates requires an active Explorer phase")
            source = Path(rules_path).expanduser().resolve()
            phase_root = Path(context["phase_root"]).resolve()
            if phase_root not in source.parents or not source.is_file():
                raise ValueError("Field rules must be an existing file inside the Explorer phase")
            payload = json.loads(source.read_text(encoding="utf-8"))
            updates_payload = json.loads(updates_json)
            updates = updates_payload.get("updates") if isinstance(updates_payload, dict) else None
            if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
                raise ValueError("field_extraction_rules.json must contain a rules list")
            if not isinstance(updates, list):
                raise ValueError("updates_json must contain an updates list")
            by_id = {
                str(rule.get("target_field_id")): rule
                for rule in payload["rules"]
                if isinstance(rule, dict) and rule.get("target_field_id")
            }
            allowed = {
                "source_files",
                "source_columns",
                "join_keys",
                "filters",
                "derivation_logic",
                "capability_type",
                "evaluation_policy",
                "evidence",
                "confidence",
                "status",
                "unsupported_reason",
            }
            for update in updates:
                if not isinstance(update, dict):
                    raise ValueError("Each field rule update must be an object")
                field_id = str(update.get("target_field_id") or "")
                if field_id not in by_id:
                    raise ValueError(f"Unknown locked target_field_id: {field_id}")
                forbidden = set(update) - allowed - {"target_field_id"}
                if forbidden:
                    raise ValueError(f"Field rule update contains forbidden keys: {sorted(forbidden)}")
                by_id[field_id].update({key: update[key] for key in allowed if key in update})
            if len(by_id) != len(payload["rules"]):
                raise ValueError("Locked field rule IDs must be unique")
            payload["field_count"] = len(payload["rules"])
            step_ctx = begin_step("publish_field_rule_updates")
            if step_ctx is None:
                target_dir = Path(context["artifacts_dir"]) / "field_rule_updates"
                target_dir.mkdir(parents=True, exist_ok=True)
            else:
                target_dir = Path(step_ctx["step_dir"])
            output = target_dir / "field_extraction_rules.json"
            task_plan_path = target_dir / "extraction_task_plan.json"
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            task_plan = compile_extraction_task_plan(
                payload["rules"],
                record_grain=str(payload.get("record_grain") or "case"),
            )
            task_plan["target_categories"] = list(payload.get("target_categories") or [])
            task_plan_path.write_text(json.dumps(task_plan, ensure_ascii=False, indent=2), encoding="utf-8")
            record = record_step(
                "publish_field_rule_updates",
                f"Enriched {len(updates)} locked field rules.",
                [output, task_plan_path],
                handoffs={
                    "field_extraction_rules": output,
                    "extraction_task_plan": task_plan_path,
                },
                metadata={
                    "field_count": len(payload["rules"]),
                    "task_count": task_plan["task_count"],
                    "update_count": len(updates),
                },
                _step_ctx=step_ctx,
            )
            return _response(
                "SUCCESS",
                "Locked field rules enriched.",
                {
                    "file_path": str(output),
                    "task_plan_path": str(task_plan_path),
                    "field_count": len(payload["rules"]),
                    "task_count": task_plan["task_count"],
                    "update_count": len(updates),
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Field rule update publication failed.", issues=[str(exc)])


def register_explorer_report_tools(toolkit: Toolkit) -> ExplorerReportTools:
    tools = ExplorerReportTools()
    toolkit.register_tool_function(tools.publish_analysis_report)
    toolkit.register_tool_function(tools.publish_field_rule_updates)
    return tools
