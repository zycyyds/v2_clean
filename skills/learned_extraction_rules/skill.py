from __future__ import annotations

import json
from pathlib import Path

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

from agent_artifacts import begin_step, record_step
from gold_guidance import freeze_rules, load_learned_rules, promote_feedback_rules
from skills import _common  # noqa: F401

V2_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RULES_DIR = V2_DIR / "program" / "output" / "learned_extraction_rules" / "rules"


def _statuses(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()} or {"active", "frozen"}


def load_learned_rules_tool(
    task_text: str = "",
    rules_dir: str = "",
    include_statuses: str = "active,frozen",
) -> ToolResponse:
    """Load learned extraction rules for the current task."""
    step_ctx = begin_step("load_learned_rules")
    resolved_rules_dir = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
    loaded = load_learned_rules(resolved_rules_dir, include_statuses=_statuses(include_statuses))
    manifest_record = record_step(
        "load_learned_rules",
        f"已加载 {loaded['rule_count']} 条 learned extraction rules。",
        [],
        metadata={
            "task_text": task_text,
            "rules_dir": loaded["rules_dir"],
            "rule_count": loaded["rule_count"],
            "include_statuses": loaded["include_statuses"],
        },
        _step_ctx=step_ctx,
    )
    payload = {
        "status": "SUCCESS",
        "summary": f"已加载 {loaded['rule_count']} 条 learned extraction rules。",
        "artifacts": {
            **loaded,
            "manifest_path": (manifest_record or {}).get("manifest_path", ""),
            "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", ""),
        },
        "issues": [],
    }
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


def promote_validation_feedback_tool(
    feedback_report_path: str,
    rules_dir: str = "",
    min_metric_delta: float = 0.0,
) -> ToolResponse:
    """Promote validation feedback into active learned extraction rules."""
    step_ctx = begin_step("promote_validation_feedback")
    resolved_rules_dir = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
    try:
        promoted = promote_feedback_rules(
            feedback_report_path=feedback_report_path,
            rules_dir=resolved_rules_dir,
            min_metric_delta=min_metric_delta,
        )
        artifact_paths = [promoted["rules_path"]] if promoted.get("rules_path") else []
        manifest_record = record_step(
            "promote_validation_feedback",
            f"已晋升 {promoted['promoted_count']} 条验证反馈规则。",
            artifact_paths,
            handoffs={"learned_rules": promoted.get("rules_path", "")},
            metadata={
                "feedback_report_path": promoted["feedback_report_path"],
                "promoted_count": promoted["promoted_count"],
                "rules_dir": promoted["rules_dir"],
            },
            _step_ctx=step_ctx,
        )
        payload = {
            "status": "SUCCESS",
            "summary": f"已晋升 {promoted['promoted_count']} 条验证反馈规则。",
            "artifacts": {
                **promoted,
                "manifest_path": (manifest_record or {}).get("manifest_path", ""),
                "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", ""),
                "handoffs": (manifest_record or {}).get("handoffs", {}),
            },
            "issues": [],
        }
    except Exception as exc:
        manifest_record = record_step(
            "promote_validation_feedback",
            "验证反馈规则晋升失败。",
            [],
            status="NEEDS_REPAIR",
            issues=[str(exc)],
            _step_ctx=step_ctx,
        )
        payload = {
            "status": "NEEDS_REPAIR",
            "summary": "验证反馈规则晋升失败。",
            "artifacts": {
                "manifest_path": (manifest_record or {}).get("manifest_path", ""),
                "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", ""),
            },
            "issues": [str(exc)],
        }
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


def freeze_learned_rules_tool(rules_dir: str = "") -> ToolResponse:
    """Freeze learned extraction rules before running a test split."""
    step_ctx = begin_step("freeze_learned_rules")
    resolved_rules_dir = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
    frozen = freeze_rules(resolved_rules_dir)
    manifest_record = record_step(
        "freeze_learned_rules",
        f"已冻结 {frozen['frozen_count']} 条 learned extraction rules。",
        [frozen["rules_path"]],
        handoffs={"learned_rules": frozen["rules_path"]},
        metadata={
            "frozen_count": frozen["frozen_count"],
            "rules_dir": frozen["rules_dir"],
        },
        _step_ctx=step_ctx,
    )
    payload = {
        "status": "SUCCESS",
        "summary": f"已冻结 {frozen['frozen_count']} 条 learned extraction rules。",
        "artifacts": {
            **frozen,
            "manifest_path": (manifest_record or {}).get("manifest_path", ""),
            "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", ""),
            "handoffs": (manifest_record or {}).get("handoffs", {}),
        },
        "issues": [],
    }
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


SKILL = {
    "name": "learned_extraction_rules",
    "layer": "util",
    "description": "固定的可学习抽取规则 skill：加载 active/frozen 规则、把有效验证反馈晋升为 active rules、在测试前冻结规则集。",
    "when_to_use": "FeatureEngineer 开始抽取前先加载；验证集评估产生有效补充规则后晋升；进入测试集前冻结，测试阶段只读 frozen rules。",
    "when_to_skip": "当前工作流由 experiment active/candidate bundle 管理规则，或不存在旧版全局 learned rules 时跳过。",
    "capability_types": ["legacy_rule_load", "legacy_rule_promote", "legacy_rule_freeze"],
    "inputs": ["task_text?", "rules_dir?", "include_statuses?", "feedback_report_path?", "min_metric_delta?"],
    "outputs": ["rules", "rule_count", "rules_path"],
    "prerequisites": ["legacy_rules_directory_available"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": False,
    "lib_entrypoints": [
        "gold_guidance.load_learned_rules",
        "gold_guidance.promote_feedback_rules",
        "gold_guidance.freeze_rules",
    ],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(load_learned_rules_tool)
    toolkit.register_tool_function(promote_validation_feedback_tool)
    toolkit.register_tool_function(freeze_learned_rules_tool)
