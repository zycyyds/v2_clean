from __future__ import annotations

import json

from agentscope.tool import Toolkit

from skills._rule_tools import record, response, step_output
from workflow.rule_execution import join_source_records


def join_source_records_tool(sources_json: str, join_keys_json: str, how: str = "left"):
    """按规则声明的键连接多个结构化来源。"""
    step, output = step_output("join_source_records")
    try:
        sources = json.loads(sources_json)
        join_keys = json.loads(join_keys_json)
        if not isinstance(sources, list) or not isinstance(join_keys, list):
            raise ValueError("sources_json and join_keys_json must be JSON lists")
        path = join_source_records(sources, join_keys, output / "joined_records.csv", how=how)
        manifest = record("join_source_records", "Joined structured sources.", [path], step=step, metadata={"join_keys": join_keys, "how": how})
        return response("SUCCESS", "Structured sources joined.", {"output_path": path, "manifest_path": (manifest or {}).get("manifest_path", "")})
    except Exception as exc:
        return response("NEEDS_REPAIR", "Source join failed.", issues=[str(exc)])


SKILL = {
    "name": "join_source_records",
    "layer": "process",
    "description": "按规则中的一个或多个 join key 连接结构化来源，保留连接键并报告失败。",
    "when_to_use": "目标字段跨多个来源文件，且 Explorer 已明确 join_keys 和连接粒度时使用。",
    "when_to_skip": "只有单一来源，join key 不明确，或连接会造成未解释的一对多扩张时跳过。",
    "capability_types": ["structured_join", "record_alignment"],
    "inputs": ["sources_json", "join_keys_json", "how?"],
    "outputs": ["joined_records.csv"],
    "prerequisites": ["join_keys_known", "source_artifacts_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["workflow.rule_execution.join_source_records"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(join_source_records_tool)
