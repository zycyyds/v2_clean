from __future__ import annotations

import json
from pathlib import Path

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse, Toolkit

from lib.agent_artifacts import begin_step, record_step
from lib.gold_guidance import build_gold_guidance, build_gold_guidance_from_analysis
from skills import _common  # noqa: F401
from workflow.gold_analysis import analyze_training_examples

V2_DIR = Path(__file__).resolve().parents[2]


def analyze_gold_examples_tool(
    task_text: str,
    raw_data_root: str = "",
    gold_examples_path: str = "",
    prior_report_path: str = "",
    learned_rules_dir: str = "",
    analysis_findings: str = "",
) -> ToolResponse:
    """Generate standard gold-guidance reports from paths or atomic-tool findings."""
    step_ctx = begin_step("analyze_gold_examples")
    output_dir = Path(step_ctx["step_dir"]) if step_ctx is not None else V2_DIR / "output" / "code_outputs" / "gold_guidance"
    rules_dir = (
        output_dir / "learned_rules"
        if step_ctx is not None
        else Path(learned_rules_dir).expanduser().resolve()
        if learned_rules_dir
        else output_dir / "learned_rules"
    )
    try:
        dataset_agnostic = bool(
            gold_examples_path
            and Path(gold_examples_path).expanduser().is_dir()
            and not analysis_findings
        )
        if dataset_agnostic:
            artifacts = analyze_training_examples(gold_examples_path, output_dir)
            rules_payload = json.loads(
                Path(artifacts["field_extraction_rules"]).read_text(encoding="utf-8"),
            )
            field_count = int(rules_payload.get("field_count", 0))
            manifest_record = record_step(
                "analyze_gold_examples",
                "Generated dataset-agnostic field extraction rules and Explorer reports.",
                list(artifacts.values()),
                handoffs={
                    "field_extraction_rules": artifacts["field_extraction_rules"],
                    "extraction_task_plan": artifacts["extraction_task_plan"],
                    "analysis_report": artifacts["data_analysis_report"],
                    "explorer_run_record": artifacts["explorer_run_record"],
                },
                metadata={
                    "field_count": field_count,
                    "gold_examples_path": str(Path(gold_examples_path).expanduser().resolve()),
                    "analysis_mode": "deterministic_dataset_agnostic",
                },
                _step_ctx=step_ctx,
            )
            payload = {
                "status": "SUCCESS",
                "summary": "Generated deterministic dataset-agnostic extraction rules.",
                "artifacts": {
                    **artifacts,
                    "field_count": field_count,
                    "manifest_path": (manifest_record or {}).get("manifest_path", ""),
                    "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", str(output_dir)),
                    "handoffs": (manifest_record or {}).get("handoffs", {}),
                },
                "issues": [],
            }
            return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])
        if analysis_findings:
            findings = json.loads(analysis_findings)
            report = build_gold_guidance_from_analysis(
                task_text=task_text,
                analysis_findings=findings,
                output_dir=str(output_dir),
                learned_rules_dir=str(rules_dir),
            )
        else:
            report = build_gold_guidance(
                task_text=task_text,
                raw_data_root=raw_data_root,
                gold_examples_path=gold_examples_path,
                prior_report_path=prior_report_path,
                output_dir=str(output_dir),
                learned_rules_dir=str(rules_dir),
            )
        artifacts = report.get("artifacts", {})
        artifact_paths = [value for value in artifacts.values() if value]
        manifest_record = record_step(
            "analyze_gold_examples",
            "已生成 gold 字段来源分析和 planner 抽取指导报告。",
            artifact_paths,
            handoffs={
                "gold_field_provenance_report": artifacts.get("gold_field_provenance_report", ""),
                "planner_extraction_brief": artifacts.get("planner_extraction_brief", ""),
                "learned_rules": artifacts.get("learned_rules_path", ""),
            },
            metadata={
                "field_count": len(report.get("field_provenance", [])),
                "learned_rule_count": report.get("learned_rules", {}).get("rule_count", 0),
                "raw_data_root": report.get("raw_data_root", ""),
                "gold_examples_path": report.get("gold_examples_path", ""),
            },
            _step_ctx=step_ctx,
        )
        payload = {
            "status": "SUCCESS",
            "summary": "已生成 gold-guided 抽取规格。",
            "artifacts": {
                **artifacts,
                "field_count": len(report.get("field_provenance", [])),
                "learned_rule_count": report.get("learned_rules", {}).get("rule_count", 0),
                "manifest_path": (manifest_record or {}).get("manifest_path", ""),
                "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", str(output_dir)),
                "handoffs": (manifest_record or {}).get("handoffs", {}),
            },
            "issues": [],
        }
    except Exception as exc:
        manifest_record = record_step(
            "analyze_gold_examples",
            "gold 示例分析失败。",
            [],
            status="NEEDS_REPAIR",
            issues=[str(exc)],
            _step_ctx=step_ctx,
        )
        payload = {
            "status": "NEEDS_REPAIR",
            "summary": "gold 示例分析失败。",
            "artifacts": {
                "manifest_path": (manifest_record or {}).get("manifest_path", ""),
                "step_artifact_dir": (manifest_record or {}).get("step_artifact_dir", str(output_dir)),
            },
            "issues": [str(exc)],
        }
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))])


SKILL = {
    "name": "analyze_gold_examples",
    "layer": "explore",
    "description": "确定性展开训练 gold 的全部叶子字段并生成抽取规则、数据分析报告和执行记录；analysis_findings 保留旧流程兼容。",
    "when_to_use": "用户输入包含训练 raw/gold 示例对时调用；必须使用完整字段清单，不能只总结顶层类别。",
    "when_to_skip": "没有成对的训练 raw/gold 示例，或当前阶段只执行既有 extraction task 时跳过。",
    "capability_types": ["gold_schema_expand", "field_provenance", "extraction_rule_build"],
    "inputs": ["task_text", "analysis_findings?", "raw_data_root?", "gold_examples_path?", "prior_report_path?", "learned_rules_dir?"],
    "outputs": ["field_extraction_rules", "extraction_task_plan", "data_analysis_report", "explorer_run_record"],
    "prerequisites": ["train_examples_available"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": [
        "workflow.gold_analysis.analyze_training_examples",
        "lib.gold_guidance.build_gold_guidance",
        "lib.gold_guidance.build_gold_guidance_from_analysis",
    ],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(analyze_gold_examples_tool)
