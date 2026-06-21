from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from skills._phase_adapter import begin_phase_skill, finalize_phase_skill
from step23_tools import (
    run_step23_directory_pipeline_tool as _legacy_run_step23_directory_pipeline_tool,
)


async def run_step23_directory_pipeline_tool(
    input_path: str,
    output_root: str | None = None,
    concurrency: int = 8,
    use_llm: bool = True,
):
    """Run the legacy text pipeline with phase-local output isolation."""
    step_ctx, target_output = begin_phase_skill("extract_text_features", output_root)
    response = await _legacy_run_step23_directory_pipeline_tool(
        input_path=input_path,
        output_root=target_output,
        concurrency=concurrency,
        use_llm=use_llm,
    )
    return finalize_phase_skill("extract_text_features", response, step_ctx)

SKILL = {
    "name": "extract_text_features",
    "layer": "process",
    "description": "从临床文本（放射报告、出院记录等）中用 LLM 抽取结构化字段，生成宽表 CSV。",
    "when_to_use": "数据包含非结构化文本列（report_text、notes、text 等），需要抽取实体/特征时。",
    "when_to_skip": "目标字段均可从结构化列获得，或输入没有可读文本时跳过；图片文字应先调用 run_ocr。",
    "capability_types": ["text_extract", "clinical_entity_extract"],
    "inputs": ["input_path", "output_root?", "concurrency?", "use_llm?"],
    "outputs": ["output_csv_path", "summary_path"],
    "prerequisites": ["text_source_available", "target_text_schema_defined"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step23_tools.run_step23_directory_pipeline_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(run_step23_directory_pipeline_tool)
