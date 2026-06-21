from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from skills._phase_adapter import begin_phase_skill, finalize_phase_skill
from step23_tools import run_step23_ocr_tool as _legacy_run_step23_ocr_tool


def run_step23_ocr_tool(
    input_path: str,
    output_root: str | None = None,
    ocr_workers: int = 8,
    cache_path: str | None = None,
):
    """Run legacy OCR with outputs isolated in the active phase step."""
    step_ctx, target_output = begin_phase_skill("run_ocr", output_root)
    response = _legacy_run_step23_ocr_tool(
        input_path=input_path,
        output_root=target_output,
        ocr_workers=ocr_workers,
        cache_path=cache_path,
    )
    return finalize_phase_skill("run_ocr", response, step_ctx)

SKILL = {
    "name": "run_ocr",
    "layer": "process",
    "description": "对图像/PDF 文件做 OCR，将视觉内容转为文本缓存，供后续文本抽取使用。",
    "when_to_use": "数据包含图像（CXR、扫描件等）且需要从中提取文字信息时。纯结构化数据可跳过。",
    "when_to_skip": "输入没有图片/PDF 文字，已有可复用 OCR 缓存，或目标字段不依赖视觉文字时跳过。",
    "capability_types": ["ocr", "document_text_recognition"],
    "inputs": ["input_path", "output_root?", "ocr_workers?", "cache_path?"],
    "outputs": ["ocr_cache_path", "processed_count"],
    "prerequisites": ["visual_text_source_available"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step23_tools.run_step23_ocr_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(run_step23_ocr_tool)
