from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step7_runtime import load_task_context

SKILL = {
    "name": "load_task_context",
    "layer": "util",
    "description": "加载已有的任务上下文（cleaned CSV、selection report、passed_patients、task_text），用于续跑。",
    "when_to_use": "从中间产物续跑时，先调用此 skill 恢复上下文，避免重复处理。",
    "when_to_skip": "新任务、没有旧 Step7 固定上下文，或当前实验使用 manifest/bundle 断点续跑时跳过。",
    "capability_types": ["legacy_context_restore"],
    "inputs": ["output_root?", "task_text?"],
    "outputs": ["cleaned_csv_path", "selection_report_path", "passed_patients_path", "task_text"],
    "prerequisites": ["legacy_step7_artifacts_available"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step7_runtime.load_task_context"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(load_task_context)
