from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step1_tools import validate_step1_output_tool

SKILL = {
    "name": "validate_output",
    "layer": "util",
    "description": "校验 Step1 重组输出目录的完整性（Step1 专用，非通用产物校验器）。",
    "when_to_use": "仅当你需要验证 Step1 风格的重组输出目录是否完整，再决定是否进入后续步骤时使用。通用产物验证请用 run_python_code 自行检查。",
    "when_to_skip": "产物不是 Step1 重组目录，或需要验证字段 mapping、表格主键及最终 handoff 时跳过。",
    "capability_types": ["legacy_step1_output_validation"],
    "inputs": ["output_root", "expected_min_files?"],
    "outputs": ["is_valid", "issues"],
    "prerequisites": ["step1_reorganized_output_available"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step1_tools.validate_step1_output_tool"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(validate_step1_output_tool)
