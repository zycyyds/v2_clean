from __future__ import annotations
from agentscope.tool import Toolkit
from skills import _common  # noqa: F401
from step6_tools import save_step6_report

SKILL = {
    "name": "save_report",
    "layer": "util",
    "description": "保存 Step6 一致性检查报告及其配套 sidecar 文件（Step6 专用，非通用报告保存器）。",
    "when_to_use": "仅当你已经生成了 Step6 的一致性检查报告内容，需要落盘为 Step6 标准报告产物时使用。通用 JSON/文本报告请用 run_python_code 直接写文件。",
    "when_to_skip": "报告不是 Step6 一致性报告，或当前工作流使用标准 manifest/publish 工具管理报告时跳过。",
    "capability_types": ["legacy_consistency_report_save"],
    "inputs": ["report_content", "output_root?"],
    "outputs": ["report_path"],
    "prerequisites": ["step6_consistency_report_ready"],
    "adapter_policy": "direct_only",
    "preserves_raw_values": True,
    "lib_entrypoints": ["step6_tools.save_step6_report"],
}

def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(save_step6_report)
