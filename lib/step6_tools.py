from __future__ import annotations

from agentscope.tool import Toolkit

import step6_runtime as runtime


def load_data_overview():
    return runtime.load_data_overview()


def analyze_all_patients_consistency():
    return runtime.analyze_all_patients_consistency()


def save_step6_report(report_content: str):
    return runtime.save_step6_report(report_content)


def register_step6_tools(toolkit: Toolkit) -> Toolkit:
    for tool in (
        load_data_overview,
        analyze_all_patients_consistency,
        save_step6_report,
    ):
        toolkit.register_tool_function(tool)
    return toolkit
