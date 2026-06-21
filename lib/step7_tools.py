from __future__ import annotations

from agentscope.tool import Toolkit

from step7_runtime import (
    build_label_series,
    create_icd_binary_label,
    create_notnull_binary_label,
    discover_diagnosis_fields,
    format_and_save_ml_dataset,
    get_column_distribution,
    load_task_context,
    propose_label_mapping,
)


def register_step7_tools(toolkit: Toolkit) -> Toolkit:
    for tool in (
        load_task_context,
        get_column_distribution,
        discover_diagnosis_fields,
        create_icd_binary_label,
        create_notnull_binary_label,
        propose_label_mapping,
        build_label_series,
        format_and_save_ml_dataset,
    ):
        toolkit.register_tool_function(tool)
    return toolkit
