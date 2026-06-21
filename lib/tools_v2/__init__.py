# -*- coding: utf-8 -*-
"""tools 包 —— 所有工具函数的统一导出"""

from .explore_tools import (
    detect_input_type,
    read_csv_sample,
    get_csv_full_info,
    scan_directory,
    list_directory,
    read_text_sample,
    read_jsonl_sample,
)
from .ocr_tools import (
    ocr_image,
    preprocess_text,
    ocr_and_clean,
)
from .extract_tools import (
    extract_from_text,
    extract_from_text_sync,
    standardize_entities,
)
from .io_tools import (
    read_csv_all,
    save_rows_to_csv,
    save_result_json,
    load_excel_table,
    get_patient_row,
    build_output_dir,
)

__all__ = [
    # explore
    "detect_input_type",
    "read_csv_sample",
    "get_csv_full_info",
    "scan_directory",
    "list_directory",
    "read_text_sample",
    "read_jsonl_sample",
    # ocr
    "ocr_image",
    "preprocess_text",
    "ocr_and_clean",
    # extract
    "extract_from_text",
    "extract_from_text_sync",
    "standardize_entities",
    # io
    "read_csv_all",
    "save_rows_to_csv",
    "save_result_json",
    "load_excel_table",
    "get_patient_row",
    "build_output_dir",
]
