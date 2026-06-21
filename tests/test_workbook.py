from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from workflow.workbook import export_gold_workbook


def test_exports_dynamic_category_sheets_and_control_sheets(tmp_path: Path) -> None:
    result = export_gold_workbook(
        output_dir=tmp_path,
        cases=[{"case_id": "case-a"}, {"case_id": "case-b"}],
        field_values=[
            {
                "case_id": "case-a",
                "target_field_id": "profile_name",
                "target_field_path": "客户/资料.姓名",
                "value": "Alice",
            },
            {
                "case_id": "case-a",
                "target_field_id": "order_code",
                "target_field_path": "订单明细[].编码",
                "occurrence_id": "order-1",
                "value": "A-1",
            },
            {
                "case_id": "case-a",
                "target_field_id": "order_amount",
                "target_field_path": "订单明细[].金额",
                "occurrence_id": "order-1",
                "value": 10,
            },
            {
                "case_id": "case-b",
                "target_field_id": "order_code",
                "target_field_path": "订单明细[].编码",
                "occurrence_id": "order-1",
                "value": "B-1",
            },
        ],
        provenance=[{"target_field_id": "profile_name", "skill": "structured_lookup"}],
        unsupported=[{"target_field_id": "missing", "reason": "not present"}],
    )

    workbook = Path(result["workbook"])
    manifest = json.loads(Path(result["result_manifest"]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(result["target_field_mapping"]).read_text(encoding="utf-8"))

    assert workbook.name == "final_dataset.xlsx"
    assert pd.ExcelFile(workbook).sheet_names == [
        "_cases",
        "_provenance",
        "_unsupported",
        "客户_资料",
        "订单明细",
    ]
    assert all(
        {
            "alias",
            "path",
            "sheet_name",
            "case_id_column",
            "occurrence_id_column",
            "grain",
            "row_count",
        }
        <= set(item)
        for item in manifest["artifacts"]
    )
    order_artifact = next(item for item in manifest["artifacts"] if item["sheet_name"] == "订单明细")
    assert order_artifact["grain"] == "occurrence"
    assert order_artifact["occurrence_id_column"] == "occurrence_id"
    order_rows = pd.read_excel(workbook, sheet_name="订单明细").to_dict(orient="records")
    assert order_rows[0] == {
        "case_id": "case-a",
        "occurrence_id": "order-1",
        "编码": "A-1",
        "金额": 10.0,
    }
    assert order_rows[1]["case_id"] == "case-b"
    assert order_rows[1]["occurrence_id"] == "order-1"
    assert order_rows[1]["编码"] == "B-1"
    assert pd.isna(order_rows[1]["金额"])
    by_field = {item["target_field_id"]: item for item in mapping["mappings"]}
    assert by_field["order_amount"] == {
        "target_field_id": "order_amount",
        "artifact": order_artifact["alias"],
        "sheet_name": "订单明细",
        "source_column": "金额",
        "case_id_column": "case_id",
        "occurrence_id_column": "occurrence_id",
    }


def test_sanitizes_truncates_and_deduplicates_dynamic_sheet_names(tmp_path: Path) -> None:
    long_prefix = "这是一个超过Excel三十一字符限制的类别名称用于测试截断行为"
    result = export_gold_workbook(
        output_dir=tmp_path,
        cases=["case-a"],
        category_rows={
            "bad/name": [{"case_id": "case-a", "value": 1}],
            "bad:name": [{"case_id": "case-a", "value": 2}],
            long_prefix + "甲": [{"case_id": "case-a", "value": 3}],
            long_prefix + "乙": [{"case_id": "case-a", "value": 4}],
            "_cases": [{"case_id": "case-a", "value": 5}],
        },
    )

    sheets = pd.ExcelFile(result["workbook"]).sheet_names
    dynamic = sheets[3:]
    assert len(dynamic) == 5
    assert len({name.casefold() for name in sheets}) == len(sheets)
    assert all(len(name) <= 31 for name in sheets)
    assert all(not any(char in name for char in "[]:*?/\\") for name in dynamic)
    assert "bad_name" in dynamic
    assert "bad_name_2" in dynamic
    assert "_cases_2" in dynamic


def test_emits_declared_gold_categories_even_when_no_values_were_observed(tmp_path: Path) -> None:
    result = export_gold_workbook(
        output_dir=tmp_path,
        cases=["case-a"],
        field_values=[
            {
                "case_id": "case-a",
                "target_field_id": "profile_name",
                "target_field_path": "profile.name",
                "value": "Alice",
            }
        ],
        target_categories=["profile", "procedures", "free_text"],
    )

    sheets = pd.ExcelFile(result["workbook"]).sheet_names
    assert sheets == ["_cases", "_provenance", "_unsupported", "profile", "procedures", "free_text"]
    assert pd.read_excel(result["workbook"], sheet_name="procedures")["case_id"].tolist() == ["case-a"]


def test_exports_case_level_csv_alongside_review_workbook(tmp_path: Path) -> None:
    result = export_gold_workbook(
        output_dir=tmp_path,
        cases=["case-a", "case-b"],
        field_values=[
            {
                "case_id": "case-a",
                "target_field_id": "profile_age",
                "target_field_path": "profile.age",
                "value": 42,
            },
            {
                "case_id": "case-a",
                "target_field_id": "diagnosis_code",
                "target_field_path": "diagnoses[].code",
                "occurrence_id": "d1",
                "value": "A10",
            },
            {
                "case_id": "case-a",
                "target_field_id": "diagnosis_code",
                "target_field_path": "diagnoses[].code",
                "occurrence_id": "d2",
                "value": "B20",
            },
        ],
        target_categories=["profile", "diagnoses"],
    )

    case_csv = pd.read_csv(result["csv"])
    manifest = json.loads(Path(result["result_manifest"]).read_text(encoding="utf-8"))

    assert case_csv["case_id"].tolist() == ["case-a", "case-b"]
    assert case_csv["case_id"].is_unique
    assert case_csv.loc[0, "profile_age"] == 42
    assert json.loads(case_csv.loc[0, "diagnosis_code"]) == ["A10", "B20"]
    assert {Path(item["path"]).name for item in manifest["artifacts"]} >= {
        "final_dataset.csv",
        "final_dataset.xlsx",
    }
