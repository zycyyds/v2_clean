from agent_4.medical_column_selector.main import TaskDrivenColumnSelector


def _profile(modality, missing_rate=0.0, is_phi=False):
    return {
        "inferred_modality": modality,
        "missing_rate": missing_rate,
        "is_potential_phi": is_phi,
    }


def test_force_keep_structured_fields_generalizes_beyond_mimic():
    selector = TaskDrivenColumnSelector(config={"require_llm": False})
    schema_profile = {
        "columns": {
            "patient_id": _profile("Identifiers/Keys"),
            "visit_date": _profile("Time"),
            "一般资料-一般资料-性别": _profile("Demographics"),
            "一般资料-一般资料-年龄": _profile("Demographics"),
            "diagnosis_code": _profile("Diagnosis"),
            "note_text": _profile("ClinicalNotes"),
            "empty_lab_value": _profile("Labs", missing_rate=1.0),
        }
    }
    selection_result = {
        "must_have_columns": [],
        "useful_columns": [],
        "maybe_columns": [],
        "drop_columns": [
            "patient_id",
            "visit_date",
            "一般资料-一般资料-性别",
            "一般资料-一般资料-年龄",
            "diagnosis_code",
            "note_text",
            "empty_lab_value",
        ],
        "reason_by_column": {},
    }

    result = selector._force_keep_structured_columns(selection_result, schema_profile)

    assert result["must_have_columns"] == [
        "patient_id",
        "visit_date",
        "一般资料-一般资料-性别",
        "一般资料-一般资料-年龄",
    ]
    assert "patient_id" not in result["drop_columns"]
    assert "visit_date" not in result["drop_columns"]
    assert "一般资料-一般资料-性别" not in result["drop_columns"]
    assert "一般资料-一般资料-年龄" not in result["drop_columns"]
    assert "diagnosis_code" in result["drop_columns"]
    assert "note_text" in result["drop_columns"]
    assert "empty_lab_value" in result["drop_columns"]
    assert result["structured_keep_overrides"][0]["action"] == "forced_keep_structured_field"


def test_force_keep_structured_fields_does_not_override_strict_phi():
    selector = TaskDrivenColumnSelector(config={"require_llm": False, "privacy_mode": "strict"})
    schema_profile = {
        "columns": {
            "patient_name": _profile("Demographics", is_phi=True),
            "encounter_id": _profile("Identifiers/Keys", is_phi=False),
        }
    }
    selection_result = {
        "must_have_columns": [],
        "useful_columns": [],
        "maybe_columns": [],
        "drop_columns": ["patient_name", "encounter_id"],
        "reason_by_column": {},
    }

    result = selector._force_keep_structured_columns(selection_result, schema_profile)

    assert "patient_name" not in result["must_have_columns"]
    assert "patient_name" in result["drop_columns"]
    assert "encounter_id" in result["must_have_columns"]


def test_force_keep_structured_fields_does_not_keep_all_clinical_table_fields():
    selector = TaskDrivenColumnSelector(config={"require_llm": False})
    schema_profile = {
        "columns": {
            "一般资料-一般资料-性别": _profile("Demographics"),
            "症状学-本次发作-眩晕特点": _profile("Diagnosis"),
            "辅助检查-眼震视图等检查-HIT试验(详细)-左水平减低": _profile("Labs"),
            "第一次复诊-复诊信息-头晕/眩晕复发": _profile("Encounter"),
        }
    }
    selection_result = {
        "must_have_columns": [],
        "useful_columns": [],
        "maybe_columns": [],
        "drop_columns": list(schema_profile["columns"].keys()),
        "reason_by_column": {},
    }

    result = selector._force_keep_structured_columns(selection_result, schema_profile)

    assert result["must_have_columns"] == ["一般资料-一般资料-性别"]
    assert "症状学-本次发作-眩晕特点" in result["drop_columns"]
    assert "辅助检查-眼震视图等检查-HIT试验(详细)-左水平减低" in result["drop_columns"]
    assert "第一次复诊-复诊信息-头晕/眩晕复发" in result["drop_columns"]
