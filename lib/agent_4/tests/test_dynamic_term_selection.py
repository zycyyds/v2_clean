import csv
import json
import os
from pathlib import Path

import pandas as pd

from agent_4.medical_column_selector.main import TaskDrivenColumnSelector
from agent_4.medical_column_selector.agents.task_planner import TaskPlannerAgent


def _write_csv(path: Path, columns: list[str]) -> None:
    rows = [
        {column: f"{column}_value_1" for column in columns},
        {column: f"{column}_value_2" for column in columns},
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_default_config_does_not_cap_final_columns():
    selector = TaskDrivenColumnSelector(config={"require_llm": False})

    assert selector.config["max_final_columns"] is None


def test_task_planner_falls_back_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    planner = TaskPlannerAgent(model_name="gpt-4.1-mini", allow_fallback=True)

    spec = planner.generate_task_spec("肝癌的诊断")

    assert planner.last_call_mode == "fallback"
    assert "Diagnosis" in spec["need_modalities"]


def test_dynamic_term_profile_selects_task_columns_and_protects_mimic_core(tmp_path, monkeypatch):
    columns = [
        "subject_id",
        "hadm_id",
        "gender",
        "anchor_age",
        "admittime",
        "diagnoses_icd_codes",
        "radiology_Diagnosis_hcc_status",
        "radiology_Diagnosis_hepatocellular_carcinoma_status",
        "discharge_LabResult_alt_value",
        "discharge_LabResult_ast_value",
        "radiology_Finding_mediastinal_contours_value",
        "discharge_Medication_atorvastatin_dose",
        "radiology_Finding_mastoid_air_cells_value",
        "unrelated_column",
    ]
    input_csv = tmp_path / "input.csv"
    _write_csv(input_csv, columns)

    term_profile = {
        "task_domain": "liver_cancer_diagnosis",
        "trigger_terms": ["肝癌", "HCC"],
        "column_terms": ["hcc", "hepatocellular"],
        "direct_diagnosis_terms": ["hcc", "hepatocellular carcinoma", "cholangiocarcinoma"],
        "supportive_evidence_terms": ["cirrhosis", "hepatitis", "liver lesion", "liver mass"],
        "background_disease_terms": ["cirrhosis", "hepatitis b", "hepatitis c"],
        "imaging_finding_terms": ["liver lesion", "hepatic lesion", "liver mass"],
        "lab_marker_terms": ["afp", "alt", "ast", "bilirubin", "albumin", "inr"],
        "treatment_context_terms": ["hepatectomy", "transplant"],
        "short_terms": ["alt", "ast"],
        "negative_terms": ["mediastinal", "atorvastatin", "mastoid"],
        "reason": "mocked liver cancer term profile",
    }

    class FakeTermPlanner:
        last_call_mode = "llm"
        last_call_error = None
        last_raw_response = json.dumps(term_profile)

        def generate_term_profile(self, task_text, task_spec=None):
            return term_profile

    monkeypatch.setattr(
        "agent_4.medical_column_selector.main.TermPlannerAgent",
        lambda model_name=None, allow_fallback=False: FakeTermPlanner(),
    )

    selector = TaskDrivenColumnSelector(
        config={
            "require_llm": False,
            "selector_input_mode": "heuristic_prefilter",
            "keep_free_text": True,
        }
    )
    result = selector.run(str(input_csv), "肝癌的诊断", str(tmp_path))

    with open(result["filtered_csv_path"], encoding="utf-8", newline="") as file:
        final_columns = next(csv.reader(file))

    assert "subject_id" in final_columns
    assert "hadm_id" in final_columns
    assert "diagnoses_icd_codes" in final_columns
    assert "radiology_Diagnosis_hcc_status" in final_columns
    assert "radiology_Diagnosis_hepatocellular_carcinoma_status" in final_columns
    assert "discharge_LabResult_alt_value" in final_columns
    assert "discharge_LabResult_ast_value" in final_columns
    assert "radiology_Finding_mediastinal_contours_value" not in final_columns
    assert "discharge_Medication_atorvastatin_dose" not in final_columns
    assert "radiology_Finding_mastoid_air_cells_value" not in final_columns

    report = json.loads(Path(result["selection_report_path"]).read_text(encoding="utf-8"))
    assert report["term_profile"]["task_domain"] == "liver_cancer_diagnosis"
    assert report["term_profile"]["trigger_terms"] == ["肝癌", "hcc"]
    assert "bilirubin" in report["term_profile"]["lab_marker_terms"]
    assert "liver lesion" in report["term_profile"]["imaging_finding_terms"]
    assert report["term_profile"]["negative_terms"] == ["mediastinal", "atorvastatin", "mastoid"]
    assert report["llm_input_profile"]["dataset_info"]["task_terms_used"]["short_terms"] == ["alt", "ast"]
    assert report["llm_input_profile"]["dataset_info"]["term_recall_stats"]["lab_marker_terms"]["count"] >= 2
    assert report["final_columns"] == final_columns


def test_layered_liver_cancer_terms_recall_supportive_evidence_and_context():
    selector = TaskDrivenColumnSelector(
        config={
            "require_llm": False,
            "selector_input_mode": "heuristic_prefilter",
        }
    )
    columns = [
        "subject_id",
        "hadm_id",
        "diagnoses_icd_codes",
        "radiology_Diagnosis_hepatocellular_carcinoma_status",
        "radiology_Diagnosis_cholangiocarcinoma_status",
        "discharge_LabResult_afp_value",
        "discharge_LabResult_alt_value",
        "discharge_LabResult_ast_value",
        "discharge_LabResult_total_bilirubin_value",
        "discharge_LabResult_albumin_value",
        "discharge_LabResult_inr_value",
        "discharge_Diagnosis_cirrhosis_status",
        "discharge_Diagnosis_hepatitis_b_status",
        "radiology_Diagnosis_liver_lesion_status",
        "radiology_Finding_liver_mass_value",
        "radiology_Finding_mediastinal_contours_value",
        "discharge_Note_altered_mental_status",
        "discharge_Medication_atorvastatin_dose",
    ]
    schema_profile = _schema_profile_for_columns(columns)
    term_profile = {
        "task_domain": "liver_cancer_diagnosis",
        "trigger_terms": ["肝癌"],
        "direct_diagnosis_terms": ["hepatocellular carcinoma", "cholangiocarcinoma"],
        "supportive_evidence_terms": ["cirrhosis", "hepatitis", "liver lesion", "liver mass"],
        "background_disease_terms": ["cirrhosis", "hepatitis b"],
        "imaging_finding_terms": ["liver lesion", "liver mass"],
        "lab_marker_terms": ["afp", "alt", "ast", "bilirubin", "albumin", "inr"],
        "treatment_context_terms": [],
        "short_terms": ["afp", "alt", "ast", "inr"],
        "negative_terms": ["mediastinal", "altered", "atorvastatin"],
    }

    llm_input_profile = selector._build_llm_input_profile(
        schema_profile=schema_profile,
        task_spec={"need_modalities": ["Diagnosis", "Labs"]},
        task_text="肝癌的诊断",
        term_profile=term_profile,
    )

    candidate_columns = set(llm_input_profile["columns"])
    expected = {
        "subject_id",
        "hadm_id",
        "diagnoses_icd_codes",
        "radiology_Diagnosis_hepatocellular_carcinoma_status",
        "radiology_Diagnosis_cholangiocarcinoma_status",
        "discharge_LabResult_afp_value",
        "discharge_LabResult_alt_value",
        "discharge_LabResult_ast_value",
        "discharge_LabResult_total_bilirubin_value",
        "discharge_LabResult_albumin_value",
        "discharge_LabResult_inr_value",
        "discharge_Diagnosis_cirrhosis_status",
        "discharge_Diagnosis_hepatitis_b_status",
        "radiology_Diagnosis_liver_lesion_status",
        "radiology_Finding_liver_mass_value",
    }
    assert expected <= candidate_columns
    assert "radiology_Finding_mediastinal_contours_value" not in candidate_columns
    assert "discharge_Note_altered_mental_status" not in candidate_columns
    assert "discharge_Medication_atorvastatin_dose" not in candidate_columns
    stats = llm_input_profile["dataset_info"]["term_recall_stats"]
    assert stats["direct_diagnosis_terms"]["count"] == 2
    assert stats["lab_marker_terms"]["count"] >= 6
    assert stats["imaging_finding_terms"]["count"] >= 2


def test_layered_terms_generalize_to_heart_failure_and_pneumonia_tasks():
    selector = TaskDrivenColumnSelector(
        config={
            "require_llm": False,
            "selector_input_mode": "heuristic_prefilter",
        }
    )
    columns = [
        "subject_id",
        "hadm_id",
        "diagnoses_icd_codes",
        "discharge_Diagnosis_heart_failure_status",
        "discharge_Diagnosis_chf_status",
        "discharge_LabResult_bnp_value",
        "discharge_LabResult_nt_probnp_value",
        "radiology_Finding_pulmonary_edema_value",
        "echo_Finding_ejection_fraction_value",
        "discharge_Medication_diuretic_status",
        "radiology_Diagnosis_pneumonia_status",
        "radiology_Finding_consolidation_value",
        "radiology_Finding_infiltrate_value",
        "discharge_LabResult_wbc_value",
        "discharge_LabResult_crp_value",
        "discharge_Symptom_fever_status",
        "discharge_Medication_antibiotic_status",
        "discharge_Note_after_visit_summary",
        "discharge_Note_altered_status",
    ]
    schema_profile = _schema_profile_for_columns(columns)

    heart_failure_profile = {
        "task_domain": "heart_failure_diagnosis",
        "direct_diagnosis_terms": ["heart failure", "chf"],
        "supportive_evidence_terms": ["ejection fraction", "pulmonary edema"],
        "lab_marker_terms": ["bnp", "nt probnp"],
        "imaging_finding_terms": ["pulmonary edema"],
        "treatment_context_terms": ["diuretic"],
        "short_terms": ["ef", "bnp"],
        "negative_terms": ["after", "altered"],
    }
    pneumonia_profile = {
        "task_domain": "pneumonia_diagnosis",
        "direct_diagnosis_terms": ["pneumonia"],
        "supportive_evidence_terms": ["consolidation", "infiltrate", "fever"],
        "lab_marker_terms": ["wbc", "crp"],
        "imaging_finding_terms": ["consolidation", "infiltrate"],
        "treatment_context_terms": ["antibiotic"],
        "short_terms": ["wbc", "crp"],
        "negative_terms": ["after", "altered"],
    }

    heart_candidates = set(selector._build_llm_input_profile(
        schema_profile=schema_profile,
        task_spec={"need_modalities": ["Diagnosis", "Labs"]},
        task_text="心衰诊断",
        term_profile=heart_failure_profile,
    )["columns"])
    pneumonia_candidates = set(selector._build_llm_input_profile(
        schema_profile=schema_profile,
        task_spec={"need_modalities": ["Diagnosis", "Labs"]},
        task_text="肺炎诊断",
        term_profile=pneumonia_profile,
    )["columns"])

    assert {
        "discharge_Diagnosis_heart_failure_status",
        "discharge_Diagnosis_chf_status",
        "discharge_LabResult_bnp_value",
        "discharge_LabResult_nt_probnp_value",
        "radiology_Finding_pulmonary_edema_value",
        "echo_Finding_ejection_fraction_value",
        "discharge_Medication_diuretic_status",
    } <= heart_candidates
    assert {
        "radiology_Diagnosis_pneumonia_status",
        "radiology_Finding_consolidation_value",
        "radiology_Finding_infiltrate_value",
        "discharge_LabResult_wbc_value",
        "discharge_LabResult_crp_value",
        "discharge_Symptom_fever_status",
        "discharge_Medication_antibiotic_status",
    } <= pneumonia_candidates
    assert "discharge_Note_after_visit_summary" not in heart_candidates
    assert "discharge_Note_altered_status" not in pneumonia_candidates


def test_wide_schema_prefilter_does_not_keep_all_diagnosis_and_time_columns():
    selector = TaskDrivenColumnSelector(
        config={
            "require_llm": False,
            "selector_input_mode": "heuristic_prefilter",
        }
    )
    columns = [
        "subject_id",
        "hadm_id",
        "diagnoses_icd_codes",
        "radiology_Diagnosis_hcc_status",
        "discharge_LabResult_alt_value",
    ]
    columns.extend(f"radiology_Diagnosis_unrelated_{idx}_status" for idx in range(600))
    columns.extend(f"discharge_Symptom_unrelated_{idx}_onset" for idx in range(600))
    schema_profile = {
        "dataset_info": {
            "total_rows": 2,
            "sampled_rows": 2,
            "total_columns": len(columns),
        },
        "columns": {
            column: {
                "missing_rate": 0.0,
                "is_constant": False,
                "is_potential_phi": False,
                "inferred_modality": "Diagnosis" if "Diagnosis" in column else "Time",
                "dtype_guess": "object",
            }
            for column in columns
        },
    }
    schema_profile["columns"]["subject_id"]["inferred_modality"] = "Identifiers/Keys"
    schema_profile["columns"]["hadm_id"]["inferred_modality"] = "Identifiers/Keys"
    schema_profile["columns"]["diagnoses_icd_codes"]["inferred_modality"] = "Diagnosis"

    term_profile = {
        "task_domain": "liver_cancer_diagnosis",
        "trigger_terms": ["肝癌"],
        "column_terms": ["hcc"],
        "short_terms": ["alt"],
        "negative_terms": [],
        "reason": "mocked",
    }

    llm_input_profile = selector._build_llm_input_profile(
        schema_profile=schema_profile,
        task_spec={"need_modalities": ["Diagnosis", "Time"]},
        task_text="肝癌的诊断",
        term_profile=term_profile,
    )

    candidate_columns = list(llm_input_profile["columns"])
    assert candidate_columns == [
        "subject_id",
        "hadm_id",
        "diagnoses_icd_codes",
        "radiology_Diagnosis_hcc_status",
        "discharge_LabResult_alt_value",
    ]
    assert llm_input_profile["dataset_info"]["wide_schema_mode"] is True


def _schema_profile_for_columns(columns: list[str]) -> dict:
    return {
        "dataset_info": {
            "total_rows": 2,
            "sampled_rows": 2,
            "total_columns": len(columns),
        },
        "columns": {
            column: {
                "missing_rate": 0.0,
                "is_constant": False,
                "is_potential_phi": False,
                "inferred_modality": "Diagnosis" if "Diagnosis" in column else "Labs",
                "dtype_guess": "object",
            }
            for column in columns
        },
    }
