from __future__ import annotations

import json
from pathlib import Path

from lib.gold_guidance import (
    build_gold_guidance,
    freeze_rules,
    load_gold_examples,
    load_learned_rules,
    promote_feedback_rules,
    summarize_gold_schema,
)
from lib.agent_artifacts import init_phase_session, resolve_phase_handoff


def _write_csv(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _sample_raw_root(tmp_path: Path) -> Path:
    raw_root = tmp_path / "mimic"
    _write_csv(
        raw_root / "structured" / "admissions.csv",
        "subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,race,hospital_expire_flag",
    )
    _write_csv(
        raw_root / "structured" / "patients.csv",
        "subject_id,gender,anchor_age,anchor_year,dod",
    )
    _write_csv(
        raw_root / "structured" / "diagnoses_icd.csv",
        "subject_id,hadm_id,seq_num,icd_code,icd_version",
    )
    _write_csv(
        raw_root / "structured" / "prescriptions.csv",
        "subject_id,hadm_id,drug,dose_val_rx,dose_unit_rx,starttime,stoptime,route",
    )
    _write_csv(
        raw_root / "structured" / "labevents.csv",
        "labevent_id,subject_id,hadm_id,itemid,charttime,value,valuenum,valueuom,ref_range_lower,ref_range_upper,flag",
    )
    _write_csv(
        raw_root / "notes" / "discharge_notes.csv",
        "note_id,subject_id,hadm_id,note_type,charttime,text",
    )
    return raw_root


def _sample_gold_records() -> list[dict]:
    return [
        {
            "病历": {
                "入院时间": "2153-03-19 06:32:00",
                "住院天数": 13.34,
                "人口学": {"年龄_入院时": 67, "性别": "M"},
            },
            "诊断列表": [
                {
                    "icd_version": 10,
                    "icd_code": "K7581",
                    "是否主诊断": True,
                    "source": {"table": "hosp/diagnoses_icd", "row_id": None},
                }
            ],
            "住院用药": [
                {
                    "药名": "Ursodiol",
                    "剂量单位": "mg",
                    "source": {"table": "hosp/prescriptions", "row_id": None},
                }
            ],
            "就诊文本": {
                "notes": [
                    {
                        "text": "nodular liver",
                        "charttime": "2153-03-19 06:32:00",
                        "type": "discharge",
                    }
                ],
                "num_notes": 1,
            },
            "实验室检验": {
                "ALT_2153-03-18": {
                    "检验套餐": "ALT",
                    "项目明细": [
                        {
                            "itemid": 50861,
                            "名称": "Alanine Aminotransferase",
                            "结果数值": 7.0,
                            "单位": "IU/L",
                        }
                    ],
                }
            },
            "生命体征": {"体温_C": None},
        }
    ]


def test_summarize_gold_schema_normalizes_nested_arrays_and_dynamic_keys() -> None:
    summary = summarize_gold_schema(_sample_gold_records())
    paths = {field["field_path"]: field for field in summary["fields"]}

    assert "病历.人口学.年龄_入院时" in paths
    assert "诊断列表[].icd_code" in paths
    assert "住院用药[].剂量单位" in paths
    assert "就诊文本.notes[].text" in paths
    assert "实验室检验.*.项目明细[].结果数值" in paths
    assert paths["实验室检验.*.项目明细[].结果数值"]["dynamic_segments"] == ["实验室检验.*"]


def test_build_gold_guidance_maps_gold_fields_to_mimic_sources(tmp_path: Path) -> None:
    raw_root = _sample_raw_root(tmp_path)
    gold_path = tmp_path / "gold" / "train.jsonl"
    output_dir = tmp_path / "out"
    rules_dir = tmp_path / "rules"
    _write_jsonl(gold_path, _sample_gold_records())

    report = build_gold_guidance(
        task_text=f"处理 {raw_root} 训练示例 {gold_path}",
        raw_data_root=str(raw_root),
        gold_examples_path=str(gold_path),
        output_dir=str(output_dir),
        learned_rules_dir=str(rules_dir),
    )

    provenance = {item["target_field_path"]: item for item in report["field_provenance"]}
    assert provenance["病历.入院时间"]["source_files"] == ["structured/admissions.csv"]
    assert "admittime" in provenance["病历.入院时间"]["source_columns"]
    assert provenance["诊断列表[].icd_code"]["source_files"] == ["structured/diagnoses_icd.csv"]
    assert provenance["住院用药[].剂量单位"]["source_files"] == ["structured/prescriptions.csv"]
    assert provenance["实验室检验.*.项目明细[].结果数值"]["source_files"] == ["structured/labevents.csv"]
    assert provenance["生命体征.体温_C"]["source_status"] == "source_schema_not_found"

    assert (output_dir / "gold_schema_summary.json").exists()
    assert (output_dir / "gold_field_provenance_report.json").exists()
    assert (output_dir / "gold_field_provenance_report.md").exists()
    assert (output_dir / "planner_extraction_brief.json").exists()
    assert load_learned_rules(str(rules_dir), include_statuses={"draft"})["rule_count"] >= 5


def test_build_gold_guidance_accepts_tabular_gold_examples(tmp_path: Path) -> None:
    raw_root = _sample_raw_root(tmp_path)
    gold_csv = tmp_path / "gold" / "train.csv"
    output_dir = tmp_path / "out"
    rules_dir = tmp_path / "rules"
    gold_csv.parent.mkdir(parents=True, exist_ok=True)
    gold_csv.write_text(
        "病历.入院时间,诊断列表[].icd_code,住院用药[].剂量单位\n"
        "2153-03-19 06:32:00,K7581,mg\n",
        encoding="utf-8",
    )

    records = load_gold_examples(gold_csv)
    assert records == [
        {
            "病历.入院时间": "2153-03-19 06:32:00",
            "诊断列表[].icd_code": "K7581",
            "住院用药[].剂量单位": "mg",
        }
    ]

    report = build_gold_guidance(
        task_text=f"处理 {raw_root} 表格金标准 {gold_csv}",
        raw_data_root=str(raw_root),
        gold_examples_path=str(gold_csv),
        output_dir=str(output_dir),
        learned_rules_dir=str(rules_dir),
    )
    provenance = {item["target_field_path"]: item for item in report["field_provenance"]}
    assert provenance["病历.入院时间"]["source_files"] == ["structured/admissions.csv"]
    assert provenance["诊断列表[].icd_code"]["source_files"] == ["structured/diagnoses_icd.csv"]
    assert provenance["住院用药[].剂量单位"]["source_files"] == ["structured/prescriptions.csv"]


def test_learned_rules_promote_and_freeze_feedback(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "target_field_path": "病历.出院时间",
                        "source_files": ["structured/admissions.csv"],
                        "source_columns": ["dischtime"],
                        "join_keys": ["subject_id", "hadm_id"],
                        "derivation_logic": "保留 admissions.dischtime。",
                        "validation_metric_delta": 0.2,
                        "examples_or_failure_cases": [{"hadm_id": 1}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    promoted = promote_feedback_rules(str(feedback_path), str(rules_dir), min_metric_delta=0.05)
    assert promoted["promoted_count"] == 1
    active = load_learned_rules(str(rules_dir), include_statuses={"active"})
    assert active["rule_count"] == 1
    assert active["rules"][0]["status"] == "active"

    frozen = freeze_rules(str(rules_dir))
    assert frozen["frozen_count"] == 1
    assert load_learned_rules(str(rules_dir), include_statuses={"frozen"})["rules"][0]["status"] == "frozen"


def test_analysis_report_handoff_prefers_canonical_data_analysis_report(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    phase = init_phase_session(run_root, "explorer")
    artifacts = Path(phase["artifacts_dir"])
    gold_report = artifacts / "step01" / "gold_field_provenance_report.json"
    analysis_report = artifacts / "step02" / "data_analysis_report.json"
    gold_report.parent.mkdir(parents=True, exist_ok=True)
    analysis_report.parent.mkdir(parents=True, exist_ok=True)
    gold_report.write_text('{"kind": "gold"}', encoding="utf-8")
    analysis_report.write_text('{"kind": "analysis"}', encoding="utf-8")

    assert resolve_phase_handoff(run_root, "explorer", "analysis_report") == str(analysis_report.resolve())


def test_load_learned_rules_accepts_canonical_rule_file(tmp_path: Path) -> None:
    rules_file = tmp_path / "learned_rules.json"
    rules_file.write_text(
        json.dumps({
            "rules": [{
                "target_field_path": "patient.age",
                "source_files": ["patients.csv"],
                "source_columns": ["anchor_age"],
                "join_keys": ["subject_id"],
                "derivation_logic": "直接读取",
                "examples_or_failure_cases": [],
                "validation_metric_delta": 0,
                "status": "draft",
            }],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = load_learned_rules(rules_file, include_statuses={"draft"})

    assert loaded["rule_count"] == 1
    assert loaded["rules"][0]["target_field_path"] == "patient.age"
