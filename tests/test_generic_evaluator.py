from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from workflow.evaluator import create_embedding_semantic_scorer, evaluate_result_package
from workflow.workbook import export_gold_workbook


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_evaluates_multiple_artifacts_by_stable_field_mapping(tmp_path: Path) -> None:
    gold = tmp_path / "validation_gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                {"case_key": "a", "metrics": {"score": 2}, "tags": ["x", "y"], "note": "alpha text"},
                {"case_key": "b", "metrics": {"score": 3}, "tags": ["z"], "note": "beta text"},
            ]
        ),
        encoding="utf-8",
    )
    rules = tmp_path / "field_extraction_rules.json"
    _write_json(
        rules,
        {
            "record_grain": "case_key",
            "rules": [
                _rule("case", "case_key", "canonical"),
                _rule("score", "metrics.score", "numeric_exact"),
                _rule("tags", "tags[]", "collection_contains"),
                _rule("note", "note", "semantic_text"),
            ],
        },
    )
    main = tmp_path / "main.csv"
    tags = tmp_path / "tags.csv"
    pd.DataFrame([{"case_key": "a", "score": 2, "note": "alpha"}, {"case_key": "b", "score": 3, "note": "beta"}]).to_csv(main, index=False)
    pd.DataFrame(
        [
            {"case_key": "a", "tag": "x"},
            {"case_key": "a", "tag": "y"},
            {"case_key": "a", "tag": "extra"},
            {"case_key": "b", "tag": "z"},
        ]
    ).to_csv(tags, index=False)
    manifest = tmp_path / "result_manifest.json"
    _write_json(
        manifest,
        {
            "artifacts": [
                {"alias": "main", "path": str(main), "case_id_column": "case_key"},
                {"alias": "tags", "path": str(tags), "case_id_column": "case_key"},
            ]
        },
    )
    mapping = tmp_path / "target_field_mapping.json"
    _write_json(
        mapping,
        {
            "mappings": [
                {"target_field_id": "case", "artifact": "main", "source_column": "case_key"},
                {"target_field_id": "score", "artifact": "main", "source_column": "score"},
                {"target_field_id": "tags", "artifact": "tags", "source_column": "tag"},
                {"target_field_id": "note", "artifact": "main", "source_column": "note"},
            ]
        },
    )

    result = evaluate_result_package(
        gold_path=gold,
        rules_path=rules,
        result_manifest_path=manifest,
        target_mapping_path=mapping,
        output_dir=tmp_path / "evaluation",
        semantic_scorer=lambda _gold, _pred: 0.9,
    )

    report = json.loads(Path(result["evaluation_report"]).read_text(encoding="utf-8"))
    assert report["metrics"] == {
        "field_coverage": 1.0,
        "value_correctness": 1.0,
        "sample_coverage": 1.0,
        "source_explainability": 1.0,
        "composite_score": 1.0,
    }
    assert report["artifact_validation"]["valid"] is True
    assert report["detail_metrics"]["field_value_macro_score"] == 1.0
    assert report["detail_metrics"]["field_value_micro_score"] == 1.0
    assert report["detail_metrics"]["semantic_text_match_score"] == 1.0
    assert report["detail_metrics"]["per_case_scores"] == [
        {"case_id": "a", "passed": 4, "total": 4, "score": 1.0},
        {"case_id": "b", "passed": 4, "total": 4, "score": 1.0},
    ]


def test_public_feedback_never_contains_gold_or_prediction_values(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    pd.DataFrame([{"record_id": "a", "answer": "secret-gold-value"}]).to_csv(gold, index=False)
    rules = tmp_path / "rules.json"
    _write_json(rules, {"record_grain": "record_id", "rules": [_rule("answer", "answer", "canonical")]})
    prediction = tmp_path / "prediction.csv"
    pd.DataFrame([{"record_id": "a", "answer": "wrong-prediction-value"}]).to_csv(prediction, index=False)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"artifacts": [{"alias": "main", "path": str(prediction), "case_id_column": "record_id"}]})
    mapping = tmp_path / "mapping.json"
    _write_json(mapping, {"mappings": [{"target_field_id": "answer", "artifact": "main", "source_column": "answer"}]})

    result = evaluate_result_package(gold, rules, manifest, mapping, tmp_path / "evaluation")
    public_text = Path(result["public_feedback"]).read_text(encoding="utf-8")
    private_text = Path(result["private_report"]).read_text(encoding="utf-8")

    assert "secret-gold-value" not in public_text
    assert "wrong-prediction-value" not in public_text
    assert "secret-gold-value" in private_text
    assert "wrong-prediction-value" in private_text
    public = json.loads(public_text)
    assert public["field_feedback"][0]["error_types"] == {"canonical_mismatch": 1}
    assert public["field_feedback"][0]["source_files"] == ["source.csv"]


def test_validation_gold_fields_missing_from_training_rules_are_reported(tmp_path: Path) -> None:
    gold = tmp_path / "gold.json"
    _write_json(gold, {"record_id": "a", "known": 1, "validation_only": "new"})
    rules = tmp_path / "rules.json"
    _write_json(
        rules,
        {
            "record_grain": "record_id",
            "rules": [
                _rule("record", "record_id", "canonical"),
                _rule("known", "known", "numeric_exact"),
            ],
        },
    )
    prediction = tmp_path / "prediction.csv"
    pd.DataFrame([{"record_id": "a", "known": 1}]).to_csv(prediction, index=False)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"artifacts": [{"alias": "main", "path": str(prediction), "case_id_column": "record_id"}]})
    mapping = tmp_path / "mapping.json"
    _write_json(
        mapping,
        {
            "mappings": [
                {"target_field_id": "record", "artifact": "main", "source_column": "record_id"},
                {"target_field_id": "known", "artifact": "main", "source_column": "known"},
            ]
        },
    )

    result = evaluate_result_package(gold, rules, manifest, mapping, tmp_path / "evaluation")
    public = json.loads(Path(result["public_feedback"]).read_text())
    missing = next(item for item in public["field_feedback"] if item["target_field_path"] == "validation_only")

    assert missing["error_types"] == {"missing_rule": 1}
    assert missing["target_field_id"].startswith("field_")
    assert public["metrics"]["field_coverage"] == 0.6667
    assert "new" not in json.dumps(missing)


def test_validation_dynamic_keys_reuse_training_field_ids(tmp_path: Path) -> None:
    gold_root = tmp_path / "validation_gold"
    for case_id, dynamic_key, value in (("case-a", "ALT_2180-07-01-22:03", 12.0),):
        case_dir = gold_root / case_id
        case_dir.mkdir(parents=True)
        _write_json(
            case_dir / "gold.json",
            {
                "case_id": case_id,
                "实验室检验": {
                    dynamic_key: {
                        "项目明细": [{"结果数值": value}],
                    }
                },
            },
        )
    rules = tmp_path / "rules.json"
    _write_json(
        rules,
        {
            "record_grain": "case_id",
            "rules": [
                _rule("case", "case_id", "canonical"),
                _rule(
                    "lab_value",
                    "实验室检验[].项目明细[].结果数值",
                    "collection_contains",
                ),
            ],
        },
    )
    prediction = tmp_path / "prediction.csv"
    pd.DataFrame(
        [{"case_id": "case-a", "lab_value": 12.0}]
    ).to_csv(prediction, index=False)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {"artifacts": [{"alias": "main", "path": str(prediction), "case_id_column": "case_id"}]},
    )
    mapping = tmp_path / "mapping.json"
    _write_json(
        mapping,
        {
            "mappings": [
                {"target_field_id": "case", "artifact": "main", "source_column": "case_id"},
                {"target_field_id": "lab_value", "artifact": "main", "source_column": "lab_value"},
            ]
        },
    )

    result = evaluate_result_package(gold_root, rules, manifest, mapping, tmp_path / "evaluation")
    report = json.loads(Path(result["evaluation_report"]).read_text(encoding="utf-8"))
    public = json.loads(Path(result["public_feedback"]).read_text(encoding="utf-8"))

    assert report["field_count"] == 2
    assert report["comparison_count"] == 2
    assert report["metrics"]["field_coverage"] == 1.0
    assert report["metrics"]["value_correctness"] == 1.0
    assert public["field_feedback"] == []


def test_embedding_semantic_scorer_uses_cosine_similarity() -> None:
    class FakeEmbeddings:
        def create(self, **_kwargs):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0]), SimpleNamespace(embedding=[1.0, 0.0])]
            )

    scorer = create_embedding_semantic_scorer(
        client=SimpleNamespace(embeddings=FakeEmbeddings()),
        model="fake-model",
    )

    assert scorer is not None
    assert scorer("gold", "prediction") == 1.0


def test_source_explainability_requires_real_source_file_and_column(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    pd.DataFrame([{"record_id": "a", "answer": 1}]).to_csv(gold, index=False)
    rules = tmp_path / "rules.json"
    bad_rule = _rule("answer", "answer", "numeric_exact")
    bad_rule["source_files"] = ["missing.csv"]
    _write_json(rules, {"record_grain": "record_id", "rules": [bad_rule]})
    prediction = tmp_path / "prediction.csv"
    pd.DataFrame([{"record_id": "a", "answer": 1}]).to_csv(prediction, index=False)
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([{"record_id": "a", "answer": 1}]).to_csv(raw / "source.csv", index=False)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"artifacts": [{"alias": "main", "path": str(prediction), "case_id_column": "record_id"}]})
    mapping = tmp_path / "mapping.json"
    _write_json(mapping, {"mappings": [{"target_field_id": "answer", "artifact": "main", "source_column": "answer"}]})

    result = evaluate_result_package(
        gold,
        rules,
        manifest,
        mapping,
        tmp_path / "evaluation",
        source_data_path=raw,
    )
    report = json.loads(Path(result["evaluation_report"]).read_text())

    assert report["metrics"]["source_explainability"] == 0.0


def test_evaluates_excel_artifacts_by_sheet_name(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"case_id": "a", "profile": {"name": "Alice"}, "tags": ["x", "y"]}),
        encoding="utf-8",
    )
    rules = tmp_path / "rules.json"
    _write_json(
        rules,
        {
            "record_grain": "case_id",
            "rules": [
                _rule("case", "case_id", "canonical"),
                _rule("name", "profile.name", "canonical"),
                _rule("tags", "tags[]", "collection_contains"),
            ],
        },
    )
    package = export_gold_workbook(
        output_dir=tmp_path / "result",
        cases=[{"case_id": "a"}],
        field_values=[
            {
                "case_id": "a",
                "target_field_id": "case",
                "target_field_path": "case_id",
                "value": "a",
            },
            {
                "case_id": "a",
                "target_field_id": "name",
                "target_field_path": "profile.name",
                "value": "Alice",
            },
            {
                "case_id": "a",
                "target_field_id": "tags",
                "target_field_path": "tags[]",
                "occurrence_id": "tag-1",
                "value": "x",
            },
            {
                "case_id": "a",
                "target_field_id": "tags",
                "target_field_path": "tags[]",
                "occurrence_id": "tag-2",
                "value": "y",
            },
        ],
    )

    result = evaluate_result_package(
        gold,
        rules,
        package["result_manifest"],
        package["target_field_mapping"],
        tmp_path / "evaluation",
    )
    report = json.loads(Path(result["evaluation_report"]).read_text(encoding="utf-8"))

    assert report["status"] == "SUCCESS"
    assert report["metrics"]["field_coverage"] == 1.0
    assert report["metrics"]["value_correctness"] == 1.0


def test_excel_artifact_rejects_missing_sheet_and_empty_mapped_column(tmp_path: Path) -> None:
    workbook = tmp_path / "result.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame([{"case_id": "a", "answer": None}]).to_excel(
            writer,
            sheet_name="actual",
            index=False,
        )
    gold = tmp_path / "gold.csv"
    pd.DataFrame([{"case_id": "a", "answer": "expected"}]).to_csv(gold, index=False)
    rules = tmp_path / "rules.json"
    _write_json(rules, {"record_grain": "case_id", "rules": [_rule("answer", "answer", "canonical")]})
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "artifacts": [
                {
                    "alias": "missing",
                    "path": str(workbook),
                    "sheet_name": "not-there",
                    "case_id_column": "case_id",
                },
                {
                    "alias": "actual",
                    "path": str(workbook),
                    "sheet_name": "actual",
                    "case_id_column": "case_id",
                },
            ]
        },
    )
    mapping = tmp_path / "mapping.json"
    _write_json(
        mapping,
        {
            "mappings": [
                {
                    "target_field_id": "answer",
                    "artifact": "actual",
                    "sheet_name": "actual",
                    "source_column": "answer",
                }
            ]
        },
    )

    result = evaluate_result_package(gold, rules, manifest, mapping, tmp_path / "evaluation")
    report = json.loads(Path(result["evaluation_report"]).read_text(encoding="utf-8"))

    assert report["status"] == "NEEDS_REPAIR"
    assert any("not-there" in issue and "sheet" in issue.lower() for issue in report["artifact_validation"]["issues"])
    assert any("no non-empty values" in issue for issue in report["artifact_validation"]["issues"])


def _rule(field_id: str, path: str, policy: str) -> dict:
    return {
        "target_field_id": field_id,
        "target_field_path": path,
        "evaluation_policy": policy,
        "source_files": ["source.csv"],
        "source_columns": [path.rsplit(".", 1)[-1].replace("[]", "")],
        "join_keys": ["case_key"],
        "derivation_logic": {"operation": "direct"},
        "status": "supported",
    }
