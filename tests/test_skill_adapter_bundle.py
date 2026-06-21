from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from workflow.skill_adapter import (
    parse_variant_result,
    persist_validated_variants,
    restore_bundle_variants,
    validate_adapter_request,
    write_variant_execution_receipt,
)


def test_adapter_request_and_result_protocol() -> None:
    request = {
        "rule_ids": ["field_a"],
        "input_artifacts": [{"alias": "raw", "path": "/tmp/raw.csv"}],
        "parameters": {"batch_size": 4},
        "output_contract": {"format": "csv"},
    }
    assert validate_adapter_request(request) == request

    result = parse_variant_result(
        'log line\nVARIANT_RESULT_JSON={"status":"SUCCESS","artifacts":[],"field_mappings":[],"metrics":{},"issues":[]}\n'
    )
    assert result["status"] == "SUCCESS"

    with pytest.raises(ValueError, match="field_mappings"):
        parse_variant_result('VARIANT_RESULT_JSON={"status":"SUCCESS","artifacts":[]}')


def test_validated_variants_persist_and_restore_across_rounds(tmp_path: Path) -> None:
    base_source = tmp_path / "base_skill.py"
    base_source.write_text("VALUE = 1\n", encoding="utf-8")
    source_hash = hashlib.sha256(base_source.read_bytes()).hexdigest()
    variant_root = tmp_path / "round1" / "skill_variants"
    variant = variant_root / "entity_adapter"
    variant.mkdir(parents=True)
    (variant / "variant.py").write_text("print('adapter')\n", encoding="utf-8")
    (variant / "request.json").write_text(
        json.dumps({"rule_ids": [], "input_artifacts": [], "parameters": {}, "output_contract": {}}),
        encoding="utf-8",
    )
    (variant / "SKILL.md").write_text("# Entity adapter\n", encoding="utf-8")
    (variant / "variant.json").write_text(
        json.dumps(
            {
                "base_skill": "extract_text_features",
                "mode": "adapter",
                "status": "validated",
                "protocol_version": 1,
                "base_source_hashes": {str(base_source): source_hash},
            }
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "round1" / "artifact.csv"
    artifact.write_text("record_id,value\na,1\n", encoding="utf-8")
    write_variant_execution_receipt(
        variant,
        {
            "status": "SUCCESS",
            "artifacts": [{"path": str(artifact)}],
            "field_mappings": [],
            "metrics": {},
            "issues": [],
        },
        output_files=[artifact],
    )
    candidate = tmp_path / "experiment" / "candidates" / "round_0001"
    (candidate / "capabilities").mkdir(parents=True)

    persisted = persist_validated_variants(variant_root, candidate)
    assert persisted == ["entity_adapter"]
    assert (candidate / "capabilities" / "entity_adapter" / "variant.py").is_file()

    next_variant_root = tmp_path / "round2" / "skill_variants"
    restored = restore_bundle_variants(candidate, next_variant_root)
    assert restored == ["entity_adapter"]
    restored_meta = json.loads((next_variant_root / "entity_adapter" / "variant.json").read_text())
    assert restored_meta["status"] == "validated"
    assert restored_meta["restored_from_bundle"] is True

    base_source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        restore_bundle_variants(candidate, tmp_path / "round3" / "skill_variants")


def test_validated_variant_cannot_persist_without_matching_execution_receipt(tmp_path: Path) -> None:
    base_source = tmp_path / "base_skill.py"
    base_source.write_text("VALUE = 1\n", encoding="utf-8")
    variant = tmp_path / "variants" / "field_adapter"
    variant.mkdir(parents=True)
    (variant / "SKILL.md").write_text("# Adapter\n", encoding="utf-8")
    (variant / "variant.py").write_text("print('ok')\n", encoding="utf-8")
    (variant / "request.json").write_text(
        json.dumps({"rule_ids": [], "input_artifacts": [], "parameters": {}, "output_contract": {}}),
        encoding="utf-8",
    )
    (variant / "variant.json").write_text(
        json.dumps(
            {
                "base_skill": "extract_text_features",
                "mode": "adapter",
                "status": "validated",
                "protocol_version": 1,
                "base_source_hashes": {
                    str(base_source): hashlib.sha256(base_source.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate"
    (candidate / "capabilities").mkdir(parents=True)

    with pytest.raises(ValueError, match="execution receipt"):
        persist_validated_variants(variant.parent, candidate)

    artifact = tmp_path / "artifact.csv"
    artifact.write_text("id\na\n", encoding="utf-8")
    write_variant_execution_receipt(
        variant,
        {
            "status": "SUCCESS",
            "artifacts": [{"path": str(artifact)}],
            "field_mappings": [],
            "metrics": {},
            "issues": [],
        },
        output_files=[artifact],
    )
    (variant / "variant.py").write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="receipt does not match"):
        persist_validated_variants(variant.parent, candidate)
