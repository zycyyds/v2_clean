from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pandas as pd


def _payload(response) -> dict:
    block = response.content[0]
    return json.loads(block.text if hasattr(block, "text") else block["text"])


def _context(tmp_path: Path, *read_roots: Path):
    from agent_tools.context import EngineerToolContext

    return EngineerToolContext.from_task(
        task_text="quality evidence tools",
        engineer_phase_root=tmp_path / "phase",
        additional_read_roots=read_roots,
        require_skill_plan=False,
        split_mode="correction",
    )


def test_profile_data_quality_reports_missing_duplicates_types_and_patterns(tmp_path: Path) -> None:
    from agent_tools.data_quality_tools import DataQualityEvidenceTools

    source = tmp_path / "source.csv.gz"
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 2, 4],
            "value": [1.0, 2.5, 2.5, None],
            "code": ["A-01", "B-02", "B-02", "bad"],
        }
    )
    frame.to_csv(source, index=False, compression="gzip")

    result = _payload(
        DataQualityEvidenceTools(_context(tmp_path, source)).ProfileDataQuality(
            file_path=str(source),
            columns_json='["stay_id", "value", "code"]',
            top_k=3,
        )
    )

    assert result["status"] == "SUCCESS"
    profile = result["artifacts"]
    assert profile["row_count"] == 4
    assert profile["duplicate_row_count"] == 1
    assert profile["columns"]["value"]["missing_count"] == 1
    assert profile["columns"]["value"]["logical_type"] == "numeric"
    assert profile["columns"]["code"]["top_values"][0] == {"value": "B-02", "count": 2}
    assert profile["columns"]["code"]["format_patterns"]


def test_profile_by_group_reports_group_size_missing_and_inconsistency(tmp_path: Path) -> None:
    from agent_tools.data_quality_tools import DataQualityEvidenceTools

    source = tmp_path / "groups.csv"
    pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 2],
            "sex": ["F", "M", "M", "M", "M"],
            "value": [1.0, None, 10.0, 12.0, 14.0],
        }
    ).to_csv(source, index=False)

    result = _payload(
        DataQualityEvidenceTools(_context(tmp_path, source)).ProfileByGroup(
            file_path=str(source),
            group_columns_json='["subject_id"]',
            value_columns_json='["sex", "value"]',
            max_groups=10,
        )
    )

    assert result["status"] == "SUCCESS"
    profile = result["artifacts"]
    assert profile["group_count"] == 2
    first = next(item for item in profile["groups"] if item["key"] == {"subject_id": 1})
    assert first["row_count"] == 2
    assert first["columns"]["value"]["missing_count"] == 1
    assert first["columns"]["sex"]["distinct_count"] == 2
    assert first["columns"]["sex"]["inconsistent"] is True


def test_profile_by_group_rejects_unbounded_high_cardinality(tmp_path: Path) -> None:
    from agent_tools.data_quality_tools import DataQualityEvidenceTools

    source = tmp_path / "groups.csv"
    pd.DataFrame({"id": range(30), "value": range(30)}).to_csv(source, index=False)

    result = _payload(
        DataQualityEvidenceTools(_context(tmp_path, source)).ProfileByGroup(
            file_path=str(source),
            group_columns_json='["id"]',
            value_columns_json='["value"]',
            max_groups=10,
        )
    )

    assert result["status"] == "NEEDS_REPAIR"
    assert "max_groups" in " ".join(result["issues"])


def test_data_constraints_cover_supported_declarative_types(tmp_path: Path) -> None:
    from agent_tools.data_quality_tools import DataQualityEvidenceTools

    source = tmp_path / "source.csv"
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 2, 4],
            "group": ["a", "a", "b", "b"],
            "label": ["x", "y", "z", "z"],
            "value": [1, 20, 3, None],
            "code": ["A01", "BAD", "C03", "D04"],
            "start": [1, 3, 5, 9],
            "end": [2, 2, 6, 9],
        }
    ).to_csv(source, index=False)
    pd.DataFrame({"id": [1, 2, 3]}).to_csv(reference, index=False)
    constraints = [
        {"id": "nn", "type": "not_null", "file_path": str(source), "columns": ["value"]},
        {"id": "uq", "type": "unique", "file_path": str(source), "columns": ["id"]},
        {"id": "av", "type": "allowed_values", "file_path": str(source), "column": "label", "values": ["x", "z"]},
        {"id": "rg", "type": "range", "file_path": str(source), "column": "value", "min": 0, "max": 10},
        {"id": "rx", "type": "regex", "file_path": str(source), "column": "code", "pattern": "^[A-Z][0-9]{2}$"},
        {"id": "fd", "type": "functional_dependency", "file_path": str(source), "determinant": ["group"], "dependent": ["label"]},
        {"id": "fk", "type": "foreign_key", "file_path": str(source), "columns": ["id"], "reference_file_path": str(reference), "reference_columns": ["id"]},
        {"id": "tm", "type": "temporal_order", "file_path": str(source), "start_column": "start", "end_column": "end"},
    ]

    result = _payload(
        DataQualityEvidenceTools(_context(tmp_path, source, reference)).TestDataConstraint(
            constraints_json=json.dumps(constraints),
            sample_limit=3,
        )
    )

    assert result["status"] == "SUCCESS"
    reports = {item["id"]: item for item in result["artifacts"]["constraints"]}
    assert set(reports) == {"nn", "uq", "av", "rg", "rx", "fd", "fk", "tm"}
    assert all(item["violation_count"] > 0 for item in reports.values())
    assert reports["fk"]["violation_count"] == 1
    assert reports["tm"]["violation_count"] == 1


def test_data_constraints_reject_arbitrary_expressions_and_denied_paths(tmp_path: Path) -> None:
    from agent_tools.data_quality_tools import DataQualityEvidenceTools

    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    source = allowed / "source.csv"
    hidden = denied / "reference_private.csv"
    pd.DataFrame({"id": [1]}).to_csv(source, index=False)
    pd.DataFrame({"id": [1]}).to_csv(hidden, index=False)
    context = _context(tmp_path, allowed)
    tools = DataQualityEvidenceTools(context)

    arbitrary = _payload(
        tools.TestDataConstraint(
            constraints_json=json.dumps(
                [{"id": "bad", "type": "python", "file_path": str(source), "expression": "__import__('os')"}]
            )
        )
    )
    hidden_result = _payload(
        tools.TestDataConstraint(
            constraints_json=json.dumps(
                [{"id": "hidden", "type": "not_null", "file_path": str(hidden), "columns": ["id"]}]
            )
        )
    )

    assert arbitrary["status"] == "NEEDS_REPAIR"
    assert "unsupported constraint type" in " ".join(arbitrary["issues"])
    assert hidden_result["status"] == "NEEDS_REPAIR"
    assert "outside authorized roots" in " ".join(hidden_result["issues"])


def test_audit_repair_delta_aligns_by_key_and_handles_row_order(tmp_path: Path) -> None:
    from agent_tools.data_quality_tools import DataQualityEvidenceTools

    dirty = tmp_path / "dirty.csv"
    repaired = tmp_path / "repaired.csv"
    pd.DataFrame({"id": [1, 2], "value": ["old", "keep"], "other": [1, 2]}).to_csv(dirty, index=False)
    pd.DataFrame({"id": [3, 1], "value": ["new", "fixed"], "other": [3, 1]}).to_csv(repaired, index=False)

    result = _payload(
        DataQualityEvidenceTools(_context(tmp_path, dirty, repaired)).AuditRepairDelta(
            dirty_path=str(dirty),
            repaired_path=str(repaired),
            key_columns_json='["id"]',
            sample_limit=5,
        )
    )

    assert result["status"] == "SUCCESS"
    audit = result["artifacts"]["files"][0]
    assert audit["inserted_rows"] == 1
    assert audit["deleted_rows"] == 1
    assert audit["changed_rows"] == 1
    assert audit["changed_cells"] == 1
    assert audit["changed_by_column"] == {"value": 1}

    pd.DataFrame({"id": [2, 1], "value": ["keep", "old"], "other": [2, 1]}).to_csv(repaired, index=False)
    reordered = _payload(
        DataQualityEvidenceTools(_context(tmp_path / "again", dirty, repaired)).AuditRepairDelta(
            dirty_path=str(dirty),
            repaired_path=str(repaired),
            key_columns_json='["id"]',
        )
    )
    assert reordered["artifacts"]["totals"]["changed_cells"] == 0
    assert reordered["artifacts"]["totals"]["inserted_rows"] == 0
    assert reordered["artifacts"]["totals"]["deleted_rows"] == 0


def test_audit_repair_delta_reports_duplicate_key_ambiguity_and_package_schema_changes(tmp_path: Path) -> None:
    from agent_tools.data_quality_tools import DataQualityEvidenceTools

    dirty = tmp_path / "dirty"
    repaired = tmp_path / "repaired"
    dirty.mkdir()
    repaired.mkdir()
    pd.DataFrame({"id": [1, 1], "value": ["a", "b"]}).to_csv(dirty / "table.csv", index=False)
    pd.DataFrame({"id": [1, 1], "value": ["a", "c"], "extra": [0, 0]}).to_csv(
        repaired / "table.csv", index=False
    )

    result = _payload(
        DataQualityEvidenceTools(_context(tmp_path, dirty, repaired)).AuditRepairDelta(
            dirty_path=str(dirty),
            repaired_path=str(repaired),
            key_map_json='{"table.csv": ["id"]}',
        )
    )

    assert result["status"] == "SUCCESS"
    audit = result["artifacts"]["files"][0]
    assert audit["duplicate_key_groups"] == 1
    assert audit["ambiguous_key_groups"] == 1
    assert audit["schema_added_columns"] == ["extra"]


def test_quality_tools_are_only_registered_when_explicitly_requested(tmp_path: Path) -> None:
    from agent.reference_runtime import create_reference_toolkit

    context = _context(tmp_path)

    async def schemas(include: bool) -> set[str]:
        toolkit, _ = create_reference_toolkit(
            context,
            include_pipeline_skills=False,
            include_error_view_skills=False,
            include_quality_tools=include,
        )
        return {
            item["function"]["name"]
            for item in await toolkit.get_tool_schemas()
        }

    disabled = asyncio.run(schemas(False))
    enabled = asyncio.run(schemas(True))
    names = {"ProfileDataQuality", "ProfileByGroup", "TestDataConstraint", "AuditRepairDelta"}

    assert names.isdisjoint(disabled)
    assert names <= enabled


def test_correction_explicitly_enables_quality_tools_while_shared_default_is_disabled() -> None:
    from agent.reference_runtime import create_reference_toolkit
    from workflow.reference_correction import ReferenceCorrectionWorkflow

    toolkit_signature = inspect.signature(create_reference_toolkit)
    correction_source = inspect.getsource(ReferenceCorrectionWorkflow._run_agent_session)

    assert toolkit_signature.parameters["include_quality_tools"].default is False
    assert "include_quality_tools=True" in correction_source


def test_quality_tools_do_not_mutate_input_files(tmp_path: Path) -> None:
    from agent_tools.data_quality_tools import DataQualityEvidenceTools

    source = tmp_path / "source.csv"
    pd.DataFrame({"id": [1, 2], "value": ["a", "b"]}).to_csv(source, index=False)
    before = source.read_bytes()
    tools = DataQualityEvidenceTools(_context(tmp_path, source))

    _payload(tools.ProfileDataQuality(file_path=str(source)))
    _payload(
        tools.TestDataConstraint(
            constraints_json=json.dumps(
                [{"id": "nn", "type": "not_null", "file_path": str(source), "columns": ["id"]}]
            )
        )
    )

    assert source.read_bytes() == before
