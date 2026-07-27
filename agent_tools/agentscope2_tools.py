from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import (
    Edit,
    Glob,
    Grep,
    Read,
    ToolBase,
    ToolChunk,
    ToolMiddlewareBase,
    ToolResponse,
    Write,
)

from .context import EngineerToolContext
from .data_quality_tools import DataQualityEvidenceTools
from .reference_variants import ReferenceVariantTools
from .restricted_backend import RestrictedLocalBackend
from .tools import EngineerTools


class ToolAuditMiddleware(ToolMiddlewareBase):
    def __init__(self, audit_path: str | Path) -> None:
        self.audit_path = Path(audit_path).expanduser().resolve()

    async def on_tool_call(self, tool, input_kwargs, next_handler):
        started = time.monotonic()
        status = "SUCCESS"
        try:
            async for chunk in next_handler(**input_kwargs):
                state = getattr(chunk, "state", None)
                if state in {
                    ToolResultState.ERROR,
                    ToolResultState.DENIED,
                    ToolResultState.INTERRUPTED,
                }:
                    status = str(state.value).upper()
                yield chunk
        except Exception:
            status = "ERROR"
            raise
        finally:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "tool": tool.name,
                "status": status,
                "duration_seconds": round(time.monotonic() - started, 6),
                "input_keys": sorted(input_kwargs),
            }
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class MethodTool(ToolBase):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        method: Callable[..., ToolResponse],
        read_only: bool = False,
        middlewares: list[ToolMiddlewareBase] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.method = method
        self.is_read_only = read_only
        self.is_concurrency_safe = read_only
        super().__init__(middlewares=middlewares)

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="Project-scoped tool permissions are enforced by EngineerToolContext.",
        )

    async def call(self, **kwargs: Any) -> ToolChunk:
        response = self.method(**kwargs)
        if not isinstance(response, ToolResponse):
            raise TypeError(f"{self.name} returned {type(response).__name__}, expected ToolResponse")
        text = "\n".join(
            str(block.get("text") if isinstance(block, dict) else getattr(block, "text", block))
            for block in response.content
        )
        state = ToolResultState.SUCCESS
        try:
            payload = json.loads(text)
            if str(payload.get("status") or "").upper() in {"NEEDS_REPAIR", "ERROR", "FAILED"}:
                state = ToolResultState.ERROR
        except Exception:
            pass
        return ToolChunk(content=[TextBlock(text=text)], state=state)


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def _text(description: str, default: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "string", "description": description}
    if default is not None:
        value["default"] = default
    return value


def _integer(description: str, default: int, minimum: int = 0, maximum: int = 10_000) -> dict[str, Any]:
    return {
        "type": "integer",
        "description": description,
        "default": default,
        "minimum": minimum,
        "maximum": maximum,
    }


def build_reference_tools(
    context: EngineerToolContext,
    *,
    include_quality_tools: bool = False,
) -> list[ToolBase]:
    audit = [ToolAuditMiddleware(context.workspace_dir / "tool_audit.jsonl")]
    backend = RestrictedLocalBackend(
        read_roots=context.read_roots,
        denied_read_roots=context.denied_read_roots,
        write_roots=[context.engineer_phase_root],
        cwd=context.workspace_dir,
    )
    implementation = EngineerTools(context)
    quality = DataQualityEvidenceTools(context)
    variants = ReferenceVariantTools(
        context,
        skill_names=context.executable_skill_names,
        skills_root=Path(__file__).resolve().parent.parent / "skills",
    )
    tools: list[ToolBase] = [
        Read(middlewares=audit, backend=backend),
        Glob(middlewares=audit, backend=backend),
        Grep(middlewares=audit, backend=backend),
        Write(middlewares=audit, backend=backend),
        Edit(middlewares=audit, backend=backend),
    ]
    custom = [
        MethodTool(
            name="InspectDataFile",
            description="Inspect an authorized structured data file.",
            input_schema=_object(
                {
                    "file_path": _text("CSV, TSV, gzip, parquet, JSON or spreadsheet path."),
                    "sample_rows": _integer("Sample rows.", 3, 0, 50),
                    "max_columns": _integer("Maximum columns.", 80, 1, 500),
                },
                ["file_path"],
            ),
            method=implementation.InspectDataFile,
            read_only=True,
            middlewares=audit,
        ),
        MethodTool(
            name="CompareArtifact",
            description="Compare two authorized structured artifacts.",
            input_schema=_object(
                {
                    "expected_path": _text("Expected artifact."),
                    "actual_path": _text("Actual artifact."),
                    "key_columns_json": _text("JSON key column list.", "[]"),
                    "numeric_tolerance": {"type": "number", "default": 1e-8, "minimum": 0},
                    "ignore_row_order": {"type": "boolean", "default": True},
                    "normalize_empty": {"type": "boolean", "default": True},
                    "limit": _integer("Maximum examples.", 10, 1, 100),
                },
                ["expected_path", "actual_path"],
            ),
            method=implementation.CompareArtifact,
            read_only=True,
            middlewares=audit,
        ),
        MethodTool(
            name="ValidateResultPackage",
            description="Validate a complete result package without hidden reference data.",
            input_schema=_object(
                {
                    "package_root": _text("Result package directory."),
                    "key_column": _text("Primary key column."),
                    "expected_key_count": _integer("Expected cohort keys.", 0, 0, 100_000_000),
                    "allow_empty_feature_tables": {"type": "boolean", "default": False},
                },
                ["package_root"],
            ),
            method=implementation.ValidateResultPackage,
            read_only=True,
            middlewares=audit,
        ),
        MethodTool(
            name="RunSkill",
            description="Execute an allowed Pipeline Skill by name.",
            input_schema=_object(
                {"skill_name": _text("Skill name."), "spec_json": _text("Skill JSON specification.", "{}")},
                ["skill_name"],
            ),
            method=implementation.RunSkill,
            middlewares=audit,
        ),
        MethodTool(
            name="ExecutePython",
            description="Execute a workspace Python file in the restricted project sandbox.",
            input_schema=_object(
                {
                    "file_path": _text("Workspace Python file."),
                    "args": {"type": "array", "items": {"type": "string"}, "default": []},
                    "timeout_seconds": _integer("Timeout seconds.", 300, 1, 600),
                    "task_id": _text("Optional task identifier.", ""),
                },
                ["file_path"],
            ),
            method=implementation.ExecutePython,
            middlewares=audit,
        ),
        MethodTool(
            name="PublishDirectoryArtifact",
            description="Publish a complete phase-local directory into the Agent workspace.",
            input_schema=_object(
                {
                    "directory_path": _text("Source directory."),
                    "alias": _text("Artifact alias.", "result_package"),
                    "target_name": _text("Optional target directory name.", ""),
                },
                ["directory_path"],
            ),
            method=implementation.PublishDirectoryArtifact,
            middlewares=audit,
        ),
        MethodTool(
            name="publish_artifact",
            description="Publish a phase-local file artifact.",
            input_schema=_object(
                {"file_path": _text("Source file."), "alias": _text("Artifact alias.")},
                ["file_path", "alias"],
            ),
            method=implementation.publish_artifact,
            middlewares=audit,
        ),
        MethodTool(
            name="inspect_skill",
            description="Inspect one retained Pipeline Skill.",
            input_schema=_object({"skill_name": _text("Skill name.")}, ["skill_name"]),
            method=variants.inspect_skill,
            read_only=True,
            middlewares=audit,
        ),
        MethodTool(
            name="create_skill_variant",
            description="Create a run-local Skill adapter or fork.",
            input_schema=_object(
                {
                    "base_skill": _text("Base Skill."),
                    "variant_name": _text("Variant name."),
                    "mode": {"type": "string", "enum": ["adapter", "fork"]},
                    "change_spec": _text("Required changes."),
                    "contracts": _text("JSON input/output contracts."),
                },
                ["base_skill", "variant_name", "mode", "change_spec", "contracts"],
            ),
            method=variants.create_skill_variant,
            middlewares=audit,
        ),
        MethodTool(
            name="validate_skill_variant",
            description="Validate a run-local Skill variant.",
            input_schema=_object(
                {"variant_name": _text("Variant name."), "artifact_path": _text("Optional output artifact.", "")},
                ["variant_name"],
            ),
            method=variants.validate_skill_variant,
            middlewares=audit,
        ),
        MethodTool(
            name="execute_skill_variant",
            description="Execute a ready run-local Skill variant.",
            input_schema=_object(
                {
                    "variant_name": _text("Variant name."),
                    "timeout_seconds": _integer("Timeout seconds.", 300, 1, 600),
                },
                ["variant_name"],
            ),
            method=variants.execute_skill_variant,
            middlewares=audit,
        ),
    ]
    if include_quality_tools:
        custom.extend(
            [
                MethodTool(
                    name="ProfileDataQuality",
                    description=(
                        "Profile one authorized structured data file for missingness, "
                        "types, duplicates, distributions and format patterns. Evidence only."
                    ),
                    input_schema=_object(
                        {
                            "file_path": _text("Authorized structured data file."),
                            "columns_json": _text("Optional JSON list of columns.", "[]"),
                            "top_k": _integer("Maximum frequent values and patterns per column.", 10, 1, 50),
                        },
                        ["file_path"],
                    ),
                    method=quality.ProfileDataQuality,
                    read_only=True,
                    middlewares=audit,
                ),
                MethodTool(
                    name="ProfileByGroup",
                    description=(
                        "Profile within-group consistency and distributions for explicitly selected columns. "
                        "Evidence only."
                    ),
                    input_schema=_object(
                        {
                            "file_path": _text("Authorized structured data file."),
                            "group_columns_json": _text("JSON list of grouping columns."),
                            "value_columns_json": _text("Optional JSON list of analyzed columns.", "[]"),
                            "max_groups": _integer("Maximum groups returned.", 100, 1, 1000),
                        },
                        ["file_path", "group_columns_json"],
                    ),
                    method=quality.ProfileByGroup,
                    read_only=True,
                    middlewares=audit,
                ),
                MethodTool(
                    name="TestDataConstraint",
                    description=(
                        "Test safe declarative candidate constraints and report violations. "
                        "Does not prove business correctness or repair values."
                    ),
                    input_schema=_object(
                        {
                            "constraints_json": _text("Non-empty JSON list of declarative constraints."),
                            "sample_limit": _integer("Maximum violation examples per constraint.", 10, 0, 50),
                        },
                        ["constraints_json"],
                    ),
                    method=quality.TestDataConstraint,
                    read_only=True,
                    middlewares=audit,
                ),
                MethodTool(
                    name="AuditRepairDelta",
                    description=(
                        "Compare authorized dirty and repaired files or packages by business keys. "
                        "Reports changes without judging correctness."
                    ),
                    input_schema=_object(
                        {
                            "dirty_path": _text("Dirty source file or package directory."),
                            "repaired_path": _text("Candidate repaired file or package directory."),
                            "key_columns_json": _text("Default JSON business-key column list.", "[]"),
                            "key_map_json": _text("Optional JSON map from relative file path to key columns.", "{}"),
                            "sample_limit": _integer("Maximum changed-row examples per file.", 10, 0, 50),
                        },
                        ["dirty_path", "repaired_path"],
                    ),
                    method=quality.AuditRepairDelta,
                    read_only=True,
                    middlewares=audit,
                ),
            ]
        )
    tools.extend(custom)
    return tools
