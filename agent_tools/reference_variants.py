"""Minimal run-local adapter lifecycle for the Data Cleaning Agent."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse

from lib.agent_artifacts import get_phase_context, record_step
from workflow.skill_adapter import verify_variant_execution_receipt

from .context import EngineerToolContext


_VARIANT_NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")
_BLOCKED_IMPORTS = {"requests", "httpx", "socket", "subprocess", "urllib"}


def _response(status: str, summary: str, artifacts=None, issues=()) -> ToolResponse:
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "status": status,
                        "summary": summary,
                        "artifacts": artifacts or {},
                        "issues": list(issues),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
        ],
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class ReferenceVariantTools:
    """Inspect, derive, validate, and execute adapters inside one experiment round."""

    def __init__(
        self,
        context: EngineerToolContext,
        *,
        skill_names: Iterable[str],
        skills_root: str | Path,
    ) -> None:
        self.context = context
        self.skill_names = frozenset(skill_names)
        self.skills_root = Path(skills_root).expanduser().resolve()
        self.inspected_skill_hashes: dict[str, dict[str, str]] = {}

    def inspect_skill(self, skill_name: str) -> ToolResponse:
        """Inspect one retained Pipeline Skill before deriving an adapter from it."""
        try:
            source_dir = self._base_skill_dir(skill_name)
            sources = self._source_hashes(source_dir)
            self.inspected_skill_hashes[skill_name] = sources
            skill_md = source_dir / "SKILL.md"
            return _response(
                "SUCCESS",
                f"Inspected base Skill {skill_name}.",
                {
                    "skill_name": skill_name,
                    "source_dir": str(source_dir),
                    "source_paths": list(sources),
                    "source_hashes": sources,
                    "skill_markdown": skill_md.read_text(encoding="utf-8") if skill_md.exists() else "",
                    "metadata": self._read_skill_metadata(source_dir / "skill.py"),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Skill inspection failed.", issues=[str(exc)])

    def create_skill_variant(
        self,
        base_skill: str,
        variant_name: str,
        mode: str,
        change_spec: str,
        contracts: str,
    ) -> ToolResponse:
        """Create a run-local adapter or fork without changing a base Skill."""
        try:
            source_dir = self._base_skill_dir(base_skill)
            source_hashes = self._source_hashes(source_dir)
            if self.inspected_skill_hashes.get(base_skill) != source_hashes:
                raise ValueError(f"Call inspect_skill for {base_skill} immediately before creating a variant")
            if mode not in {"adapter", "fork"}:
                raise ValueError("mode must be adapter or fork")
            if not _VARIANT_NAME_RE.fullmatch(variant_name):
                raise ValueError("variant_name must be a 3-64 character snake_case identifier")
            if not str(change_spec).strip():
                raise ValueError("change_spec is required")
            contract_value = json.loads(contracts)
            if not isinstance(contract_value, dict):
                raise ValueError("contracts must be a JSON object")
            input_contract = contract_value.get("input_contract")
            output_contract = contract_value.get("output_contract")
            if not isinstance(input_contract, dict) or not isinstance(output_contract, dict):
                raise ValueError("contracts must contain input_contract and output_contract objects")

            variant_dir = (self.context.variant_root / variant_name).resolve()
            if variant_dir.exists():
                raise ValueError(f"Variant already exists in this run: {variant_name}")
            variant_dir.mkdir(parents=True)
            phase_context = get_phase_context() or {}
            metadata = {
                "protocol_version": 1,
                "base_skill": base_skill,
                "mode": mode,
                "base_source_hashes": source_hashes,
                "change_spec": str(change_spec).strip(),
                "input_contract": input_contract,
                "output_contract": output_contract,
                "status": "draft",
                "created_run_id": phase_context.get("run_id", Path(self.context.engineer_phase_root).name),
                "created_round": self._created_round(phase_context),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            _write_json(variant_dir / "variant.json", metadata)
            _write_json(
                variant_dir / "request.json",
                {"rule_ids": [], "input_artifacts": [], "parameters": {}, "output_contract": output_contract},
            )
            (variant_dir / "SKILL.md").write_text(
                f"# {variant_name}\n\nRun-local {mode} derived from `{base_skill}`.\n\n"
                f"## Change specification\n\n{change_spec.strip()}\n",
                encoding="utf-8",
            )
            (variant_dir / "variant.py").write_text(
                self._variant_scaffold(base_skill, mode, output_contract),
                encoding="utf-8",
            )
            record = record_step(
                "create_skill_variant",
                f"Created draft {mode} variant {variant_name} from {base_skill}.",
                list(variant_dir.iterdir()),
                metadata={"variant_name": variant_name, "base_skill": base_skill, "mode": mode},
            )
            return _response(
                "SUCCESS",
                f"Created draft Skill variant {variant_name}.",
                {
                    "variant_dir": str(variant_dir),
                    "variant_path": str(variant_dir / "variant.py"),
                    "request_path": str(variant_dir / "request.json"),
                    "metadata_path": str(variant_dir / "variant.json"),
                    "variant_status": "draft",
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            return _response("NEEDS_REPAIR", "Skill variant creation failed.", issues=[str(exc)])

    def validate_skill_variant(self, variant_name: str, artifact_path: str = "") -> ToolResponse:
        """Validate variant files and optionally a produced CSV/JSON artifact."""
        try:
            variant_dir = self._variant_dir(variant_name)
            metadata_path = variant_dir / "variant.json"
            metadata = _load_json(metadata_path)
            required = [variant_dir / name for name in ("SKILL.md", "variant.json", "variant.py", "request.json")]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ValueError(f"Variant files are missing: {missing}")
            self._verify_source_hashes(metadata)
            self._validate_variant_source(variant_dir / "variant.py")
            _load_json(variant_dir / "request.json")
            if artifact_path:
                artifact = self.context.resolve_read_path(artifact_path)
                self._validate_artifact(artifact, metadata.get("output_contract") or {})
                receipt = verify_variant_execution_receipt(variant_dir)
                output_files = {str(Path(value).expanduser().resolve()) for value in receipt.get("output_files", [])}
                if str(artifact.resolve()) not in output_files:
                    raise ValueError("Validated artifact was not produced by this variant execution")
                metadata["status"] = "validated"
                metadata["validated_artifact"] = str(artifact)
            else:
                metadata["status"] = "ready"
            metadata["validated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(metadata_path, metadata)
            record = record_step(
                "validate_skill_variant",
                f"Variant {variant_name} is {metadata['status']}.",
                [metadata_path],
                metadata={"variant_name": variant_name, "variant_status": metadata["status"]},
            )
            return _response(
                "SUCCESS",
                f"Skill variant {variant_name} is {metadata['status']}.",
                {
                    "variant_dir": str(variant_dir),
                    "variant_status": metadata["status"],
                    "artifact_path": str(artifact_path or ""),
                    "manifest_path": (record or {}).get("manifest_path", ""),
                },
            )
        except Exception as exc:
            self._mark_variant_failed(variant_name, str(exc))
            return _response("NEEDS_REPAIR", "Skill variant validation failed.", issues=[str(exc)])

    def execute_skill_variant(self, variant_name: str, timeout_seconds: int = 300) -> ToolResponse:
        """Run a ready variant using its lifecycle-managed request.json."""
        try:
            variant_dir = self._variant_dir(variant_name)
            metadata = _load_json(variant_dir / "variant.json")
            if metadata.get("status") not in {"ready", "validated"}:
                raise ValueError("Derived Skill must be ready before execution")
            from .tools import EngineerTools

            response = EngineerTools(self.context).ExecutePython(
                str(variant_dir / "variant.py"),
                args=[str(variant_dir / "request.json")],
                timeout_seconds=timeout_seconds,
            )
            payload = json.loads(response.content[0]["text"])
            self.context.record_skill_call(
                str(metadata.get("base_skill") or ""),
                f"variant:{variant_name}",
                str(payload.get("status") or "UNKNOWN"),
                artifacts=(payload.get("artifacts") or {}).get("output_files") or [],
            )
            return response
        except Exception as exc:
            self._mark_variant_failed(variant_name, str(exc))
            return _response("NEEDS_REPAIR", "Skill variant execution failed.", issues=[str(exc)])

    def _base_skill_dir(self, skill_name: str) -> Path:
        if skill_name not in self.skill_names:
            raise ValueError(f"Skill is not available to Data Cleaning Agent: {skill_name}")
        path = (self.skills_root / skill_name).resolve()
        if path.parent != self.skills_root or not path.is_dir():
            raise ValueError(f"Base Skill directory does not exist: {path}")
        return path

    @staticmethod
    def _source_hashes(source_dir: Path) -> dict[str, str]:
        paths = [path for path in (source_dir / "SKILL.md", source_dir / "skill.py") if path.is_file()]
        if not paths:
            raise ValueError(f"Base Skill has no inspectable source files: {source_dir}")
        return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    @staticmethod
    def _read_skill_metadata(skill_py: Path) -> dict[str, Any]:
        if not skill_py.exists():
            return {}
        try:
            tree = ast.parse(skill_py.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "SKILL" for target in node.targets
                ):
                    value = ast.literal_eval(node.value)
                    return value if isinstance(value, dict) else {}
        except Exception:
            return {}
        return {}

    def _variant_dir(self, variant_name: str) -> Path:
        if not _VARIANT_NAME_RE.fullmatch(variant_name):
            raise ValueError("Invalid variant_name")
        path = (self.context.variant_root / variant_name).resolve()
        if path.parent != self.context.variant_root.resolve() or not path.is_dir():
            raise ValueError(f"Variant does not exist in this run: {variant_name}")
        return path

    def _verify_source_hashes(self, metadata: dict[str, Any]) -> None:
        expected = metadata.get("base_source_hashes")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Variant is missing base_source_hashes")
        current = self._source_hashes(self._base_skill_dir(str(metadata.get("base_skill") or "")))
        if current != expected:
            raise ValueError(f"Base Skill source changed after variant creation: {metadata.get('base_skill')}")

    @staticmethod
    def _validate_variant_source(path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        if "IMPLEMENTATION_REQUIRED" in source:
            raise ValueError("variant.py still contains IMPLEMENTATION_REQUIRED")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                if names & _BLOCKED_IMPORTS:
                    raise ValueError(f"Forbidden import in variant.py: {sorted(names & _BLOCKED_IMPORTS)}")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _BLOCKED_IMPORTS:
                raise ValueError(f"Forbidden import in variant.py: {node.module}")

    @staticmethod
    def _validate_artifact(path: Path, contract: dict[str, Any]) -> None:
        if not path.is_file():
            raise ValueError(f"Variant artifact does not exist: {path}")
        fmt = str(contract.get("format") or path.suffix.lstrip(".")).lower()
        if fmt == "csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                if not csv.DictReader(handle).fieldnames:
                    raise ValueError("CSV artifact has no header")
        elif fmt == "json":
            json.loads(path.read_text(encoding="utf-8"))
        elif fmt == "jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
        else:
            raise ValueError(f"Unsupported output contract format: {fmt}")

    def _mark_variant_failed(self, variant_name: str, issue: str) -> None:
        try:
            metadata_path = self._variant_dir(variant_name) / "variant.json"
            metadata = _load_json(metadata_path)
            metadata["status"] = "failed"
            metadata["last_error"] = issue
            _write_json(metadata_path, metadata)
        except Exception:
            return

    def _created_round(self, phase_context: dict[str, Any]) -> int:
        values = [str(phase_context.get("run_id") or ""), str(self.context.engineer_phase_root)]
        for value in values:
            match = re.search(r"round[_-](\d+)", value)
            if match:
                return int(match.group(1))
        return 0

    @staticmethod
    def _variant_scaffold(base_skill: str, mode: str, output_contract: dict[str, Any]) -> str:
        return f'''from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# IMPLEMENTATION_REQUIRED
# Base Skill: {base_skill}
# Mode: {mode}
# Output contract: {output_contract!r}
request_path = Path(sys.argv[1]).resolve()
request = json.loads(request_path.read_text(encoding="utf-8"))
output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
raise NotImplementedError("IMPLEMENTATION_REQUIRED")
# Successful implementations must print: VARIANT_RESULT_JSON={{...}}
'''


def register_reference_variant_tools(
    toolkit: Toolkit,
    context: EngineerToolContext,
    *,
    skill_names: Iterable[str],
    skills_root: str | Path,
) -> ReferenceVariantTools:
    tools = ReferenceVariantTools(context, skill_names=skill_names, skills_root=skills_root)
    for tool in (
        tools.inspect_skill,
        tools.create_skill_variant,
        tools.validate_skill_variant,
        tools.execute_skill_variant,
    ):
        toolkit.register_tool_function(tool)
    return tools
