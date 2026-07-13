from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROTOCOL_KEYS = ("rule_ids", "input_artifacts", "parameters", "output_contract")
RESULT_KEYS = ("status", "artifacts", "field_mappings", "metrics", "issues")
RESULT_PREFIX = "VARIANT_RESULT_JSON="
EXECUTION_RECEIPT = "execution_receipt.json"


def validate_adapter_request(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Adapter request must be a JSON object")
    missing = [key for key in PROTOCOL_KEYS if key not in value]
    if missing:
        raise ValueError(f"Adapter request is missing keys: {', '.join(missing)}")
    if not isinstance(value["rule_ids"], list) or not all(isinstance(item, str) for item in value["rule_ids"]):
        raise ValueError("rule_ids must be a list of strings")
    if not isinstance(value["input_artifacts"], list) or not all(
        isinstance(item, dict) and item.get("path") for item in value["input_artifacts"]
    ):
        if value["input_artifacts"]:
            raise ValueError("input_artifacts must contain objects with path")
    if not isinstance(value["parameters"], dict):
        raise ValueError("parameters must be an object")
    if not isinstance(value["output_contract"], dict):
        raise ValueError("output_contract must be an object")
    return value


def parse_variant_result(stdout: str) -> dict[str, Any]:
    payload_text = ""
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload_text = line[len(RESULT_PREFIX) :].strip()
            break
    if not payload_text:
        raise ValueError("stdout is missing VARIANT_RESULT_JSON")
    value = json.loads(payload_text)
    if not isinstance(value, dict):
        raise ValueError("VARIANT_RESULT_JSON must be an object")
    missing = [key for key in RESULT_KEYS if key not in value]
    if missing:
        raise ValueError(f"VARIANT_RESULT_JSON is missing {', '.join(missing)}")
    if value["status"] not in {"SUCCESS", "NEEDS_REPAIR"}:
        raise ValueError("Variant status must be SUCCESS or NEEDS_REPAIR")
    for key in ("artifacts", "field_mappings", "issues"):
        if not isinstance(value[key], list):
            raise ValueError(f"Variant result {key} must be a list")
    if not isinstance(value["metrics"], dict):
        raise ValueError("Variant result metrics must be an object")
    return value


def persist_validated_variants(variant_root: str | Path, candidate_bundle: str | Path) -> list[str]:
    source_root = Path(variant_root).expanduser().resolve()
    bundle = Path(candidate_bundle).expanduser().resolve()
    capabilities = bundle / "capabilities"
    capabilities.mkdir(parents=True, exist_ok=True)
    persisted: list[str] = []
    bindings: list[dict[str, Any]] = []
    if not source_root.is_dir():
        return persisted
    for variant in sorted(path for path in source_root.iterdir() if path.is_dir()):
        metadata = _load_metadata(variant)
        if metadata.get("status") != "validated":
            continue
        _verify_source_hashes(metadata)
        receipt = _verify_execution_receipt(variant)
        destination = capabilities / variant.name
        temporary = capabilities / f".{variant.name}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(variant, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        persisted.append(variant.name)
        bindings.append(_binding_for_variant(destination, metadata, receipt, bundle))
    if bindings:
        _merge_skill_bindings(bundle / "skill_bindings.json", bindings)
    return persisted


def restore_bundle_variants(bundle_dir: str | Path, variant_root: str | Path) -> list[str]:
    capabilities = Path(bundle_dir).expanduser().resolve() / "capabilities"
    destination_root = Path(variant_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    if not capabilities.is_dir():
        return restored
    for capability in sorted(path for path in capabilities.iterdir() if path.is_dir()):
        if not (capability / "variant.json").is_file():
            continue
        metadata = _load_metadata(capability)
        if metadata.get("status") != "validated":
            continue
        _verify_source_hashes(metadata)
        _verify_execution_receipt(capability)
        destination = destination_root / capability.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(capability, destination)
        restored_metadata = json.loads((destination / "variant.json").read_text(encoding="utf-8"))
        restored_metadata["restored_from_bundle"] = True
        (destination / "variant.json").write_text(
            json.dumps(restored_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        restored.append(capability.name)
    return restored


def _load_metadata(variant_dir: Path) -> dict[str, Any]:
    required = ("SKILL.md", "variant.json", "variant.py", "request.json")
    missing = [name for name in required if not (variant_dir / name).is_file()]
    if missing:
        raise ValueError(f"Variant {variant_dir.name} is missing files: {', '.join(missing)}")
    request = json.loads((variant_dir / "request.json").read_text(encoding="utf-8"))
    validate_adapter_request(request)
    metadata = json.loads((variant_dir / "variant.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("protocol_version") != 1:
        raise ValueError(f"Variant {variant_dir.name} has an unsupported protocol")
    return metadata


def _verify_source_hashes(metadata: dict[str, Any]) -> None:
    expected = metadata.get("base_source_hashes")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Variant is missing base source hashes")
    for raw_path, expected_hash in expected.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Base Skill source hash changed: {path}")


def write_variant_execution_receipt(
    variant_dir: str | Path,
    result: dict[str, Any],
    *,
    output_files: list[str | Path],
) -> Path:
    """Bind a successful variant run to the exact implementation and request."""
    variant = Path(variant_dir).expanduser().resolve()
    parsed = parse_variant_result(f"{RESULT_PREFIX}{json.dumps(result, ensure_ascii=False)}")
    if parsed["status"] != "SUCCESS":
        raise ValueError("Cannot write an execution receipt for a failed variant")
    variant_path = variant / "variant.py"
    request_path = variant / "request.json"
    if not variant_path.is_file() or not request_path.is_file():
        raise ValueError("Variant execution receipt requires variant.py and request.json")
    receipt = {
        "schema_version": 1,
        "status": "SUCCESS",
        "variant_sha256": _sha256(variant_path),
        "request_sha256": _sha256(request_path),
        "result": parsed,
        "output_files": [str(Path(path).expanduser().resolve()) for path in output_files],
        "executed_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = variant / EXECUTION_RECEIPT
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def verify_variant_execution_receipt(variant_dir: str | Path) -> dict[str, Any]:
    return _verify_execution_receipt(Path(variant_dir).expanduser().resolve())


def _verify_execution_receipt(variant: Path) -> dict[str, Any]:
    receipt_path = variant / EXECUTION_RECEIPT
    if not receipt_path.is_file():
        raise ValueError(f"Variant {variant.name} is missing execution receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("status") != "SUCCESS":
        raise ValueError(f"Variant {variant.name} has an invalid execution receipt")
    expected = {
        "variant_sha256": _sha256(variant / "variant.py"),
        "request_sha256": _sha256(variant / "request.json"),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Variant {variant.name} execution receipt does not match current files")
    result = receipt.get("result")
    if not isinstance(result, dict) or result.get("status") != "SUCCESS":
        raise ValueError(f"Variant {variant.name} execution receipt has no successful result")
    return receipt


def _binding_for_variant(
    variant: Path,
    metadata: dict[str, Any],
    receipt: dict[str, Any],
    bundle: Path,
) -> dict[str, Any]:
    request = json.loads((variant / "request.json").read_text(encoding="utf-8"))
    result = receipt.get("result") if isinstance(receipt.get("result"), dict) else {}
    capability_path = variant.relative_to(bundle).as_posix() if bundle in variant.parents else str(variant)
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), list) else []
    field_mappings = result.get("field_mappings") if isinstance(result.get("field_mappings"), list) else []
    input_artifacts = request.get("input_artifacts") if isinstance(request.get("input_artifacts"), list) else []
    return {
        "binding_type": "skill_variant",
        "variant_name": variant.name,
        "base_skill": str(metadata.get("base_skill") or ""),
        "mode": str(metadata.get("mode") or ""),
        "status": str(metadata.get("status") or ""),
        "capability_path": capability_path,
        "created_round": metadata.get("created_round", 0),
        "created_run_id": metadata.get("created_run_id", ""),
        "change_spec": metadata.get("change_spec", ""),
        "rule_ids": [str(value) for value in request.get("rule_ids") or []],
        "task_ids": [str(value) for value in metadata.get("task_ids") or []],
        "reference_artifacts": [
            str(item.get("alias") or item.get("path") or "")
            for item in input_artifacts
            if isinstance(item, dict) and (item.get("alias") or item.get("path"))
        ],
        "input_contract": metadata.get("input_contract") or {},
        "output_contract": metadata.get("output_contract") or {},
        "artifacts": artifacts,
        "field_mappings": field_mappings,
        "metrics": result.get("metrics") if isinstance(result.get("metrics"), dict) else {},
        "base_source_hashes": metadata.get("base_source_hashes") or {},
        "execution_receipt": str((variant / EXECUTION_RECEIPT).resolve()),
        "variant_sha256": receipt.get("variant_sha256", ""),
        "request_sha256": receipt.get("request_sha256", ""),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _merge_skill_bindings(path: Path, new_bindings: list[dict[str, Any]]) -> None:
    payload = {"schema_version": 1, "bindings": []}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
                if not isinstance(payload.get("bindings"), list):
                    payload["bindings"] = []
        except Exception:
            payload = {"schema_version": 1, "bindings": []}
    existing = [
        item for item in payload.get("bindings", [])
        if isinstance(item, dict)
    ]
    by_variant = {
        str(item.get("variant_name") or ""): item
        for item in existing
        if item.get("variant_name")
    }
    for binding in new_bindings:
        by_variant[str(binding.get("variant_name") or "")] = binding
    payload["bindings"] = [
        by_variant[name]
        for name in sorted(name for name in by_variant if name)
    ]
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
