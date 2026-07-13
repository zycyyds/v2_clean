from __future__ import annotations

import json
import shutil
from pathlib import Path

from agentscope.tool import Toolkit

from skills._pipeline_tools import run_pipeline_step, safe_target, table_info


SOURCE_PIPELINE_FILES = [
    "local reference package adapter wrapping MIMIC-IV-Data-Pipeline outputs",
]
EXPECTED_RAW_FILES: list[str] = []


def pipeline_assemble_reference_package_tool(spec_json: str):
    """Assemble a result_package matching train/reference shape.

    spec_json fields:
    - files: list of {source, target, role?}; target is relative under result_package.
    - package_name: optional, default result_package.
    - write_manifest: optional, default false.
    """

    def runner(spec, output: Path):
        package_name = str(spec.get("package_name") or "result_package")
        if ".." in Path(package_name).parts:
            raise ValueError("Unsafe package_name")
        package_root = output / package_name
        package_root.mkdir(parents=True, exist_ok=True)
        files = spec.get("files") or []
        if not isinstance(files, list) or not files:
            raise ValueError("spec_json.files must be a non-empty list")
        copied: list[dict] = []
        artifact_paths: list[str] = []
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("Each file entry must be an object")
            source = item.get("source")
            target = item.get("target")
            if not source or not target:
                raise ValueError("Each file entry requires source and target")
            source_path = Path(source).expanduser().resolve()
            target_path = safe_target(package_root, str(target))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            rows, cols = table_info(target_path)
            artifact_paths.append(str(target_path))
            copied.append(
                {
                    "source": str(source_path),
                    "target": str(target),
                    "path": str(target_path),
                    "role": item.get("role") or "",
                    "row_count": rows,
                    "columns": cols,
                }
            )
        manifest_path = ""
        if spec.get("write_manifest"):
            manifest = {
                "schema_version": 1,
                "package_name": package_name,
                "files": copied,
            }
            manifest_target = package_root / "package_manifest.json"
            manifest_target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path = str(manifest_target)
            artifact_paths.append(manifest_path)
        return {
            "status": "SUCCESS",
            "result_package": str(package_root),
            "package_manifest": manifest_path,
            "files": copied,
            "row_counts": {Path(item["path"]).name: item["row_count"] for item in copied},
            "columns": {Path(item["path"]).name: item["columns"] for item in copied},
            "source_files": [item["source"] for item in copied],
            "artifact_paths": artifact_paths,
        }

    return run_pipeline_step(
        "pipeline_assemble_reference_package",
        spec_json,
        runner,
        summary="Assembled reference-shaped result package without requiring reference.csv.",
        artifact_keys=("result_package", "package_manifest"),
        source_pipeline_files=SOURCE_PIPELINE_FILES,
        expected_raw_files=EXPECTED_RAW_FILES,
    )


SKILL = {
    "name": "pipeline_assemble_reference_package",
    "layer": "process",
    "description": "Pipeline 化结果包组装能力：把 cohort/features/明细 CSV 复制成与 train/reference 同构的 result_package。reference.csv 和 package_manifest.json 都是可选。",
    "when_to_use": "已经生成目标 CSV，需要按 train/reference 的相对路径组装 validation result_package 时使用。",
    "when_to_skip": "核心 CSV 尚未生成，或需要先修复 train 回归差异时跳过。",
    "capability_types": ["pipeline_package_assembly", "reference_package_assembly"],
    "inputs": ["spec_json"],
    "outputs": ["result_package/"],
    "prerequisites": ["target_artifacts_available", "reference_shape_known"],
    "adapter_policy": "direct_then_adapter_then_fork",
    "preserves_raw_values": True,
    "lib_entrypoints": ["skills.pipeline_assemble_reference_package.pipeline_assemble_reference_package_tool"],
    "source_pipeline_files": [
        "local reference package adapter wrapping MIMIC-IV-Data-Pipeline outputs",
    ],
    "expected_raw_files": [],
    "output_contract": ["result_package contains the requested relative files; manifest is optional"],
    "common_failure_modes": ["source artifact missing", "unsafe target path", "empty file list"],
}


def register(toolkit: Toolkit) -> None:
    toolkit.register_tool_function(pipeline_assemble_reference_package_tool)
