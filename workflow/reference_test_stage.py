from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from workflow.reference_guided import evaluate_reference_directory_package, infer_reference_contract


@dataclass(frozen=True)
class ReferenceTestStageConfig:
    dataset_split: str | Path
    experiment_dir: str | Path
    adapter_script: str | Path

    def normalized(self) -> "ReferenceTestStageConfig":
        split = Path(self.dataset_split).expanduser().resolve()
        if not split.is_dir():
            raise ValueError(f"dataset_split does not exist: {split}")
        experiment = Path(self.experiment_dir).expanduser().resolve()
        adapter = Path(self.adapter_script).expanduser().resolve()
        if not adapter.is_file():
            raise ValueError(f"adapter_script does not exist: {adapter}")
        return ReferenceTestStageConfig(
            dataset_split=split,
            experiment_dir=experiment,
            adapter_script=adapter,
        )


def run_reference_test_stage(config: ReferenceTestStageConfig) -> dict[str, Any]:
    cfg = config.normalized()
    cfg.experiment_dir.mkdir(parents=True, exist_ok=True)
    contract = infer_reference_contract(cfg.dataset_split)
    paths = contract.get("paths") or {}
    test_raw = _required_path(paths, "test_raw")
    test_keys = _required_path(paths, "test_keys")
    train_reference_root = _required_path(paths, "train_reference_root")
    test_reference_root = (
        _optional_existing_path(paths.get("test_reference_private_root"))
        or _optional_existing_path(paths.get("test_reference_root"))
        or _optional_existing_path(paths.get("test_reference_private"))
    )
    if test_reference_root is None:
        raise ValueError("dataset_split does not expose a test reference directory")

    run_root = cfg.experiment_dir / "test_run"
    output_package = run_root / "result_package"
    if output_package.exists():
        shutil.rmtree(output_package)
    run_root.mkdir(parents=True, exist_ok=True)

    adapter_snapshot = cfg.experiment_dir / "adapter_script.py"
    if cfg.adapter_script.resolve() != adapter_snapshot.resolve():
        shutil.copy2(cfg.adapter_script, adapter_snapshot)
    run_log = cfg.experiment_dir / "test_adapter_run.log"
    env = os.environ.copy()
    env.update(
        {
            "RAW_ROOT": str(test_raw),
            "KEYS_CSV": str(test_keys),
            "TRAIN_REF_ROOT": str(train_reference_root),
            "OUTPUT_DIR": str(run_root),
        }
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    with run_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            [sys.executable, str(adapter_snapshot)],
            cwd=str(cfg.experiment_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finished_at = datetime.now().isoformat(timespec="seconds")

    evaluation_paths: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    if process.returncode == 0:
        evaluation_paths = evaluate_reference_directory_package(
            result_package=output_package,
            validation_reference_root=test_reference_root,
            output_dir=cfg.experiment_dir / "test_evaluation",
            key_column=str(contract.get("key_column") or ""),
        )
        report_path = Path(evaluation_paths["evaluation_report"])
        if report_path.is_file():
            evaluation_report = json.loads(report_path.read_text(encoding="utf-8"))
            metrics = evaluation_report.get("metrics") or {}

    report = {
        "schema_version": 1,
        "status": "SUCCESS" if process.returncode == 0 and metrics else "FAILED",
        "workflow": "reference-test-evaluate",
        "started_at": started_at,
        "finished_at": finished_at,
        "dataset_split": str(cfg.dataset_split),
        "experiment_dir": str(cfg.experiment_dir),
        "adapter_script": str(cfg.adapter_script),
        "adapter_snapshot": str(adapter_snapshot),
        "run_log": str(run_log),
        "returncode": process.returncode,
        "result_package": str(output_package),
        "evaluation": evaluation_paths,
        "metrics": metrics,
        "contract_paths": {
            "test_raw": str(test_raw),
            "test_keys": str(test_keys),
            "train_reference_root": str(train_reference_root),
            "test_reference_root": str(test_reference_root),
        },
    }
    report_path = cfg.experiment_dir / "reference_test_stage_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _required_path(paths: dict[str, Any], key: str) -> Path:
    value = paths.get(key)
    path = Path(str(value or "")).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"contract path {key} does not exist: {path}")
    return path


def _optional_existing_path(value: Any) -> Path | None:
    if not str(value or "").strip():
        return None
    path = Path(str(value)).expanduser().resolve()
    return path if path.exists() else None
