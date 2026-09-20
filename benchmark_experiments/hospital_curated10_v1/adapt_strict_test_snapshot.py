"""Adapt the frozen Hospital submission to the strict Test replay contract.

The generated pipeline accepts both Train dirty and Train reference paths, but
its learning phase derives every artifact from the reference rows.  Strict Test
replay intentionally exposes only the Train reference.  This adapter creates a
new, provenance-recorded validation bundle whose submission supplies the Train
reference for both arguments.  It does not alter the cleaning implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from agent.pi_harness import directory_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def adapt_snapshot(source_experiment: Path, output_experiment: Path) -> dict[str, Any]:
    source_experiment = source_experiment.expanduser().resolve()
    output_experiment = output_experiment.expanduser().resolve()
    source_host = source_experiment / "host"
    source_snapshot = source_host / "reproducible_snapshot"
    source_report_path = source_host / "run_report.json"
    source_manifest_path = source_host / "run_manifest.json"

    report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "SUCCESS_REPRODUCIBLE":
        raise ValueError("source experiment must be SUCCESS_REPRODUCIBLE")
    if Path(str(report.get("reproducible_snapshot") or "")).resolve() != source_snapshot:
        raise ValueError("source report does not identify its reproducible snapshot")
    if output_experiment.exists():
        raise FileExistsError(f"output experiment already exists: {output_experiment}")

    output_host = output_experiment / "host"
    output_snapshot = output_host / "reproducible_snapshot"
    output_snapshot.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_snapshot, output_snapshot)

    submission_path = output_snapshot / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    argv = submission.get("replay", {}).get("argv")
    if not isinstance(argv, list):
        raise ValueError("source submission replay.argv is invalid")
    train_raw_occurrences = sum(item.count("{train_raw}") for item in argv)
    train_raw_flags = {"--train-raw", "--train_raw"}
    if train_raw_occurrences != 1 or not train_raw_flags.intersection(argv):
        raise ValueError("expected exactly one explicit {train_raw} argument")

    source_submission_sha256 = _sha256(submission_path)
    submission["replay"]["argv"] = [
        item.replace("{train_raw}", "{train_reference}") for item in argv
    ]
    _write_json(submission_path, submission)

    output_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_manifest["strict_test_adapter"] = {
        "name": "hospital_train_reference_alias_v1",
        "source_experiment": str(source_experiment),
        "source_snapshot_sha256": directory_sha256(source_snapshot),
    }
    _write_json(output_host / "run_manifest.json", output_manifest)

    output_report = dict(report)
    output_report["reproducible_snapshot"] = str(output_snapshot)
    output_report["strict_test_adapter"] = "hospital_train_reference_alias_v1"
    _write_json(output_host / "run_report.json", output_report)

    adaptation = {
        "schema_version": 1,
        "adapter": "hospital_train_reference_alias_v1",
        "source_experiment": str(source_experiment),
        "source_snapshot_sha256": directory_sha256(source_snapshot),
        "adapted_snapshot_sha256": directory_sha256(output_snapshot),
        "source_submission_sha256": source_submission_sha256,
        "adapted_submission_sha256": _sha256(submission_path),
        "semantic_change": "alias unused train_raw input to train_reference",
        "cleaning_implementation_modified": False,
        "test_data_accessed": False,
    }
    _write_json(output_host / "strict_test_adaptation.json", adaptation)
    return adaptation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-experiment", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            adapt_snapshot(args.source_experiment, args.output_experiment),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
