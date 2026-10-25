from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_ROOT = Path("/Users/mac/PycharmProjects/MIMIC-IV-Data-Pipeline-main")
DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[1] / "reproductions" / "mimic_icu_mortality_v3_1_full_v1"
MINIMUM_FREE_BYTES = 20 * 1024**3

FEATURE_FILES = (
    "preproc_chart_icu.csv.gz",
    "preproc_diag_icu.csv.gz",
    "preproc_med_icu.csv.gz",
    "preproc_out_icu.csv.gz",
    "preproc_proc_icu.csv.gz",
)
SUMMARY_FILES = (
    "diag_summary.csv",
    "diag_features.csv",
    "med_summary.csv",
    "med_features.csv",
    "proc_summary.csv",
    "proc_features.csv",
    "out_summary.csv",
    "out_features.csv",
    "chart_summary.csv",
    "chart_features.csv",
)
GOLD_RELATIVE_PATHS = (
    "data/cohort/cohort_icu_mortality_0__.csv.gz",
    "data/csv/labels.csv",
    *(f"data/features/{name}" for name in FEATURE_FILES),
    *(f"data/summary/{name}" for name in SUMMARY_FILES),
)
STAGES = ("cohort", "features", "diagnosis", "summaries", "stay_generation", "validation")


@dataclass(frozen=True)
class ReproductionConfig:
    source_root: Path = DEFAULT_SOURCE_ROOT
    run_root: Path = DEFAULT_RUN_ROOT
    mimic_version: str = "3.1"
    care_setting: str = "ICU"
    task: str = "Mortality"
    disease_filter: str = "No Disease Filter"
    disease_label: str = ""
    cohort_time: int = 0
    diagnosis_mode: str = "Convert ICD-9 to ICD-10 and group ICD-10 codes"
    include_hours: int = 72
    prediction_hours: int = 2
    bucket_hours: int = 1
    imputation: bool = False
    feature_flags: dict[str, bool] = field(
        default_factory=lambda: {
            "diagnosis": True,
            "output": True,
            "chart": True,
            "procedures": True,
            "medications": True,
        }
    )
    cohort_output: str = "cohort_icu_mortality_0__"
    summary_output: str = "summary_icu_mortality_0__"

    def __post_init__(self) -> None:
        source_root = Path(self.source_root).expanduser().resolve()
        run_root = Path(self.run_root).expanduser().resolve()
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "run_root", run_root)
        original_data = source_root / "data"
        if run_root == original_data or original_data in run_root.parents:
            raise ValueError("run_root cannot be inside the original project data directory")
        if run_root == source_root or source_root in run_root.parents:
            raise ValueError("run_root cannot be inside the original source project")
        if self.include_hours <= 0 or self.prediction_hours < 0 or self.bucket_hours <= 0:
            raise ValueError("time-window values must be positive")
        if not all(self.feature_flags.values()) or set(self.feature_flags) != {
            "diagnosis",
            "output",
            "chart",
            "procedures",
            "medications",
        }:
            raise ValueError("this reproduction profile requires all five ICU feature families")

    @property
    def raw_root(self) -> Path:
        return self.source_root / "mimiciv" / self.mimic_version

    @property
    def mapping_path(self) -> Path:
        return self.source_root / "utils" / "mappings" / "ICD9_to_ICD10_mapping.txt"

    @staticmethod
    def required_raw_files() -> tuple[str, ...]:
        return (
            "hosp/admissions.csv.gz",
            "hosp/patients.csv.gz",
            "hosp/diagnoses_icd.csv.gz",
            "icu/icustays.csv.gz",
            "icu/outputevents.csv.gz",
            "icu/chartevents.csv.gz",
            "icu/procedureevents.csv.gz",
            "icu/inputevents.csv.gz",
        )

    @staticmethod
    def required_source_files() -> tuple[str, ...]:
        return (
            "preprocessing/day_intervals_preproc/day_intervals_cohort_v3.py",
            "preprocessing/day_intervals_preproc/disease_cohort.py",
            "preprocessing/hosp_module_preproc/feature_selection_icu.py",
            "utils/icu_preprocess_util.py",
            "utils/outlier_removal.py",
            "utils/uom_conversion.py",
            "model/data_generation_icu.py",
        )

    def as_serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_root"] = str(self.source_root)
        result["run_root"] = str(self.run_root)
        result["feature_flags"] = dict(sorted(self.feature_flags.items()))
        return result


class RunLayout:
    def __init__(self, config: ReproductionConfig):
        self.config = config
        self.root = config.run_root
        self.data_root = self.root / "data"
        self.cohort_dir = self.data_root / "cohort"
        self.features_dir = self.data_root / "features"
        self.intermediate_dir = self.data_root / "intermediate"
        self.summary_dir = self.data_root / "summary"
        self.csv_dir = self.data_root / "csv"
        self.dict_dir = self.data_root / "dict"
        self.logs_dir = self.root / "logs"
        self.checkpoints_dir = self.root / "checkpoints"
        self.raw_link = self.root / "mimiciv" / config.mimic_version
        self.mapping_link = self.root / "utils" / "mappings" / "ICD9_to_ICD10_mapping.txt"
        self.metadata_path = self.root / "reproduction_metadata.json"
        self.state_path = self.root / "run_state.json"
        self.validation_path = self.root / "validation_report.json"
        self.manifest_path = self.root / "reproduction_manifest.json"
        self.source_snapshot_path = self.root / "source_data_snapshot.json"
        self.lock_path = self.root / ".reproduction.lock"

    @property
    def intermediate_diag_path(self) -> Path:
        return self.intermediate_dir / "preproc_diag_icu_extracted.csv.gz"

    @property
    def cohort_path(self) -> Path:
        return self.cohort_dir / f"{self.config.cohort_output}.csv.gz"

    @property
    def cohort_summary_path(self) -> Path:
        return self.cohort_dir / f"{self.config.summary_output}.txt"

    @property
    def labels_path(self) -> Path:
        return self.csv_dir / "labels.csv"

    @staticmethod
    def expected_dict_names() -> tuple[str, ...]:
        return (
            "dataDic",
            "hadmDic",
            "ethVocab",
            "ageVocab",
            "insVocab",
            "medVocab",
            "outVocab",
            "chartVocab",
            "condVocab",
            "procVocab",
            "metaDic",
        )


def config_fingerprint(config: ReproductionConfig) -> str:
    payload = json.dumps(config.as_serializable(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_fingerprint(config: ReproductionConfig) -> str:
    source_files = tuple(config.source_root / relative for relative in config.required_source_files())
    runner_files = (
        Path(__file__),
        Path(__file__).with_name("mimic_stage_worker.py"),
        Path(__file__).with_name("mimic_validation.py"),
    )
    inputs = [
        *(config.raw_root / relative for relative in config.required_raw_files()),
        config.mapping_path,
        *source_files,
        *runner_files,
    ]
    metadata = [file_fingerprint(path) for path in inputs]
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def directory_metadata_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "total_size": 0, "digest": None}
    for root, directories, files in os.walk(path):
        directories.sort()
        files.sort()
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            stat = file_path.stat()
            relative = file_path.relative_to(path).as_posix()
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
            file_count += 1
            total_size += stat.st_size
    return {
        "path": str(path.resolve()),
        "exists": True,
        "file_count": file_count,
        "total_size": total_size,
        "digest": digest.hexdigest(),
    }


def _ensure_link(link: Path, target: Path, *, directory: bool) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"existing link {link} points to {link.resolve()}, expected {target}")
        return
    if link.exists():
        raise RuntimeError(f"input link path already exists and is not a symlink: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=directory)


def validate_source_inputs(config: ReproductionConfig) -> None:
    if not config.source_root.is_dir():
        raise FileNotFoundError(f"source project not found: {config.source_root}")
    missing = [
        str(config.raw_root / relative)
        for relative in config.required_raw_files()
        if not (config.raw_root / relative).is_file() or (config.raw_root / relative).stat().st_size == 0
    ]
    missing.extend(
        str(config.source_root / relative)
        for relative in config.required_source_files()
        if not (config.source_root / relative).is_file() or (config.source_root / relative).stat().st_size == 0
    )
    if not config.mapping_path.is_file() or config.mapping_path.stat().st_size == 0:
        missing.append(str(config.mapping_path))
    if missing:
        raise FileNotFoundError("missing or empty required inputs:\n" + "\n".join(missing))


def initialize_workspace(config: ReproductionConfig) -> RunLayout:
    validate_source_inputs(config)
    layout = RunLayout(config)
    disk_probe = layout.root
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"insufficient free disk space: {free_bytes} bytes available, {MINIMUM_FREE_BYTES} required"
        )
    if layout.root.exists() and any(layout.root.iterdir()) and not layout.metadata_path.is_file():
        raise RuntimeError(f"refusing to use unmanaged non-empty run root: {layout.root}")
    layout.root.mkdir(parents=True, exist_ok=True)
    for directory in (
        layout.cohort_dir,
        layout.features_dir,
        layout.intermediate_dir,
        layout.summary_dir,
        layout.csv_dir,
        layout.dict_dir,
        layout.logs_dir,
        layout.checkpoints_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _ensure_link(layout.raw_link, config.raw_root, directory=True)
    _ensure_link(layout.mapping_link, config.mapping_path, directory=False)
    metadata = {
        "owner": "mimic-reproduction-runner",
        "schema_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "input_fingerprint": input_fingerprint(config),
    }
    atomic_write_json(layout.metadata_path, metadata)
    current_source_snapshot = directory_metadata_fingerprint(config.source_root / "data")
    if layout.source_snapshot_path.is_file():
        previous_source_snapshot = json.loads(layout.source_snapshot_path.read_text(encoding="utf-8"))
        if previous_source_snapshot != current_source_snapshot:
            raise RuntimeError("original project data metadata changed since this reproduction run was initialized")
    else:
        atomic_write_json(layout.source_snapshot_path, current_source_snapshot)
    return layout


def _fingerprints(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [file_fingerprint(path) for path in paths if path.exists() and path.is_file()]


def write_checkpoint(
    layout: RunLayout,
    stage: str,
    config: ReproductionConfig,
    outputs: Iterable[Path],
    *,
    elapsed_seconds: float,
    details: dict[str, Any] | None = None,
) -> Path:
    output_paths = list(outputs)
    missing = [str(path) for path in output_paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"stage {stage} did not produce required outputs: {missing}")
    checkpoint = {
        "schema_version": 1,
        "stage": stage,
        "status": "success",
        "config_fingerprint": config_fingerprint(config),
        "input_fingerprint": input_fingerprint(config),
        "elapsed_seconds": elapsed_seconds,
        "outputs": _fingerprints(output_paths),
        "details": details or {},
    }
    path = layout.checkpoints_dir / f"{stage}.json"
    atomic_write_json(path, checkpoint)
    return path


def checkpoint_is_reusable(
    layout: RunLayout,
    stage: str,
    config: ReproductionConfig,
    outputs: Iterable[Path],
) -> bool:
    path = layout.checkpoints_dir / f"{stage}.json"
    if not path.is_file():
        return False
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if checkpoint.get("status") != "success" or checkpoint.get("stage") != stage:
        return False
    if checkpoint.get("config_fingerprint") != config_fingerprint(config):
        return False
    if checkpoint.get("input_fingerprint") != input_fingerprint(config):
        return False
    output_paths = list(outputs)
    if any(not item.is_file() or item.stat().st_size == 0 for item in output_paths):
        return False
    return checkpoint.get("outputs") == _fingerprints(output_paths)


def stage_outputs(layout: RunLayout, stage: str) -> list[Path]:
    if stage == "cohort":
        return [layout.cohort_path, layout.cohort_summary_path]
    if stage == "features":
        return [
            *(layout.features_dir / name for name in FEATURE_FILES if name != "preproc_diag_icu.csv.gz"),
            layout.intermediate_diag_path,
        ]
    if stage == "diagnosis":
        return [layout.features_dir / "preproc_diag_icu.csv.gz"]
    if stage == "summaries":
        return [layout.summary_dir / name for name in SUMMARY_FILES]
    if stage == "stay_generation":
        return [layout.labels_path, *(layout.dict_dir / name for name in layout.expected_dict_names())]
    if stage == "validation":
        return [layout.validation_path, layout.manifest_path]
    raise ValueError(f"unknown stage: {stage}")


def stage_worker_arguments(config: ReproductionConfig, stage: str) -> list[str]:
    worker = Path(__file__).with_name("mimic_stage_worker.py")
    return [
        sys.executable,
        str(worker),
        "--source-root",
        str(config.source_root),
        "--run-root",
        str(config.run_root),
        "--cohort-output",
        config.cohort_output,
        "--summary-output",
        config.summary_output,
        "--stage",
        stage,
    ]


def _run_child_stage(config: ReproductionConfig, layout: RunLayout, stage: str) -> float:
    started = time.monotonic()
    log_path = layout.logs_dir / f"{stage}.log"
    command = stage_worker_arguments(config, stage)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== stage={stage} command={json.dumps(command)} ===\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=layout.root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"stage {stage} failed with exit code {completed.returncode}; see {log_path}")
    return elapsed


def _run_validation(config: ReproductionConfig, layout: RunLayout) -> float:
    from reproduction.mimic_validation import validate_reproduction

    started = time.monotonic()
    report, manifest = validate_reproduction(config, layout)
    atomic_write_json(layout.validation_path, report)
    atomic_write_json(layout.manifest_path, manifest)
    if not report["passed"]:
        raise RuntimeError(f"validation failed; see {layout.validation_path}")
    return time.monotonic() - started


def _prepare_stage(layout: RunLayout, stage: str) -> None:
    if stage == "features":
        chunk_dir = layout.features_dir / "chartevents"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True)
    if stage == "stay_generation":
        for directory in (layout.csv_dir, layout.dict_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)


def _write_state(layout: RunLayout, payload: dict[str, Any]) -> None:
    atomic_write_json(layout.state_path, payload)


def run_reproduction(
    config: ReproductionConfig,
    *,
    force_stage: str | None = None,
    validate_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if force_stage is not None and force_stage not in STAGES:
        raise ValueError(f"force_stage must be one of: {', '.join(STAGES)}")
    validate_source_inputs(config)
    if dry_run:
        return {
            "status": "dry_run",
            "config": config.as_serializable(),
            "stages": ["validation"] if validate_only else list(STAGES),
            "commands": {
                stage: stage_worker_arguments(config, stage)
                for stage in STAGES
                if stage != "validation"
            },
        }

    layout = initialize_workspace(config)
    try:
        lock_fd = os.open(layout.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another reproduction process is using {layout.root}") from exc
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)

    state: dict[str, Any] = {
        "status": "running",
        "config_fingerprint": config_fingerprint(config),
        "stages": {},
    }
    _write_state(layout, state)
    selected_stages = ["validation"] if validate_only else list(STAGES)
    force_index = STAGES.index(force_stage) if force_stage else None
    try:
        for stage in selected_stages:
            outputs = stage_outputs(layout, stage)
            stage_index = STAGES.index(stage)
            reusable = checkpoint_is_reusable(layout, stage, config, outputs)
            if force_index is not None and stage_index >= force_index:
                reusable = False
            if reusable:
                state["stages"][stage] = {"status": "skipped", "reason": "valid_checkpoint"}
                _write_state(layout, state)
                continue
            state["stages"][stage] = {"status": "running"}
            _write_state(layout, state)
            _prepare_stage(layout, stage)
            elapsed = _run_validation(config, layout) if stage == "validation" else _run_child_stage(config, layout, stage)
            details: dict[str, Any] = {}
            if stage == "stay_generation":
                details["stay_directory_count"] = sum(
                    1 for path in layout.csv_dir.iterdir() if path.is_dir() and path.name.isdigit()
                )
            write_checkpoint(layout, stage, config, outputs, elapsed_seconds=elapsed, details=details)
            state["stages"][stage] = {"status": "success", "elapsed_seconds": elapsed}
            _write_state(layout, state)
        state["status"] = "success"
        _write_state(layout, state)
        return state
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        _write_state(layout, state)
        raise
    finally:
        layout.lock_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce MIMIC-IV 3.1 ICU mortality data through Block 7")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--force-stage", choices=STAGES)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ReproductionConfig(source_root=args.source_root, run_root=args.run_root)
    result = run_reproduction(
        config,
        force_stage=args.force_stage,
        validate_only=args.validate_only,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
