from __future__ import annotations

import json
import shutil
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from workflow.mimic_pipeline import split_teacher_reference


TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}


@dataclass(frozen=True)
class ReferenceSplitConfig:
    raw_root: str | Path
    reference: str | Path
    split_output: str | Path
    reference_root: str | Path | None = None
    task_text: str = ""
    counts: tuple[int, int, int | None] = (10, 100, None)
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    seed: int = 42
    key_column: str = ""
    raw_mode: str = "copy"
    decompress_gzip: bool = False
    case_table_copy_limit: int = 200
    feature_table_copy_limit: int = 1_000

    def normalized(self) -> "ReferenceSplitConfig":
        raw = Path(self.raw_root).expanduser().resolve()
        if not raw.exists():
            raise ValueError(f"raw_root does not exist: {raw}")
        reference = Path(self.reference).expanduser().resolve() if str(self.reference or "").strip() else Path()
        reference_root = (
            Path(self.reference_root).expanduser().resolve()
            if self.reference_root and str(self.reference_root).strip()
            else None
        )
        if reference_root is None and not str(self.reference or "").strip():
            raise ValueError("Either reference or reference_root must be provided")
        if reference_root is not None and not reference_root.exists():
            raise ValueError(f"reference_root does not exist: {reference_root}")
        if reference_root is None and not reference.exists():
            raise ValueError(f"reference does not exist: {reference}")
        if self.raw_mode not in {"copy", "symlink"}:
            raise ValueError("raw_mode must be copy or symlink")
        return ReferenceSplitConfig(
            raw_root=raw,
            reference=reference,
            split_output=Path(self.split_output).expanduser().resolve(),
            reference_root=reference_root,
            task_text=self.task_text,
            counts=self.counts,
            ratios=self.ratios,
            seed=self.seed,
            key_column=self.key_column,
            raw_mode=self.raw_mode,
            decompress_gzip=self.decompress_gzip,
            case_table_copy_limit=self.case_table_copy_limit,
            feature_table_copy_limit=self.feature_table_copy_limit,
        )


def prepare_reference_splits(config: ReferenceSplitConfig) -> dict[str, Any]:
    cfg = config.normalized()
    output = Path(cfg.split_output)
    output.mkdir(parents=True, exist_ok=True)
    reference_package = (
        discover_reference_package(cfg.reference_root, key_column=cfg.key_column)
        if cfg.reference_root
        else None
    )
    primary_reference = Path(reference_package["primary_reference"]) if reference_package else Path(cfg.reference)
    work = output / ".reference_split_work"
    if work.exists():
        shutil.rmtree(work)
    reference_split = split_teacher_reference(
        primary_reference,
        work,
        ratios=cfg.ratios,
        counts=cfg.counts,
        seed=cfg.seed,
        key_column=cfg.key_column,
    )
    reference_manifest = reference_split["manifest"]
    split_key = str(reference_manifest["split_key"])
    raw_root = Path(cfg.raw_root)
    dataset_profile = infer_dataset_profile(raw_root)
    records: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        split_dir = output / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        raw_dir = split_dir / "raw"
        reference_dir = split_dir / ("reference" if split_name == "train" else "reference_private")
        raw_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_source = Path(reference_split[f"{split_name}_reference"])
        key_source = Path(reference_split[f"{split_name}_keys"])
        reference_target = reference_dir / "reference.csv"
        keys_target = split_dir / "keys.csv"
        shutil.copy2(reference_source, reference_target)
        shutil.copy2(key_source, keys_target)
        if reference_package:
                package_record = materialize_reference_package_split(
                    reference_package,
                    reference_dir,
                    keys_target,
                    split_key=split_key,
                    split_name=split_name,
                    decompress_gzip=cfg.decompress_gzip,
                    case_table_copy_limit=cfg.case_table_copy_limit,
                    feature_table_copy_limit=cfg.feature_table_copy_limit,
                )
        else:
            package_record = None
        keys_frame = pd.read_csv(keys_target, dtype=object)
        key_count = int(keys_frame[split_key].nunique()) if split_key in keys_frame.columns else 0
        if key_count == 0:
            raw_record = {"mode": "empty_split", "file_count": 0, "selected_counts": {split_key: 0}, "files": []}
        elif cfg.raw_mode == "symlink":
            raw_record = link_raw_root(raw_root, raw_dir)
        elif dataset_profile == "mimic_iv_3_1":
            raw_record = copy_mimic_raw_subset(
                raw_root,
                raw_dir,
                keys_target,
                split_key=split_key,
                decompress_gzip=cfg.decompress_gzip,
            )
        else:
            raw_record = copy_generic_raw_subset(
                raw_root,
                raw_dir,
                keys_target,
                split_key=split_key,
                decompress_gzip=cfg.decompress_gzip,
            )
        record = {
            "schema_version": 1,
            "split": split_name,
            "dataset_profile": dataset_profile,
            "split_key": split_key,
            "raw_root": str(raw_dir),
            "reference_root": str(reference_dir),
            "keys_path": str(keys_target),
            "reference_path": str(reference_target),
            "key_count": key_count,
            "raw_record": raw_record,
            "reference_package": package_record,
        }
        _write_json(split_dir / "split_record.json", record)
        records[split_name] = record
    if work.exists():
        shutil.rmtree(work)
    manifest = {
        "schema_version": 1,
        "status": "SUCCESS",
        "task_text": cfg.task_text,
        "dataset_profile": dataset_profile,
        "source_raw_root": str(raw_root),
        "source_reference": str(primary_reference),
        "source_reference_root": str(cfg.reference_root or ""),
        "raw_mode": cfg.raw_mode,
        "decompress_gzip": cfg.decompress_gzip,
        "reference_mode": "package" if reference_package else "single_table",
        "reference_package": reference_package,
        "split_key": split_key,
        "split_mode": reference_manifest.get("split_mode"),
        "requested_counts": reference_manifest.get("requested_counts"),
        "row_counts": reference_manifest.get("row_counts"),
        "splits": {
            name: {
                "raw": records[name]["raw_root"],
                "reference": records[name]["reference_root"],
                "keys": records[name]["keys_path"],
                "record": str(output / name / "split_record.json"),
            }
            for name in records
        },
    }
    manifest_path = output / "split_manifest.json"
    _write_json(manifest_path, manifest)
    return {"status": "SUCCESS", "split_output": str(output), "split_manifest": str(manifest_path), "manifest": manifest}


def infer_dataset_profile(raw_root: str | Path) -> str:
    root = Path(raw_root)
    if (root / "hosp").is_dir() or (root / "icu").is_dir():
        return "mimic_iv_3_1"
    return "generic"


def discover_reference_package(reference_root: str | Path, *, key_column: str = "") -> dict[str, Any]:
    root = Path(reference_root).expanduser().resolve()
    metadata = _read_optional_json(root / "metadata.json")
    primary = _resolve_package_path(root, metadata.get("primary_reference") or "")
    cohort_files = [_resolve_package_path(root, value) for value in metadata.get("cohort_files", [])]
    feature_files = [_resolve_package_path(root, value) for value in metadata.get("feature_files", [])]
    case_roots = [_resolve_package_path(root, value) for value in metadata.get("case_table_roots", [])]

    if not cohort_files:
        cohort_files = _reference_files_under(root / "cohort")
    if not feature_files:
        feature_files = _reference_files_under(root / "features")
    if primary is None:
        primary = _choose_primary_reference(cohort_files or _reference_files_under(root), key_column=key_column)
    if primary is None:
        raise ValueError(f"Could not infer a primary reference table under: {root}")
    if primary not in cohort_files and (root / "cohort") in primary.parents:
        cohort_files.insert(0, primary)
    return {
        "schema_version": 1,
        "reference_root": str(root),
        "primary_reference": str(primary),
        "cohort_files": [str(path) for path in cohort_files if path and path.exists()],
        "feature_files": [str(path) for path in feature_files if path and path.exists()],
        "case_table_roots": [str(path) for path in case_roots if path and path.exists()],
        "metadata": metadata,
    }


def materialize_reference_package_split(
    package: dict[str, Any],
    reference_dir: str | Path,
    keys_path: str | Path,
    *,
    split_key: str,
    split_name: str,
    decompress_gzip: bool = False,
    case_table_copy_limit: int = 200,
    feature_table_copy_limit: int = 1_000,
) -> dict[str, Any]:
    root = Path(str(package["reference_root"])).expanduser().resolve()
    target = Path(reference_dir).expanduser().resolve()
    keys = pd.read_csv(keys_path, dtype=object)
    allowed = set(keys[split_key].dropna().astype(str)) if split_key in keys.columns else set()
    copied: list[dict[str, Any]] = []
    for kind, files in (("cohort", package.get("cohort_files") or []), ("features", package.get("feature_files") or [])):
        for raw_file in files:
            source = Path(raw_file).expanduser().resolve()
            if not source.is_file():
                continue
            relative = _package_relative(root, source)
            destination = target / _maybe_uncompressed_relative(relative, decompress_gzip=decompress_gzip)
            if kind == "features" and len(allowed) > feature_table_copy_limit:
                copied.append(
                    {
                        "kind": kind,
                        "source": str(source),
                        "path": str(destination),
                        "mode": "skipped_too_many_cases",
                        "case_count": len(allowed),
                        "copy_limit": feature_table_copy_limit,
                    }
                )
                continue
            row_count = copy_reference_table_for_keys(
                source,
                destination,
                split_key=split_key,
                allowed=allowed,
                decompress_gzip=decompress_gzip,
            )
            copied.append({"kind": kind, "source": str(source), "path": str(destination), "mode": "filtered_or_copied", "row_count": row_count})
    case_tables = []
    for raw_root in package.get("case_table_roots") or []:
        source_root = Path(raw_root).expanduser().resolve()
        destination_root = target / "case_tables" / source_root.name
        if len(allowed) > case_table_copy_limit:
            case_tables.append(
                {
                    "source_root": str(source_root),
                    "path": str(destination_root),
                    "mode": "skipped_too_many_cases",
                    "case_count": len(allowed),
                    "copy_limit": case_table_copy_limit,
                }
            )
            continue
        copied_count = 0
        for key in sorted(allowed):
            source_case = source_root / key
            if not source_case.is_dir():
                continue
            destination_case = destination_root / key
            destination_case.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_case, destination_case)
            copied_count += 1
        case_tables.append(
            {
                "source_root": str(source_root),
                "path": str(destination_root),
                "mode": "copied_case_dirs",
                "case_count": copied_count,
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "SUCCESS",
        "split": split_name,
        "split_key": split_key,
        "key_count": len(allowed),
        "source_reference_root": str(root),
        "primary_reference": str(package.get("primary_reference") or ""),
        "cohort_files": [item for item in copied if item["kind"] == "cohort"],
        "feature_files": [item for item in copied if item["kind"] == "features"],
        "case_tables": case_tables,
    }
    _write_json(target / "package_manifest.json", manifest)
    return manifest


def copy_reference_table_for_keys(
    source: str | Path,
    destination: str | Path,
    *,
    split_key: str,
    allowed: set[str],
    decompress_gzip: bool = False,
) -> int:
    src = Path(source)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not _is_tabular_file(src):
        _copy_or_decompress(src, dst, decompress_gzip=decompress_gzip)
        return -1
    header = _read_header(src)
    if split_key not in header:
        _copy_or_decompress(src, dst, decompress_gzip=decompress_gzip)
        return -1
    return _filter_tabular_file(src, dst, column=split_key, allowed=allowed, chunksize=250_000)


def link_raw_root(raw_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    source = Path(raw_root).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    links = []
    children = [child for child in source.iterdir() if child.name in {"hosp", "icu", "note", "notes"}]
    if not children:
        children = [source]
    for child in children:
        destination = target / child.name if child != source else target / "source"
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.symlink_to(child, target_is_directory=child.is_dir())
        links.append({"path": str(destination), "source": str(child)})
    return {"mode": "symlink", "source_raw_root": str(source), "links": links}


def copy_mimic_raw_subset(
    raw_root: str | Path,
    output_root: str | Path,
    keys_path: str | Path,
    *,
    split_key: str,
    decompress_gzip: bool = False,
    chunksize: int = 250_000,
    include_relative_paths: set[str] | None = None,
) -> dict[str, Any]:
    source = Path(raw_root).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    selected = _selected_mimic_ids(source, keys_path, split_key)
    normalized_include = {value.replace("\\", "/") for value in include_relative_paths or set()}
    files = [
        path
        for folder in ("hosp", "icu")
        for path in sorted((source / folder).rglob("*"))
        if path.is_file()
        and (not normalized_include or path.relative_to(source).as_posix() in normalized_include)
    ]
    if normalized_include:
        found = {path.relative_to(source).as_posix() for path in files}
        missing = sorted(normalized_include - found)
        if missing:
            raise ValueError(f"requested MIMIC raw tables are missing: {missing}")
    outputs = []
    for path in files:
        relative = path.relative_to(source)
        destination = target / _maybe_uncompressed_relative(relative, decompress_gzip=decompress_gzip)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not _is_tabular_file(path):
            _copy_or_decompress(path, destination, decompress_gzip=decompress_gzip)
            outputs.append({"path": str(destination), "mode": "copied_non_tabular"})
            continue
        header = _read_header(path)
        filter_column = _mimic_filter_column(header, selected)
        if filter_column is None:
            if not _should_copy_unkeyed_mimic_file(relative):
                outputs.append(
                    {
                        "path": str(destination),
                        "mode": "skipped_unfilterable_unkeyed",
                        "reason": "table has no subject_id/hadm_id/stay_id and is not a lookup dictionary",
                    }
                )
                continue
            _copy_or_decompress(path, destination, decompress_gzip=decompress_gzip)
            outputs.append({"path": str(destination), "mode": "copied_lookup_dictionary", "row_count": -1})
            continue
        row_count = _filter_tabular_file(
            path,
            destination,
            column=filter_column,
            allowed=selected[filter_column],
            chunksize=chunksize,
        )
        outputs.append({"path": str(destination), "mode": "filtered", "filter_column": filter_column, "row_count": row_count})
    return {"file_count": len(outputs), "selected_counts": {key: len(value) for key, value in selected.items()}, "files": outputs}


def copy_generic_raw_subset(
    raw_root: str | Path,
    output_root: str | Path,
    keys_path: str | Path,
    *,
    split_key: str,
    decompress_gzip: bool = False,
    chunksize: int = 250_000,
) -> dict[str, Any]:
    source = Path(raw_root).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    keys = set(pd.read_csv(keys_path, dtype=object)[split_key].dropna().astype(str))
    outputs = []
    case_dirs = [child for child in source.iterdir() if child.is_dir() and child.name in keys]
    if case_dirs:
        for child in case_dirs:
            destination = target / child.name
            shutil.copytree(child, destination)
            outputs.append({"path": str(destination), "mode": "copied_case_dir"})
        return {"file_count": len(outputs), "selected_counts": {split_key: len(keys)}, "files": outputs}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        destination = target / _maybe_uncompressed_relative(relative, decompress_gzip=decompress_gzip)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _is_tabular_file(path):
            header = _read_header(path)
            if split_key in header:
                row_count = _filter_tabular_file(path, destination, column=split_key, allowed=keys, chunksize=chunksize)
                outputs.append({"path": str(destination), "mode": "filtered", "filter_column": split_key, "row_count": row_count})
                continue
        _copy_or_decompress(path, destination, decompress_gzip=decompress_gzip)
        outputs.append({"path": str(destination), "mode": "copied_unkeyed"})
    return {"file_count": len(outputs), "selected_counts": {split_key: len(keys)}, "files": outputs}


def _selected_mimic_ids(raw_root: Path, keys_path: str | Path, split_key: str) -> dict[str, set[str]]:
    key_frame = pd.read_csv(keys_path, dtype=object)
    requested = set(key_frame[split_key].dropna().astype(str)) if split_key in key_frame.columns else set()
    selected: dict[str, set[str]] = {"subject_id": set(), "hadm_id": set(), "stay_id": set()}
    admissions = _read_existing_columns(raw_root / "hosp" / "admissions.csv", ["subject_id", "hadm_id"])
    icu = _read_existing_columns(raw_root / "icu" / "icustays.csv", ["subject_id", "hadm_id", "stay_id"])
    if split_key == "stay_id" and not icu.empty:
        matched = icu[icu["stay_id"].astype(str).isin(requested)]
        _add_ids(selected, matched)
        if selected["hadm_id"] and not admissions.empty:
            matched = admissions[admissions["hadm_id"].astype(str).isin(selected["hadm_id"])]
            _add_ids(selected, matched)
        return selected
    if split_key == "hadm_id":
        selected["hadm_id"].update(requested)
        if selected["hadm_id"] and not admissions.empty:
            matched = admissions[admissions["hadm_id"].astype(str).isin(selected["hadm_id"])]
            _add_ids(selected, matched)
        if selected["hadm_id"] and not icu.empty and "hadm_id" in icu.columns:
            matched = icu[icu["hadm_id"].astype(str).isin(selected["hadm_id"])]
            _add_ids(selected, matched)
        return selected
    if split_key == "subject_id":
        selected["subject_id"].update(requested)
        if selected["subject_id"] and not admissions.empty:
            matched = admissions[admissions["subject_id"].astype(str).isin(selected["subject_id"])]
            _add_ids(selected, matched)
        if selected["subject_id"] and not icu.empty and "subject_id" in icu.columns:
            matched = icu[icu["subject_id"].astype(str).isin(selected["subject_id"])]
            _add_ids(selected, matched)
        return selected
    if split_key in selected:
        selected[split_key].update(requested)
    return selected


def _add_ids(selected: dict[str, set[str]], frame: pd.DataFrame) -> None:
    for column in selected:
        if column in frame.columns:
            selected[column].update(frame[column].dropna().astype(str))


def _mimic_filter_column(header: list[str], selected: dict[str, set[str]]) -> str | None:
    for column in ("stay_id", "hadm_id", "subject_id"):
        if column in header and selected.get(column):
            return column
    return None


def _should_copy_unkeyed_mimic_file(relative: Path) -> bool:
    name = relative.name.lower()
    return name.startswith("d_")


def _read_existing_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    path = _resolve_existing_tabular_path(path)
    if not path.exists():
        return pd.DataFrame()
    header = _read_header(path)
    usecols = [column for column in columns if column in header]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=object, usecols=usecols)


def _resolve_existing_tabular_path(path: Path) -> Path:
    if path.exists():
        return path
    gzip_path = path.with_suffix(path.suffix + ".gz")
    if gzip_path.exists():
        return gzip_path
    return path


def _maybe_uncompressed_relative(relative: Path, *, decompress_gzip: bool) -> Path:
    if decompress_gzip and relative.name.lower().endswith(".gz"):
        return relative.with_name(relative.name[:-3])
    return relative


def _copy_or_decompress(source: Path, destination: Path, *, decompress_gzip: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if decompress_gzip and source.name.lower().endswith(".gz"):
        with gzip.open(source, "rb") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return
    shutil.copy2(source, destination)


def _is_tabular_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in (".csv", ".csv.gz", ".tsv", ".tsv.gz"))


def _is_reference_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"))


def _reference_files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if _is_reference_file(root) else []
    return sorted(path for path in root.rglob("*") if path.is_file() and _is_reference_file(path))


def _choose_primary_reference(files: list[Path], *, key_column: str = "") -> Path | None:
    preferred = sorted(files, key=lambda path: (0 if "cohort" in path.name.lower() else 1, len(path.parts), path.name))
    for path in preferred:
        try:
            header = _read_header(path)
        except Exception:
            continue
        if key_column and key_column in header:
            return path
        if any(column in header for column in ("stay_id", "hadm_id", "subject_id", "case_id")) and any(
            column in header for column in ("label", "mortality", "readmission", "outcome")
        ):
            return path
    for path in preferred:
        try:
            header = _read_header(path)
        except Exception:
            continue
        if any(column in header for column in ("stay_id", "hadm_id", "subject_id", "case_id")):
            return path
    return preferred[0] if preferred else None


def _resolve_package_path(root: Path, value: str | Path) -> Path | None:
    if not str(value or "").strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _package_relative(root: Path, source: Path) -> Path:
    try:
        return source.relative_to(root)
    except ValueError:
        return Path(source.name)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _read_header(path: Path) -> list[str]:
    sep = "\t" if path.name.lower().endswith((".tsv", ".tsv.gz")) else ","
    return list(pd.read_csv(path, nrows=0, sep=sep).columns)


def _filter_tabular_file(
    source: Path,
    destination: Path,
    *,
    column: str,
    allowed: set[str],
    chunksize: int,
) -> int:
    sep = "\t" if source.name.lower().endswith((".tsv", ".tsv.gz")) else ","
    wrote_header = False
    row_count = 0
    for chunk in pd.read_csv(source, dtype=object, sep=sep, chunksize=chunksize):
        filtered = chunk[chunk[column].astype(str).isin(allowed)]
        if filtered.empty:
            continue
        filtered.to_csv(destination, sep=sep, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        row_count += len(filtered)
    if not wrote_header:
        header = pd.read_csv(source, nrows=0, sep=sep)
        header.to_csv(destination, sep=sep, index=False)
    return row_count


def _safe_row_count(path: Path) -> int:
    try:
        sep = "\t" if path.name.lower().endswith((".tsv", ".tsv.gz")) else ","
        return int(sum(len(chunk) for chunk in pd.read_csv(path, sep=sep, chunksize=250_000)))
    except Exception:
        return -1


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
