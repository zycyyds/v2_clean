"""把 4 个 v2 rawerror 数据集重建到 datasets/注入错误数据集/{name}/

每个数据集：
  - baseline raw (decompressed) → datasets/注入错误数据集/{name}/{split}/raw/
  - baseline reference → datasets/注入错误数据集/{name}/{split}/reference/
  - baseline keys.csv → datasets/注入错误数据集/{name}/{split}/keys.csv
  - modification_log.csv → datasets/注入错误数据集/{name}/{split}/modification_log.csv

行为完全等价于之前的 rebuild_*.py（同一 seed → 同一错误集）。
"""
from __future__ import annotations

import csv
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
sys.path.insert(0, str(ROOT / "reproduction"))

from build_v2_error_configs import build_modification_log, validate_stats
from apply_modification_log import apply_modification_log

LOGS_DIR = Path("/tmp/v2_logs")
DST_ROOT = ROOT / "datasets" / "注入错误数据集"

# (new_name, old_baseline, split_key)
DATASETS = [
    ("K29_clean_v2_rawerror_10train_400_800_1800",          "K29_clean_eval_10_20_rest",                 "hadm_id"),
    ("icu_5000_v2_rawerror_10train_400_800_1800",          "icu_5000_clean_eval_10_20_rest",            "stay_id"),
    ("N18_ckd_ED_v2_rawerror_10train_400_800_1800",        "N18_ckd_ED_notes_eval_10_20_rest",          "subject_id"),
    ("I48_atrial_fibrillation_v2_rawerror_10train_400_800_1800", "I48_atrial_fibrillation_ECG_eval_10_20_rest", "subject_id"),
]

SPLIT_ERROR_COUNTS = {"train": 400, "validation": 800, "test": 1800}


def open_csv_gz_or_csv(path: Path, mode: str = "r"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def read_keys(keys_csv: Path) -> list[str]:
    with open(keys_csv) as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader if row]


def decompress_copytree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_dir():
            decompress_copytree(p, dst / p.name)
        elif p.is_file():
            if p.name.endswith(".gz"):
                dst_name = p.name[:-3]
                with open_csv_gz_or_csv(p) as fin, open(dst / dst_name, "w", encoding="utf-8", newline="") as fout:
                    fout.write(fin.read())
            else:
                shutil.copy2(p, dst / p.name)


def build_one_split(*, new_root: Path, old_root: Path, new_name: str, split_name: str,
                    split_key: str, target_errors: int) -> dict:
    print(f"\n  === {split_name} (target {target_errors} errors) ===", flush=True)
    t0 = time.time()

    # icu_5000 baseline 用 val/ 不是 validation/；常规数据集都用 validation/
    baseline_split_dir = old_root / ("val" if split_name == "validation" and (old_root / "val").is_dir() else split_name)

    # 1. baseline keys
    sel_keys = read_keys(baseline_split_dir / "keys.csv")
    print(f"    baseline keys: {len(sel_keys)}", flush=True)

    # 2. raw
    raw_src = baseline_split_dir / "raw"
    raw_dst = new_root / split_name / "raw"
    if raw_dst.exists():
        shutil.rmtree(raw_dst)
    decompress_copytree(raw_src, raw_dst)
    print(f"    copied raw/ (decompressed)", flush=True)

    # 3. reference
    ref_src = baseline_split_dir / "reference"
    ref_dst = new_root / split_name / "reference"
    if ref_dst.exists():
        shutil.rmtree(ref_dst)
    if ref_src.is_dir():
        shutil.copytree(ref_src, ref_dst)
    print(f"    copied reference/", flush=True)

    # 4. keys.csv
    keys_dst = new_root / split_name / "keys.csv"
    with open(keys_dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([split_key])
        for k in sel_keys:
            w.writerow([k])

    # 5. subject_id_lookup
    subject_id_lookup = None
    if split_key == "subject_id":
        ids = set(sel_keys)
        for sub_file in [raw_dst / "hosp" / "admissions.csv", raw_dst / "icu" / "icustays.csv"]:
            if sub_file.is_file():
                with open(sub_file, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for c in ("subject_id", "hadm_id", "stay_id"):
                            v = row.get(c)
                            if v:
                                ids.add(v)
        subject_id_lookup = ids
        print(f"    derived {len(subject_id_lookup)} ids (subject+hadm+stay)", flush=True)
    elif split_key == "stay_id":
        icu_path = raw_dst / "icu" / "icustays.csv"
        if icu_path.is_file():
            subject_ids, stay_ids, hadm_ids = set(), set(), set()
            with open(icu_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("subject_id"):
                        subject_ids.add(row["subject_id"])
                    if row.get("hadm_id"):
                        hadm_ids.add(row["hadm_id"])
                    if row.get("stay_id"):
                        stay_ids.add(row["stay_id"])
            subject_id_lookup = subject_ids | stay_ids | hadm_ids
            print(f"    derived {len(subject_id_lookup)} ids (subject+hadm+stay from icustays)", flush=True)

    # 6. Build modification log
    log_path = LOGS_DIR / f"{new_name}_{split_name}_modification_log.csv"
    seed = hash((new_name, split_name)) & 0xFFFFFFFF
    stats = build_modification_log(
        split_name=split_name,
        target_count=target_errors,
        baseline_raw_dir=raw_dst,
        output_log_path=log_path,
        split_key=split_key,
        all_keys=sel_keys,
        random_seed=seed,
        subject_id_lookup=subject_id_lookup,
    )
    print(f"    mod_log: {stats['actual_count']} rows, {stats['n_subtypes_covered']}/18 subtypes, "
          f"{stats['n_files_touched']} files, {stats['n_stays_touched']} stays", flush=True)

    # 7. validate
    passed, failures = validate_stats(stats, split_name=split_name)
    if not passed:
        print(f"    [FAIL] validation: {failures}", flush=True)
    else:
        print(f"    [PASS] validation", flush=True)

    # 8. inject errors
    apply_stats = apply_modification_log(raw_dst, log_path)
    print(f"    applied: cell_updates={apply_stats['n_cell_updates']}, "
          f"duplicates={apply_stats['n_duplicates']}, "
          f"inserts={apply_stats['n_inserts']}, "
          f"skipped={apply_stats['n_skipped']}", flush=True)

    # 9. copy modification_log into dataset dir
    shutil.copy2(log_path, new_root / split_name / "modification_log.csv")

    print(f"    [done] {time.time() - t0:.1f}s", flush=True)
    return {
        "target_errors": target_errors,
        "actual_errors": stats["actual_count"],
        "n_baseline_keys": len(sel_keys),
        "subtype_coverage": stats["n_subtypes_covered"],
        "validation_passed": passed,
        "applied": apply_stats,
        "log_path": str(log_path),
    }


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    report_all: list[dict] = []
    for new_name, old_baseline, split_key in DATASETS:
        new_root = DST_ROOT / new_name
        old_root = ROOT / "datasets" / "常规数据集" / old_baseline
        if not old_root.is_dir():
            print(f"[ERROR] baseline not found: {old_root}")
            sys.exit(1)

        if new_root.exists():
            print(f"\n>>> Removing stale: {new_root}")
            shutil.rmtree(new_root)

        print(f"\n{'#' * 80}")
        print(f"Building: {new_name}")
        print(f"  baseline: datasets/常规数据集/{old_baseline}")
        print(f"  target:   {DST_ROOT.relative_to(ROOT)}/{new_name}")
        print(f"  split_key: {split_key}")
        print(f"{'#' * 80}")

        ds_report = {"name": new_name, "old_baseline": f"常规数据集/{old_baseline}", "splits": {}}
        for split_name, target in SPLIT_ERROR_COUNTS.items():
            try:
                ds_report["splits"][split_name] = build_one_split(
                    new_root=new_root, old_root=old_root, new_name=new_name,
                    split_name=split_name, split_key=split_key, target_errors=target,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                ds_report["splits"][split_name] = {"error": str(e)}
        report_all.append(ds_report)

    # 也写到 root datasets/v2_rawerror_validation_report.json（保持兼容）
    out = ROOT / "datasets" / "v2_rawerror_validation_report.json"
    existing = []
    if out.is_file():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            pass
    existing_names = {r.get("name") for r in existing}
    for r in report_all:
        if r["name"] in existing_names:
            existing = [x for x in existing if x.get("name") != r["name"]]
        existing.append(r)
    out.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nReport updated: {out}")

    # 验证最终目录结构
    print(f"\n{'#' * 80}")
    print("Final structure under datasets/注入错误数据集/:")
    print(f"{'#' * 80}")
    for new_name, _, _ in DATASETS:
        ds_path = DST_ROOT / new_name
        if ds_path.is_dir():
            n_errors = sum(1 for _ in (ds_path / "test" / "modification_log.csv").open()) - 1
            sub_dirs = sorted(p.name for p in (ds_path / "test" / "raw").iterdir() if p.is_dir())
            print(f"  {new_name}/")
            for sp in SPLIT_ERROR_COUNTS:
                n = sum(1 for _ in (ds_path / sp / "modification_log.csv").open()) - 1
                print(f"    {sp}/: {n} errors injected; log at modification_log.csv")
            print(f"    test/raw subdirs: {sub_dirs}")


if __name__ == "__main__":
    main()