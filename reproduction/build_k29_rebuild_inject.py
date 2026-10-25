"""K29 v2 rawerror 重建 driver：保留原 keys 划分，加大错误注入量。

目的：在原 K29_clean_v2_rawerror_10train_400_800_1800 数据集上重新注入
- train: 400 → 5,000 cell_update (10 hadm_id)
- validation: 800 → 7,000 cell_update (20 hadm_id)
- test: 1,800 → 26,500,000 cell_update (8092 hadm_id, cell-level 10%)

保持原 keys 划分（来自 K29_clean_v2_rawerror_10train_400_800_1800/{split}/keys.csv）。
删除原 raw/ 和 modification_log.csv，从 K29_clean_eval_10_20_rest baseline 重建 raw 并重新注入。
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
sys.path.insert(0, str(ROOT / "reproduction"))

from build_v2_error_configs import build_modification_log, validate_stats
from apply_modification_log import apply_modification_log
from build_all_v2_rawerror_datasets import (
    LOGS_DIR, read_keys, filter_reference_for_keys,
    build_split_raw_from_test, open_csv, HOSP_KEY_COLS, ICU_KEY_COLS,
)

# ============== 配置 ==============
BASELINE_ROOT = ROOT / "datasets/常规数据集/K29_clean_eval_10_20_rest"
TARGET_ROOT = ROOT / "datasets/注入错误数据集/K29_clean_v2_rawerror_10train_400_800_1800"
SPLIT_KEY = "hadm_id"

# baseline split 目录名 → target split 目录名
# baseline: train/val/test, target: train/validation/test
BASELINE_SPLIT_DIR = {"train": "train", "validation": "val", "test": "test"}

# 重建目标
SPLIT_ERROR_COUNTS = {
    "train": 3_500,          # 10 hadm_id (baseline K29 train 288k cells → ~3.5k)
    "validation": 7_000,     # 20 hadm_id
    "test": 26_500_000,      # 8092 hadm_id → 实际 10% cell-level
}


def apply_per_file_logs(raw_dir: Path, log_path: Path) -> dict:
    """按 raw_file 拆 log，逐文件调 apply_modification_log。避免一次性 load OOM。"""
    log = pd.read_csv(log_path, dtype=str)
    print(f"    [apply] {len(log)} entries across {log['raw_file'].nunique()} files", flush=True)
    total_updates = total_dups = total_inserts = total_skipped = 0
    file_stats = {}
    for raw_file, sub_log in log.groupby("raw_file", sort=False):
        sub_log_path = log_path.parent / f"{log_path.stem}__{raw_file.replace('/', '_')}.csv"
        sub_log.to_csv(sub_log_path, index=False)
        try:
            s = apply_modification_log(raw_dir, sub_log_path)
            total_updates += s.get("n_cell_updates", 0)
            total_dups += s.get("n_duplicates", 0)
            total_inserts += s.get("n_inserts", 0)
            total_skipped += s.get("n_skipped", 0)
            file_stats[raw_file] = s.get("n_cell_updates", 0)
            print(f"    [apply] {raw_file:<35} {len(sub_log):>9} entries "
                  f"(+{s.get('n_cell_updates', 0)} updates, "
                  f"+{s.get('n_duplicates', 0)} dups, "
                  f"+{s.get('n_inserts', 0)} inserts)", flush=True)
        finally:
            if sub_log_path.exists():
                sub_log_path.unlink()
    return {
        "n_cell_updates": total_updates, "n_duplicates": total_dups,
        "n_inserts": total_inserts, "n_skipped": total_skipped, "by_file": file_stats,
    }


def rebuild_split(split_name: str, error_count: int) -> dict:
    """重建单个 split 的 raw + 注入错误。"""
    print(f"\n  --- {split_name} (target {error_count:,}) ---", flush=True)
    sel_keys = read_keys(TARGET_ROOT / split_name / "keys.csv")
    print(f"    [keys] {len(sel_keys)} (preserved from existing)", flush=True)
    sel_set = set(sel_keys)

    # 删旧 raw (保留 keys.csv + reference/ + modification_log.csv)
    raw_dst = TARGET_ROOT / split_name / "raw"
    if raw_dst.is_dir():
        import shutil
        shutil.rmtree(raw_dst)
        print(f"    [clean] removed old raw/", flush=True)

    # 从 baseline 重建 raw（直接复制 baseline 对应 split 的 raw/）
    baseline_split_dir = BASELINE_ROOT / BASELINE_SPLIT_DIR[split_name] / "raw"
    if not baseline_split_dir.is_dir():
        print(f"    [ERROR] baseline {baseline_split_dir} not found", flush=True)
        sys.exit(1)
    if split_name == "test":
        from build_all_v2_rawerror_datasets import decompress_copytree
        decompress_copytree(baseline_split_dir, raw_dst)
        print(f"    [raw] copied from baseline/test/raw", flush=True)
    else:
        # train/validation: baseline 已有对应 split 的 raw/，直接复制
        import shutil
        if baseline_split_dir.is_dir():
            shutil.copytree(baseline_split_dir, raw_dst)
            n_files = sum(1 for _ in raw_dst.rglob("*.csv"))
            print(f"    [raw] copied from baseline/{BASELINE_SPLIT_DIR[split_name]}/raw ({n_files} files)", flush=True)
        else:
            # fallback: 从 test/raw filter
            test_raw = BASELINE_ROOT / "test" / "raw"
            extended = set()
            if SPLIT_KEY in ("stay_id", "hadm_id"):
                icu_baseline = test_raw / "icu" / "icustays.csv"
                if not icu_baseline.is_file():
                    icu_baseline = test_raw / "icu" / "icustays.csv.gz"
                if icu_baseline.is_file():
                    with open_csv(icu_baseline) as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        si = header.index("subject_id") if "subject_id" in header else -1
                        hi = header.index("hadm_id") if "hadm_id" in header else -1
                        ki = header.index(SPLIT_KEY)
                        for row in reader:
                            if len(row) > ki and row[ki] in sel_set:
                                if si >= 0 and len(row) > si: extended.add(row[si])
                                if hi >= 0 and len(row) > hi: extended.add(row[hi])
            counts = build_split_raw_from_test(
                test_raw_dir=test_raw, output_raw_dir=raw_dst,
                split_key=SPLIT_KEY, selected_keys=sel_set, extended_keys=extended,
            )
            n_total = sum(v for v in counts.values() if v >= 0)
            n_files = sum(1 for v in counts.values() if v >= 0)
            print(f"    [raw] filtered from test/raw: {n_files} files, {n_total} rows", flush=True)

    # 派生 subject_id_lookup
    subject_id_lookup = None
    if SPLIT_KEY == "hadm_id":
        adm_path = raw_dst / "hosp" / "admissions.csv"
        if adm_path.is_file():
            subject_ids = set()
            with open_csv(adm_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                if "subject_id" in header and "hadm_id" in header:
                    si = header.index("subject_id")
                    hi = header.index("hadm_id")
                    for row in reader:
                        if len(row) > hi and row[hi] in sel_set:
                            subject_ids.add(row[si])
            subject_id_lookup = subject_ids
            print(f"    [subject_id] derived {len(subject_id_lookup)} from admissions", flush=True)

    # 删旧 modification_log.csv
    log_path = TARGET_ROOT / split_name / "modification_log.csv"
    if log_path.exists():
        log_path.unlink()

    # build new modification log
    seed = int(hashlib.md5(f"K29_v2_rebuild|{split_name}".encode()).hexdigest(), 16) & 0xFFFFFFFF
    t0 = time.time()
    stats = build_modification_log(
        split_name=split_name,
        target_count=error_count,
        baseline_raw_dir=raw_dst,
        output_log_path=log_path,
        split_key=SPLIT_KEY,
        all_keys=sel_keys,
        random_seed=seed,
        subject_id_lookup=subject_id_lookup,
    )
    t1 = time.time()
    print(f"    [build] {stats['actual_count']:,} rows, {stats['n_subtypes_covered']} subtypes, "
          f"{stats['n_files_touched']} files, {stats['n_stays_touched']} stays, "
          f"max/min={stats['subtype_ratio_max_over_min']:.1f}, built in {t1-t0:.1f}s", flush=True)

    passed, failures = validate_stats(stats, split_name=split_name)
    if not passed:
        print(f"    [FAIL] {failures}", flush=True)
    else:
        print(f"    [PASS] validation", flush=True)

    # apply via streaming wrapper
    t2 = time.time()
    apply_stats = apply_per_file_logs(raw_dst, log_path)
    t3 = time.time()
    print(f"    [apply] {t3-t2:.1f}s: cell_updates={apply_stats['n_cell_updates']:,}, "
          f"duplicates={apply_stats['n_duplicates']:,}, "
          f"inserts={apply_stats['n_inserts']:,}, skipped={apply_stats['n_skipped']:,}", flush=True)

    # cell-level rate
    total_cells = 0
    for sub in ["hosp", "icu"]:
        for f in raw_dst.rglob("*.csv"):
            with f.open() as fh:
                lines = fh.readlines()
            if lines:
                r = len(lines) - 1
                c = len(lines[0].strip().split(","))
                total_cells += r * c
    cell_rate = 100.0 * apply_stats["n_cell_updates"] / total_cells if total_cells else 0
    print(f"    [cell-level] {cell_rate:.2f}% ({apply_stats['n_cell_updates']:,} / {total_cells:,})", flush=True)

    return {
        "target_errors": error_count, "actual_errors": stats["actual_count"],
        "subtype_coverage": stats["n_subtypes_covered"],
        "subtype_ratio": stats["subtype_ratio_max_over_min"],
        "validation_passed": passed, "validation_failures": failures,
        "applied": apply_stats, "total_cells": total_cells,
        "cell_level_error_rate": round(cell_rate, 2),
        "build_seconds": round(t1 - t0, 1), "apply_seconds": round(t3 - t2, 1),
    }


def main():
    only_split = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"\n========== K29 v2 rawerror rebuild ==========")
    print(f"  baseline: {BASELINE_ROOT.name}")
    print(f"  target:   {TARGET_ROOT.name}")
    print(f"  targets:  {SPLIT_ERROR_COUNTS}")

    report = {"name": TARGET_ROOT.name, "baseline": BASELINE_ROOT.name, "splits": {}}
    for split_name, error_count in SPLIT_ERROR_COUNTS.items():
        if only_split and split_name != only_split:
            continue
        report["splits"][split_name] = rebuild_split(split_name, error_count)

    report_path = TARGET_ROOT.parent / "v2_rawerror_k29_rebuild_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\n[OK] report → {report_path}")


if __name__ == "__main__":
    main()
