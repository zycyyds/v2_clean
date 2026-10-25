"""K29 highrate_10pct_v2 注入器（重建版本）。

设计：
- 假设 NEW_ROOT/{train,validation,test}/raw + reference + keys.csv 已经从 baseline + orig 复制好
- 本脚本只做：build_modification_log + apply_per_file_logs + cell-level 校验
- 不做任何 raw/ref/keys 重生成（避免 ICU filter bug）
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
sys.path.insert(0, str(ROOT / "reproduction"))

from build_v2_error_configs import build_modification_log, validate_stats
from apply_modification_log import apply_modification_log
from build_all_v2_rawerror_datasets import (
    LOGS_DIR, read_keys, open_csv,
)

NEW_NAME = "K29_clean_v2_rawerror_highrate_10pct_v2"
NEW_ROOT = ROOT / "datasets/注入错误数据集" / NEW_NAME
DATASETS_ROOT = ROOT / "datasets"
SPLIT_KEY = "hadm_id"

# cell-level 10% 目标（按 raw 总 cell 数 10% 估算）
# train raw cells 约 113k, 10% = ~11k
# val raw cells 约 728k, 10% = ~73k
# test raw cells 约 66M, 10% = ~6.6M
SPLIT_ERROR_COUNTS = {
    "train": 5_000,         # 10 hadm_id → 物理上限
    "validation": 20_000,   # 20 hadm_id
    "test": 2_000_000,      # 8092 hadm_id
}


def apply_per_file_logs(raw_dir: Path, log_path: Path) -> dict:
    """按 raw_file 拆 log，逐文件调 apply_modification_log。避免一次性 load OOM。"""
    log_path = Path(log_path)
    print(f"    [apply] loading {log_path} ...", flush=True)
    log = pd.read_csv(log_path, dtype=str)
    print(f"    [apply] {len(log)} entries across {log['raw_file'].nunique()} files", flush=True)

    total_updates = 0
    total_dups = 0
    total_inserts = 0
    total_skipped = 0
    file_stats = {}

    for raw_file, sub_log in log.groupby("raw_file", sort=False):
        sub_log_path = log_path.parent / f"{log_path.stem}__{raw_file.replace('/', '_')}.csv"
        sub_log.to_csv(sub_log_path, index=False)
        try:
            apply_stats = apply_modification_log(raw_dir, sub_log_path)
            total_updates += apply_stats.get("n_cell_updates", 0)
            total_dups += apply_stats.get("n_duplicates", 0)
            total_inserts += apply_stats.get("n_inserts", 0)
            total_skipped += apply_stats.get("n_skipped", 0)
            file_stats[raw_file] = apply_stats.get("n_cell_updates", 0)
            print(f"    [apply] {raw_file:<35} {len(sub_log):>9} entries "
                  f"(+{apply_stats.get('n_cell_updates', 0)} updates, "
                  f"+{apply_stats.get('n_duplicates', 0)} dups, "
                  f"+{apply_stats.get('n_inserts', 0)} inserts)", flush=True)
        finally:
            if sub_log_path.exists():
                sub_log_path.unlink()

    return {
        "n_cell_updates": total_updates,
        "n_duplicates": total_dups,
        "n_inserts": total_inserts,
        "n_skipped": total_skipped,
        "by_file": file_stats,
    }


def main():
    only_split = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"\n========== {NEW_NAME} ==========")
    print(f"  root: {NEW_ROOT}")
    print(f"  split_key: {SPLIT_KEY}")
    print(f"  targets: {SPLIT_ERROR_COUNTS}")

    report = {"name": NEW_NAME, "splits": {}}

    for split_name, error_count in SPLIT_ERROR_COUNTS.items():
        if only_split and split_name != only_split:
            continue
        print(f"\n  --- {split_name} (target {error_count:,} errors) ---", flush=True)

        split_dir = NEW_ROOT / split_name
        keys_csv = split_dir / "keys.csv"
        raw_dst = split_dir / "raw"
        ref_dst = split_dir / "reference"

        if not keys_csv.exists():
            print(f"    [ERROR] keys.csv not found: {keys_csv}", flush=True)
            continue
        if not raw_dst.exists():
            print(f"    [ERROR] raw/ not found: {raw_dst}", flush=True)
            continue
        if not ref_dst.exists():
            print(f"    [ERROR] reference/ not found: {ref_dst}", flush=True)
            continue

        sel_keys = read_keys(keys_csv)
        sel_set = set(sel_keys)
        print(f"    keys: {len(sel_keys)} ({SPLIT_KEY})", flush=True)
        print(f"    raw/ has {len(list(raw_dst.rglob('*.csv')))} csv files", flush=True)

        # 派生 subject_id_lookup
        subject_id_lookup = None
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
            print(f"    derived {len(subject_id_lookup)} subject_ids from admissions", flush=True)

        # build modification log
        log_path = LOGS_DIR / f"{NEW_NAME}_{split_name}_modification_log.csv"
        seed = int(hashlib.md5(f"{NEW_NAME}|{split_name}".encode()).hexdigest(), 16) & 0xFFFFFFFF
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
        print(f"    modification_log: {stats['actual_count']:,} rows, "
              f"{stats['n_subtypes_covered']} subtypes, "
              f"{stats['n_files_touched']} files, "
              f"{stats['n_stays_touched']} stays, "
              f"max/min={stats['subtype_ratio_max_over_min']:.1f}, "
              f"built in {t1-t0:.1f}s", flush=True)

        # validate
        passed, failures = validate_stats(stats, split_name=split_name)
        if not passed:
            print(f"    [FAIL] validation: {failures}", flush=True)
        else:
            print(f"    [PASS] validation", flush=True)

        # apply via streaming wrapper
        t2 = time.time()
        apply_stats = apply_per_file_logs(raw_dst, log_path)
        t3 = time.time()
        print(f"    applied in {t3-t2:.1f}s: cell_updates={apply_stats['n_cell_updates']:,}, "
              f"duplicates={apply_stats['n_duplicates']:,}, "
              f"inserts={apply_stats['n_inserts']:,}, "
              f"skipped={apply_stats['n_skipped']:,}", flush=True)

        # 计算 cell-level 错误率
        total_cells = 0
        for sub in ["hosp", "icu"]:
            for f in (raw_dst / sub).glob("*.csv"):
                with f.open() as fh:
                    lines = fh.readlines()
                if lines:
                    r = len(lines) - 1
                    c = len(lines[0].strip().split(","))
                    total_cells += r * c
        cell_rate = 100.0 * apply_stats["n_cell_updates"] / total_cells if total_cells else 0
        print(f"    cell-level error rate: {cell_rate:.2f}% "
              f"({apply_stats['n_cell_updates']:,} / {total_cells:,})", flush=True)

        report["splits"][split_name] = {
            "target_errors": error_count,
            "actual_errors": stats["actual_count"],
            "subtype_coverage": stats["n_subtypes_covered"],
            "subtype_min": stats["subtype_min"],
            "subtype_max": stats["subtype_max"],
            "subtype_ratio": stats["subtype_ratio_max_over_min"],
            "files_touched": stats["n_files_touched"],
            "stays_touched": stats["n_stays_touched"],
            "validation_passed": passed,
            "validation_failures": failures,
            "applied": apply_stats,
            "total_cells": total_cells,
            "cell_level_error_rate": round(cell_rate, 2),
            "log_path": str(log_path),
            "build_seconds": round(t1 - t0, 1),
            "apply_seconds": round(t3 - t2, 1),
        }

    # 写报告
    report_path = DATASETS_ROOT / "v2_rawerror_highrate_v2_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\n[OK] report → {report_path}")


if __name__ == "__main__":
    main()
