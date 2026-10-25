"""K29 cell-level 10% 错误率数据集构建器。

与原版 K29_clean_v2_rawerror_10train_400_800_1800 并列，不冲突。

参数：
- split_key: hadm_id (train=10, val=20, test=1719)
- 错误率: test cell-level 10% (26M cell_update / 260M cells)
- train/val: 重点是 16 subtype 全部覆盖 + 尽量多错误数 (物理上限约 7-15k)
- subtype: 16 个 (去 schema_range_violation + enum_violation)
- subtype 分布: 大致均分 (max/min ≤ 10.0)
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
    LOGS_DIR, read_keys, filter_reference_for_keys,
    build_split_raw_from_test, open_csv, HOSP_KEY_COLS, ICU_KEY_COLS,
)

# ============== 配置 ==============
NEW_NAME = "K29_clean_v2_rawerror_highrate_10pct"
OLD_ROOT = ROOT / "datasets/常规数据集/K29_clean_eval_10_20_rest"
NEW_ROOT = ROOT / "datasets/注入错误数据集" / NEW_NAME
DATASETS_ROOT = ROOT / "datasets"
SPLIT_KEY = "hadm_id"
TEST_N = 8092
N_TRAIN_KEYS = 10
N_VAL_KEYS = 20

# train/val 重点：18 subtype 全覆盖 + 尽量多 (eligible rows 物理上限)
# test 重点：cell-level 10% (260M cells * 10% = 26M)
SPLIT_ERROR_COUNTS = {
    "train": 5_000,         # 10 hadm_id → ~4.6k actual (cell-level 0.01%)
    "validation": 7_000,    # 20 hadm_id → ~6.6k actual (cell-level 0.01%)
    "test": 26_503_810,     # 1719 hadm_id → cell-level 10%
}


# ============== streaming wrapper ==============
def apply_per_file_logs(raw_dir: Path, log_path: Path) -> dict:
    """按 raw_file 拆 log，逐文件调 apply_modification_log。

    关键：避免一次性 load 26.5M entries (5.3GB) OOM。
    实现：pandas 流式按 raw_file 分组，每组写子 log → 调 apply → 删子 log。
    """
    log_path = Path(log_path)
    # 用 pandas 读、按 raw_file 分组（pandas 比 csv.DictReader 省内存 ~30%）
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


# ============== main ==============
def main():
    only_split = sys.argv[1] if len(sys.argv) > 1 else None  # 可选 train/validation/test

    print(f"\n========== {NEW_NAME} ==========")
    print(f"  baseline: {OLD_ROOT.name}")
    print(f"  split_key: {SPLIT_KEY}")
    print(f"  targets: {SPLIT_ERROR_COUNTS}")

    # 1. 读 test keys
    test_keys = read_keys(OLD_ROOT / "test" / "keys.csv")
    print(f"  test keys: {len(test_keys)}")

    # 派生 icu universe
    valid_keys: set[str] | None = None
    if SPLIT_KEY in ("hadm_id", "stay_id"):
        icu_stay_path = OLD_ROOT / "test" / "raw" / "icu" / "icustays.csv"
        if not icu_stay_path.is_file():
            icu_stay_path = OLD_ROOT / "test" / "raw" / "icu" / "icustays.csv.gz"
        if icu_stay_path.is_file():
            valid_keys = set()
            with open_csv(icu_stay_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                ki = header.index(SPLIT_KEY)
                for row in reader:
                    if len(row) > ki:
                        valid_keys.add(row[ki])
            print(f"  icu-derived valid {SPLIT_KEY}: {len(valid_keys)}")
            test_keys = [k for k in test_keys if k in valid_keys]
            print(f"  test keys (filtered to icu universe): {len(test_keys)}")

    # 派生 train/val keys (复用 baseline)
    import random
    baseline_train_keys_path = OLD_ROOT / "train" / "keys.csv"
    if baseline_train_keys_path.is_file():
        baseline_train_keys = read_keys(baseline_train_keys_path)
        if valid_keys is not None:
            baseline_train_keys = [k for k in baseline_train_keys if k in valid_keys]
        test_set = set(test_keys)
        baseline_train_keys = [k for k in baseline_train_keys if k not in test_set]
        if len(baseline_train_keys) >= N_TRAIN_KEYS:
            train_keys = baseline_train_keys[:N_TRAIN_KEYS]
        else:
            extras = [k for k in test_keys if k not in baseline_train_keys][:N_TRAIN_KEYS - len(baseline_train_keys)]
            train_keys = baseline_train_keys + extras
    else:
        rng = random.Random(hash(NEW_NAME) & 0xFFFFFFFF)
        shuffled = list(test_keys)
        rng.shuffle(shuffled)
        train_keys = shuffled[:N_TRAIN_KEYS]

    train_set = set(train_keys)
    test_keys = [k for k in test_keys if k not in train_set]

    rng = random.Random(int(hashlib.md5(f"{NEW_NAME}|val".encode()).hexdigest(), 16) & 0xFFFFFFFF)
    shuffled_for_val = list(test_keys)
    rng.shuffle(shuffled_for_val)
    val_keys = shuffled_for_val[:N_VAL_KEYS]
    test_keys_remaining = shuffled_for_val[N_VAL_KEYS:]
    print(f"  train keys: {len(train_keys)}, val keys: {len(val_keys)}, remaining test: {len(test_keys_remaining)}")

    test_raw = OLD_ROOT / "test" / "raw"
    report = {"name": NEW_NAME, "old_baseline": OLD_ROOT.name, "splits": {}}

    for split_name, error_count in SPLIT_ERROR_COUNTS.items():
        if only_split and split_name != only_split:
            continue
        print(f"\n  --- {split_name} (target {error_count:,} errors) ---", flush=True)

        # 选 keys
        if split_name == "train":
            sel_keys = train_keys
        elif split_name == "validation":
            sel_keys = val_keys
        else:
            sel_keys = test_keys_remaining

        # 写 keys.csv
        split_dir = NEW_ROOT / split_name
        keys_csv = split_dir / "keys.csv"
        keys_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(keys_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([SPLIT_KEY])
            for k in sel_keys:
                w.writerow([k])
        print(f"    wrote keys.csv ({len(sel_keys)} keys)", flush=True)

        # 拷 reference
        ref_src = OLD_ROOT / "test" / "reference"
        ref_dst = split_dir / "reference"
        if ref_src.is_dir() and not ref_dst.is_dir():
            sel_set = set(sel_keys)
            n_files = filter_reference_for_keys(ref_src, ref_dst, SPLIT_KEY, sel_set)
            print(f"    filtered reference/ from test/reference: {n_files} files", flush=True)
        elif ref_dst.is_dir():
            print(f"    reference/ already exists, skipping", flush=True)
        else:
            print(f"    [WARN] test/reference not found", flush=True)

        # raw
        raw_dst = split_dir / "raw"
        if split_name == "test":
            if not raw_dst.is_dir():
                from build_all_v2_rawerror_datasets import decompress_copytree
                decompress_copytree(test_raw, raw_dst)
                print(f"    copied raw/ (test baseline)", flush=True)
        else:
            sel_set = set(sel_keys)
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
                                if si >= 0 and len(row) > si:
                                    extended.add(row[si])
                                if hi >= 0 and len(row) > hi:
                                    extended.add(row[hi])
            counts = build_split_raw_from_test(
                test_raw_dir=test_raw,
                output_raw_dir=raw_dst,
                split_key=SPLIT_KEY,
                selected_keys=sel_set,
                extended_keys=extended,
            )
            n_total = sum(v for v in counts.values() if v >= 0)
            n_files = sum(1 for v in counts.values() if v >= 0)
            print(f"    filtered raw/: {n_files} files, {n_total} rows total", flush=True)

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
    report_path = DATASETS_ROOT / "v2_rawerror_highrate_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\n[OK] report → {report_path}")


if __name__ == "__main__":
    main()