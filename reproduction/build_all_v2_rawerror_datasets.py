"""驱动器：构建 4 个 v2 rawerror 数据集（K29 / icu5000 / N18 / I48）。

策略：
- 每个数据集用 existing test raw（已含 hosp+icu）作 master source
- 从 test keys 中 sample 10 → train keys, sample 20 → val keys, 余下留在 test
- 过滤 test raw 生成 train/val raw（拷 hosp+icu 子目录，过滤 csv 行）
- inject 400/800/1800 错误到各 split raw
- reference 直接复用现有 reference（已 clean）
- J18 v2 跳过（baseline icu 已被破坏）

输出结构：
  datasets/{name}/train/{raw,reference,keys.csv}
  datasets/{name}/validation/{raw,reference,keys.csv}
  datasets/{name}/test/{raw,reference,keys.csv}
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from build_v2_error_configs import build_modification_log, validate_stats
from apply_modification_log import apply_modification_log


ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
DATASETS_ROOT = ROOT / "datasets"
LOGS_DIR = Path("/tmp/v2_logs")


# 4 个 v2 数据集配置（跳过 J18）
DATASETS = [
    {
        "new_name": "K29_clean_v2_rawerror_10train_400_800_1800",
        "old_name": "K29_clean_eval_10_20_rest",
        "split_key": "hadm_id",
        "test_n": 8092,
    },
    {
        "new_name": "icu_5000_v2_rawerror_10train_400_800_1800",
        "old_name": "icu_5000_clean_eval_10_20_rest",
        "split_key": "stay_id",
        "test_n": 7180,
    },
    {
        "new_name": "N18_ckd_ED_v2_rawerror_10train_400_800_1800",
        "old_name": "N18_ckd_ED_notes_eval_10_20_rest",
        "split_key": "subject_id",
        "test_n": 2970,
    },
    {
        "new_name": "I48_atrial_fibrillation_v2_rawerror_10train_400_800_1800",
        "old_name": "I48_atrial_fibrillation_ECG_eval_10_20_rest",
        "split_key": "subject_id",
        "test_n": 2970,
    },
]


SPLIT_ERROR_COUNTS = {"train": 400, "validation": 800, "test": 1800}
N_TRAIN_KEYS = 10
N_VAL_KEYS = 20


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def open_csv(path: Path, mode: str = "r"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def read_keys(keys_csv: Path) -> list[str]:
    with open(keys_csv) as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader if row]


def filter_csv_by_keys(src: Path, dst: Path, key_col: str, allowed_keys: set[str]) -> int:
    """过滤 csv 文件：保留 key_col 命中 allowed_keys 的行。返回写入行数。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_csv(src) as fin, open_csv(dst, "w") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        if key_col not in header:
            # 没有 key_col，直接拷贝原表（d_items, d_labitems 等字典表）
            n = sum(1 for _ in reader)
            with open_csv(src) as fin2, open_csv(dst, "w") as fout2:
                r2 = csv.reader(fin2)
                w2 = csv.writer(fout2)
                hdr = next(r2)
                w2.writerow(hdr)
                n = 0
                for row in r2:
                    w2.writerow(row)
                    n += 1
            return n
        ki = header.index(key_col)
        n = 0
        for row in reader:
            if len(row) > ki and row[ki] in allowed_keys:
                writer.writerow(row)
                n += 1
        return n


def filter_reference_for_keys(
    ref_src: Path,
    ref_dst: Path,
    split_key: str,
    selected_keys: set[str],
    *,
    icu_also: bool = True,
) -> int:
    """从 ref_src 的所有 csv 过滤到 ref_dst，保留 split_key 命中 selected_keys 的行。
    icu/ 子目录同时接受 split_key 和 stay_id。"""
    ref_dst.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for src_path in sorted(ref_src.rglob("*.csv*")):
        if not src_path.is_file():
            continue
        rel = src_path.relative_to(ref_src)
        dst_path = ref_dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # 决定 key cols：顶层按 split_key，icu/ 也接受 stay_id
        cols_to_check = [split_key]
        if icu_also and "icu" in rel.parts:
            cols_to_check = list(set([split_key, "stay_id"]))
        n = filter_csv_by_any_col(src_path, dst_path, cols_to_check, selected_keys)
        n_files += 1
    return n_files


def decompress_copytree(src: Path, dst: Path) -> None:
    """递归拷贝 src → dst，所有 .csv.gz 解压为 .csv。"""
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_dir():
            decompress_copytree(p, dst / p.name)
        elif p.is_file():
            if p.name.endswith(".gz"):
                dst_name = p.name[:-3]
                dst_file = dst / dst_name
                with open_csv(p) as fin, open(dst_file, "w", encoding="utf-8", newline="") as fout:
                    fout.write(fin.read())
            else:
                shutil.copy2(p, dst / p.name)


def filter_csv_by_any_col(src: Path, dst: Path, key_cols: list[str], allowed_keys: set[str]) -> int:
    """多 key 列：保留任一 key 列命中 allowed_keys 的行。
    字典表（key_cols 为空 → ki_idx 为空）→ 全拷贝。
    非字典表如果 0 行 → 不 fallback，避免错误地把整个 test 数据复制进来。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_csv(src) as fin, open_csv(dst, "w") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        ki_idx = [i for i, h in enumerate(header) if h in key_cols]
        if ki_idx:
            n = 0
            for row in reader:
                if any(len(row) > i and row[i] in allowed_keys for i in ki_idx):
                    writer.writerow(row)
                    n += 1
            return n
        # 字典表（key_cols 在文件 header 里没有）→ 全拷贝
        n = 0
        for row in reader:
            writer.writerow(row)
            n += 1
        return n


# 各 hosp/icu 表的主键列（用于过滤）
HOSP_KEY_COLS = {
    "admissions.csv": ["subject_id", "hadm_id"],
    "patients.csv": ["subject_id"],
    "diagnoses_icd.csv": ["subject_id", "hadm_id"],
    "drgcodes.csv": ["subject_id", "hadm_id"],
    "emar.csv": ["subject_id", "hadm_id"],
    "hcpcsevents.csv": ["subject_id", "hadm_id"],
    "labevents.csv": ["subject_id", "hadm_id"],
    "microbiologyevents.csv": ["subject_id", "hadm_id"],
    "pharmacy.csv": ["subject_id", "hadm_id"],
    "poe.csv": ["subject_id", "hadm_id"],
    "prescriptions.csv": ["subject_id", "hadm_id"],
    "procedures_icd.csv": ["subject_id", "hadm_id"],
    "services.csv": ["subject_id", "hadm_id"],
    "transfers.csv": ["subject_id", "hadm_id"],
    "provider.csv": [],  # 无 key，全留
    "d_icd_diagnoses.csv": [],
    "d_icd_procedures.csv": [],
    "d_hcpcs.csv": [],
    "d_labitems.csv": [],
}
ICU_KEY_COLS = {
    "icustays.csv": ["subject_id", "hadm_id", "stay_id"],
    "chartevents.csv": ["subject_id", "hadm_id", "stay_id"],
    "inputevents.csv": ["subject_id", "hadm_id", "stay_id"],
    "outputevents.csv": ["subject_id", "hadm_id", "stay_id"],
    "procedureevents.csv": ["subject_id", "hadm_id", "stay_id"],
    "datetimeevents.csv": ["subject_id", "hadm_id", "stay_id"],
    "ingredientevents.csv": ["subject_id", "hadm_id", "stay_id"],
    "caregiver.csv": [],
    "d_items.csv": [],
}


def build_split_raw_from_test(
    *,
    test_raw_dir: Path,
    output_raw_dir: Path,
    split_key: str,
    selected_keys: set[str],
    extended_keys: set[str] | None = None,
) -> dict[str, int]:
    """从 test raw 过滤出 selected_keys 对应的 hosp+icu 副本到 output_raw_dir。
    其他子目录（note/ed/ecg/cxr 等多模态）整体保留，不注入错误。
    支持 .csv 和 .csv.gz 两种源格式；输出**统一解压为 .csv**。

    extended_keys
        额外的 key 集合（如从 icustays 派生的 subject_id / hadm_id），用于 hosp 表过滤。
    """
    counts: dict[str, int] = {}
    extra = extended_keys or set()

    # 1) 多模态 / 其他子目录（note/ed/ecg/cxr 等）整体复制，解压 .gz
    INJECT_SUBS = {"hosp", "icu"}
    for sub_dir in sorted(test_raw_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        if sub_dir.name in INJECT_SUBS:
            continue
        dst_sub = output_raw_dir / sub_dir.name
        decompress_copytree(sub_dir, dst_sub)

    # 2) hosp/icu 按 key 过滤
    for sub, key_cols_map in [("hosp", HOSP_KEY_COLS), ("icu", ICU_KEY_COLS)]:
        src_dir = test_raw_dir / sub
        dst_dir = output_raw_dir / sub
        if not src_dir.is_dir():
            continue
        for src_path in sorted(src_dir.iterdir()):
            if not src_path.is_file():
                continue
            fname = src_path.name
            if fname.endswith(".gz"):
                dst_name = fname[:-3]
            else:
                dst_name = fname
            fname_no_gz = dst_name
            key_cols = key_cols_map.get(fname_no_gz, [])
            dst_path = dst_dir / dst_name
            if not key_cols:
                # 字典表整体拷贝（解压成 .csv）
                dst_dir.mkdir(parents=True, exist_ok=True)
                with open_csv(src_path) as fin, open(dst_path, "w", encoding="utf-8", newline="") as fout:
                    fout.write(fin.read())
                counts[f"{sub}/{dst_name}"] = -1
                continue
            # hosp 表：如果 extended_keys 非空，把它们也算 accept（派生 subject_id / hadm_id）
            if sub == "hosp" and extra:
                accept = selected_keys | extra
            else:
                accept = selected_keys
            n = filter_csv_by_any_col(src_path, dst_path, key_cols, accept)
            counts[f"{sub}/{dst_name}"] = n
    return counts


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def build_dataset(cfg: dict, *, only_split: str | None = None, skip_apply: bool = False) -> dict:
    new_name = cfg["new_name"]
    old_name = cfg["old_name"]
    split_key = cfg["split_key"]
    new_root = DATASETS_ROOT / new_name
    old_root = DATASETS_ROOT / old_name

    print(f"\n========== {new_name} ==========")
    print(f"  baseline: {old_name}")
    print(f"  split_key: {split_key}")

    # 1. 读 test keys, sample train / val keys
    test_keys = read_keys(old_root / "test" / "keys.csv")
    print(f"  test keys: {len(test_keys)}")

    # 对于 hadm_id / stay_id split：从 test raw 的 icustays 派生有效 keys
    # （K29 / icu_5000 baseline 的 hosp 和 icu 有不交叉的 id，需要限制 keys 在有 icu 数据的范围内）
    valid_keys: set[str] | None = None
    if split_key in ("hadm_id", "stay_id"):
        icu_stay_path = old_root / "test" / "raw" / "icu" / "icustays.csv"
        if not icu_stay_path.is_file():
            icu_stay_path = old_root / "test" / "raw" / "icu" / "icustays.csv.gz"
        if icu_stay_path.is_file():
            valid_keys = set()
            with open_csv(icu_stay_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                ki = header.index(split_key)
                for row in reader:
                    if len(row) > ki:
                        valid_keys.add(row[ki])
            print(f"  icu-derived valid {split_key}: {len(valid_keys)}")
            test_keys = [k for k in test_keys if k in valid_keys]
            print(f"  test keys (filtered to icu universe): {len(test_keys)}")

    # train keys 优先复用 baseline/train/keys.csv（保证 v2 train 与 baseline train 同源）
    baseline_train_keys_path = old_root / "train" / "keys.csv"
    if baseline_train_keys_path.is_file():
        baseline_train_keys = read_keys(baseline_train_keys_path)
        # 限制到 icu universe（如果适用）
        if valid_keys is not None:
            baseline_train_keys = [k for k in baseline_train_keys if k in valid_keys]
        # 排除与 test 重叠的（保险）
        test_set = set(test_keys)
        baseline_train_keys = [k for k in baseline_train_keys if k not in test_set]
        # 如果 baseline train 不足 N_TRAIN_KEYS，从 test 里补足
        if len(baseline_train_keys) >= N_TRAIN_KEYS:
            # baseline train 全部都有 ICU，可以直接用
            train_keys = baseline_train_keys[:N_TRAIN_KEYS]
            print(f"  train keys (from baseline/train): {len(train_keys)}")
        else:
            # baseline train 缺乏 ICU 数据，从 test 里抽 10 个有 ICU 的
            print(f"  baseline train only has {len(baseline_train_keys)} keys with ICU data, supplementing from test")
            extras = [k for k in test_keys if k not in baseline_train_keys][:N_TRAIN_KEYS - len(baseline_train_keys)]
            train_keys = baseline_train_keys + extras
            print(f"  train keys (supplemented): {len(train_keys)}")
    else:
        # baseline 没有 train/keys.csv（罕见）：从 test 里随机抽
        rng = random.Random(hash(new_name) & 0xFFFFFFFF)
        shuffled = list(test_keys)
        rng.shuffle(shuffled)
        train_keys = shuffled[:N_TRAIN_KEYS]
        print(f"  train keys (sampled from test): {len(train_keys)}")

    train_set = set(train_keys)
    test_keys = [k for k in test_keys if k not in train_set]
    print(f"  test keys (excluding train): {len(test_keys)}")

    rng = random.Random(hash((new_name, "val")) & 0xFFFFFFFF)
    shuffled_for_val = list(test_keys)
    rng.shuffle(shuffled_for_val)
    val_keys = shuffled_for_val[:N_VAL_KEYS]
    # 余下保持原顺序（保持与原 test 一致）
    test_keys_remaining = shuffled_for_val[N_VAL_KEYS:]

    print(f"  train keys: {len(train_keys)}, val keys: {len(val_keys)}, remaining test: {len(test_keys_remaining)}")

    test_raw = old_root / "test" / "raw"

    report: dict = {"name": new_name, "old_baseline": old_name, "splits": {}}

    # 2. 对每个 split 建 raw + inject errors
    for split_name, error_count in SPLIT_ERROR_COUNTS.items():
        if only_split and split_name != only_split:
            continue
        print(f"\n  --- {split_name} (target {error_count} errors) ---")

        # 选 keys
        if split_name == "train":
            sel_keys = train_keys
        elif split_name == "validation":
            sel_keys = val_keys
        else:
            sel_keys = test_keys_remaining

        # 写 keys.csv
        split_dir = new_root / split_name
        keys_csv = split_dir / "keys.csv"
        keys_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(keys_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([split_key])
            for k in sel_keys:
                w.writerow([k])
        print(f"    wrote keys.csv ({len(sel_keys)} keys)")

        # 拷 reference：原 baseline 通常只有 train + test 有 reference，val 没有。
        # 策略：从 test/reference 过滤出 selected_keys 对应的行作为新 split 的 reference
        ref_src = old_root / "test" / "reference"  # 从 test/reference 过滤（最完整）
        ref_dst = split_dir / "reference"
        if ref_src.is_dir() and not ref_dst.is_dir():
            sel_set = set(sel_keys)
            n_files = filter_reference_for_keys(ref_src, ref_dst, split_key, sel_set)
            print(f"    filtered reference/ from test/reference: {n_files} files")
        elif ref_dst.is_dir():
            print(f"    reference/ already exists, skipping")
        else:
            print(f"    [WARN] test/reference not found, skipping reference copy")

        # 过滤 raw（train/val/test 都从 test raw 过滤；test 拷整目录）
        raw_dst = split_dir / "raw"
        if split_name == "test":
            # 直接符号链接原 test raw 以节省时间；先试试硬链接/拷贝
            if not raw_dst.is_dir():
                # 解压：递归复制 test_raw，解压所有 .gz 为 .csv
                decompress_copytree(test_raw, raw_dst)
                print(f"    copied raw/ (test baseline, decompressed to .csv)")
        else:
            sel_set = set(sel_keys)
            # 派生 subject_id / hadm_id（hosp 表用 stay_id 过滤需要）
            extended = set()
            if split_key in ("stay_id", "hadm_id"):
                # 从 test baseline icustays 派生出 subject_id/hadm_id
                icu_baseline = test_raw / "icu" / "icustays.csv"
                if not icu_baseline.is_file():
                    icu_baseline = test_raw / "icu" / "icustays.csv.gz"
                if icu_baseline.is_file():
                    with open_csv(icu_baseline) as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        si = header.index("subject_id") if "subject_id" in header else -1
                        hi = header.index("hadm_id") if "hadm_id" in header else -1
                        for row in reader:
                            if split_key == "stay_id":
                                ki = header.index("stay_id")
                            else:
                                ki = header.index("hadm_id")
                            if len(row) > ki and row[ki] in sel_set:
                                if si >= 0 and len(row) > si:
                                    extended.add(row[si])
                                if hi >= 0 and len(row) > hi:
                                    extended.add(row[hi])
            counts = build_split_raw_from_test(
                test_raw_dir=test_raw,
                output_raw_dir=raw_dst,
                split_key=split_key,
                selected_keys=sel_set,
                extended_keys=extended,
            )
            n_total = sum(v for v in counts.values() if v >= 0)
            n_files = sum(1 for v in counts.values() if v >= 0)
            print(f"    filtered raw/: {n_files} files, {n_total} rows total")

        # 3. 构建 modification log
        # 对 hadm_id / stay_id 数据集：从 admissions.csv / icustays.csv 派生 subject_id / hadm_id lookup
        # （让 schema_range_violation 等需要多 key 匹配的 subtype 也能找到行）
        subject_id_lookup = None
        if split_key == "hadm_id":
            adm_path = raw_dst / "hosp" / "admissions.csv"
            if adm_path.is_file():
                subject_ids = set()
                with open_csv(adm_path) as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    rows = list(reader)
                    if "subject_id" in header and "hadm_id" in header:
                        si = header.index("subject_id")
                        hi = header.index("hadm_id")
                        for row in rows:
                            if len(row) > hi and row[hi] in sel_set:
                                subject_ids.add(row[si])
                subject_id_lookup = subject_ids
                print(f"    derived {len(subject_id_lookup)} subject_ids from admissions")
        elif split_key == "stay_id":
            icu_path = raw_dst / "icu" / "icustays.csv"
            if icu_path.is_file():
                # collect subject_ids (for hosp tables) + stay_ids (for icu tables)
                subject_ids = set()
                stay_ids = set()
                hadm_ids = set()
                with open_csv(icu_path) as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    rows = list(reader)
                    if "subject_id" in header:
                        si = header.index("subject_id")
                        for row in rows:
                            if len(row) > si and row[si]:
                                subject_ids.add(row[si])
                    if "hadm_id" in header:
                        hi = header.index("hadm_id")
                        for row in rows:
                            if len(row) > hi and row[hi]:
                                hadm_ids.add(row[hi])
                    if "stay_id" in header:
                        sti = header.index("stay_id")
                        for row in rows:
                            if len(row) > sti and row[sti]:
                                stay_ids.add(row[sti])
                # subject_id_lookup 必须包含 subject_ids（hosp 表的主键）+ stay_ids（icu 表的主键）
                subject_id_lookup = subject_ids | stay_ids | hadm_ids
                print(f"    derived {len(subject_id_lookup)} ids from icustays: "
                      f"{len(subject_ids)} subj + {len(hadm_ids)} hadm + {len(stay_ids)} stay")

        log_path = LOGS_DIR / f"{new_name}_{split_name}_modification_log.csv"
        seed = hash((new_name, split_name)) & 0xFFFFFFFF
        stats = build_modification_log(
            split_name=split_name,
            target_count=error_count,
            baseline_raw_dir=raw_dst,
            output_log_path=log_path,
            split_key=split_key,
            all_keys=sel_keys,
            random_seed=seed,
            subject_id_lookup=subject_id_lookup,
        )
        print(f"    modification_log: {stats['actual_count']} rows, "
              f"{stats['n_subtypes_covered']}/{len(set(c['subtype'] for c in [])) or 18} subtypes, "
              f"{stats['n_files_touched']} files, "
              f"{stats['n_stays_touched']} stays")

        # 4. 校验
        passed, failures = validate_stats(stats, split_name=split_name)
        if not passed:
            print(f"    [FAIL] validation: {failures}")
        else:
            print(f"    [PASS] validation")

        # 5. 应用 modification log（注入错误）
        if not skip_apply:
            apply_stats = apply_modification_log(raw_dst, log_path)
            print(f"    applied: cell_updates={apply_stats['n_cell_updates']}, "
                  f"duplicates={apply_stats['n_duplicates']}, "
                  f"inserts={apply_stats['n_inserts']}, "
                  f"skipped={apply_stats['n_skipped']}")
        else:
            apply_stats = {"n_cell_updates": 0, "n_duplicates": 0, "n_inserts": 0, "n_skipped": 0, "by_file": {}, "n_files": 0}
            print(f"    [skip-apply] would inject {stats['actual_count']} errors")

        report["splits"][split_name] = {
            "target_errors": error_count,
            "actual_errors": stats["actual_count"],
            "subtype_coverage": stats["n_subtypes_covered"],
            "subtype_min": stats["subtype_min"],
            "subtype_max": stats["subtype_max"],
            "subtype_ratio": stats["subtype_ratio_max_over_min"],
            "files_touched": stats["n_files_touched"],
            "file_min": stats["file_min"],
            "file_max": stats["file_max"],
            "stays_touched": stats["n_stays_touched"],
            "stay_min": stats["stay_min"],
            "stay_max": stats["stay_max"],
            "stay_stddev": round(stats["stay_stddev"], 2),
            "validation_passed": passed,
            "validation_failures": failures,
            "applied": apply_stats,
            "log_path": str(log_path),
        }

    return report


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only-dataset", help="只跑指定数据集 new_name", default=None)
    p.add_argument("--only-split", help="只跑指定 split (train/validation/test)", default=None)
    p.add_argument("--skip-apply", action="store_true", help="只 build modification_log，不注入 raw")
    args = p.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    all_reports = []
    for cfg in DATASETS:
        if args.only_dataset and cfg["new_name"] != args.only_dataset:
            continue
        try:
            report = build_dataset(cfg, only_split=args.only_split, skip_apply=args.skip_apply)
            all_reports.append(report)
        except Exception as e:
            print(f"\n[ERROR] {cfg['new_name']}: {e}")
            import traceback
            traceback.print_exc()
            all_reports.append({"name": cfg["new_name"], "error": str(e)})

    out_path = ROOT / "datasets" / "v2_rawerror_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()