"""icu_5000 v2 重建（用 baseline train/val/test 划分，注入错误）

策略：
- baseline train keys (10) → v2 train keys
- baseline val keys (20) → v2 val keys
- baseline test keys (7180) → v2 test keys
- 每个 split 的 raw 直接从 baseline/{split}/raw 复制（解 .gz）
- 每个 split 的 reference 直接从 baseline/{split}/reference 复制
- 错误只注入到 hosp/ + icu/ 子目录（note/, multimodal/, ecg/ 保留原样）
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


OLD_NAME = "icu_5000_clean_eval_10_20_rest"
NEW_NAME = "icu_5000_v2_rawerror_10train_400_800_1800"
SPLIT_KEY = "stay_id"
SPLIT_ERROR_COUNTS = {"train": 400, "validation": 800, "test": 1800}

LOGS_DIR = Path("/tmp/v2_logs")


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
    """递归拷贝 src → dst，所有 .csv.gz 解压为 .csv。"""
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


def build_split_with_baseline_keys(*, new_root: Path, old_root: Path, split_name: str, split_key: str, target_errors: int) -> dict:
    """按 baseline/{split}/keys.csv 直接拷贝 raw + reference，再注入错误。"""
    print(f"\n  === {split_name} (target {target_errors} errors) ===", flush=True)
    t0 = time.time()

    # 1. 读 baseline keys
    baseline_keys_path = old_root / ("val" if split_name == "validation" else split_name) / "keys.csv"
    sel_keys = read_keys(baseline_keys_path)
    print(f"    baseline keys: {len(sel_keys)}", flush=True)

    # 2. 拷 raw（解压 .gz）
    raw_src = old_root / ("val" if split_name == "validation" else split_name) / "raw"
    raw_dst = new_root / split_name / "raw"
    if raw_dst.exists():
        shutil.rmtree(raw_dst)
    decompress_copytree(raw_src, raw_dst)
    print(f"    copied raw/ from baseline/{split_name}/raw (decompressed)", flush=True)

    # 3. 拷 reference
    ref_src = old_root / ("val" if split_name == "validation" else split_name) / "reference"
    ref_dst = new_root / split_name / "reference"
    if ref_dst.exists():
        shutil.rmtree(ref_dst)
    if ref_src.is_dir():
        shutil.copytree(ref_src, ref_dst)
    print(f"    copied reference/ from baseline/{split_name}/reference", flush=True)

    # 4. 拷 keys.csv
    keys_dst = new_root / split_name / "keys.csv"
    keys_dst.parent.mkdir(parents=True, exist_ok=True)
    with open(keys_dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([split_key])
        for k in sel_keys:
            w.writerow([k])
    print(f"    wrote keys.csv ({len(sel_keys)} keys)", flush=True)

    # 5. 派生 subject_id_lookup（让 hosp 表错误也能注入到对应行）
    subject_id_lookup = None
    if split_key == "stay_id":
        icu_path = raw_dst / "icu" / "icustays.csv"
        if icu_path.is_file():
            subject_ids = set()
            stay_ids = set()
            hadm_ids = set()
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
            print(f"    derived {len(subject_id_lookup)} ids from icustays: "
                  f"{len(subject_ids)} subj + {len(hadm_ids)} hadm + {len(stay_ids)} stay", flush=True)

    # 6. Build modification log
    log_path = LOGS_DIR / f"{NEW_NAME}_{split_name}_modification_log.csv"
    seed = hash((NEW_NAME, split_name)) & 0xFFFFFFFF
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
    print(f"    modification_log: {stats['actual_count']} rows, "
          f"{stats['n_subtypes_covered']}/18 subtypes, "
          f"{stats['n_files_touched']} files, "
          f"{stats['n_stays_touched']} stays", flush=True)

    # 7. 校验
    passed, failures = validate_stats(stats, split_name=split_name)
    if not passed:
        print(f"    [FAIL] validation: {failures}", flush=True)
    else:
        print(f"    [PASS] validation", flush=True)

    # 8. 注入错误
    apply_stats = apply_modification_log(raw_dst, log_path)
    print(f"    applied: cell_updates={apply_stats['n_cell_updates']}, "
          f"duplicates={apply_stats['n_duplicates']}, "
          f"inserts={apply_stats['n_inserts']}, "
          f"skipped={apply_stats['n_skipped']}", flush=True)
    print(f"    [done] {time.time()-t0:.1f}s", flush=True)

    return {
        "target_errors": target_errors,
        "actual_errors": stats["actual_count"],
        "n_baseline_keys": len(sel_keys),
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


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    new_root = ROOT / "datasets" / NEW_NAME
    old_root = ROOT / "datasets" / OLD_NAME

    if new_root.exists():
        shutil.rmtree(new_root)
    new_root.mkdir(parents=True)

    print(f"========== {NEW_NAME} ==========")
    print(f"  baseline: {OLD_NAME}")
    print(f"  split_key: {SPLIT_KEY}")

    report = {"name": NEW_NAME, "old_baseline": OLD_NAME, "splits": {}}
    for split_name, target in SPLIT_ERROR_COUNTS.items():
        try:
            split_report = build_split_with_baseline_keys(
                new_root=new_root,
                old_root=old_root,
                split_name=split_name,
                split_key=SPLIT_KEY,
                target_errors=target,
            )
            report["splits"][split_name] = split_report
        except Exception as e:
            print(f"\n[ERROR] {split_name}: {e}")
            import traceback
            traceback.print_exc()
            report["splits"][split_name] = {"error": str(e)}

    out_path = ROOT / "datasets" / "v2_rawerror_validation_report.json"
    existing = []
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text())
            existing = [r for r in existing if r.get("name") != NEW_NAME]
        except Exception:
            existing = []
    existing.append(report)
    out_path.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()