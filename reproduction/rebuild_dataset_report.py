#!/usr/bin/env python3
"""从磁盘上的成品目录重建每个数据集的 final_report.json。

为什么需要它：build_rawerror_by_ratio.py 一次只跑一个 split，报告是覆盖写
的 —— 最后一次跑的那一个 split 会把前面 split 的条目冲掉。这个脚本直接
读成品的 modification_log.csv 和 raw 目录重算，不碰注入结果。

口径（对齐 K29 4.72pct）：
  cell 级错误率 = 实际错误条数 / **注入后**的 raw cell 总数
  （duplicate / insert 会新增行，所以分母比 baseline 大）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(sys.maxsize)

from build_rawerror_by_ratio import (  # noqa: E402
    DATASETS, OUT_ROOT, RATE_PRESETS, count_raw_cells, deterministic_seed,
    ENGINE_VERSION,
)

CLEAN_ROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/常规数据集")
SPLITS = ("train", "validation", "test")


def scan_log(path: Path):
    """流式扫日志 → (总数, subtype 计数, file 计数, op 计数, class 计数, stay 计数)。"""
    n = 0
    by_subtype, by_file, by_op, by_class = (Counter() for _ in range(4))
    by_stay = Counter()
    with path.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            n += 1
            by_subtype[e["error_subtype"]] += 1
            by_file[e["raw_file"]] += 1
            by_op[e["operation"]] += 1
            by_class[e["error_class"]] += 1
            sid = (e.get("canonical_stay_id") or "").strip()
            if sid:
                by_stay[sid] += 1
    return n, by_subtype, by_file, by_op, by_class, by_stay


def _min_max(c: Counter):
    if not c:
        return 0, 0
    return min(c.values()), max(c.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--rates", choices=sorted(RATE_PRESETS), default="4.7pct")
    args = ap.parse_args()

    key = args.dataset
    cfg = DATASETS[key]
    preset = RATE_PRESETS[args.rates]
    out_name = cfg["out_base"].format(tag=preset["tag"])
    out_dir = OUT_ROOT / out_name

    splits_report = {}
    for split in SPLITS:
        sd = out_dir / split
        log = sd / "modification_log.csv"
        raw = sd / "raw"
        if not log.is_file() or not raw.is_dir():
            print(f"  [SKIP] {split}: 目录不完整")
            continue

        n, by_subtype, by_file, by_op, by_class, by_stay = scan_log(log)
        final_cells, final_rows = count_raw_cells(raw)
        clean_dir = CLEAN_ROOT / cfg["clean"] / cfg["split_dir"][split]
        base_cells, base_rows = count_raw_cells(clean_dir / "raw")
        sub_min, sub_max = _min_max(by_subtype)
        file_min, file_max = _min_max(by_file)
        stay_min, stay_max = _min_max(by_stay)

        splits_report[split] = {
            "target_errors": int(round(base_cells * preset["rates"][split])),
            "actual_errors": n,
            "cell_level_error_rate": round(n / final_cells, 6) if final_cells else 0,
            "total_cells": final_cells,
            "baseline_total_cells": base_cells,
            "total_rows": final_rows,
            "baseline_total_rows": base_rows,
            "subtype_counts": dict(by_subtype),
            "subtype_min": sub_min,
            "subtype_max": sub_max,
            "file_counts": dict(by_file),
            "file_min": file_min,
            "file_max": file_max,
            "stays_touched": len(by_stay),
            "stay_min": stay_min,
            "stay_max": stay_max,
            "operation_counts": dict(by_op),
            "error_class_counts": dict(by_class),
            "n_subtypes_covered": len(by_subtype),
            "natural_noise_count": by_class.get("class0_baseline_gap", 0),
            "seed": deterministic_seed(key, split),
        }
        print(f"  {split}: {n:,} 条 / {final_cells:,} cells = "
              f"{splits_report[split]['cell_level_error_rate'] * 100:.4f}%  "
              f"(目标 {preset['rates'][split] * 100:.2f}%)  "
              f"subtypes={len(by_subtype)}  自然噪声={by_class.get('class0_baseline_gap', 0)}")

    report = {
        "name": out_name,
        "baseline": cfg["clean"],
        "injection_engine_version": ENGINE_VERSION,
        "rate_preset": args.rates,
        "cell_rates": preset["rates"],
        "naming_convention": (
            "目录名里的 {N}M = 该数据集**注入前**的 raw cell 总数（490M/411M/614M）；"
            "K29 的 K29_v2_rawerror_test4.7pct_14.1M 用的是**错误条数**（14,123,558）。"
            "两套命名并存，看目录名时别混。"
        ),
        "splits": splits_report,
    }
    p = out_dir / f"v2_rawerror_{key}_final_report.json"
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告已写出: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
