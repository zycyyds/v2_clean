#!/usr/bin/env python3
"""make_icu5000_manifest.py —— 给 icu5000 的 mimicrawerror 两档生成 dataset manifest。

harness 的 `dataset_manifest` 只被检查「存在」并记进 run_manifest（路径 + sha256），
不影响评分 —— 但如果直接把另一个档位的 manifest 复制过来，里面记的
cell_level_error_rate 是**别的档位**的值，档位信息就错了。

本脚本从**权威来源**取真值，两档字段结构与单位统一：

  16 (1.6pct) ← datasets/v2_rawerror_validation_report_1.6pct.json 里 icu5000 那条
  47 (4.7pct) ← 源数据集自带的 v2_rawerror_icu5000_final_report.json

单位约定（两档一致、便于直接比）：
  cell_level_error_rate          = **百分数**（1.58 / 4.72）
  cell_level_error_rate_fraction = 分数（0.0158 / 0.0472）

用法:
  python3 make_icu5000_manifest.py --preset 16|47
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
DSROOT = PROJ / "datasets/注入错误数据集/icu5000"
REPORT_16 = PROJ / "datasets/v2_rawerror_validation_report_1.6pct.json"

NAMING = ("目录名里的 {N}M = 该数据集**注入前**的 raw cell 总数（614M）；"
          "与 K29 的 K29_v2_rawerror_test4.7pct_14.1M（用错误条数命名）不是一套，"
          "看 icu5000 的目录名时别混。")

PRESETS = {
    "16": {"src_name": "icu_5000_v2_rawerror_test1.6pct_614M",
           "preset": "1.6pct"},
    "47": {"src_name": "icu_5000_v2_rawerror_test4.7pct_614M",
           "preset": "4.7pct"},
}


def load_source(preset: str):
    if preset == "16":
        entries = json.loads(REPORT_16.read_text(encoding="utf-8"))
        entry = next(e for e in entries
                     if e["name"] == PRESETS["16"]["src_name"])
        return {
            "name": entry["name"],
            "baseline": entry.get("old_baseline") or entry.get("baseline"),
            "injection_engine_version": entry.get("injection_engine_version"),
            "cell_rates": entry.get("cell_rates"),
            "test": entry["splits"]["test"],
        }
    src = DSROOT / PRESETS["47"]["src_name"] / "v2_rawerror_icu5000_final_report.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    return {
        "name": d["name"],
        "baseline": d.get("baseline"),
        "injection_engine_version": d.get("injection_engine_version"),
        "cell_rates": d.get("cell_rates"),
        "test": d["splits"]["test"],
    }


def build_applied(t: dict) -> dict:
    """两档的源数据结构不完全一样（16 档有 applied，47 档没有），这里补齐成同一形状。"""
    if t.get("applied"):
        a = t["applied"]
        return {
            "n_cell_updates": a.get("n_cell_updates", 0),
            "n_duplicates": a.get("n_duplicates", 0),
            "n_inserts": a.get("n_inserts", 0),
            "n_skipped": a.get("n_skipped", 0),
            "by_file": a.get("by_file", {}),
        }
    op = t.get("operation_counts", {})
    return {
        "n_cell_updates": op.get("cell_update", 0),
        "n_duplicates": op.get("duplicate", 0),
        "n_inserts": op.get("insert", 0),
        "n_skipped": 0,
        "by_file": t.get("file_counts", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    args = ap.parse_args()
    preset = args.preset
    src = load_source(preset)
    t = src["test"]

    frac = t["cell_level_error_rate"]
    rate_pct = round(frac * 100, 2)
    ratio = t.get("subtype_ratio")
    failures = []
    if ratio is not None and ratio > 500.0:
        failures.append("subtype_ratio %.2f > 500.0" % ratio)
    gap = abs(t["actual_errors"] - t["target_errors"]) / max(1, t["target_errors"])
    if gap > 0.05:
        failures.append("actual_count %d too far from target %d"
                        % (t["actual_errors"], t["target_errors"]))

    manifest = {
        "name": src["name"],
        "baseline": src["baseline"],
        "injection_engine_version": src["injection_engine_version"],
        "rate_preset": PRESETS[preset]["preset"],
        "cell_rates": src["cell_rates"],
        "naming_convention": NAMING,
        "splits": {
            "test": {
                "target_errors": t["target_errors"],
                "actual_errors": t["actual_errors"],
                "total_cells": t["total_cells"],
                "total_rows": t["total_rows"],
                "cell_level_error_rate": rate_pct,
                "cell_level_error_rate_fraction": frac,
                "subtype_counts": t.get("subtype_counts", {}),
                "operation_counts": t.get("operation_counts", {}),
                "error_class_counts": t.get("error_class_counts", {}),
                "file_counts": t.get("file_counts", {}),
                "natural_noise_count": t.get("natural_noise_count"),
                "stays_touched": t.get("stays_touched"),
                "subtype_coverage": t.get("subtype_coverage"),
                "subtype_ratio": ratio,
                "validation_passed": not failures,
                "validation_failures": failures,
                "applied": build_applied(t),
            },
        },
        "files": {
            "driver": str(PROJ / "reproduction/build_icu5000_rebuild_inject.py"),
            "config": str(PROJ / "reproduction/build_v2_error_configs.py"),
            "authoritative_source": (
                str(REPORT_16) if preset == "16" else
                str(DSROOT / PRESETS["47"]["src_name"]
                    / "v2_rawerror_icu5000_final_report.json")),
        },
    }

    out = DSROOT / (src["name"] + "_mimicrawerror") / "v2_rawerror_icu5000_final_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写出:", out)
    print("  cell_level_error_rate = %s%%  (fraction %s)"
          % (rate_pct, frac))
    print("  actual_errors = %s  total_cells = %s  total_rows = %s"
          % (t["actual_errors"], t["total_cells"], t["total_rows"]))
    print("  validation_failures =", failures)


if __name__ == "__main__":
    main()
