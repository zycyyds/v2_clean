#!/usr/bin/env python3
"""make_n18_manifest.py —— 给 N18 的 mimicraw 两档生成统一的 dataset manifest。

harness 的 `dataset_manifest` 只被检查「存在」并记进 run_manifest（路径 + sha256），
不影响评分 —— 但如果直接把另一个档位的 manifest 复制过来，里面记的
cell_level_error_rate 是**别的档位**的值，档位信息就错了。

本脚本从**权威来源**取真值，两档字段结构与单位统一：

  · 注入层的数字 ← **源数据集**（无原生错误层）的 manifest
  · 原生错误层的数字 ← **产物实测**（mimic_native_error_log.csv +
    modification_log.csv 重新统计，不抄构建器的自检）

单位约定（两档一致、便于直接比）：
  cell_level_error_rate          = **百分数**（1.58 / 4.72）
  cell_level_error_rate_fraction = 分数（0.0158 / 0.0472）

用法:
  python3 make_i48_manifest.py --preset 16|47
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
DSROOT = PROJ / "datasets/注入错误数据集/n18"

NAMING = ("目录名里的 {N}M = 该数据集**注入前**的 raw cell 总数（490M）；"
          "与 K29 / ICU5000 的命名（用错误条数）不是一套，看 n18 的目录名时别混。")

PRESETS = {
    "16": {"src_name": "N18_ckd_ED_v2_rawerror_test1.6pct_490M",
           "dst_name": "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M",
           "preset": "1.6pct"},
    "47": {"src_name": "N18_ckd_ED_v2_rawerror_test4.7pct_490M",
           "dst_name": "N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M",
           "preset": "4.7pct"},
}

SPLITS = ("train", "validation", "test")


def native_layer_stats(dst: Path):
    """从**产物**实测原生错误层（不信构建器的自检报告）。"""
    out = {"by_split": {}, "by_itemid": {}, "by_detection_mode": {}}
    tot_mode = Counter()
    tot_inj = 0
    for sp in SPLITS:
        p = dst / sp / "mimic_native_error_log.csv"
        if not p.is_file():
            continue
        rows = []
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_item = Counter(r.get("itemid", "") for r in rows)
        by_mode = Counter(r.get("detection_mode", "") for r in rows)
        n_inj = sum(1 for r in rows if r.get("also_had_injection") == "yes")
        out["by_split"][sp] = {
            "cells": len(rows),
            "by_itemid": dict(by_item.most_common()),
            "by_detection_mode": dict(by_mode),
            "also_had_injection": n_inj,
        }
        tot_mode.update(by_mode)
        tot_inj += n_inj
        for k, v in by_item.most_common():
            out["by_itemid"][k] = out["by_itemid"].get(k, 0) + v
    out["by_detection_mode"] = dict(tot_mode)
    out["also_had_injection_total"] = tot_inj
    return out


def log_operation_counts(dst: Path):
    """从 mimicraw2 的 modification_log.csv 实测各 operation / subtype 的条数。"""
    out = {}
    for sp in SPLITS:
        p = dst / sp / "modification_log.csv"
        if not p.is_file():
            continue
        ops = Counter()
        subs = Counter()
        with p.open(newline="", encoding="utf-8") as f:
            for e in csv.DictReader(f):
                ops[e.get("operation", "")] += 1
                if e.get("error_subtype", "").startswith("mimic_native"):
                    subs[e["error_subtype"]] += 1
        out[sp] = {"operation_counts": dict(ops),
                   "native_subtype_counts": dict(subs),
                   "total": sum(ops.values())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    args = ap.parse_args()
    preset = args.preset
    src_dir = DSROOT / PRESETS[preset]["src_name"]
    dst_dir = DSROOT / PRESETS[preset]["dst_name"]
    src_manifest = json.loads(
        (src_dir / "v2_rawerror_n18_final_report.json").read_text(encoding="utf-8"))
    build_report = {}
    br = dst_dir / "mimic_native_error_build_report.json"
    if br.is_file():
        build_report = json.loads(br.read_text(encoding="utf-8"))

    nat = native_layer_stats(dst_dir)
    ops = log_operation_counts(dst_dir)

    splits = {}
    failures = []
    for sp in SPLITS:
        s = src_manifest["splits"].get(sp)
        if not s:
            continue
        frac = s.get("cell_level_error_rate")
        rate_pct = round(frac * 100, 2) if frac is not None else None
        gap = abs(s.get("actual_errors", 0) - s.get("target_errors", 0)) / \
            max(1, s.get("target_errors", 1))
        if gap > 0.05:
            failures.append("%s: actual_count %d too far from target %d"
                            % (sp, s.get("actual_errors", 0), s.get("target_errors", 0)))
        splits[sp] = {
            "target_errors": s.get("target_errors"),
            "actual_errors": s.get("actual_errors"),
            "total_cells": s.get("total_cells"),
            "total_rows": s.get("total_rows"),
            "cell_level_error_rate": rate_pct,
            "cell_level_error_rate_fraction": frac,
            "subtype_counts": s.get("subtype_counts", {}),
            "operation_counts": s.get("operation_counts", {}),
            "error_class_counts": s.get("error_class_counts", {}),
            "file_counts": s.get("file_counts", {}),
            "natural_noise_count": s.get("natural_noise_count"),
            "stays_touched": s.get("stays_touched"),
            "n_subtypes_covered": s.get("n_subtypes_covered"),
            "seed": s.get("seed"),
            # mimicraw2 的 log 实测（含原生层追加的条目）
            "mimicraw_log": ops.get(sp, {}),
        }

    manifest = {
        "name": src_manifest["name"],
        "baseline": src_manifest.get("baseline"),
        "injection_engine_version": src_manifest.get("injection_engine_version"),
        "rate_preset": PRESETS[preset]["preset"],
        "cell_rates": src_manifest.get("cell_rates"),
        "naming_convention": NAMING,
        "splits": splits,
        "mimic_native_error_layer": {
            "what": ("各 split 的 reference 里，MIMIC 自身的原生错误（T1 哨兵占位码 + "
                     "T2 生理不可能区间）已按「该患者该 itemid 的可用读数中位数」修好，"
                     "并在 modification_log 里登记为 error_subtype=mimic_native_*；"
                     "raw 里与原生格重叠的那部分已恢复成 MIMIC 原生错值。"),
            "rule": "R2_patient_itemid_median_of_usable_readings",
            "detection": ("T1 哨兵占位码（对所有 itemid 生效）+ T2 生理不可能区间"
                          "（按 PhysioNet d_items 的权威 label 文本匹配 N18 自己的区间）"
                          "；值 0 一律不算错（MIMIC 里 0 = 未测到/呼吸机暂停）"),
            "sentinel_codes": build_report.get("sentinels"),
            "hard_by_label": build_report.get("hard_by_label"),
            "unrepairable_note": ("整段都是坏值的格（该患者该项一个可用读数都没有）"
                                  "无法算中位数 ⇒ 真值不可判定 ⇒ 不修也不登记"),
            "train_val_examples_note": (
                "train/validation 的 reference 里原生错误极少（N18：train 0 格、"
                "validation 3 格，而 test 有 715 格）⇒ 往 train/validation 的 raw 里"
                "注入了**同形态**的孤立尖峰（T1 哨兵码 / T2 越界各占一半，脏值取 N18 "
                "自己的真实原生代表值），并登记在 log 里 ⇒ agent 能从 train 学到"
                "「越界值/哨兵码 ⇒ 用该患者该项的可用读数中位数」这条规则。"
                "⚠️ 注入的格一律避开「key 被注入器改过」或「valuenum 被注入器改过」的行"
                "（两层错误叠加、形态同构、不可判定 ⇒ 会把 agent 教反）。"),
            "itemid_clash_note": ("itemid 也被注入器改过的格 = 两层错误叠加、形态同构、"
                                  "不可判定 ⇒ 不进 native 层，交给注入层评测"),
            "by_split": nat["by_split"],
            "by_itemid_total": nat["by_itemid"],
            "by_detection_mode_total": nat["by_detection_mode"],
            "also_had_injection_total": nat["also_had_injection_total"],
            "build_report": build_report.get("splits", {}),
        },
        "validation_failures": failures,
        "files": {
            "driver": str(PROJ / "reproduction/build_n18_mimic_raw_error.py"),
            "verifier": str(PROJ / "reproduction/verify_n18_mimic_raw_error.py"),
            "authoritative_source": str(
                src_dir / "v2_rawerror_i48_final_report.json"),
            "dataset": str(dst_dir),
        },
    }

    out = dst_dir / "v2_rawerror_n18_final_report.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("已写出:", out)
    for sp, v in splits.items():
        nl = nat["by_split"].get(sp, {})
        print("  %-11s 注入 %-10s 原生层 %-6s（T1 %s / T2 %s / 示例 %s）"
              % (sp, v["actual_errors"], nl.get("cells", 0),
                 nl.get("by_detection_mode", {}).get("T1_sentinel", 0),
                 nl.get("by_detection_mode", {}).get("T2_impossible", 0),
                 sum(nl.get("by_detection_mode", {}).get(k, 0)
                     for k in nl.get("by_detection_mode", {})
                     if k.endswith("_injected_example"))))


if __name__ == "__main__":
    main()
