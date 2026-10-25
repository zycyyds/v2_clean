#!/usr/bin/env python3
"""重新生成 datasets/v2_rawerror_validation_report.json（共享报告）。

四个数据集各一项，K29 在前。字段沿用旧报告的命名，再补上这次新注入
需要的口径字段：

  cell_level_error_rate  错误条数 / **注入后**的 raw cell 总数
                         （duplicate / insert 会新增行，分母比 baseline 大；
                          旧 K29 报告用 baseline 当分母且按物理行计数，
                          所以 4.72% 被算成 1.66%）
  total_cells            注入后的 cell 总数（csv.reader 数记录数）
  baseline_total_cells   注入前的 cell 总数
  natural_noise_count    class0_baseline_gap 条数（自然噪声）
  verification           verify_rawerror_dataset.py 的 10 项检查结果

用法:
    python3 rebuild_shared_report.py [--rates 4.7pct] [--datasets n18 i48 icu5000]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(sys.maxsize)

from build_rawerror_by_ratio import (  # noqa: E402
    DATASETS, OUT_ROOT, RATE_PRESETS, count_raw_cells,
)

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
CLEAN_ROOT = ROOT / "datasets/常规数据集"
REPORT = ROOT / "datasets/v2_rawerror_validation_report.json"
SPLITS = ("train", "validation", "test")

K29 = {
    # 两档的目录名不同 —— 4.72pct 是 14.1M 错误，1.58pct 是 4.2M 错误
    "name": {"4.7pct": "K29_v2_rawerror_test4.7pct_14.1M",
             "1.6pct": "K29_v2_rawerror_test1.6pct_4.2M"},
    "old_baseline": "K29_clean_eval_10_20_rest",
    "injection_engine_version": "legacy",
    # 注入产物目录用 validation，干净 baseline 用 val —— 两个不一样
    "src_split_dir": {"train": "train", "validation": "validation", "test": "test"},
    "clean_split_dir": {"train": "train", "validation": "val", "test": "test"},
    "semantic_deviations": [
        "日志由 diff 重建路径生成（reconstruct_mod_log_highrate.py），"
        "含 class0_baseline_gap 自然噪声 —— raw 与 gold 天然就有的差异被记成了错误",
        "future_window_leakage 退化为「清空 charttime」（chartevents 无 outtime 列）",
        "stay_hadm_mismatch 退化为写入字面量 (error generating)",
        "raw_row_index 是**最终** raw 行号（注入引擎那条路径是注入前行号，需位移修正）",
    ],
}


def scan_log(path: Path):
    """流式扫日志。返回 (总数, subtype, file, op, class, stay 计数)。"""
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


def _stddev(c: Counter) -> float:
    if not c:
        return 0.0
    return round(statistics.pstdev(list(c.values())), 2)


def entry_for(name: str, baseline: str, root: Path, src_split_dir: dict,
              engine: str, rates: dict, deviations: list | None = None,
              clean_split_dir: dict | None = None) -> dict:
    clean_split_dir = clean_split_dir or src_split_dir
    splits = {}
    for split in SPLITS:
        sd = root / src_split_dir[split]
        log = sd / "modification_log.csv"
        raw = sd / "raw"
        if not log.is_file() or not raw.is_dir():
            continue
        n, by_subtype, by_file, by_op, by_class, by_stay = scan_log(log)
        final_cells, final_rows = count_raw_cells(raw)
        base_cells = base_rows = 0
        clean = CLEAN_ROOT / baseline / clean_split_dir[split]
        if (clean / "raw").is_dir():
            base_cells, base_rows = count_raw_cells(clean / "raw")

        sub_min, sub_max = _min_max(by_subtype)
        file_min, file_max = _min_max(by_file)
        stay_min, stay_max = _min_max(by_stay)
        s = {
            "target_errors": int(round(base_cells * rates[split])) if rates else None,
            "actual_errors": n,
            "cell_level_error_rate": round(n / final_cells, 6) if final_cells else 0,
            "total_cells": final_cells,
            "baseline_total_cells": base_cells,
            "total_rows": final_rows,
            "baseline_total_rows": base_rows,
            "subtype_coverage": len(by_subtype),
            "subtype_min": sub_min,
            "subtype_max": sub_max,
            "subtype_ratio": round(sub_max / max(1, sub_min), 4),
            "files_touched": len(by_file),
            "file_min": file_min,
            "file_max": file_max,
            "stays_touched": len(by_stay),
            "stay_min": stay_min,
            "stay_max": stay_max,
            "stay_stddev": _stddev(by_stay),
            "natural_noise_count": by_class.get("class0_baseline_gap", 0),
            "error_class_counts": dict(by_class),
            "operation_counts": dict(by_op),
            "subtype_counts": dict(by_subtype),
            "applied": {
                "n_cell_updates": by_op.get("cell_update", 0),
                "n_duplicates": by_op.get("duplicate", 0),
                "n_inserts": by_op.get("insert", 0),
                "by_file": dict(by_file),
            },
            "log_path": str(log),
        }
        vpath = sd / "verify_report.json"
        if vpath.is_file():
            try:
                s["verification"] = json.loads(vpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        splits[split] = s
        rate_txt = f"{s['cell_level_error_rate'] * 100:.4f}%"
        print(f"    {split:11s} {n:>12,} 条 / {final_cells:>14,} cells = {rate_txt}"
              f"  subtypes={len(by_subtype)}  自然噪声={s['natural_noise_count']:,}")
    e = {
        "name": name,
        "old_baseline": baseline,
        "injection_engine_version": engine,
        "cell_rates": rates,
        "splits": splits,
    }
    if deviations:
        e["semantic_deviations"] = deviations
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", choices=sorted(RATE_PRESETS), default="4.7pct")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--no-k29", action="store_true")
    args = ap.parse_args()
    preset = RATE_PRESETS[args.rates]
    tag = preset["tag"]
    rates = preset["rates"]

    out = []
    if not args.no_k29:
        k29_name = K29["name"].get(tag)
        if not k29_name or not (OUT_ROOT / k29_name).is_dir():
            print(f"  K29 (legacy): 跳过（{k29_name} 不存在）")
        else:
            print("  K29 (legacy):")
            out.append(entry_for(
                k29_name, K29["old_baseline"], OUT_ROOT / k29_name,
                K29["src_split_dir"], K29["injection_engine_version"], rates,
                K29["semantic_deviations"], K29["clean_split_dir"]))

    for key in (args.datasets or sorted(DATASETS)):
        cfg = DATASETS[key]
        name = cfg["out_base"].format(tag=tag)
        # 新数据集直接读 rebuild_dataset_report.py 产出的独立报告 ——
        # 那个脚本已经把 cell 数和日志统计量算过了，重扫一遍几十 GB 是浪费
        rep_path = OUT_ROOT / name / f"v2_rawerror_{key}_final_report.json"
        if rep_path.is_file():
            try:
                rep = json.loads(rep_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                rep = None
            if rep and rep.get("splits"):
                print(f"  {key} (fixed-v1) ← 读 {rep_path.name}")
                entry = {
                    "name": name,
                    "old_baseline": cfg["clean"],
                    "injection_engine_version": rep.get("injection_engine_version", "fixed-v1"),
                    "cell_rates": rep.get("cell_rates", rates),
                    "splits": {},
                }
                for split, s in rep["splits"].items():
                    st = dict(s)
                    st["subtype_coverage"] = st.get("n_subtypes_covered", 0)
                    st["subtype_ratio"] = round(
                        st.get("subtype_max", 0) / max(1, st.get("subtype_min", 1)), 4)
                    st["files_touched"] = len(st.get("file_counts", {}))
                    st["file_min"] = min(st.get("file_counts", {}).values(), default=0)
                    st["file_max"] = max(st.get("file_counts", {}).values(), default=0)
                    st["applied"] = {
                        "n_cell_updates": st.get("operation_counts", {}).get("cell_update", 0),
                        "n_duplicates": st.get("operation_counts", {}).get("duplicate", 0),
                        "n_inserts": st.get("operation_counts", {}).get("insert", 0),
                        "by_file": st.get("file_counts", {}),
                    }
                    vpath = OUT_ROOT / name / split / "verify_report.json"
                    if vpath.is_file():
                        try:
                            st["verification"] = json.loads(vpath.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError):
                            pass
                    entry["splits"][split] = st
                out.append(entry)
                continue
        print(f"  {key} (fixed-v1):")
        out.append(entry_for(
            name, cfg["clean"], OUT_ROOT / name, cfg["split_dir"],
            "fixed-v1", rates))

    # 4.7pct 用旧路径（保持兼容），1.6pct 另存 —— 两个口径不能互相覆盖
    path = REPORT if tag == "4.7pct" else REPORT.with_name(
        f"v2_rawerror_validation_report_{tag}.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n共享报告已写出: {path}（{len(out)} 个数据集）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
