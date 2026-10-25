"""Merge build-stage mod_log subtype 分类到 reconstruct-stage mod_log.

原因：
- reconstruct mod_log 是 ground truth（raw vs ref 实际 diff，行号 100% 对应）
- 但 infer_subtype() 只识别 11 个 subtype，剩 7 个 catch-all 到 cell_value_mismatch
- build 阶段 mod_log 有 18 subtype 全分类，但 row_index 错位（不能直接用）

策略：
- 用 (raw_file, column, dirty_value, clean_value) 4 元组作为 join key（避开 row_index）
- reconstruct 的 entry 命中 build 的 entry → 拿 build 的 error_subtype
- 未命中（baseline 自身差异） → 保留 reconstruct 的 infer_subtype() 结果
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

NEW_ROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/K29_clean_v2_rawerror_highrate_10pct_v2")
BUILD_LOG = Path("/tmp/v2_logs")


def enrich_split(split: str) -> dict:
    print(f"\n========== {split} ==========", flush=True)
    rec_path = NEW_ROOT / split / "modification_log.csv"
    build_path = BUILD_LOG / f"K29_clean_v2_rawerror_highrate_10pct_v2_{split}_modification_log.csv"
    if not build_path.exists():
        print(f"  [SKIP] build log not found: {build_path}")
        return {}

    # 1. 加载 build log 建索引 (raw_file, column, dirty_value, clean_value) -> subtype
    print(f"  loading build log: {build_path}", flush=True)
    build_index: dict[tuple, str] = {}
    build_classes: dict[tuple, str] = {}
    with build_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # cell_update/duplicate 用 (file, col, dirty) 三元组
            # insert/duplicate 用 (file, dirty="(insert)" or "(duplicate)") 标识
            k = (r["raw_file"], r["column"], r["dirty_value"], r["clean_value"])
            if k not in build_index:
                build_index[k] = r["error_subtype"]
                build_classes[k] = r["error_class"]
    print(f"  build log: {len(build_index)} unique keys", flush=True)

    # 2. 加载 reconstruct log
    print(f"  loading reconstruct log: {rec_path}", flush=True)
    rec_rows = list(csv.DictReader(rec_path.open(newline="", encoding="utf-8")))
    print(f"  reconstruct log: {len(rec_rows)} entries", flush=True)

    # 3. enrich
    n_matched = 0
    n_unmatched = 0
    new_subtype_counts: dict[str, int] = defaultdict(int)
    old_subtype_counts: dict[str, int] = defaultdict(int)

    for r in rec_rows:
        old_subtype_counts[r["error_subtype"]] += 1
        k = (r["raw_file"], r["column"], r["dirty_value"], r["clean_value"])
        if k in build_index:
            r["error_subtype"] = build_index[k]
            r["error_class"] = build_classes[k]
            n_matched += 1
        else:
            n_unmatched += 1
        new_subtype_counts[r["error_subtype"]] += 1

    # 4. 写回
    fieldnames = list(rec_rows[0].keys())
    with rec_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rec_rows)
    print(f"  matched: {n_matched}, unmatched: {n_unmatched}", flush=True)
    print(f"  new subtype distribution:", flush=True)
    for s, n in sorted(new_subtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:<40} {n:>10,}", flush=True)

    return {
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "old_subtypes": dict(old_subtype_counts),
        "new_subtypes": dict(new_subtype_counts),
    }


def main():
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["train", "validation", "test"]
    summary = {}
    for s in splits:
        summary[s] = enrich_split(s)
    print(f"\n[OK] all splits enriched")


if __name__ == "__main__":
    main()
