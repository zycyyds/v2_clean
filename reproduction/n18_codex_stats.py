#!/usr/bin/env python3
"""算 n18 mimicraw 1.6pct 的错误统计（给 codex prompt 用）。

口径对齐 k29_codex_16_prompt.md：
  错误条数 = modification_log 行数
  相关 cell = distinct (raw_file, raw_row_index, column)
  cell-level% = 相关 cell / 涉及 raw 表的总 cell（非 key 列）
  ICU stay 数 = distinct canonical_stay_id
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)
P = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18/"
         "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M")


def stats(split):
    log_p = P / split / "modification_log.csv"
    nat_p = P / split / "mimic_native_error_log.csv"
    rows = []
    with open(log_p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    cells = {(r["raw_file"], r["raw_row_index"], r["column"]) for r in rows}
    by_op = Counter(r["operation"] for r in rows)
    by_cls = Counter(r["error_class"] for r in rows)
    by_sub = Counter(r["error_subtype"] for r in rows)
    by_file = Counter(r["raw_file"] for r in rows)
    stays = {r["canonical_stay_id"] for r in rows if r.get("canonical_stay_id")}
    subjects = {r.get("matching_columns") for r in rows[:1]}

    # 各 raw 文件的 cell 总量（非 key 列）
    KEYCOLS = defaultdict(set)
    total_cells = 0
    per_file_cells = {}
    for rf in sorted(by_file):
        rp = P / split / "raw" / rf
        if not rp.exists():
            continue
        with open(rp, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            hdr = next(rd)
            n = 0
            for _ in rd:
                n += 1
        # key 列：hadm_id / subject_id / stay_id / charttime 之类，按列名判断
        keyc = sum(1 for c in hdr if c in ("hadm_id", "subject_id", "stay_id", "charttime",
                                           "seq_num", "icd_version", "study_id"))
        cellsn = n * max(1, len(hdr) - keyc)
        per_file_cells[rf] = (n, len(hdr), cellsn)
        total_cells += cellsn

    nat_n = 0
    if nat_p.exists():
        with open(nat_p, newline="", encoding="utf-8") as f:
            nat_n = sum(1 for _ in csv.DictReader(f))

    print("=" * 88)
    print("【%s】错误条数 %d" % (split, len(rows)))
    print("  相关 cell（distinct file,row,col）: %d" % len(cells))
    print("  涉及 raw 表 %d 张；这些表总 cell（非 key 列）: %d  ⇒ cell-level %.2f%%"
          % (len(by_file), total_cells, 100.0 * len(cells) / total_cells if total_cells else 0))
    print("  影响的 ICU stay: %d" % len(stays))
    print("  原生错误（mimic_native_error_log）: %d 格" % nat_n)
    print("  按操作类型: %s" % dict(by_op.most_common()))
    print("  按 error_class:")
    for k, v in by_cls.most_common():
        print("     %-42s %d" % (k, v))
    print("  按 error_subtype:")
    for k, v in by_sub.most_common():
        print("     %-42s %d" % (k, v))
    print("  按 raw 表:")
    for k, v in by_file.most_common():
        n, nc, cn = per_file_cells.get(k, (0, 0, 0))
        print("     %-28s 注入 %9d  (该表 %d 行 × %d 列 ⇒ %d cell)"
              % (k, v, n, nc, cn))
    return {
        "entries": len(rows), "cells": len(cells), "files": len(by_file),
        "total_cells": total_cells, "stays": len(stays), "native": nat_n,
        "by_op": dict(by_op), "by_cls": dict(by_cls), "by_sub": dict(by_sub),
        "by_file": dict(by_file),
    }


out = {}
for s in ("train", "test"):
    out[s] = stats(s)
Path("/Users/mac/.claude/jobs/448aad57/tmp/n18_codex_stats.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n写出 n18_codex_stats.json")
