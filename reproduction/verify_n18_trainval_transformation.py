#!/usr/bin/env python3
"""verify_n18_trainval_transformation.py —— 核查 N18 mimicraw 的 train/validation
**到底改造了什么**（与母数据集逐格对比）。

三项核查：
  ① reference：mimicraw 的 reference 与母数据集的 reference 差了几格？差在哪？
  ② raw     ：mimicraw 的 raw 与母数据集的 raw 差了几格？差在哪？
  ③ log     ：mimicraw 的 modification_log 里登记了多少 mimic_native_* 条目？
外加：非 chartevents 的输出文件（labs / medications / …）是否**完全没动**。

只输出聚合，不含病人级标识。
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)

ROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18")
PAIRS = {
    "1.6pct": ("N18_ckd_ED_v2_rawerror_test1.6pct_490M",
               "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"),
    "4.7pct": ("N18_ckd_ED_v2_rawerror_test4.7pct_490M",
               "N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"),
}
# 同一套口径对照 i48 mimicraw2（回答「和前面几类一样吗」）
I48 = {
    "16": ("I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M",
           "I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M"),
    "47": ("I48_atrial_fibrillation_v2_rawerror_test4.7pct_411M",
           "I48_atrial_fibrillation_v2_rawerror_mimicraw2_test4.7pct_411M"),
}
CE = "icu/chartevents.csv"
KEY5 = ("subject_id", "hadm_id", "stay_id", "charttime", "itemid")


def load_by_key(path: Path):
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(tuple(r.get(c, "") for c in KEY5), []).append(r)
    return out


def diff_cells(a: Path, b: Path, cols):
    """a、b 两个 csv 按 5 段 key 对齐后，cols 里的格有几处不同。"""
    A, B = load_by_key(a), load_by_key(b)
    n_diff = 0
    by_col = Counter()
    for k, rows_a in A.items():
        rows_b = B.get(k)
        if not rows_b:
            continue
        for ra, rb in zip(rows_a, rows_b):
            for c in cols:
                va, vb = (ra.get(c) or "").strip(), (rb.get(c) or "").strip()
                if va != vb:
                    n_diff += 1
                    by_col[c] += 1
    return n_diff, by_col, len(A), len(B)


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "1.6pct"
    if tier in I48:
        src, dst = (ROOT.parent / "i48" / I48[tier][0],
                    ROOT.parent / "i48" / I48[tier][1])
        print("（对照：i48 mimicraw2 %s）" % tier)
    else:
        src_name, dst_name = PAIRS[tier]
        src, dst = ROOT / src_name, ROOT / dst_name

    for split in ("train", "validation"):
        print("=" * 78)
        print("[%s] %s" % (tier, split))
        # ---- ① reference ----
        n, by_col, nk_a, nk_b = diff_cells(
            src / split / "reference" / CE, dst / split / "reference" / CE,
            ("valuenum",))
        print("  ① reference/chartevents：与母数据集差 %d 格（按列：%s）；"
              "key 数 %d → %d" % (n, dict(by_col), nk_a, nk_b))
        # ---- ② raw ----
        n, by_col, nk_a, nk_b = diff_cells(
            src / split / "raw" / CE, dst / split / "raw" / CE, ("valuenum",))
        print("  ② raw/chartevents      ：与母数据集差 %d 格（按列：%s）；"
              "key 数 %d → %d" % (n, dict(by_col), nk_a, nk_b))
        # ---- ③ log ----
        nat = 0
        ops = Counter()
        subs = Counter()
        with (dst / split / "modification_log.csv").open(
                newline="", encoding="utf-8") as f:
            for e in csv.DictReader(f):
                ops[e.get("operation", "")] += 1
                if e.get("error_subtype", "").startswith("mimic_native"):
                    nat += 1
                    subs[e["error_subtype"]] += 1
        print("  ③ modification_log：mimic_native_* 条目 %d 条 %s"
              % (nat, dict(subs)))
        # ---- ④ 别的输出文件动没动 ----
        moved = []
        ref_dir = dst / split / "reference"
        for p in sorted(ref_dir.rglob("*.csv")):
            rel = p.relative_to(ref_dir).as_posix()
            if rel == CE:
                continue
            s = src / split / "reference" / rel
            if not s.is_file():
                continue
            nd, _, _, _ = diff_cells(s, p, None) if False else (0, {}, 0, 0)
            # 逐列比（这些表没有 5 段 key，直接按行号比）
            with s.open(newline="", encoding="utf-8") as f1, \
                    p.open(newline="", encoding="utf-8") as f2:
                r1 = list(csv.DictReader(f1))
                r2 = list(csv.DictReader(f2))
            if len(r1) != len(r2):
                moved.append("%s(行数 %d→%d)" % (rel, len(r1), len(r2)))
                continue
            c1 = Counter()
            for a, b in zip(r1, r2):
                for k in (a.keys() if a else []):
                    if (a.get(k) or "").strip() != (b.get(k) or "").strip():
                        c1[k] += 1
            if c1:
                moved.append("%s(%s)" % (rel, dict(c1)))
        print("  ④ 其余输出文件：%s"
              % ("**完全没动**" if not moved else "动了 → " + "; ".join(moved)))


if __name__ == "__main__":
    main()
