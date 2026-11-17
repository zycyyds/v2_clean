#!/usr/bin/env python3
"""⚠️ **未采用的备选方案（gold 对齐模式）—— 默认别用它。**

> **当前采用的是同目录的 `fix_k29_log_rawform.py`**（把 `clean_value` 改成
> **与 4.7pct 同一约定** = MIMIC 原始串形式；实测对 4.7pct 的 log 零改动）。
> 本脚本是另一种思路（把 `clean_value` 改成**与 gold 逐字节同形式**），
> 2026-11-17 被明确否决，**没有在任何数据集上执行过**，只留作备选记录。
> ⇒ 想统一两档口径时，用 `fix_k29_log_rawform.py`。

修 mimicrawerror 数据集 modification_log.csv 里 `clean_value` 的**字符串形式**。

## 为什么
评分器（老口径 `--string-compare`）拿 **log 的 `clean_value`** 当"这一格该修成什么"的判据。
但两档的 log 里 `clean_value` 的写法不一样：
- 4.7pct（v2 重建引擎）：保留 MIMIC 原始字符串（`'160'`）—— 而它自己的 gold 是 float repr（`'160.0'`）
  ⇒ **agent 写对了（与 gold 逐字节相同）却被判「没修好」**
- 1.6pct（legacy 引擎）：写 float repr ⇒ 与 gold 同形式 ⇒ 不罚

⇒ 所以「老口径的 Det/Rep 绝对值」会随**写 log 的引擎**变，同一份 agent 输出在两档能差 0.11。

## 怎么修
把每条 `cell_update` 的 `clean_value` 换成 **gold（reference）同一 (行号, 列) 上的字符串**。
- gold 的行号 L 就是 log 的索引空间（log 的 `raw_row_index` = reference 行号）✓
- **只在两者数值相等时替换**；数值不等 ⇒ 保留原值并计数（那是另一个问题，要单独查）
- **`dirty_value` 一律不动**：它必须与 raw 里实际写入的内容一致，否则 `apply_modification_log.py`
  重建出来的 raw 会变（raw 必须与源数据集逐字节相同，除被恢复的叠脏格）
- 其它列、其它行、行序全部原样保留 ⇒ 写临时文件后原子替换

用法：
    python3 fix_k29_log_cleanvalue.py --dataset <mimicraw 数据集目录>            # 只报告
    python3 fix_k29_log_cleanvalue.py --dataset <mimicraw 数据集目录> --fix      # 真改
只输出聚合，不打印行级内容。
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

# raw 文件 → (agent 输出文件名, 键列)。与 score_only/error_metrics/compute_error_metrics.py
# 里的 RAW_TO_OUTPUT 一致。
RAW_TO_OUTPUT = {
    "hosp/patients.csv": ("demographics.csv",
                           ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
    "hosp/admissions.csv": ("admissions.csv",
                            ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
    "hosp/diagnoses_icd.csv": ("diagnoses.csv",
                               ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
    "icu/icustays.csv": ("icu/icustays.csv",
                         ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
    "icu/chartevents.csv": ("icu/chartevents.csv",
                            ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
    "icu/inputevents.csv": ("icu/inputevents.csv",
                            ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
    "icu/outputevents.csv": ("icu/outputevents.csv",
                             ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
}
SAMPLE = 500


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def same_num(a, b):
    fa, fb = fnum(a), fnum(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) < 1e-9


def gold_rows(path, want):
    """{row: {col: string}}"""
    out = defaultdict(dict)
    if not want:
        return out
    mx = max(want)
    with path.open(newline="", encoding="utf-8") as f:
        for L, r in enumerate(csv.DictReader(f)):
            if L in want:
                out[L] = dict(r)
            if L >= mx:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    ds = Path(args.dataset)
    log_path = ds / args.split / "modification_log.csv"
    ref_root = ds / args.split / "reference"

    # ---- 1) 扫 log：按 (file, column) 采样 ----
    samples = defaultdict(list)
    counts = Counter()
    with log_path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames
        for e in rd:
            if e.get("operation") != "cell_update":
                continue
            key = (e.get("raw_file"), e.get("column"))
            counts[key] += 1
            s = samples[key]
            if len(s) < SAMPLE:
                s.append((int(e["raw_row_index"]), e.get("clean_value")))
    print("log 里 cell_update 的 (file, column) 共 %d 种，条目合计 %d"
          % (len(counts), sum(counts.values())))

    # ---- 2) 抽样判定：哪些 (file, column) 受影响（数值相等但字符串不同）----
    affected = set()
    for (rf, col), s in sorted(samples.items()):
        out_rel = RAW_TO_OUTPUT.get(rf, (None, None))[0]
        if out_rel is None:
            print("   %-26s %-14s ⚠️ 不在 RAW_TO_OUTPUT 里，跳过" % (rf, col))
            continue
        refp = ref_root / out_rel
        if not refp.is_file():
            continue
        g = gold_rows(refp, {L for L, _ in s})
        same = diff = numeq = other = 0
        for L, cv in s:
            row = g.get(L)
            if row is None:
                other += 1
                continue
            gs = row.get(col)
            if gs is None:
                other += 1
                continue
            if str(cv) == str(gs):
                same += 1
            elif same_num(cv, gs):
                diff += 1
                numeq += 1
            else:
                other += 1
        tag = ""
        if numeq:
            affected.add((rf, col))
            tag = "  ⚠️ **受影响**"
        print("   %-26s %-14s 条 %9d | 与 gold 同串 %4d / 数值同串不同 %4d / 其他 %4d%s"
              % (rf, col, counts[(rf, col)], same, numeq, other, tag))

    if not affected:
        print("\n没有受影响的列 ⇒ log 的 clean_value 形式已经与 gold 一致，无需修改。")
        return
    print("\n受影响列 %d 种：%s" % (len(affected), sorted(affected)))
    if not args.fix:
        print("（--dry-run：只报告，不改文件）")
        return

    # ---- 3) 载入受影响列的 gold 字符串 ----
    print("\n载入 gold 受影响列 …", flush=True)
    gold_vals = {}
    for rf, col in sorted(affected):
        out_rel = RAW_TO_OUTPUT[rf][0]
        refp = ref_root / out_rel
        vals = []
        with refp.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vals.append(r.get(col))
        gold_vals[(rf, col)] = vals
        print("   %-26s %-14s %d 行" % (rf, col, len(vals)), flush=True)

    # ---- 4) 流式重写 log ----
    tmp = log_path.with_suffix(".csv.fixing")
    stat = Counter()
    with log_path.open(newline="", encoding="utf-8") as fin, \
            tmp.open("w", newline="", encoding="utf-8") as fout:
        rd = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for e in rd:
            if e.get("operation") == "cell_update":
                k = (e.get("raw_file"), e.get("column"))
                if k in gold_vals:
                    vals = gold_vals[k]
                    L = int(e["raw_row_index"])
                    cv = e.get("clean_value")
                    if 0 <= L < len(vals):
                        gs = vals[L]
                        if gs is not None and str(cv) != str(gs):
                            if same_num(cv, gs):
                                e["clean_value"] = gs
                                stat["已改成 gold 的字符串形式"] += 1
                            else:
                                stat["数值不等 ⇒ 保留原值（另有原因）"] += 1
                        else:
                            stat["本来就与 gold 同串"] += 1
            stat["总条目"] += 1
            w.writerow(e)
    print("\n改写统计：")
    for k, v in stat.most_common():
        print("   %-30s %12s" % (k, f"{v:,}"))

    # ---- 5) 原子替换 ----
    os.replace(tmp, log_path)
    print("\n已写回：%s" % log_path)


if __name__ == "__main__":
    main()
