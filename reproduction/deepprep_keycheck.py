#!/usr/bin/env python3
"""精确核对：4 个 deepprep 目录里，哪些输出表**缺工具要的 key 列**或**是空表**。

工具要的 key（score_only/error_metrics/compute_error_metrics.py 的 RAW_TO_OUTPUT）：
  hosp/patients.csv     → demographics.csv      : subject_id
  hosp/admissions.csv   → admissions.csv        : hadm_id
  hosp/diagnoses_icd.csv→ diagnoses.csv        : subject_id, hadm_id, seq_num, icd_code, icd_version
  icu/icustays.csv      → icu/icustays.csv      : stay_id
  icu/chartevents.csv   → icu/chartevents.csv   : subject_id, hadm_id, stay_id, charttime, itemid
  icu/inputevents.csv   → icu/inputevents.csv   : subject_id, hadm_id, stay_id, starttime, itemid
  icu/outputevents.csv  → icu/outputevents.csv  : subject_id, hadm_id, stay_id, charttime, itemid
"""
import csv
import os
from pathlib import Path

csv.field_size_limit(2 ** 31)
DL = Path("/Users/mac/Downloads/ceshi1")

NEED = {
    "demographics.csv": ["subject_id"],
    "admissions.csv": ["hadm_id"],
    "diagnoses.csv": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "icu/icustays.csv": ["stay_id"],
    "icu/chartevents.csv": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"],
    "icu/inputevents.csv": ["subject_id", "hadm_id", "stay_id", "starttime", "itemid"],
    "icu/outputevents.csv": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"],
}

for rel in ("n18/deepprep-less/output", "n18/deepprep-more/output",
            "I48/deepprep-less/output", "I48/deepprep-more/output"):
    root = DL / rel
    print("【%s】" % rel)
    fatal = []
    for tbl, keys in NEED.items():
        p = root / tbl
        if not p.exists():
            print("   %-22s ❌ 缺文件" % tbl)
            fatal.append(tbl + "(缺文件)")
            continue
        size = p.stat().st_size
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            rd = csv.reader(f)
            hdr = next(rd, None) or []
            n = sum(1 for _ in rd)
        miss = [k for k in keys if k not in hdr]
        prob = []
        if not hdr:
            prob.append("空文件(无表头) ⇒ 建表语法错")
        if miss:
            prob.append("缺 key 列 %s" % ",".join(miss))
        if n == 0:
            prob.append("0 行数据")
        if prob:
            fatal.append("%s(%s)" % (tbl, "；".join(prob)))
            print("   %-22s ❌ %s" % (tbl, "；".join(prob)))
        else:
            print("   %-22s ✅ %d 行，key 齐" % (tbl, n))
    print("   ⇒ 致命项: %s" % ("；".join(fatal) if fatal else "无（这个目录可算错误级）"))
    print()
