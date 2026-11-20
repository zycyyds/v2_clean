#!/usr/bin/env python3
"""n18_unrecoverable_fields.py —— 列出 N18 里「raw 里没有信息、reference 里有值」的字段。

为什么需要它：这类字段是**结构性不可判定**的 —— raw 里根本没有线索，
任何 pipeline 都修不出来 ⇒ 它们是 composite 分数的**上限**，不是 agent 的错。

（与台账里已有的「`demographics.anchor_age` 三档全中」同型；
 N18 还多一个 `admissions.insurance`。）

用法: python3 n18_unrecoverable_fields.py [train|validation] [视图目录]
只输出字段名与统计，不含病人级标识。
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

DEFAULT_VIEW = Path("/Users/mac/PycharmProjects/v2_clean/datasets/val_views/n18_mimicraw_16")

# 输出文件 → raw 源表（一个输出文件可能由多张 raw 表拼出来）
PAIRS = {
    "admissions.csv": ("hosp/admissions.csv",),
    "cohort.csv": ("hosp/admissions.csv", "hosp/diagnoses_icd.csv"),
    # ⚠️ demographics 的源表**不止 patients.csv**：MIMIC-IV-ED 的 ed/edstays.csv
    #    也带 gender 列，而 patients.csv 里的 gender 被去标识化成 UNKNOWN_GENDER_CODE
    #    ⇒ gender 其实**可以**从 ed/edstays.csv join 回来（实测 agent 做到了，
    #    M:16 / F:4 与 reference 完全一致）。
    "demographics.csv": ("hosp/patients.csv", "ed/edstays.csv"),
    "diagnoses.csv": ("hosp/diagnoses_icd.csv",),
    "procedures.csv": ("hosp/procedures_icd.csv",),
    "medications.csv": ("hosp/prescriptions.csv",),
    "labs.csv": ("hosp/labevents.csv",),
    "icu/icustays.csv": ("icu/icustays.csv",),
    "icu/chartevents.csv": ("icu/chartevents.csv",),
    "icu/inputevents.csv": ("icu/inputevents.csv",),
    "icu/outputevents.csv": ("icu/outputevents.csv",),
    "multimodal/notes.csv": ("note/discharge.csv", "note/radiology.csv",
                             "note/discharge_detail.csv", "note/radiology_detail.csv"),
    "multimodal/edstays.csv": ("ed/edstays.csv",),
    "multimodal/ed_triage.csv": ("ed/triage.csv",),
    "multimodal/ed_vitalsign.csv": ("ed/vitalsign.csv",),
    "multimodal/ed_pyxis.csv": ("ed/pyxis.csv",),
    "multimodal/ed_medrecon.csv": ("ed/medrecon.csv",),
    "multimodal/ed_diagnosis.csv": ("ed/diagnosis.csv",),
}


def raw_profile(raw: Path, srcs):
    """(raw 里出现过的列, 每列非空的行数, 每列非空值的取值集合)"""
    cols, nonempty = set(), Counter()
    values = defaultdict(set)
    for s in srcs:
        p = raw / s
        if not p.is_file():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for c, v in r.items():
                    if c is None:
                        continue
                    cols.add(c)
                    if v is not None and str(v).strip() != "":
                        nonempty[c] += 1
                        if len(values[c]) < 8:      # 只留前几个不同取值，够判「是否常量」
                            values[c].add(str(v).strip())
    return cols, nonempty, values


# agent 推导列：raw 里本来就没有，但**能推出来**（从 ICD 过滤条件 / 年龄分桶）
# ⇒ 不算「不可判定」，只是不能照抄。
DERIVABLE = {
    "cohort.csv": {"cohort_type", "cohort_label", "icd_root"},
    "demographics.csv": {"anchor_year_group"},
}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    view = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_VIEW
    raw, ref = view / split / "raw", view / split / "reference"

    print("=== %s 分片：reference 有值、但 raw 里**无任何信息**的字段 ===" % split)
    n_lost_total = 0
    injected_all = []
    for rel in sorted(PAIRS):
        rp = ref / rel
        if not rp.is_file():
            continue
        cols, nonempty, rvalues = raw_profile(raw, PAIRS[rel])
        with rp.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        derivable = DERIVABLE.get(rel, set())
        lost, derived, injected = [], [], []
        for c in rows[0].keys():
            if c is None:
                continue
            n_ref = sum(1 for r in rows
                        if r.get(c) is not None and str(r.get(c)).strip() != "")
            if n_ref == 0:
                continue
            if c in derivable:
                derived.append((c, n_ref))
                continue
            ref_vals = {str(r.get(c)).strip() for r in rows
                        if r.get(c) is not None and str(r.get(c)).strip() != ""}
            if c not in cols:
                lost.append((c, n_ref, "raw 里没有这一列"))
                n_lost_total += n_ref
            elif nonempty.get(c, 0) == 0:
                lost.append((c, n_ref, "raw 里整列为空"))
                n_lost_total += n_ref
            elif len(rvalues.get(c, ())) == 1 and len(ref_vals) > 1:
                # ⚠️ raw 里整列是**同一个常量**，而 reference 里是多个不同值
                #    两种成因，分开算：
                #    ① MIMIC 去标识化把真值抹掉（anchor_age = 999 = 90+ 岁）
                #       ⇒ **真上限**，任何 pipeline 都修不出来
                #    ② 注入层把整列打成同一个哨兵（outputevents.charttime = epoch）
                #       ⇒ 是**注入层**的错误，能不能反推取决于注入规则本身
                const = list(rvalues[c])[0]
                if const == "999":
                    lost.append((c, n_ref, "去标识化常量 999（90+ 岁）"))
                    n_lost_total += n_ref
                else:
                    injected.append((rel, c, n_ref, const))
        injected_all.extend(injected)
        if derived:
            print("  %-28s [可推导，不算不可判定] %s"
                  % (rel, "、".join("%s(%d 格)" % d for d in derived)))
        if lost:
            print("  %-28s %s" % (rel, "; ".join(
                "%s（%d 格 · %s）" % l for l in lost)))
    print()
    print("  ⇒ 合计 %d 个格**真·结构性上限**（去标识化 / 源数据缺失）"
          " ⇒ 任何 pipeline 都修不出来，不是 agent 的错" % n_lost_total)
    if injected_all:
        print()
        print("  ⚠️ 以下是**注入层**把整列打成同一个哨兵（不是去标识化）：")
        for rel, c, n_ref, const in injected_all:
            print("     %-28s %s（%d 格 · raw 全是 %s）—— 能不能反推取决于注入规则本身"
                  % (rel, c, n_ref, const))


if __name__ == "__main__":
    main()
