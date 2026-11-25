#!/usr/bin/env python3
"""独立核 codex 的 train 自检：work/train_out vs 切片的 train/reference（逐表逐格）。"""
import csv
import sys
from pathlib import Path

csv.field_size_limit(2 ** 31)
RES = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/n18-codex-pipeline/"
           "agent_work/n18_codex_16/work/train_out")
REF = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_codex_16/train/reference")

KEY = {
    "admissions.csv": ["hadm_id"],
    "cohort.csv": ["subject_id", "hadm_id"],
    "demographics.csv": ["subject_id"],
    "diagnoses.csv": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "procedures.csv": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "medications.csv": ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "dose_val_rx", "dose_unit_rx", "route"],
    "labs.csv": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
    "icu/icustays.csv": ["stay_id"],
    "icu/chartevents.csv": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"],
    "icu/inputevents.csv": ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount"],
    "icu/outputevents.csv": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"],
    "multimodal/edstays.csv": ["stay_id"],
    "multimodal/ed_diagnosis.csv": ["subject_id", "stay_id", "seq_num", "icd_code", "icd_version"],
    "multimodal/ed_pyxis.csv": ["subject_id", "stay_id", "charttime", "med_rn", "name", "gsn_rn", "gsn"],
    "multimodal/ed_medrecon.csv": ["subject_id", "stay_id", "charttime", "name", "gsn", "ndc", "etc_rn", "etccode", "etcdescription"],
    "multimodal/ed_triage.csv": ["subject_id", "stay_id"],
    "multimodal/ed_vitalsign.csv": ["subject_id", "stay_id", "charttime"],
    "multimodal/notes.csv": ["note_id"],
}


def load(p, keys):
    idx = {}
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        for r in rd:
            k = tuple(r.get(c, "") for c in keys)
            idx[k] = r
    return idx, cols


print("=== codex train 自检：work/train_out vs train/reference ===")
print("%-32s %8s %8s %10s %10s" % ("表", "codex", "ref", "key 命中", "错 cell"))
tot_bad = 0
for rel, keys in KEY.items():
    a, b = RES / rel, REF / rel
    if not a.exists():
        print("%-32s %8s %8s   ❌ 缺文件" % (rel, "—", "—"))
        continue
    ia, ca = load(a, keys)
    ib, cb = load(b, keys)
    common = set(ia) & set(ib)
    bad = 0
    for k in common:
        for col in cb:
            if col in keys:
                continue
            if (ia[k].get(col, "") or "") != (ib[k].get(col, "") or ""):
                bad += 1
    tot_bad += bad
    print("%-32s %8d %8d %10d %10d%s"
          % (rel, len(ia), len(ib), len(common), bad,
             "" if bad == 0 and len(ia) == len(ib) else "   ⚠️"))
print("\n⇒ train 自检合计错 cell: %d" % tot_bad)
