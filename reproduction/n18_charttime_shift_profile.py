#!/usr/bin/env python3
"""n18_charttime_shift_profile.py —— 原生层里「行被挪过」的格，
raw 的 charttime 相对 reference 偏了多少？

目的：判断这个偏移**是不是确定性的**（能写成规则进 prompt），
还是随机的（只能靠模型猜）。

做法：对每个「raw 里 key 对不上」的原生格，
在同一个 (subject, hadm, stay) 里找 itemid 相同（别名还原后）且最接近的行，
算 delta = raw_charttime − 原生层登记的 charttime（分钟）。
只输出 delta 的分布（聚合），不输出病人级标识与逐格医学值。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

csv.field_size_limit(2 ** 31)

KEY5 = ("subject_id", "hadm_id", "stay_id", "charttime", "itemid")


def restore_itemid(i):
    if i.startswith("99") and len(i) >= 5:
        return "2" + i[3] + i[-4:]
    return i


def parse_ct(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_of(r):
    return tuple(r.get(c, "") for c in KEY5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ds, res, sp = Path(args.dataset), Path(args.result), args.split

    nat = load(ds / sp / "mimic_native_error_log.csv")
    # 关心的 (subject, hadm, stay, itemid) 四元组（原生层那些格）
    want_keys = {(r["subject_id"], r["hadm_id"], r["stay_id"], r["itemid"])
                 for r in nat}

    # 扫 raw：同一 (subject, hadm, stay, itemid还原后) 的所有 charttime
    rows_by = {}
    with (ds / sp / "raw" / "icu" / "chartevents.csv").open(
            newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            it = restore_itemid(r.get("itemid", ""))
            k = (r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"), it)
            if k in want_keys:
                rows_by.setdefault(k, []).append(r.get("charttime", ""))

    deltas = Counter()
    sign = Counter()
    n_disp = 0
    n_exact = 0
    for r in nat:
        want = (r["subject_id"], r["hadm_id"], r["stay_id"], r["itemid"])
        cands = rows_by.get(want, [])
        t0 = parse_ct(r["charttime"])
        if t0 is None or not cands:
            continue
        ds_ = []
        for c in cands:
            t1 = parse_ct(c)
            if t1 is not None:
                ds_.append((t1 - t0).total_seconds() / 60.0)
        if not ds_:
            continue
        if 0.0 in ds_:
            n_exact += 1
            continue
        n_disp += 1
        best = min(ds_, key=abs)
        d = int(round(best))
        deltas[d] += 1
        sign["raw 更晚" if d > 0 else "raw 更早"] += 1

    print("=== %s：原生格在 raw 里 charttime 的偏移分布 ===" % sp)
    print("  raw 里 charttime 正好对得上: %d 格" % n_exact)
    print("  charttime 被挪过（取最近的一个候选）: %d 格" % n_disp)
    if sign:
        print("  方向：%s" % dict(sign))
    print()
    print("  偏移量 top 15（分钟）:")
    for d, n in deltas.most_common(15):
        print("    %+8d 分钟  (%+.2f 小时)  : %d 格" % (d, d / 60.0, n))
    if deltas:
        tot = sum(deltas.values())
        top = deltas.most_common(1)[0]
        print()
        print("  ⇒ 单一偏移量占的比例: %.1f%%（%d/%d）"
              % (100.0 * top[1] / tot, top[1], tot))
        print("  ⇒ 偏移量不同的取值个数: %d" % len(deltas))


if __name__ == "__main__":
    main()
