#!/usr/bin/env python3
"""n18_order_sensitive_subset.py —— 量出「**cohort 筛选顺序敏感**」的那批原生格有多少、
以及 agent 在这批格上的修复率。

为什么要量：agent 在两档都把「修复时机」放在了 cohort 筛选**之前** ⇒
池子混进该患者**不在 cohort 里**的 stay 的读数 ⇒ 中位数偏 ⇒ 修成错值。
但这个错误**只在「该患者有多于一个 stay 且部分 stay 不在 cohort」的格上显形** ⇒
**分母里这种格越多、原生层修复率越低** ⇒ 跨档比修复率会被分母带偏。

口径：
  · 该患者在 icustays 里的 stay 数 > 该患者在 **cohort 的** stay 数 ⇒ 属「顺序敏感格」
  · 否则 ⇒ 「顺序无关格」（怎么算都一样）

用 icustays.csv（小表）算 stay 集合，不去扫 1400 万行的 chartevents。

只输出聚合，不含病人级标识。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)


def load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="mimicraw 数据集目录")
    ap.add_argument("--result", required=True, help="agent 结果包目录")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ds, res = Path(args.dataset), Path(args.result)
    sp = args.split

    # cohort 的 hadm（reference 的 cohort.csv 是权威）
    cohort_hadm = {r["hadm_id"] for r in load(ds / sp / "reference" / "cohort.csv")}
    # 每个 subject 的 stay：全部 vs cohort 内的
    all_stays, coh_stays = defaultdict(set), defaultdict(set)
    for r in load(ds / sp / "reference" / "icu" / "icustays.csv"):
        s = r["subject_id"]
        all_stays[s].add(r["stay_id"])
        if r["hadm_id"] in cohort_hadm:
            coh_stays[s].add(r["stay_id"])
    # raw 的 icustays 可能更大（cohort 外患者的 stay 只在 raw 里）
    raw_icu = ds / sp / "raw" / "icu" / "icustays.csv"
    if raw_icu.is_file():
        for r in load(raw_icu):
            all_stays[r["subject_id"]].add(r["stay_id"])

    agent_by_key = defaultdict(list)
    for r in load(res / "icu" / "chartevents.csv"):
        agent_by_key[(r["subject_id"], r["hadm_id"], r["stay_id"],
                       r["charttime"], r["itemid"])].append(r.get("valuenum"))

    nat = load(ds / sp / "mimic_native_error_log.csv")
    cnt = {"顺序敏感": [0, 0], "顺序无关": [0, 0]}   # [总格数, 修对格数]
    detail = {"顺序敏感": defaultdict(Counter), "顺序无关": defaultdict(Counter)}
    for r in nat:
        s = r["subject_id"]
        sens = len(all_stays.get(s, ())) > len(coh_stays.get(s, ()))
        bucket = "顺序敏感" if sens else "顺序无关"
        k = (s, r["hadm_id"], r["stay_id"], r["charttime"], r["itemid"])
        vals = agent_by_key.get(k, [])
        got = vals[0] if vals else None
        def num(x):
            try:
                return float(x)
            except Exception:
                return None
        gv, dirty, rep = num(got), num(r.get("mimic_native_value")), num(r.get("repaired_value"))
        if gv is None:
            verdict = "缺行"
        elif dirty is not None and abs(gv - dirty) < 1e-9:
            verdict = "没动"
        elif rep is not None and abs(gv - rep) < 1e-9:
            verdict = "修对"
        else:
            verdict = "修错"
        cnt[bucket][0] += 1
        if verdict == "修对":
            cnt[bucket][1] += 1
        detail[bucket][r.get("detection_mode", "")][verdict] += 1

    print("=== %s：原生格按「是否受 cohort 筛选顺序影响」分（%d 格）===" % (sp, len(nat)))
    for b in ("顺序敏感", "顺序无关"):
        tot, ok = cnt[b]
        pct = 100.0 * ok / tot if tot else 0.0
        print("  %-8s %4d 格（%.1f%%）  修对 %4d（%.1f%%）  %s"
              % (b, tot, 100.0 * tot / len(nat), ok, pct,
                 dict(detail[b])))
    sens = cnt["顺序敏感"][0]
    print()
    print("  ⇒ 全部 %d 格的修复率 = %d 格（%.1f%%）"
          % (len(nat), cnt["顺序敏感"][1] + cnt["顺序无关"][1],
             100.0 * (cnt["顺序敏感"][1] + cnt["顺序无关"][1]) / len(nat)))
    print("  ⇒ 但其中 **%d 格（%.1f%%）是顺序敏感格** —— 它们的正确性取决于"
          "「先 cohort 筛选再修原生错误」这个顺序有没有做对"
          % (sens, 100.0 * sens / len(nat)))
    print("  ⇒ 跨档比原生层修复率时，**必须先看两档的顺序敏感格占比**，"
          "否则会被分母带偏（1.6pct 6/31 vs 4.7pct 14/31）")


if __name__ == "__main__":
    main()
