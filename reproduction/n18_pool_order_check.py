#!/usr/bin/env python3
"""n18_pool_order_check.py —— 验证「修复时机早于 cohort 筛选 ⇒ 池子被非 cohort stay 污染」。

对每个原生格：
  · 该 subject 在 **raw** 里共有几个 stay
  · 其中几个 stay 在 **cohort** 里（= 该 subject 的 cohort hadm 对应的 stay）
  · agent 的修复值 == 「cohort 内 stay 的池中位数」（A）还是「全部 stay 的池中位数」（H）

只输出聚合统计，不含病人级标识。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)
HERE = Path(__file__).resolve().parent
SENT = {-999.0, -998.0, -997.0, -990.0, 9999.0, 99999.0,
        999999.0, 9999999.0, 220000.0, 220045.0}


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "n18b", HERE / "build_n18_mimic_raw_error.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def med(a):
    s = sorted(a)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()
    B = load_builder()
    di = B.load_di(B.D_ITEMS)
    ds, src, res = Path(args.dataset), Path(args.source), Path(args.result)
    sp = args.split

    # cohort hadm（从 reference 的 cohort.csv 拿，这是权威）
    cohort_hadm = {r["hadm_id"] for r in load(src / sp / "reference" / "cohort.csv")}

    # raw 的 (subject,itemid) 可用读数 + 每行的 stay 是否在 cohort
    pool_all = defaultdict(list)
    pool_coh = defaultdict(list)
    stays_per_subj = defaultdict(set)
    coh_stays_per_subj = defaultdict(set)
    for r in load(src / sp / "raw" / "icu" / "chartevents.csv"):
        try:
            it = int(r["itemid"])
        except (ValueError, KeyError):
            continue
        v = num(r.get("valuenum"))
        if v is None or v == 0 or v in SENT:
            continue
        rg = B.hard_range(di, it)
        if rg is not None and not (rg[0] <= v <= rg[1]):
            continue
        s = r["subject_id"]
        stays_per_subj[s].add(r["stay_id"])
        in_coh = r["hadm_id"] in cohort_hadm
        if in_coh:
            coh_stays_per_subj[s].add(r["stay_id"])
        pool_all[(s, it)].append((r["stay_id"], r["charttime"], r["valuenum"], v))
        if in_coh:
            pool_coh[(s, it)].append((r["stay_id"], r["charttime"], r["valuenum"], v))

    agent_by_key = defaultdict(list)
    for r in load(res / "icu" / "chartevents.csv"):
        agent_by_key[(r["subject_id"], r["hadm_id"], r["stay_id"],
                       r["charttime"], r["itemid"])].append(r.get("valuenum"))

    nat = load(ds / sp / "mimic_native_error_log.csv")
    stat = {"有多余 stay": 0, "只有 cohort stay": 0}
    hitA = hitH = 0
    for r in nat:
        s = r["subject_id"]
        it = int(r["itemid"])
        n_all = len(stays_per_subj.get(s, ()))
        n_coh = len(coh_stays_per_subj.get(s, ()))
        stat["有多余 stay" if n_all > n_coh else "只有 cohort stay"] += 1
        k = (s, r["hadm_id"], r["stay_id"], r["charttime"], r["itemid"])
        gv = num(agent_by_key.get(k, [None])[0])
        for label, pool in (("A", pool_coh), ("H", pool_all)):
            lst = pool.get((s, it), [])
            if not lst:
                continue
            vs = [v for *_, v in lst]
            ma = med(vs)
            i_bad = max(range(len(lst)), key=lambda i: abs(lst[i][3] - ma))
            m = med([lst[i][3] for i in range(len(lst)) if i != i_bad])
            if gv is not None and m is not None and abs(gv - m) < 1e-6:
                if label == "A":
                    hitA += 1
                else:
                    hitH += 1
    print("=== %s：池子口径归属 ===" % sp)
    print("  agent 值 == A（**cohort 内** stay 的池中位数，真值口径）: %d" % hitA)
    print("  agent 值 == H（**全部** stay 的池中位数，即过滤 cohort 之前算的）: %d" % hitH)
    print()
    print("  原生格按「该患者是否有非 cohort 的 stay」分：")
    for k2, v in stat.items():
        print("    %-16s %d 格" % (k2, v))


if __name__ == "__main__":
    main()
