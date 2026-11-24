#!/usr/bin/env python3
"""n18_native_missed_cells.py —— N18 原生层里**没修对**的格，到底还能不能修？

对每个非 b_clean 的原生格（b_dirty / b_wrong / b_missing）判断：
  ① 该格在 agent 输出里有没有对应行？**在 raw 里呢？**
     · raw 有、输出没有 ⇒ **agent 把这行删了**（多半是去重规则误删）
     · raw 也没有   ⇒ 这行本来就不在输入里
  ② 该 (subject_id, itemid) 在 **cohort 内**还有没有别的可用读数可以当池子？
     （没有 ⇒ 这一格**结构上不可判定**，谁都修不出来）
  ③ 如果有别的读数：agent 的值 vs 池中位数 vs 真值 —— 差在哪。

只输出聚合统计与 itemid，不出现 subject_id / hadm_id / stay_id。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)

KEY5 = ("subject_id", "hadm_id", "stay_id", "charttime", "itemid")


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_of(r):
    return tuple(r.get(c, "") for c in KEY5)


def median(vals):
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ds, res, sp = Path(args.dataset), Path(args.result), args.split

    nat = load(ds / sp / "mimic_native_error_log.csv")
    agent = load(res / "icu" / "chartevents.csv")
    agent_by_key = {}
    for r in agent:
        agent_by_key.setdefault(key_of(r), []).append(r)

    cohort_hadm = {r["hadm_id"] for r in load(ds / sp / "reference" / "cohort.csv")}
    coh_stays = set()
    for r in load(ds / sp / "reference" / "icu" / "icustays.csv"):
        if r["hadm_id"] in cohort_hadm:
            coh_stays.add((r["subject_id"], r["stay_id"]))

    pool = {}
    for r in agent:
        v = num(r.get("valuenum"))
        if v is None:
            continue
        pool.setdefault((r["subject_id"], r["itemid"]), []).append(
            (r.get("stay_id", ""), r.get("charttime", ""), v))

    # ---- 找出没修对的格 ----
    problems = []
    for r in nat:
        k = key_of(r)
        rep = num(r.get("repaired_value"))
        dirty = num(r.get("mimic_native_value"))
        rows = agent_by_key.get(k)
        got = num(rows[0].get("valuenum")) if rows else None
        if got is None:
            verdict = "缺行"
        elif dirty is not None and abs(got - dirty) < 1e-9:
            verdict = "没动"
        elif rep is not None and abs(got - rep) < 1e-9:
            continue          # 修对
        else:
            verdict = "修错"
        problems.append((r, verdict, got, rep, dirty))

    # ---- 流式扫 raw，找这些 key 在不在 ----
    want = {key_of(r) for r, *_ in problems}
    raw_hit = {}
    raw_path = ds / sp / "raw" / "icu" / "chartevents.csv"
    if raw_path.is_file():
        with raw_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                k = key_of(r)
                if k in want:
                    raw_hit.setdefault(k, []).append(r)

    kinds = Counter()
    detail = []
    for r, verdict, got, rep, dirty in problems:
        k = key_of(r)
        in_cohort = (r["subject_id"], r["stay_id"]) in coh_stays
        pr = pool.get((r["subject_id"], r["itemid"]), [])
        others = [v for (st, ct, v) in pr
                  if not (st == r["stay_id"] and ct == r["charttime"])]
        med = median(others) if others else None
        in_raw = k in raw_hit
        n_raw_rows = len(raw_hit.get(k, []))

        if verdict == "缺行":
            kind = ("agent 把这行删了（raw 里有）" if in_raw
                    else "raw 里本来就没有这行")
            if not others:
                kind += " + 池里没有别的读数 ⇒ 结构上不可判定"
        elif verdict == "修错":
            if not others:
                kind = "池里没有别的读数 ⇒ 结构上不可判定"
            elif med is not None and abs(got - med) < 1e-9:
                kind = "值 = 池中位数 ⇒ 池口径不对（混进非 cohort stay）"
            else:
                kind = "值 ≠ 池中位数 ⇒ 规则错"
        else:
            kind = "脏值原样留着"
        if not in_cohort:
            kind += " ⚠(不在 cohort 内)"

        kinds[(verdict, "raw有" if in_raw else "raw无",
               "池有别的读数" if others else "池空")] += 1
        detail.append((r["itemid"], verdict, kind, len(others),
                       (got, rep, dirty, med)))

    print("=== %s：原生层没修对的格（%d / %d）===" % (sp, len(detail), len(nat)))
    for kk, n in sorted(kinds.items()):
        print("  %-6s / %-6s / %-12s : %d" % (kk[0], kk[1], kk[2], n))
    print()
    agg = Counter(d[2] for d in detail)
    for k, n in agg.most_common():
        print("  %2d 格  %s" % (n, k))
    print()
    # 只输出聚合关系，**不输出逐格的数值**（遵用户的「只输出聚合统计」边界）
    rel = Counter()
    for it, v, kind, n_o, vals in detail:
        got, rep, dirty, med = vals
        if v == "修错":
            rel["修错：agent 值 == 池中位数" if (med is not None and got is not None
                                                 and abs(got - med) < 1e-9)
                else "修错：agent 值 != 池中位数"] += 1
            rel["修错：池里还有别的读数（可判定）" if n_o else "修错：池空（不可判定）"] += 1
        elif v == "缺行":
            rel["缺行：raw 里没有 ⇒ 输入里就没这行" if "raw 里本来就没有" in kind
                else "缺行：raw 里有 ⇒ agent 把它删了"] += 1
    print("  关系（聚合）：")
    for k, n in rel.most_common():
        print("    %2d 格  %s" % (n, k))
    print()
    print("  按 itemid（仅测量项类型，不含任何病人标识）：")
    for it, n in Counter(d[0] for d in detail).most_common():
        print("    itemid %s : %d 格" % (it, n))


if __name__ == "__main__":
    main()
