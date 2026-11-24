#!/usr/bin/env python3
"""n18_native_missing_rows.py —— 原生层里「agent 输出里没有这一行」的那些格，
raw 里到底有没有？用**宽松匹配**找：只看 (subject, hadm, stay)，
再看 itemid 是不是别名形态、charttime 是不是差了几分钟。

分三类（聚合统计，不输出任何逐格的医学值与病人标识）：
  A. raw 里完全没有这个 stay 的行      ⇒ 这一行真的不在输入里 ⇒ 结构上不可恢复
  B. raw 里有行，但 itemid 是别名形态  ⇒ 可恢复（是 agent 没还原 itemid 别名）
  C. raw 里有行且 itemid 正常，只是 charttime 不同 ⇒ 可恢复（时间锚点选错）
  D. raw 里有一模一样的行             ⇒ agent 把它删了（去重误删）

只输出聚合计数。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ds, res, sp = Path(args.dataset), Path(args.result), args.split

    nat = load(ds / sp / "mimic_native_error_log.csv")
    agent = load(res / "icu" / "chartevents.csv")
    agent_keys = {key_of(r) for r in agent}
    agent_stays = {(r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"))
                   for r in agent}

    # ---- 找「输出里缺行」的原生格 ----
    missing = []
    for r in nat:
        k = key_of(r)
        if k in agent_keys:
            continue
        missing.append(r)

    # ---- 流式扫 raw，按 (subject, hadm, stay) 收集候选行 ----
    want_stays = {(r["subject_id"], r["hadm_id"], r["stay_id"]) for r in missing}
    cand = {}
    raw_path = ds / sp / "raw" / "icu" / "chartevents.csv"
    with raw_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sk = (r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"))
            if sk in want_stays:
                cand.setdefault(sk, []).append(r)

    kinds = Counter()
    alias_cnt = Counter()
    for r in missing:
        sk = (r["subject_id"], r["hadm_id"], r["stay_id"])
        rows = cand.get(sk, [])
        it = r["itemid"]
        ct = r["charttime"]
        if not rows:
            kinds["A. raw 里这个 stay 一行都没有 ⇒ 结构上不可恢复"] += 1
            continue
        # 这一格在 raw 里到底有几行（itemid 按别名还原后算）
        def is_this(x):
            i = x.get("itemid", "")
            if i == it:
                return True
            return i.startswith("99") and len(i) >= 5 and "2" + i[3] + i[-4:] == it
        same_key = [x for x in rows
                    if is_this(x) and x.get("charttime") == ct]
        same_it = [x for x in rows if is_this(x)]
        aliased = [x for x in same_key if x.get("itemid", "") != it]
        if len(same_key) > 1:
            kinds["F. raw 里同一 key 有 %d 行 ⇒ 去重挑了另一个副本" % len(same_key)] += 1
        elif len(same_key) == 1 and aliased:
            kinds["B. raw 里这一行 itemid 是别名形态 ⇒ 可恢复（没还原别名）"] += 1
            alias_cnt["别名 itemid"] += 1
        elif len(same_key) == 1:
            kinds["D. raw 里有完全一样的行 ⇒ agent 去重误删"] += 1
        elif len(same_it) > 1:
            kinds["C. raw 里 itemid 对但 charttime 不同（%d 个候选 charttime）" % len(same_it)] += 1
        else:
            kinds["E. raw 里有行但 itemid 既非原值也非可还原别名"] += 1

    print("=== %s：原生层里「输出缺行」的格（%d / %d）===" % (sp, len(missing), len(nat)))
    for k, n in kinds.most_common():
        print("  %2d 格  %s" % (n, k))
    print()
    print("  ⇒ 其中 raw 里有线索（= 理论上可恢复）的：%d 格；"
          "raw 里根本没这行的：%d 格"
          % (sum(n for k, n in kinds.items() if k[0] in "BCD"), kinds.get(
              "A. raw 里这个 stay 一行都没有 ⇒ 结构上不可恢复", 0)))


if __name__ == "__main__":
    main()
