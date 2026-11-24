#!/usr/bin/env python3
"""n18_alias_residue.py —— agent 的输出里**还残留多少没还原的 itemid 别名**，
以及漏掉的那几格对应的行在输出里长什么样。

只输出聚合统计（不输出逐格医学值、不输出病人标识）。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)

KEY5 = ("subject_id", "hadm_id", "stay_id", "charttime", "itemid")


def restore(alias):
    if len(alias) < 5:
        return None
    return "2" + alias[3] + alias[-4:]


def load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ds, res, sp = Path(args.dataset), Path(args.result), args.split

    nat = load(ds / sp / "mimic_native_error_log.csv")
    agent = load(res / "icu" / "chartevents.csv")

    # 输出里残留的别名形态 itemid
    residue = Counter()
    alias_rows = []
    for r in agent:
        it = r.get("itemid", "")
        if it.startswith("99") and len(it) >= 5:
            residue["前缀形态 " + it[:4]] += 1
            alias_rows.append(r)
    print("=== agent 输出里残留的别名 itemid 行 ===")
    if not alias_rows:
        print("  0 行（别名全部还原了）")
    else:
        print("  合计 %s 行" % "{:,}".format(len(alias_rows)))
        for k, v in residue.most_common():
            print("    %-14s : %s" % (k, "{:,}".format(v)))

    # 漏掉的原生格：它的真值 itemid 在输出里有没有对应行（别名形态 / 别的 charttime）
    agent_keys = {tuple(r.get(c, "") for c in KEY5) for r in agent}
    agent_alias_by_stay = {}
    for r in alias_rows:
        agent_alias_by_stay.setdefault(
            (r.get("subject_id"), r.get("hadm_id"), r.get("stay_id")), []).append(r)

    print()
    print("=== 漏掉的原生格：在 agent 输出里对应到哪一步 ===")
    c = Counter()
    for r in nat:
        k = tuple(r.get(x, "") for x in KEY5)
        if k in agent_keys:
            continue
        sk = (r["subject_id"], r["hadm_id"], r["stay_id"])
        al = [x for x in agent_alias_by_stay.get(sk, [])
              if restore(x.get("itemid", "")) == r["itemid"]]
        if al:
            c["输出里有这一行，但 itemid 还是别名形态（没还原）"] += 1
        else:
            c["输出里没有这一行（整行丢了）"] += 1
    for k, v in c.most_common():
        print("    %2d 格  %s" % (v, k))


if __name__ == "__main__":
    main()
