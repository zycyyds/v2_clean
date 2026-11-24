#!/usr/bin/env python3
"""n18_native_key_vs_reference.py —— 原生层登记用的 5 段 key，
**跟 reference（真值）对不对得上**？

这是决定性的一查：
  · 若 reference 里**有**这个 key ⇒ agent 把这一行改跑了（charttime/itemid 变了）⇒ 真的漏检
  · 若 reference 里**没有**这个 key ⇒ 原生层登记用的是 **raw 侧**的 charttime，
    而 reference 那一行的 charttime 是别的 ⇒ 按 key 找不到
    ⇒ **是评估口径的问题，不是 agent 的错**

流式实现：只把「原生层那几百个 key」放进内存，扫一遍 reference / agent 输出 / raw
（raw 先按别名规则还原 itemid）各判一次命中。
只输出聚合统计。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)

KEY5 = ("subject_id", "hadm_id", "stay_id", "charttime", "itemid")


def restore_itemid(i):
    """prompt 里的别名还原规则：原 itemid = "2" + alias[3] + alias[-4:]"""
    if i.startswith("99") and len(i) >= 5:
        return "2" + i[3] + i[-4:]
    return i


def load(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_of(r):
    return tuple(r.get(c, "") for c in KEY5)


def scan(path, wanted, do_restore=False):
    """流式扫一张表，返回 wanted 里命中的 key 集合。"""
    hit = set()
    if not path.is_file():
        return hit
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = key_of(r)
            if k in wanted:
                hit.add(k)
                continue
            if do_restore:
                i2 = restore_itemid(r.get("itemid", ""))
                if i2 != r.get("itemid", ""):
                    k2 = (r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"),
                          r.get("charttime"), i2)
                    if k2 in wanted:
                        hit.add(k2)
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    ds, res, sp = Path(args.dataset), Path(args.result), args.split

    nat = load(ds / sp / "mimic_native_error_log.csv")
    nat_keys = {key_of(r) for r in nat}
    print("原生层登记格数: %d" % len(nat_keys))

    ref_hit = scan(ds / sp / "reference" / "icu" / "chartevents.csv", nat_keys)
    agent_hit = scan(res / "icu" / "chartevents.csv", nat_keys)
    raw_hit = scan(ds / sp / "raw" / "icu" / "chartevents.csv", nat_keys,
                   do_restore=True)

    print()
    print("=== key 命中情况 ===")
    print("  raw 里有（别名还原后）   : %d / %d" % (len(raw_hit), len(nat_keys)))
    print("  reference 里有           : %d / %d" % (len(ref_hit), len(nat_keys)))
    print("  agent 输出里有           : %d / %d" % (len(agent_hit), len(nat_keys)))
    print()
    no_ref = nat_keys - ref_hit
    print("  ⚠️ reference 里**没有**这个 key 的原生格: %d 格" % len(no_ref))
    if no_ref:
        got = len(no_ref & agent_hit)
        print("     其中 agent 输出里也没有 : %d 格" % (len(no_ref) - got))
        print("     其中 agent 输出里居然有 : %d 格" % got)
        print("     ⇒ 这 %d 格是**评估口径问题**（原生层登记用 raw 侧 charttime，"
              "reference 用的是别的 charttime）" % len(no_ref))
    in_ref = nat_keys & ref_hit
    print()
    print("  reference 里有的 %d 格中，agent 命中 %d 格（漏 %d 格）"
          % (len(in_ref), len(in_ref & agent_hit), len(in_ref - agent_hit)))

    # ---- 交叉表：漏检是不是只发生在「raw 里 key 对不上（行被挪过）」的格子上 ----
    aligned = nat_keys & raw_hit          # raw 里 key 对得上
    displaced = nat_keys - raw_hit        # raw 里 key 对不上（charttime/itemid 被注入器动过）
    print()
    print("=== 交叉表：raw 里这一行的 key 对不对得上 × agent 有没有命中 ===")
    print("  %-34s %6s %6s %6s" % ("分组", "格数", "命中", "漏"))
    for name, s in (("raw 里 key 对得上", aligned),
                    ("raw 里 key 对不上（行被挪过）", displaced)):
        hit = len(s & agent_hit)
        print("  %-34s %6d %6d %6d" % (name, len(s), hit, len(s) - hit))
    a = len(aligned & agent_hit)
    d = len(displaced & agent_hit)
    if aligned:
        print("  ⇒ 对得上的格子里 agent 的漏检率: %.2f%%"
              % (100.0 * (len(aligned) - a) / len(aligned)))
    if displaced:
        print("  ⇒ 被挪过的格子里 agent 的漏检率: %.2f%%"
              % (100.0 * (len(displaced) - d) / len(displaced)))
    if displaced:
        print("  ⇒ **被挪过 %d 格，agent 仍然救回 %d 格（%.1f%%）** —— "
              "这 %d 格在 raw 里根本没有这个 key，"
              "agent 靠自己重建的时间锚点把它们找回来了"
              % (len(displaced), d, 100.0 * d / len(displaced), len(displaced)))


if __name__ == "__main__":
    main()
