#!/usr/bin/env python3
"""n18_time_anchor_rule_check.py —— v8 prompt 的前提验证：

agent 自己发明的规则「**charttime 落在 stay 时间轴之外时，取该 stay 内该 itemid 的
最近的前一个有效 charttime**」，**能不能救回那几格漏检**？

对每个「agent 输出里缺这一行」的原生格：
  ① raw 里该 (stay, itemid) 的所有读数的 charttime（按顺序）
  ② stay 的 intime/outtime
  ③ 按规则算出「最近的前一个有效 charttime」⇒ 与 reference 的 charttime 比对
  ④ 结论：这格是**规则可救**（规则能算对）还是**规则也救不了**

只输出聚合统计（命中数、偏差分布），不输出病人标识与逐格医学值。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta
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
    agent_keys = {key_of(r) for r in load(res / "icu" / "chartevents.csv")}

    # 漏检格（agent 输出里没有这一行）
    missed = [r for r in nat if key_of(r) not in agent_keys]
    want = {(r["subject_id"], r["hadm_id"], r["stay_id"], r["itemid"]) for r in missed}

    # 扫 raw：这些 (stay, itemid) 的所有 charttime
    raw_rows = {}
    with (ds / sp / "raw" / "icu" / "chartevents.csv").open(
            newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            it = restore_itemid(r.get("itemid", ""))
            k = (r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"), it)
            if k in want:
                raw_rows.setdefault(k, []).append(r.get("charttime", ""))

    # stay 的 intime/outtime
    stays = {}
    for r in load(ds / sp / "raw" / "icu" / "icustays.csv"):
        stays[(r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"))] = (
            parse_ct(r.get("intime", "")), parse_ct(r.get("outtime", "")))

    print("=== %s：漏检的原生格 —— 「最近的前一个有效 charttime」规则能不能救 ===" % sp)
    print("  漏检格数: %d" % len(missed))

    c = Counter()
    deltas = []
    for r in missed:
        k4 = (r["subject_id"], r["hadm_id"], r["stay_id"], r["itemid"])
        cands = [parse_ct(c0) for c0 in raw_rows.get(k4, [])]
        cands = [x for x in cands if x is not None]
        if not cands:
            c["raw 里这个 (stay,itemid) 没有别的读数 ⇒ 规则救不了"] += 1
            continue
        intime, outtime = stays.get(
            (r["subject_id"], r["hadm_id"], r["stay_id"]), (None, None))
        t_ref = parse_ct(r["charttime"])
        if t_ref is None:
            c["reference 的 charttime 解析不了"] += 1
            continue
        # 该格在 raw 里的那个被挪走的时刻 = 最接近 t_ref 的候选
        raw_t = min(cands, key=lambda x: abs((x - t_ref).total_seconds()))
        # 规则：raw_t 落在 stay 时间轴之外（> outtime） ⇒ 取最近的前一个有效 charttime
        valid = [x for x in cands
                 if x < raw_t and (outtime is None or x <= outtime)]
        if not valid:
            c["没有「更早的有效时刻」可用 ⇒ 规则救不了"] += 1
            continue
        pred = max(valid)
        if pred == t_ref:
            c["规则算出的时刻 == reference ✅ 规则能救"] += 1
        else:
            c["规则算出的时刻 != reference ⇒ 规则救不了（但接近）"] += 1
            deltas.append(int((pred - t_ref).total_seconds() / 60))

    print()
    for k, n in c.most_common():
        print("    %2d 格  %s" % (n, k))
    if deltas:
        print()
        print("  规则算错时的偏差（分钟）分布: %s" % sorted(deltas))

    # ---- 诊断：为什么规则救不了 ----
    print()
    print("=== 诊断（聚合）===")
    d = Counter()
    gaps = []
    for r in missed:
        k4 = (r["subject_id"], r["hadm_id"], r["stay_id"], r["itemid"])
        cands = [parse_ct(c0) for c0 in raw_rows.get(k4, [])]
        cands = [x for x in cands if x is not None]
        t_ref = parse_ct(r["charttime"])
        intime, outtime = stays.get(
            (r["subject_id"], r["hadm_id"], r["stay_id"]), (None, None))
        if not cands or t_ref is None:
            d["数据不全"] += 1
            continue
        d["候选读数个数 = %s" % min(len(cands), 5) +
          ("+" if len(cands) > 5 else "")] += 1
        d["reference 的 charttime 本身 <= outtime"] += (outtime is not None and t_ref <= outtime)
        d["raw 里有候选的 charttime == reference"] += any(x == t_ref for x in cands)
        earlier = [x for x in cands if x < t_ref]
        d["存在比 reference 更早的候选"] += bool(earlier)
        if earlier:
            gaps.append(int((max(earlier) - t_ref).total_seconds() / 60))
        if outtime is not None:
            d["raw 里所有候选都 > outtime（整条 itemid 都被推走了）"] += all(
                x > outtime for x in cands)
            d["raw 里有候选 <= outtime"] += any(x <= outtime for x in cands)
    for k, n in sorted(d.items()):
        print("    %2d 格  %s" % (n, k))
    # ---- 诊断二：reference 那个时刻在 raw 里存不存在（作为别的行的时间戳）----
    print()
    print("=== 诊断二：reference 的那个 charttime，在 raw 的这个 stay 里有没有别的行用它？===")
    stay_rows = {}
    stay_want = {(r["subject_id"], r["hadm_id"], r["stay_id"]) for r in missed}
    with (ds / sp / "raw" / "icu" / "chartevents.csv").open(
            newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sk = (r.get("subject_id"), r.get("hadm_id"), r.get("stay_id"))
            if sk in stay_want:
                stay_rows.setdefault(sk, []).append(
                    (r.get("charttime", ""), restore_itemid(r.get("itemid", ""))))
    e = Counter()
    for r in missed:
        sk = (r["subject_id"], r["hadm_id"], r["stay_id"])
        t_ref = r["charttime"]
        it = r["itemid"]
        rows = stay_rows.get(sk, [])
        same_ts = [x for x in rows if x[0] == t_ref]
        e["reference 时刻在同 stay 里被 %s 行用过" % (
            "同一 itemid 的" if any(x[1] == it for x in same_ts) else "别的 itemid 的")] += \
            (1 if same_ts else 0)
        e["reference 时刻在同 stay 里根本没被任何行用过"] += (0 if same_ts else 1)
    for k, n in e.most_common():
        print("    %2d 格  %s" % (n, k))


if __name__ == "__main__":
    main()
