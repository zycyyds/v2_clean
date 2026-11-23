#!/usr/bin/env python3
"""n18_native_pool_diagnose.py —— 诊断 agent 的原生格修复值**跟哪个口径的中位数一致**。

四个候选口径：
  A = 源 reference 的 (subject_id, itemid) 池（排除坏格）  ← **真值**
  B = agent 输出的 (stay_id, itemid) 池
  C = agent 输出的 (subject_id, itemid) 池
  D = agent 输出的 (subject_id, itemid) 池，但**只取 valuenum 落在硬区间内**的
  E = 全局（同 itemid，不分患者）的中位数

用法:
  python3 n18_native_pool_diagnose.py --dataset <mimicraw 数据集> --result <agent 结果包> \
      [--split validation] [--source <母数据集>]

只输出聚合，不含病人级标识。
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
    ap.add_argument("--dataset", required=True, help="mimicraw 数据集目录")
    ap.add_argument("--source", required=True, help="母数据集目录（无原生错误层）")
    ap.add_argument("--result", required=True, help="agent 结果包目录")
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    B = load_builder()
    di = B.load_di(B.D_ITEMS)
    ds, src, res = Path(args.dataset), Path(args.source), Path(args.result)
    sp = args.split

    # 源 reference 的 (subject,itemid) 可用池
    pool_src = defaultdict(list)
    for L, r in enumerate(load(src / sp / "reference" / "icu" / "chartevents.csv")):
        try:
            it = int(r["itemid"])
        except (ValueError, KeyError):
            continue
        v = num(r.get("valuenum"))
        if v is None or v == 0 or v in SENT:
            continue
        pool_src[(r["subject_id"], it)].append((L, v))

    # agent 输出的三个池
    pb, pc, pd_, pe = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    rows = load(res / "icu" / "chartevents.csv")
    for r in rows:
        try:
            it = int(r["itemid"])
        except (ValueError, KeyError):
            continue
        v = num(r.get("valuenum"))
        if v is None or v == 0 or v in SENT:
            continue
        pb[(r["stay_id"], it)].append(v)
        pc[(r["subject_id"], it)].append(v)
        pe[(it,)].append(v)
        rg = B.hard_range(di, it)
        if rg is None or (rg[0] <= v <= rg[1]):
            pd_[(r["subject_id"], it)].append(v)

    agent_by_key = defaultdict(list)
    agent_keys = set()
    for r in rows:
        k = (r["subject_id"], r["hadm_id"], r["stay_id"], r["charttime"], r["itemid"])
        agent_by_key[k].append(r.get("valuenum"))
        agent_keys.add(k)
    # 源 reference 的 (subject,itemid) 池，带每一行的完整 key
    pool_src_keyed = defaultdict(list)
    for L, r in enumerate(load(src / sp / "reference" / "icu" / "chartevents.csv")):
        try:
            it = int(r["itemid"])
        except (ValueError, KeyError):
            continue
        v = num(r.get("valuenum"))
        if v is None or v == 0 or v in SENT:
            continue
        k = (r["subject_id"], r["hadm_id"], r["stay_id"], r["charttime"], r["itemid"])
        pool_src_keyed[(r["subject_id"], it)].append((k, v))

    nat = load(ds / sp / "mimic_native_error_log.csv")
    n = {"未动": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "其它": 0}
    debug = []
    for r in nat:
        k = (r["subject_id"], r["hadm_id"], r["stay_id"], r["charttime"], r["itemid"])
        got = agent_by_key.get(k, [None])[0]
        gv = num(got)
        dirty = num(r["mimic_native_value"])
        if gv is not None and dirty is not None and abs(gv - dirty) < 1e-9:
            n["未动"] += 1
            continue
        lst = pool_src_keyed.get((r["subject_id"], int(r["itemid"])), [])
        A = F = G = None
        if lst:
            vs = [v for _, v in lst]
            ma = med(vs)
            i_bad = max(range(len(lst)), key=lambda i: abs(lst[i][1] - ma))
            A = med([v for i, (_, v) in enumerate(lst) if i != i_bad])
            G = med(vs)                       # ← **池子没排除坏格自己**
            rest = [(kk, v) for i, (kk, v) in enumerate(lst) if i != i_bad]
            kept = [v for kk, v in rest if kk in agent_keys]
            F = med(kept) if kept else None
        cands = {
            "A": A,
            "B": med(pb.get((r["stay_id"], int(r["itemid"])), [])),
            "C": med(pc.get((r["subject_id"], int(r["itemid"])), [])),
            "D": med(pd_.get((r["subject_id"], int(r["itemid"])), [])),
            "E": med(pe.get((int(r["itemid"]),), [])),
            "F": F,
            "G": G,
        }
        hit = None
        for key, val in cands.items():
            if gv is not None and val is not None and abs(gv - val) < 1e-6:
                hit = key
                break
        n[hit or "其它"] += 1
        if not hit:
            debug.append((r.get("itemid"), r.get("detection_mode"), gv, A, G,
                           len(lst), sorted(v for _, v in lst)[:3],
                           sorted(v for _, v in lst)[-3:]))

    print("=== 原生格 %d 格：agent 的值跟哪个口径一致 ===" % len(nat))
    print("  完全没动（值 == 原生错值）        : %d" % n["未动"])
    print("  A 源reference (subject,itemid) 池 ← 真值 : %d" % n["A"])
    print("  B agent (stay_id,itemid) 池              : %d" % n["B"])
    print("  C agent (subject_id,itemid) 池           : %d" % n["C"])
    print("  D agent (subject_id,itemid) 池 + 只取区间内: %d" % n["D"])
    print("  E 全局同 itemid 中位数                    : %d" % n["E"])
    print("  F A 但剔掉 agent 没产出的行（少产行污染池子）: %d" % n["F"])
    print("  G A 但**没排除坏格自己**                  : %d" % n["G"])
    print("  其它                                    : %d" % n["其它"])
    if debug:
        print("\n  --- 其它口径的格（itemid / mode / agent值 / A / G / 池大小 / 池首尾）---")
        for d in debug:
            print("   ", d)


if __name__ == "__main__":
    main()
