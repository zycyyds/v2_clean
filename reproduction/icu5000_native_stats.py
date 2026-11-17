#!/usr/bin/env python3
"""icu5000_native_stats.py —— 实测 icu5000 mimicraw 档的统计量，供 prompt 使用。

产出（**只有聚合计数，不含任何病人级信息**）：

  1. deviation  : 原生错误格的「与自身常态偏离度」分布
                  （dev = |mimic_native_value - repaired_value| / |repaired_value|）
                  → 用来验证 prompt 里「≥20% 偏离才算错」这条判据在 icu5000 上成立
  2. cvp        : 中心静脉压（itemid 220074）原生错值范围 + 「÷100 后落回区间」
                  这条错误判据的失败率（K29 上是 245/245 全错）
  3. alias      : itemid 列被注入成 CareVue 别名的规模与规则
                  （992{2,3} + 原 itemid 末 4 位）
  4. value_eq_clean : 注入类错误里「raw 的 value 文本列 == log 登记的 clean_value」
                  的比例 —— 这是 agent 判断「注入错误 vs 原生错误」的关键线索的可靠性
  5. enum       : 枚举型 itemid（同一时刻多条不同取值合法）的行数 / 多值组数

用法:
  python3 icu5000_native_stats.py --dataset <mimicraw 数据集目录> \
      --out <stats.json> [--sample 200000]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

CE_FILE = "icu/chartevents.csv"
HARD_ITEMS = [220045, 220050, 220051, 220052, 220059, 220061, 220074,
              220179, 220180, 220181, 220210, 223762, 224688, 224689,
              224690, 227466, 224167, 224643, 227242, 227243]
ENUM_ITEMS = [230033, 229864]
CVP_ITEM = 220074
CVP_RANGE = (-20, 80)
SAMPLE = 200000


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def median_of(arr):
    n = len(arr)
    if n == 0:
        return None
    s = sorted(arr)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# ───────────────────────── 1. 偏离度 + 2. CVP ─────────────────────────
def deviation_stats(nat_log: Path):
    devs = []
    cvp_vals = []
    by_item = Counter()
    mode = Counter()
    if not nat_log.is_file():
        return None
    with nat_log.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mode[r.get("detection_mode", "")] += 1
            by_item[r.get("itemid", "")] += 1
            v = fnum(r.get("mimic_native_value"))
            m = fnum(r.get("repaired_value"))
            if v is None or m is None or m == 0:
                continue
            devs.append(abs(v - m) / abs(m))
            if str(r.get("itemid")) == str(CVP_ITEM):
                cvp_vals.append(v)
    devs.sort()
    lt20 = sum(1 for d in devs if d < 0.20)
    cvp = {}
    if cvp_vals:
        lo, hi = CVP_RANGE
        # 「÷100 之后落不落回区间」这条判据：落回区间 ⇒ 会被误判成正常 ⇒ 判据失效
        back = sum(1 for v in cvp_vals if lo <= v / 100.0 <= hi)
        cvp = {"n": len(cvp_vals),
               "min": min(cvp_vals), "max": max(cvp_vals),
               "div100_back_in_range": back,
               "div100_fail_rate": round(back / len(cvp_vals), 6)}
    return {
        "n_cells": len(devs),
        "n_lt_20pct": lt20,
        "median_dev": round(median_of(devs), 4) if devs else None,
        "p90_dev": round(devs[int(len(devs) * 0.9)], 4) if devs else None,
        "by_item": dict(by_item.most_common()),
        "by_mode": dict(mode.most_common()),
        "cvp": cvp,
    }


# ───────────────────────── 3. itemid 别名注入 ─────────────────────────
def alias_stats(log_path: Path):
    out = {"n": 0, "prefix": Counter(), "rule_ok": 0, "cols": Counter(),
           "subtypes": Counter(), "files": Counter()}
    if not log_path.is_file():
        return None
    with log_path.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e.get("column") != "itemid":
                continue
            out["n"] += 1
            out["subtypes"][e.get("error_subtype", "")] += 1
            out["files"][e.get("raw_file", "")] += 1
            dv = (e.get("dirty_value") or "").strip()
            cv = (e.get("clean_value") or "").strip()
            out["prefix"][dv[:4]] += 1
            try:
                exp = "992" + dv[3] + str(int(cv)).zfill(4)[-4:]
            except (TypeError, ValueError):
                continue
            if dv == exp:
                out["rule_ok"] += 1
    return {"n": out["n"],
            "prefix": dict(out["prefix"].most_common(6)),
            "rule_ok": out["rule_ok"],
            "rule_rate": round(out["rule_ok"] / out["n"], 6) if out["n"] else None,
            "subtypes": dict(out["subtypes"].most_common()),
            "files": dict(out["files"].most_common())}


# ───────────────── 4. value 文本列 == clean_value 的比例（抽样） ─────────────────
def value_eq_clean(log_path: Path, raw_path: Path, ins_pos, sample: int):
    """对 unit_scale_error / sentinel_chart_value 两类，抽样比对 raw 的 value 列
    与 log 登记的 clean_value 是否相等。"""
    want = {}
    tot = Counter()
    picked = Counter()
    with log_path.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e.get("column") != "valuenum":
                continue
            st = e.get("error_subtype", "")
            if st not in ("unit_scale_error", "sentinel_chart_value"):
                continue
            tot[st] += 1
            if picked[st] >= sample:
                continue
            try:
                L = int(e["raw_row_index"])
            except (TypeError, KeyError):
                continue
            picked[st] += 1
            want.setdefault(L, []).append((st, e.get("clean_value", "")))
    res = {}
    for st in ("unit_scale_error", "sentinel_chart_value"):
        res[st] = {"total_in_log": tot.get(st, 0), "sampled": picked.get(st, 0),
                   "value_eq_clean": 0, "value_nonempty": 0}
    if not want:
        return res

    # L → raw 行号：icu5000 是 bisect 模式（raw 行号 = L + |inserts < L|）
    def raw_row(L):
        return L + bisect_left(ins_pos, L)

    # L → raw 行号：raw 行号 = L + |inserts < L|（icu5000 是 bisect 模式）
    # 先把需要的 raw 行号收集起来，再扫一遍 raw 取值
    need = {}
    for L in want:
        need[raw_row(L)] = L
    vals = {}
    with raw_path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for ri, row in enumerate(rd):
            L = need.get(ri)
            if L is None:
                continue
            vals[L] = row.get("value")
    for L, lst in want.items():
        v = vals.get(L)
        if v is None:
            continue
        for st, cv in lst:
            if v == "":
                continue
            res[st]["value_nonempty"] += 1
            if v.strip() == cv.strip():
                res[st]["value_eq_clean"] += 1
    return res


# ───────────────────────── 5. 枚举型 itemid ─────────────────────────
def enum_stats(ref_path: Path):
    want = {str(i) for i in ENUM_ITEMS}
    cnt = Counter()
    grp = defaultdict(set)
    if not ref_path.is_file():
        return None
    with ref_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            it = r["itemid"]
            if it not in want:
                continue
            cnt[it] += 1
            grp[(r["subject_id"], r["hadm_id"], r["stay_id"],
                 r["charttime"], it)].add(r["valuenum"])
    multi = Counter()
    for k, v in grp.items():
        if len(v) > 1:
            multi[k[4]] += 1
    return {"rows": dict(cnt), "multi_groups": dict(multi)}


# ───────────────────────── train raw 里 HARD itemid 覆盖 ─────────────────────────
def train_hard_items(raw_path: Path):
    cnt = Counter()
    if not raw_path.is_file():
        return {}
    with raw_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                it = int(r["itemid"])
            except (TypeError, ValueError):
                continue
            if it in HARD_ITEMS:
                cnt[it] += 1
    return dict(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=SAMPLE)
    ap.add_argument("--skip-raw", action="store_true",
                    help="跳过 raw 对照（value==clean_value 比例），快很多")
    args = ap.parse_args()
    ds = Path(args.dataset)

    stats = {"dataset": str(ds)}

    print("[1/5] 偏离度 + CVP ...", flush=True)
    stats["deviation"] = deviation_stats(ds / "test" / "mimic_native_error_log.csv")

    print("[2/5] itemid 别名注入 ...", flush=True)
    stats["alias"] = alias_stats(ds / "test" / "modification_log.csv")

    print("[3/5] 枚举型 itemid ...", flush=True)
    stats["enum"] = enum_stats(ds / "test" / "reference" / CE_FILE)

    print("[4/5] train raw 的 HARD itemid 覆盖 ...", flush=True)
    stats["train_hard_items"] = train_hard_items(ds / "train" / "raw" / CE_FILE)

    if args.skip_raw:
        stats["value_eq_clean"] = None
    else:
        print("[5/5] value 文本列 vs clean_value（抽样 %d）..." % args.sample,
              flush=True)
        log_p = ds / "test" / "modification_log.csv"
        ins_pos = []
        with log_p.open(newline="", encoding="utf-8") as f:
            for e in csv.DictReader(f):
                if e.get("operation") in ("insert", "duplicate"):
                    try:
                        ins_pos.append(int(e["raw_row_index"]))
                    except (TypeError, KeyError):
                        pass
        ins_pos.sort()
        stats["value_eq_clean"] = value_eq_clean(
            log_p, ds / "test" / "raw" / CE_FILE, ins_pos, args.sample)

    Path(args.out).write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("写出:", args.out)
    for k in ("deviation", "alias", "enum", "value_eq_clean"):
        v = stats.get(k)
        print("  %-16s %s" % (k, json.dumps(v, ensure_ascii=False)[:400]))


if __name__ == "__main__":
    main()
