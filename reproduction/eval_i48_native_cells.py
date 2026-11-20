#!/usr/bin/env python3
"""eval_i48_native_cells.py —— i48 的 **MIMIC 原生错误格**专项判定。

⚠️ **不能直接用 icu5000 版的 eval_native_cells.py**：
那一版靠「log 的 raw_row_index == reference 的行号」把原生格和 gold 行对上。
· icu5000：reference = 注入前的干净表 ⇒ 行号空间与 log 一致 ✓
· **i48**：reference = **处理后的输出表**（12,768,419 行，cohort 筛选后），
  而 log 的 raw_row_index 空间是**源 raw**（16,014,304 行）⇒ 行号对不上 ❌
  （实测：溯源表 index 上限 16,014,303 > reference 行数 12,768,419）

⇒ 本版改成**按 key 定位**：溯源表 `mimic_native_error_log.csv` 里带了
subject/hadm/stay/charttime/itemid 五列（**干净 key**）⇒ 直接拿它在 agent 输出里查。
实测：reference 按 key 命中 1264 / 1264，且值 == repaired_value ✅

逐格判定：
  b_clean   ✓ agent 的值 == 修复值（repaired_value）⇒ 修对了
  b_dirty   ✗ agent 的值 == 原始的原生错值（mimic_native_value）⇒ 没修
  b_wrong   ✗ 既不是修复值也不是原生错值
  b_missing ✗ agent 输出里按这个 key 找不到这一行

「孪生行」歧义：注入器在同一 key 上造过重复行 ⇒ agent 输出里同一 key 可能有多行。
· **主口径 = 取 agent 输出里该 key 的第一行**（与官方评分器 sqlite `fetchone()` 一致）
· 另出 **any 口径**（任一行命中修复值即算修对）⇒ 用来量化歧义影响

只输出聚合，不含病人级标识。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

CE = "icu/chartevents.csv"
KEY = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def same(a, b, numeric=True):
    """数值感知比较（默认）：'0' 与 '0.0' 算相同；numeric=False 时纯字符串比。"""
    if numeric:
        fa, fb = fnum(a), fnum(b)
        if fa is not None and fb is not None:
            return abs(fa - fb) < 1e-9
    return str(a) == str(b)


def key_of(row):
    return tuple(row.get(c, "") for c in KEY)


def load_native(ds: Path, split: str):
    """原生格溯源表：{log 行号 L: 记录}。"""
    p = ds / split / "mimic_native_error_log.csv"
    if not p.is_file():
        sys.exit("找不到溯源表：%s" % p)
    out = {}
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["raw_row_index"])] = r
    if not out:
        sys.exit("溯源表是空的：%s" % p)
    return out


def collect_native_from_log(ds: Path, split: str):
    """从 modification_log.csv 里取 mimic_native_* 条目（交叉核对用）。"""
    p = ds / split / "modification_log.csv"
    out = {}
    if not p.is_file():
        return out
    with p.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if (e.get("raw_file") == CE and e.get("column") == "valuenum"
                    and e.get("operation") == "cell_update"
                    and e.get("error_subtype", "").startswith("mimic_native")):
                out[int(e["raw_row_index"])] = e
    return out


def judge(ds: Path, res: Path, split: str, numeric=True, verbose=True):
    """返回 (status, status_any, nat, meta)。

    status[L]     主口径：该 key 在 agent 输出里的**第一行**的判定
    status_any[L] 宽松口径：该 key 的**任一行**命中修复值即算 b_clean
    """
    nat = load_native(ds, split)
    mod = collect_native_from_log(ds, split)
    if verbose:
        n_mismatch = sum(
            1 for L, r in nat.items()
            if L in mod and (fnum(mod[L].get("dirty_value")) != fnum(r.get("mimic_native_value"))
                             or fnum(mod[L].get("clean_value")) != fnum(r.get("repaired_value"))))
        print("原生格：溯源表 %d 条 / modification_log %d 条（交叉核对不一致 %d）"
              % (len(nat), len(mod), n_mismatch))

    want = defaultdict(list)
    for L, r in nat.items():
        want[key_of(r)].append(L)

    agent_rows = defaultdict(list)
    n_rows = 0
    res_ce = res / CE
    if not res_ce.is_file():
        cands = sorted(res.rglob("chartevents.csv"))
        if not cands:
            sys.exit("结果包里找不到 chartevents.csv")
        res_ce = cands[0]
        if verbose:
            print("（按 glob 找到）", res_ce)
    with res_ce.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            k = key_of(r)
            if k in want:
                agent_rows[k].append(r.get("valuenum", ""))

    def verdict(v, r):
        if same(v, r.get("repaired_value", ""), numeric):
            return "b_clean"
        if same(v, r.get("mimic_native_value", ""), numeric):
            return "b_dirty"
        return "b_wrong"

    status, status_any = {}, {}
    n_ambig = 0
    for L, r in nat.items():
        vals = agent_rows.get(key_of(r), [])
        if not vals:
            status[L] = status_any[L] = "b_missing"
            continue
        if len(vals) > 1:
            n_ambig += 1
        status[L] = verdict(vals[0], r)
        vs = [verdict(v, r) for v in vals]
        status_any[L] = "b_clean" if "b_clean" in vs else vs[0]

    meta = {
        "agent_rows": n_rows,
        "ambiguous_keys": n_ambig,      # agent 输出里同一 key 有多行（孪生行）
        "hit_cells": sum(1 for L in nat if status[L] != "b_missing"),
    }
    if verbose:
        print("agent 输出行数 %d；命中 %d / %d 格；其中 %d 格存在多行歧义"
              % (n_rows, meta["hit_cells"], len(nat), n_ambig))
    return status, status_any, nat, meta


def summarize(status, nat):
    buckets = Counter(status.values())
    per_item = defaultdict(Counter)
    per_mode = defaultdict(Counter)
    for L, b in status.items():
        per_item[nat[L].get("itemid", "")][b] += 1
        per_mode[nat[L].get("detection_mode", "")][b] += 1
    n = sum(buckets.values())
    return buckets, per_item, per_mode, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--result", required=True, help="agent 的结果包目录")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--string-compare", action="store_true",
                    help="纯字符串比对（老口径）；默认数值感知")
    ap.add_argument("--label", default="i48",
                    help="报告里的数据集名（只影响打印；N18 复用同一套实现）")
    args = ap.parse_args()
    numeric = not args.string_compare
    if numeric:
        # 2026-11-20 决定：原生错误维度取消 numeric 口径，只保留 string（纯字符串比对）。
        print("✗ 原生 numeric 口径已取消（只保留 string）。请加 --string-compare。")
        sys.exit(3)
    ds, res = Path(args.dataset), Path(args.result)

    status, status_any, nat, meta = judge(ds, res, args.split, numeric)
    buckets, per_item, per_mode, n = summarize(status, nat)
    buckets_any, _, _, _ = summarize(status_any, nat)

    fixed = buckets.get("b_clean", 0)
    report = {
        "dataset": str(ds),
        "result": str(res),
        "split": args.split,
        "compare": "numeric" if numeric else "string",
        "n_native_cells": n,
        "buckets": dict(buckets),
        "buckets_any": dict(buckets_any),
        "repair_rate": round(fixed / n, 6) if n else None,
        "meta": meta,
        "per_itemid": {k: dict(v) for k, v in
                       sorted(per_item.items(), key=lambda x: -sum(x[1].values()))},
        "per_detection_mode": {k: dict(v) for k, v in per_mode.items()},
    }
    print("\n=== MIMIC 原生错误格修复情况（%s）===" % args.label)
    print("  总格数 %d" % n)
    for b in ("b_clean", "b_wrong", "b_dirty", "b_missing"):
        print("  %-10s %5d  (%.1f%%)" % (b, buckets.get(b, 0),
                                          100.0 * buckets.get(b, 0) / max(1, n)))
    print("  修复率 (== 修复值) : %.1f%%" % (100.0 * fixed / max(1, n)))
    print("  [any 口径] %s" % dict(buckets_any))
    print("\n  按 detection_mode: %s" % report["per_detection_mode"])
    print("  按 itemid top:")
    for it, c in list(report["per_itemid"].items())[:14]:
        tot = sum(c.values())
        print("    %-8s 合计 %4d  修对 %4d  改错 %4d  未动 %4d  缺行 %4d"
              % (it, tot, c.get("b_clean", 0), c.get("b_wrong", 0),
                 c.get("b_dirty", 0), c.get("b_missing", 0)))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print("\n写出:", args.out)


if __name__ == "__main__":
    main()
