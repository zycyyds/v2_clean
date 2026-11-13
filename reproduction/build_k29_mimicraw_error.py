#!/usr/bin/env python3
"""build_k29_mimicraw_error.py —— 构建 k29 的「MIMIC 原生错误」评测层。

产物 = <src>_mimicrawerror/，与源数据集同结构（train/validation/test × raw/reference），
差别只有三处：

  1. reference 里**已修好** MIMIC 原生错误（这是金标准）
  2. modification_log.csv 里**登记**这些原生错误（⇒ 进入 v29 Det/Rep 评分）
  3. 新增 mimic_native_error_log.csv 溯源表

raw 的值**只会被注入 train/val 示例时改动**；test 的 raw 一个字节都不动
（原生错误本来就在 raw 里）。

═══ 判定规则（v2，只保留没有金标准争议的）══════════════════════════════
  T1 哨兵错误码：valuenum ∈ {-999,-998,-997,-990,9999,99999,999999,9999999,
                             220000,220045}   —— 对**所有** itemid 生效
  T2 生理不可能区间：itemid 命中 HARD_RANGE 且 越界 且 != 0
  值 0 一律不算错 —— MIMIC 里 0 = 未测到 / 呼吸机暂停，真值不可判定。

  ⭐ 判定「原生」而不是「注入」的关键：我们在 **reference** 里找候选。
     reference 是**未被注入的原始 MIMIC**，所以 reference 里越界的值
     就是 MIMIC 自己录错的。若 log 里同时有该格的 valuenum 条目，
     说明注入器又在这格上叠了一层脏值（raw 里看到的不是原生错值），
     但它**仍然是原生错误格** —— 我们登记的 clean_value 是最终应有的中位数。

═══ 修复规则 R2 ═══════════════════════════════════════════════════════════
  修复值 = 该 (subject_id, itemid) 在**最终干净表**里「可用读数」的中位数
      T2 ⇒ 可用 = 落在该 itemid 区间内 且 != 0
      T1 ⇒ 可用 = 不是哨兵码 且 != 0
  该患者该 itemid 一个可用读数都没有 ⇒ **保持原样不动**（不可判定）

═══ log 语义（已 100% 验证）══════════════════════════════════════════════
  modification_log.csv 的 raw_row_index = **原始干净表的行号** = reference 的行号
      （验证：reference[L].valuenum == log.clean_value 命中率 187/187 = 100%）
  同一格出现多条 cell_update 时，评分器只认**最后一条**
  ⇒ 我们把原生条目录在 log 末尾即可**覆盖**同格的注入条目，
     语义正好是「先修注入、再修原生」的最终结果。

═══ train/validation 注入示例 ═════════════════════════════════════════════
  train 里几乎没有原生错误（k29 只有 0~1 格），agent 无从学起。
  ⇒ 往 train/val 的 raw 里注入**同形态**的孤立尖峰：
       raw 该格 = 人造越界值 d（原值 v 的量纲错形态）
       reference 该格 = 该 (subject,itemid) 可用读数的中位数 m
       log 追加 (L, valuenum, dirty=d, clean=m)
  agent 对照 train raw / train reference 就能自己发现这条规则。

用法：
  python3 build_k29_mimicraw_error.py --src <数据集目录> --dst <输出目录>
      [--splits test] [--inject-train 12] [--inject-val 20] [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

CE_FILE = "icu/chartevents.csv"
CE_KEY = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]

# ---- T2：生理不可能区间（itemid → (lo, hi)）----
HARD_RANGE = {
    220045: (20, 300), 220050: (40, 400), 220051: (20, 250), 220052: (20, 250),
    220059: (0, 150), 220061: (0, 100), 220074: (-20, 80), 220179: (40, 400),
    220180: (20, 250), 220181: (20, 250), 220210: (0, 100), 223762: (25, 45),
    224688: (0, 100), 224689: (0, 100), 224690: (0, 100), 227466: (8, 200),
    224167: (40, 400), 224643: (20, 250), 227242: (20, 250), 227243: (40, 400),
}
# ---- T1：MIMIC 哨兵错误码 ----
SENTINELS = {-999.0, -998.0, -997.0, -990.0, 9999.0, 99999.0,
              999999.0, 9999999.0, 220000.0, 220045.0}

NAT_LOG_FIELDS = ["source_error_id", "split", "raw_file", "raw_row_index", "column",
                  "subject_id", "hadm_id", "stay_id", "charttime", "itemid",
                  "item_label", "mimic_native_value", "repaired_value", "repair_rule",
                  "n_peer_readings", "plausibility_range", "raw_dirty_value",
                  "also_had_injection", "detection_mode"]

# ---- 每个 itemid 的「真实原生错误值」代表值 ----
# 统计自 k29 test 的 510 个纯原生格（per-itemid 中位数，纯聚合、不含病人信息）。
# 注入 train/validation 示例时用它造脏值 ⇒ 示例的**形态**与测试集真实分布一致，
# 而不是清一色 ×100（否则 agent 会学成「只认 ×100 形态」，在 test 上大面积漏检）。
NATIVE_REPR = {
    220045: 17.0,     # heart rate —— 真实形态是「略低于区间下界」
    220050: 30.0,     # arterial BP systolic
    220051: 18.0,     # arterial BP diastolic
    220052: 257.0,    # arterial BP mean —— 真实形态是「远高于上界」
    220059: 150.0,    # PA pressure systolic
    220061: 244.0,    # PA pressure mean
    220074: 292.0,    # central venous pressure —— 真实形态是「250~360 的大数」
    220179: 33.0,     # NIBP systolic
    220180: 16.0,     # NIBP diastolic
    220181: 785.0,    # NIBP mean
    220210: 103.0,    # respiratory rate
    223762: 98.8,     # temperature celsius —— ℉ 误写形态
    224688: 410.0,    # respiratory rate (set)
    224689: 232.0,    # respiratory rate (spontaneous)
    224690: 103.0,    # respiratory rate (total)
    227466: 999999.0,  # ptt —— 哨兵码形态
    224167: 400.0, 224643: 250.0, 227242: 250.0, 227243: 400.0,
}


# ────────────────────────── 小工具 ──────────────────────────
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(v):
    return repr(float(v))


def median_of(arr):
    n = len(arr)
    if n == 0:
        return None
    s = sorted(arr)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def classify(it, v):
    if v in SENTINELS:
        return "T1_sentinel", "mimic_native_sentinel_value"
    rg = HARD_RANGE.get(it)
    if rg and not (rg[0] <= v <= rg[1]):
        return "T2_impossible", "mimic_native_impossible_value"
    return None, None


def in_range(it, v):
    rg = HARD_RANGE.get(it)
    if rg is None:
        return True          # 无硬区间的 itemid：只要不是哨兵就算可用
    return rg[0] <= v <= rg[1]


# ───────────── 扫 reference：候选格 + 可用读数池 ─────────────
def scan_reference(ref_path):
    pools = defaultdict(lambda: {"range": [], "sentinel": []})
    cands = []
    with ref_path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for L, r in enumerate(rd):
            try:
                it = int(r["itemid"])
            except (ValueError, KeyError):
                continue
            v = fnum(r["valuenum"])
            if v is None or v == 0:
                continue
            cat, sub = classify(it, v)
            if cat is None:
                p = pools[(r["subject_id"], it)]
                p["range"].append(v)
                p["sentinel"].append(v)
            else:
                cands.append([L, it, v, cat, sub, r["subject_id"], r])
    return cands, pools


# ───────────── 扫 log：插入行位置 + valuenum 注入登记 ─────────────
def scan_log(log_path):
    ins_pos = []
    val_inj = set()
    n = 0
    with log_path.open(newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            n += 1
            if e.get("raw_file") != CE_FILE:
                continue
            ri = int(e["raw_row_index"])
            op = e.get("operation")
            if op in ("insert", "duplicate"):
                ins_pos.append(ri)
            elif op == "cell_update" and e.get("column") == "valuenum":
                val_inj.add(ri)
    ins_pos.sort()
    return ins_pos, val_inj, n


# ───────────── 定位候选格在 raw 里的当前值 ─────────────
def detect_offset_mode(rows_by_L, ins_pos, raw_path, sample_size=200):
    """探测 raw 行号 ↔ reference 行号的偏移规则。

    ⚠️ 有两种情形，必须自动判定：
       A) raw 里真的插了行（duplicate/insert）⇒ raw 行号 = L + bisect_left(ins_pos, L)
       B) raw 与 reference 逐行对应（log 里的 duplicate 条目指向已存在的行）
          ⇒ raw 行号 = L
       两种都试，用「该 raw 行的 key 是否与 reference 行一致」来判定。
       返回 True = 用 zero（raw 行号 = L）
    """
    Ls = list(rows_by_L)[:sample_size]
    if not Ls:
        return True
    need = set()
    for L in Ls:
        need.add(L)
        need.add(L + bisect_left(ins_pos, L))
    rows_at = {}
    with raw_path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for ri, row in enumerate(rd):
            if ri in need:
                rows_at[ri] = row
            if ri >= max(need):
                break

    def key_of(row):
        return tuple(row.get(c) for c in CE_KEY)

    zero_hit = bis_hit = 0
    for L in Ls:
        k = key_of(rows_by_L[L])
        rz = rows_at.get(L)
        if rz is not None and key_of(rz) == k:
            zero_hit += 1
        rb = rows_at.get(L + bisect_left(ins_pos, L))
        if rb is not None and key_of(rb) == k:
            bis_hit += 1
    use_zero = zero_hit >= bis_hit
    print("    偏移探测: zero 命中 %d / bisect 命中 %d ⇒ raw 行号 = %s"
          % (zero_hit, bis_hit, "L" if use_zero else "L + 插入偏移"), flush=True)
    return use_zero


def raw_rows_for(cands, ins_pos, raw_path, use_zero):
    """cands: [L, it, v, cat, sub, sid, row] → {L: (raw_row_index, valuenum)}"""
    if not cands:
        return {}
    want = {}
    for c in cands:
        L = c[0]
        r = L if use_zero else L + bisect_left(ins_pos, L)
        want.setdefault(r, L)
    out = {}
    with raw_path.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for ri, row in enumerate(rd):
            L = want.get(ri)
            if L is not None:
                out[L] = (ri, row.get("valuenum", ""))
    return out


# ───────────── 改写 reference（按行号） ─────────────
def rewrite_reference(ref_path, fixes):
    n = 0
    tmp = ref_path.with_suffix(".csv.tmp")
    with ref_path.open(newline="", encoding="utf-8") as f, \
            tmp.open("w", newline="", encoding="utf-8") as g:
        rd = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=list(rd.fieldnames))
        w.writeheader()
        for L, row in enumerate(rd):
            nv = fixes.get(L)
            if nv is not None and row.get("valuenum") != nv:
                row["valuenum"] = nv
                n += 1
            w.writerow(row)
    tmp.replace(ref_path)
    return n


# ───────────── 改写 raw 的指定行（用于注入示例 / 恢复叠脏格） ─────────────
def rewrite_raw(raw_path, changes):
    """changes: {raw_row_index: (new_valuenum, new_value_or_None)}

    ⚠️ **`value` 文本列必须与 `valuenum` 同步**。这是本项目实测出来的判据：
       · MIMIC 原生错误 ⇒ raw 的 `value` 文本列记的**也是那个坏数**
         （== valuenum，510/510 全部如此）⇒ 没有任何线索能还原 ⇒ 只能用中位数；
       · 注入的量纲放大 ⇒ raw 的 `value` 列记的是**干净数**（!= valuenum）
         ⇒ 可以直接还原成 value。
       ⇒ 所以注入示例 / 恢复叠脏格时，`value` 必须写成与 `valuenum` 同一个坏数，
          否则形态与测试集的真实分布相反，会把 agent 教反。
    """
    n = 0
    tmp = raw_path.with_suffix(".csv.tmp")
    with raw_path.open(newline="", encoding="utf-8") as f, \
            tmp.open("w", newline="", encoding="utf-8") as g:
        rd = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=list(rd.fieldnames))
        w.writeheader()
        for ri, row in enumerate(rd):
            nv = changes.get(ri)
            if nv is not None:
                row["valuenum"] = nv[0]
                if nv[1] is not None and "value" in row:
                    row["value"] = nv[1]
                n += 1
            w.writerow(row)
    tmp.replace(raw_path)
    return n


# ───────────── 重写 log（流式）+ 追加条目 ─────────────
def rewrite_log(log_path, entries, drop_valuenum_rows=frozenset()):
    """流式重写 modification_log.csv：
         ① **丢弃** raw_file=CE_FILE 且 column=valuenum 且 raw_row_index ∈
            drop_valuenum_rows 的 cell_update 条目 —— 那些格既是 MIMIC 原生错误格、
            又被注入器叠了一层脏值。原生层只要一条（clean=中位数），留着注入条目会
            让 agent 分不清该用「÷100」还是「中位数」。（注入层少评这几百格，
            对千万级的注入层毫无影响；原生层 883 格全部保留。）
         ② 末尾追加原生错误条目（评分器同格只认最后一条 ⇒ 追加即覆盖）。
       返回 (追加条数, 丢弃条数)
    """
    with log_path.open(newline="", encoding="utf-8") as f:
        fields = list(csv.DictReader(f).fieldnames)
    tmp = log_path.with_suffix(".csv.tmp")
    n_drop = 0
    with log_path.open(newline="", encoding="utf-8") as f, \
            tmp.open("w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for e in csv.DictReader(f):
            if (drop_valuenum_rows
                    and e.get("raw_file") == CE_FILE
                    and e.get("column") == "valuenum"
                    and e.get("operation") == "cell_update"):
                try:
                    ri = int(e["raw_row_index"])
                except (TypeError, ValueError):
                    ri = -1
                if ri in drop_valuenum_rows:
                    n_drop += 1
                    continue
            w.writerow(e)
        for e in entries:
            w.writerow(e)
    tmp.replace(log_path)
    return len(entries), n_drop


def mk_entry(split, seq, log_idx, subtype, dirty, clean):
    return {
        "source_error_id": "%s-mimic-%04d" % (split, seq),
        "canonical_stay_id": "",
        "error_class": "class1_intra_table",
        "error_subtype": subtype,
        "target_effect_contract": "(see raw_level_error_split)",
        "operation": "cell_update",
        "raw_file": CE_FILE,
        "raw_row_index": str(log_idx),
        "column": "valuenum",
        "clean_value": clean,
        "dirty_value": dirty,
        "matching_columns": "subject_id|hadm_id|stay_id",
    }


def write_native_log(dst_split, rows):
    p = dst_split / "mimic_native_error_log.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(NAT_LOG_FIELDS)
        for r in rows:
            w.writerow(r)
    return p


# ══════════════════ 阶段 A：处理已有的原生错误（test 等） ══════════════════
def process_existing(split_sp, split):
    ref_p = split_sp / "reference" / CE_FILE
    raw_p = split_sp / "raw" / CE_FILE
    log_p = split_sp / "modification_log.csv"

    t0 = time.time()
    cands, pools = scan_reference(ref_p)
    by_cat = Counter(c[3] for c in cands)
    print("  [%s] reference 里的原生错误候选 %d 格  %s  (%.0fs)"
          % (split, len(cands), dict(by_cat), time.time() - t0), flush=True)

    fixes, details = {}, {}
    for L, it, v, cat, sub, sid, row in cands:
        pool = pools.get((sid, it))
        if pool is None:
            continue
        arr = pool["range"] if cat == "T2_impossible" else pool["sentinel"]
        med = median_of(arr)
        if med is None:
            continue
        fixes[L] = fmt(med)
        details[L] = (it, v, cat, sub, med, len(arr))
    print("  [%s] 可修 %d 格；不可修 %d 格（该患者该 itemid 无可用读数 → 保持原样）"
          % (split, len(fixes), len(cands) - len(fixes)), flush=True)

    ins_pos, val_inj, n_log = scan_log(log_p)
    print("  [%s] log %d 条；插入行 %d；被注入改过 valuenum 的格 %d"
          % (split, n_log, len(ins_pos), len(val_inj)), flush=True)

    t0 = time.time()
    rows_by_L = {c[0]: c[6] for c in cands if c[0] in fixes}
    use_zero = detect_offset_mode(rows_by_L, ins_pos, raw_p)
    found = raw_rows_for([c for c in cands if c[0] in fixes], ins_pos, raw_p, use_zero)
    print("  [%s] 在 raw 里定位到 %d / %d 格 (%.0fs)"
          % (split, len(found), len(fixes), time.time() - t0), flush=True)

    entries, nat_rows = [], []
    raw_restores = {}
    seq = 0
    n_inj = 0
    for L in sorted(fixes):
        if L not in found:
            continue
        seq += 1
        it, v, cat, sub, med, npeer = details[L]
        raw_idx, dirty = found[L]
        had_inj = L in val_inj
        if had_inj:
            # 该格既是 MIMIC 原生错误格，又被注入器叠了一层脏值。
            # ⚠️ 两层错误叠在一起 ⇒ 从数据上无法判定（单格 ×100 与单格原生错误
            #    在形态上完全同构）。**把 raw 的值恢复成 MIMIC 原生错值**（= reference
            #    的原值，权威来源），让这一格只剩**一个**错误（原生错误）⇒
            #    目标唯一、可判定、可评测。注入层因此少评这几百格（可忽略）。
            n_inj += 1
            # `value` 文本列同步写成同一个坏数 ⇒ 与「纯原生格」形态一致
            # （value == valuenum == 坏数 ⇒ 没有任何线索能还原 ⇒ 只能用中位数）
            raw_restores[raw_idx] = (fmt(v), fmt(v))
            dirty = v
        entries.append(mk_entry(split, seq, L, sub, dirty, fixes[L]))
        rg = HARD_RANGE.get(it)
        nat_rows.append([
            "%s-mimic-%04d" % (split, seq), split, CE_FILE, str(L), "valuenum",
            "", "", "", "", it, "",
            v, fixes[L],
            "R2_patient_itemid_median_of_usable_readings",
            npeer,
            ("[%s,%s]" % rg) if rg else "sentinel_code",
            dirty, "yes" if had_inj else "no", cat,
        ])
    print("  [%s] 其中 %d 格同时被注入器叠了脏值 ⇒ raw 值已恢复成 MIMIC 原生错值，"
          "该格只剩一个错误" % (split, n_inj), flush=True)

    n_changed = rewrite_reference(ref_p, fixes)
    n_raw_restored = rewrite_raw(raw_p, raw_restores) if raw_restores else 0
    n_added, n_drop = rewrite_log(log_p, entries, drop_valuenum_rows=set(fixes))
    write_native_log(split_sp, nat_rows)
    print("  [%s] reference 改写 %d 行；raw 恢复 %d 格；log 追加 %d 条；"
          "丢弃与原生格重叠的注入条目 %d 条"
          % (split, n_changed, n_raw_restored, n_added, n_drop), flush=True)
    if n_changed != n_added:
        raise SystemExit("FATAL[%s]: 改了 %d 格但登记 %d 条" % (split, n_changed, n_added))
    if n_raw_restored != n_inj:
        raise SystemExit("FATAL[%s]: 恢复 %d 格但叠脏格是 %d" % (split, n_raw_restored, n_inj))
    print("  [%s] 自检 OK" % split, flush=True)
    return {"candidates": len(cands), "fixable": len(fixes), "changed": n_changed,
            "logged": n_added, "dropped_overlap_injections": n_drop,
            "raw_restored": n_raw_restored, "also_injected": n_inj,
            "by_cat": dict(by_cat)}


# ══════════════════ 阶段 B：往 train/val 注入示例 ══════════════════
def inject_examples(split_sp, split, n_target):
    ref_p = split_sp / "reference" / CE_FILE
    raw_p = split_sp / "raw" / CE_FILE
    log_p = split_sp / "modification_log.csv"

    cands, pools = scan_reference(ref_p)
    ins_pos, val_inj, n_log = scan_log(log_p)

    # 按 (subject, itemid) 分组 reference 的行（只挑 HARD_RANGE 里的 itemid）
    rows_by_group = defaultdict(list)
    rows_by_L = {}
    with ref_p.open(newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for L, row in enumerate(rd):
            try:
                it = int(row["itemid"])
            except (ValueError, KeyError):
                continue
            if it not in HARD_RANGE:
                continue
            v = fnum(row.get("valuenum"))
            if v is None or v == 0:
                continue
            rows_by_group[(row["subject_id"], it)].append((L, v))
            rows_by_L[L] = row
    use_zero = detect_offset_mode(rows_by_L, ins_pos, raw_p)

    # 候选：组内**区间内读数 ≥ 2**（破坏其中 1 个后仍剩 ≥1 个可用读数）
    # ⚠️ R2 口径：修复值 = 同 (subject,itemid) 的「区间内且非 0」读数的中位数，
    #    而坏值本身是越界的、天然不在池子里。注入示例必须复现这个口径 ⇒
    #    **被破坏的那个读数要从池子里剔除**再算中位数。
    # （train 只有 10 个 hadm / 2k 行，条件必须放宽到 usable≥2 才有足够示例）
    picks = []
    for (sid, it), lst in sorted(rows_by_group.items()):
        usable = [(L, v) for L, v in lst if in_range(it, v)]
        if len(usable) < 2:
            continue
        # 破坏「离中位数最远」的那个读数 ⇒ 破坏后的中位数与原值差别明显，例子更清晰
        allv = [v for _, v in usable]
        med_all = median_of(allv)
        if med_all is None:
            continue
        L_bad, v_bad = max(usable, key=lambda t: abs(t[1] - med_all))
        rest = [v for Lx, v in usable if Lx != L_bad]
        med = median_of(rest)
        if med is None:
            continue
        # 剔除坏格后的中位数必须与坏格原值不同，否则这个例子「无教学价值」
        # （reference 根本没变化 ⇒ agent 看不出任何东西）
        if abs(med - v_bad) < 1e-9:
            continue
        picks.append((sid, it, L_bad, v_bad, med, len(rest), len(lst)))
    # 读数多的组优先（中位数更有意义），再按 itemid 分散
    picks.sort(key=lambda p: -p[6])
    print("  [%s] 可注入的组 %d 个" % (split, len(picks)), flush=True)

    # 取前 n_target 个，按 itemid 分散（轮转，避免全挤在同一个 itemid）
    by_item = defaultdict(list)
    for p in picks:
        by_item[p[1]].append(p)
    order = sorted(by_item)
    chosen, k = [], 0
    while len(chosen) < n_target and any(by_item[i] for i in order):
        i = order[k % len(order)]
        if by_item[i]:
            chosen.append(by_item[i].pop(0))
        k += 1

    if not chosen:
        print("  [%s] 没有可注入的组，跳过" % split, flush=True)
        return {"injected": 0}

    # 造脏值：用该 itemid 的**真实原生错误代表值**（形态与测试集一致），
    #   并在「代表值」和「代表值×10」两种形态间轮换 ⇒ 形态多样，
    #   避免 agent 学成「只认某一个形态」。
    # ⚠️ 守卫：脏值必须仍落在区间外（否则看着根本不像错误）⇒ 必要时逐级加倍。
    ref_fixes, raw_changes, entries, nat_rows = {}, {}, [], []
    for seq, (sid, it, L, v, med, npeer, ngrp) in enumerate(chosen, 1):
        repr_v = NATIVE_REPR.get(it)
        if repr_v is None:
            repr_v = v * 100.0
        if seq % 2 == 0:
            repr_v = repr_v * 10.0
        dv = repr_v
        while dv == dv and in_range(it, dv) and abs(dv) < 1e12:
            dv *= 10.0
        dirty = fmt(dv)
        ref_fixes[L] = fmt(med)
        r = L if use_zero else L + bisect_left(ins_pos, L)
        # ⚠️ `value` 文本列同步写成脏值 —— 必须与测试集真实分布同构：
        #    原生错误的 value 记的就是坏数（== valuenum），不是干净数。
        raw_changes[r] = (dirty, dirty)
        entries.append(mk_entry(split, seq, L, "mimic_native_injected_example", dirty, fmt(med)))
        nat_rows.append([
            "%s-mimic-%04d" % (split, seq), split, CE_FILE, str(L), "valuenum",
            "", "", "", "", it, "",
            dirty, fmt(med),
            "R2_patient_itemid_median_of_usable_readings",
            npeer, "[%s,%s]" % (HARD_RANGE[it][0], HARD_RANGE[it][1]),
            dirty, "no", "T2_injected_example",
        ])

    n_ref = rewrite_reference(ref_p, ref_fixes)
    n_raw = rewrite_raw(raw_p, raw_changes)
    # 同一格若之前被注入器改过 valuenum，把那条注入条目也删掉 ⇒
    # 这一格的 log 条目唯一（只有我们这条原生条目），语义干净
    n_added, n_drop = rewrite_log(log_p, entries, drop_valuenum_rows=set(ref_fixes))
    write_native_log(split_sp, nat_rows)
    print("  [%s] 注入 %d 格（reference 改 %d 行 / raw 改 %d 行 / log 加 %d 条；"
          "顺带丢弃同格的注入条目 %d 条）"
          % (split, len(chosen), n_ref, n_raw, n_added, n_drop), flush=True)
    if not (n_ref == n_raw == n_added == len(chosen)):
        raise SystemExit("FATAL[%s]: 注入计数不一致" % split)
    return {"injected": len(chosen), "items": sorted({p[1] for p in chosen})}


# ══════════════════════ 主流程 ══════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--inject-train", type=int, default=0)
    ap.add_argument("--inject-val", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        sys.exit("源不存在: %s" % src)
    if dst.exists():
        if not args.force:
            sys.exit("输出已存在（加 --force 覆盖）: %s" % dst)
        shutil.rmtree(dst)
    print("复制 %s -> %s" % (src, dst), flush=True)
    t0 = time.time()
    shutil.copytree(src, dst)
    print("  完成 %.1fs\n" % (time.time() - t0), flush=True)

    summary = {}
    for split in args.splits.split(","):
        sp = dst / split
        if not (sp / "reference" / CE_FILE).is_file():
            print("跳过", split, "（无 reference/icu/chartevents.csv）\n")
            continue
        print("=" * 72)
        print("SPLIT", split, "—— 处理已有原生错误")
        summary[split] = process_existing(sp, split)
        print()

    for split, n in (("train", args.inject_train), ("validation", args.inject_val)):
        if n <= 0:
            continue
        sp = dst / split
        if not (sp / "reference" / CE_FILE).is_file():
            continue
        print("=" * 72)
        print("SPLIT", split, "—— 注入示例")
        summary.setdefault(split, {})["injected_examples"] = inject_examples(sp, split, n)
        print()

    report = dst / "mimic_native_error_build_report.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
