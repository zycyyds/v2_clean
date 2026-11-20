#!/usr/bin/env python3
"""build_n18_mimic_raw_error.py —— 构建 N18 的「MIMIC 原生错误」评测层（mimicraw 版）。

默认 源: N18_ckd_ED_v2_rawerror_test1.6pct_490M
     新: N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M
（--src / --dst 可换成 4.7pct 版）

## 模板来源
  · **i48 版**（`build_i48_mimic_raw_error.py`）—— 因为 N18 和 i48 一样是
    **cohort 型任务**（ref ⊆ raw）：
      · N18 的 reference/icu/chartevents.csv 只有 659.7 万行，raw 有 1780.8 万行
        ⇒ **log 的 raw_row_index 在「源 raw」行号空间**，和 reference 的行号对不上
        ⇒ **必须按 key 定位，不能按行号**
      · 实测行号空间 = **bisect**（4000/4000 命中；zero 只有 10.4%）
  · **icu5000 版**的「阶段 B：往 train/validation 注入示例」
    ⇒ i48 的教训：train/val 分片里几乎没有原生错误样本 ⇒ agent 学成
      「只认这几个具体值脏」而不是「哨兵码这一类」⇒ T1 检出率只有 4.3%。
      N18 这版把示例注入补上（与 icu5000 同构）。

## 检测规则（与 icu5000 / i48 同构）
  * **T1 哨兵错误码**（对**所有** itemid 生效）：
    valuenum ∈ {-999,-998,-997,-990, 9999,99999,999999,9999999,220000,220045}
  * **T2 生理不可能区间**：itemid 命中「硬不可能区间」且越界
    ⇒ 区间**按 PhysioNet d_items 的 label 文本匹配**（不照抄任何数据集的 itemid 列表）
  * **值 0 一律不算错**（MIMIC 里 0 = 未测到 / 呼吸机暂停，真值不可判定）

## 修复规则 R2（患者-itemid 中位数）
  修复值 = 该 (subject_id, itemid) 在 reference 里「可用读数」的中位数
      T2 ⇒ 可用 = 落在该 itemid 区间内 且 != 0
      T1 ⇒ 可用 = 不是哨兵码 且 != 0
  该患者该 itemid 一个可用读数都没有 ⇒ **保持原样不动**（真值不可判定）

## log 语义
  追加的原生条目用 **log 行号空间**的 raw_row_index（= 该格在「源干净表」里的行号）
  ⇒ 与同格的注入条目同空间，评分器同格只认最后一条 ⇒ 追加即覆盖。

## 断言（任何一条不过就 SystemExit）
  ① 改写 reference 的格数 == 登记进 log 的条数
  ② 改写格数 + 被排除的 itemid 叠脏格数 == 可修格数
  ③ 恢复 raw 的格数 == 同时被注入器改过的格数
"""
import argparse
import csv
import json
import shutil
import sys
from array import array
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

ROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18")
SRC = ROOT / "N18_ckd_ED_v2_rawerror_test1.6pct_490M"
DST = ROOT / "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
D_ITEMS = Path(__file__).resolve().parent / "i48_native_error_fix/d_items.csv"

# ---- T2：硬不可能区间，**按 d_items 的 label 文本**匹配（通用，不照抄 itemid）----
HARD_BY_LABEL = {
    'heart rate': (20, 300), 'blood pressure systolic': (40, 400),
    'arterial bp systolic': (40, 400), 'blood pressure diastolic': (20, 250),
    'blood pressure mean': (20, 250), 'respiratory rate': (0, 100),
    'temperature celsius': (25, 45), 'ptt': (8, 200),
    'central venous pressure': (-20, 80),
    'pulmonary artery pressure systolic': (0, 150),
    'pulmonary artery pressure mean': (0, 100),
}
EXCLUDE_LABEL_WORDS = ('loss', 'unintentional', 'note', 'comment', 'finding',
                       'change', 'alarm')

# ---- T1：MIMIC / CareVue 的哨兵占位码（对所有 itemid 生效）----
SENTINELS = {-999.0, -998.0, -997.0, -990.0, 9999.0, 99999.0,
              999999.0, 9999999.0, 220000.0, 220045.0}

CE_KEY = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid']
CE_FILE = 'icu/chartevents.csv'

NATIVE_LOG_COLS = [
    "source_error_id", "split", "raw_file", "raw_row_index", "column",
    "subject_id", "hadm_id", "stay_id", "charttime", "itemid", "item_label",
    "mimic_native_value", "repaired_value", "repair_rule", "n_peer_readings",
    "plausibility_range", "raw_dirty_value", "also_had_injection",
    "detection_mode",
]


# ────────────────────────── 小工具 ──────────────────────────
def load_di(path):
    di = {}
    with path.open(newline='', encoding='utf-8', errors='replace') as f:
        for r in csv.DictReader(f):
            try:
                di[int(r['itemid'])] = r['label']
            except (ValueError, KeyError):
                pass
    return di


def hard_range(di, itemid):
    lab = di.get(itemid)
    if not lab:
        return None
    l = lab.lower()
    for b in EXCLUDE_LABEL_WORDS:
        if b in l:
            return None
    for k, v in HARD_BY_LABEL.items():
        if k in l:
            return v
    return None


def num(x):
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


def classify(di, it, v):
    """⇒ (类别, error_subtype) 或 (None, None)"""
    if v in SENTINELS:
        return 'T1_sentinel', 'mimic_native_sentinel_value'
    rg = hard_range(di, it)
    if rg and not (rg[0] <= v <= rg[1]):
        return 'T2_impossible', 'mimic_native_impossible_value'
    return None, None


def in_range(di, it, v):
    rg = hard_range(di, it)
    if rg is None:
        return True          # 无硬区间：只要不是哨兵就算可用
    return rg[0] <= v <= rg[1]


# ───────────── 扫 reference：候选格 + 可用读数池（两遍） ─────────────
def scan_reference(di, ref_path):
    """两遍扫 reference（见 i48 版说明）：第 1 遍找候选，第 2 遍只收
    「含候选格的那些 (subject,itemid) 组」的可用读数 ⇒ 池子内存可控。"""
    cands = []
    groups = set()
    with ref_path.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                it = int(row['itemid'])
            except (ValueError, KeyError):
                continue
            v = num(row['valuenum'])
            if v is None or v == 0:
                continue
            cat, sub = classify(di, it, v)
            if cat is None:
                continue
            key = (row['subject_id'], row['hadm_id'], row['stay_id'],
                   row['charttime'], row['itemid'])
            cands.append((key, it, v, cat, sub, di.get(it), row))
            groups.add((row['subject_id'], it))

    pools = defaultdict(lambda: array('d'))
    if groups:
        with ref_path.open(newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for row in rd:
                try:
                    it = int(row['itemid'])
                except (ValueError, KeyError):
                    continue
                g = (row['subject_id'], it)
                if g not in groups:
                    continue
                v = num(row['valuenum'])
                if v is None or v == 0:
                    continue
                cat, _ = classify(di, it, v)
                if cat is None:
                    pools[g].append(v)
    return cands, pools


# ───────────── 扫 log（流式） ─────────────
def scan_log(log_path, raw_file):
    """流式扫 modification_log，只留 chartevents 需要的东西。
    ⚠️ N18 的 test log 有 2552 万条，整表读成 list 会吃掉十几 GB ⇒ 必须流式。"""
    ins_pos = array('i')
    key_patch = {}
    val_upd = array('i')
    key_cols = set(CE_KEY)
    n_log = 0
    with log_path.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for e in rd:
            n_log += 1
            if e['raw_file'] != raw_file:
                continue
            op = e['operation']
            if op in ('insert', 'duplicate'):
                ins_pos.append(int(e['raw_row_index']))
            elif op == 'cell_update':
                ri = int(e['raw_row_index'])
                col = e['column']
                if col in key_cols:
                    key_patch.setdefault(ri, {})[col] = e['clean_value']
                elif col == 'valuenum':
                    val_upd.append(ri)
    ins_pos = array('i', sorted(ins_pos))
    val_upd = array('i', sorted(val_upd))
    return ins_pos, key_patch, val_upd, n_log


# ───────────── 索引空间探测（zero / bisect） ─────────────
def probe_index_space(ins_pos, raw_path, log_path, raw_file, k=300):
    """判定 log 的 raw_row_index 是「零位」还是「加了插入偏移」。

    判据：log 记了某格的 dirty_value ⇒ raw 那一行的 valuenum 应当正好等于它。
    （与评分器 compute_error_metrics.probe_offset_mode 同一判据。）
    """
    samples = []
    with log_path.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for e in rd:
            if e.get('raw_file') != raw_file:
                continue
            if e.get('operation') != 'cell_update' or e.get('column') != 'valuenum':
                continue
            d = num(e.get('dirty_value'))
            if d is None:
                continue
            samples.append((int(e['raw_row_index']), d))
            if len(samples) >= k:
                break
    if not samples:
        return 'bisect', (0, 0)

    need = set()
    for ri, _ in samples:
        need.add(ri)
        need.add(ri + bisect_left(ins_pos, ri))
    rows_at = {}
    with raw_path.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for i, row in enumerate(rd):
            if i in need:
                rows_at[i] = row
            if i > max(need):
                break

    z = b = 0
    for ri, d in samples:
        rz = rows_at.get(ri)
        if rz is not None:
            v = num(rz.get('valuenum'))
            if v is not None and abs(v - d) < 1e-9:
                z += 1
        rb = rows_at.get(ri + bisect_left(ins_pos, ri))
        if rb is not None:
            v = num(rb.get('valuenum'))
            if v is not None and abs(v - d) < 1e-9:
                b += 1
    mode = 'zero' if z > b else 'bisect'
    return mode, (z, b)


def build_raw2log(ins_pos, n_raw, mode):
    """raw 行号 → log 行号 的映射表。
    mode='bisect' ⇒ raw 行 i 对应 log 行 L（用插入位置反推）；
    mode='zero'   ⇒ raw 行 i 对应 log 行 i。"""
    if mode == 'zero':
        return array('i', range(n_raw))
    arr = array('i', [-1]) * n_raw if n_raw > 0 else array('i')
    used = bytearray(n_raw)
    for L in range(n_raw):
        r = L + bisect_left(ins_pos, L)
        if 0 <= r < n_raw and not used[r]:
            arr[r] = L
            used[r] = 1
    return arr


# ───────────── 改写 reference（按 key + 原值） ─────────────
def rewrite_reference(ref_path, targets):
    """targets: {(key, old_value): new_value}。
    ⚠️ 按 (key, 原值) 而不是只按 key：同一 key 可能有多行（重复行），
    只按 key 会把同 key 的正常行也改掉 ⇒ 制造新错误。"""
    n = 0
    tmp = ref_path.with_suffix('.csv.tmp')
    with ref_path.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames)
        with tmp.open('w', newline='', encoding='utf-8') as g:
            w = csv.DictWriter(g, fieldnames=fields)
            w.writeheader()
            for row in rd:
                k = (row['subject_id'], row['hadm_id'], row['stay_id'],
                     row['charttime'], row['itemid'])
                newv = targets.get((k, row['valuenum']))
                if newv is not None:
                    row['valuenum'] = newv
                    n += 1
                w.writerow(row)
    tmp.replace(ref_path)
    return n


# ───────────── 改写 raw 的指定行（恢复原生错值 / 注入示例） ─────────────
def rewrite_raw(raw_path, changes):
    """changes: {raw_row_index: (new_valuenum, new_value_or_None)}

    ⚠️ **`value` 文本列必须与 `valuenum` 同步**（icu5000 实测判据）：
       · MIMIC 原生错误 ⇒ raw 的 `value` 列记的**也是那个坏数**（== valuenum）
         ⇒ 没有任何线索能还原 ⇒ 只能用中位数；
       · 注入的量纲放大 ⇒ raw 的 `value` 列记的是**干净数**（!= valuenum）
         ⇒ 可以直接还原成 value。
       ⇒ 注入示例 / 恢复叠脏格时 `value` 必须写成与 `valuenum` 同一个坏数，
          否则形态与测试集的真实分布相反，会把 agent 教反。"""
    n = 0
    tmp = raw_path.with_suffix('.csv.tmp')
    with raw_path.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames)
        with tmp.open('w', newline='', encoding='utf-8') as g:
            w = csv.DictWriter(g, fieldnames=fields)
            w.writeheader()
            for ri, row in enumerate(rd):
                nv = changes.get(ri)
                if nv is not None:
                    row['valuenum'] = nv[0]
                    if nv[1] is not None and 'value' in row:
                        row['value'] = nv[1]
                    n += 1
                w.writerow(row)
    tmp.replace(raw_path)
    return n


# ───────────── 重写 log（流式）+ 追加条目 ─────────────
def rewrite_log(log_path, entries, drop_rows=frozenset()):
    """① 丢弃与原生格重叠的注入条目（该格只留一个错误，语义干净）
       ② 末尾追加原生条目（评分器同格只认最后一条 ⇒ 追加即覆盖）"""
    with log_path.open(newline='', encoding='utf-8') as f:
        fields = list(csv.DictReader(f).fieldnames)
    tmp = log_path.with_suffix('.csv.tmp')
    n_drop = 0
    with log_path.open(newline='', encoding='utf-8') as f, \
            tmp.open('w', newline='', encoding='utf-8') as g:
        w = csv.DictWriter(g, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for e in csv.DictReader(f):
            if (drop_rows
                    and e.get('raw_file') == CE_FILE
                    and e.get('column') == 'valuenum'
                    and e.get('operation') == 'cell_update'):
                try:
                    ri = int(e['raw_row_index'])
                except (TypeError, ValueError):
                    ri = -1
                if ri in drop_rows:
                    n_drop += 1
                    continue
            w.writerow(e)
        for e in entries:
            w.writerow(e)
    tmp.replace(log_path)
    return len(entries), n_drop


def write_native_log(dst_split, rows, append=False):
    """写溯源表。
    ⚠️ **必须支持追加**：阶段 A（处理已有原生错误）先写一批，
       阶段 B（注入示例）再写一批 —— 若覆盖，阶段 A 的记录会被冲掉
       （log 里的条目还在、溯源表却少了 ⇒ 计数对不上）。
    """
    p = dst_split / 'mimic_native_error_log.csv'
    exists = p.is_file() and append
    with p.open('a' if append else 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(NATIVE_LOG_COLS)
        for r in rows:
            w.writerow(r)
    return p


def mk_entry(split, seq, L, subtype, dirty, clean):
    return {
        'source_error_id': "%s-mimic-%04d" % (split, seq),
        'canonical_stay_id': '',
        'error_class': 'class1_intra_table',
        'error_subtype': subtype,
        'target_effect_contract': '(see raw_level_error_split)',
        'operation': 'cell_update',
        'raw_file': CE_FILE,
        'raw_row_index': str(L),
        'column': 'valuenum',
        'clean_value': clean,
        'dirty_value': dirty,
        'matching_columns': 'subject_id|hadm_id|stay_id',
    }


def tier_tag(name):
    """从数据集名里抽出档位标签（1.6pct / 4.7pct）—— 用于 NATIVE_REPR 文件名。"""
    import re
    m = re.search(r'test(\d+\.\dpct)', name)
    return m.group(1) if m else 'unknown'


# ══════════════════ 阶段 A：处理已有的原生错误 ══════════════════
def process_existing(di, dst_sp, split):
    ref_path = dst_sp / 'reference' / CE_FILE
    raw_path = dst_sp / 'raw' / CE_FILE
    log_path = dst_sp / 'modification_log.csv'

    cands, pools = scan_reference(di, ref_path)
    by_cat = defaultdict(int)
    for c in cands:
        by_cat[c[3]] += 1
    print("  原生错误候选：%d 格（T1 哨兵码 %d + T2 越界 %d）"
          % (len(cands), by_cat['T1_sentinel'], by_cat['T2_impossible']), flush=True)
    byitem = defaultdict(int)
    for c in cands:
        byitem[c[1]] += 1
    for it, n in sorted(byitem.items(), key=lambda kv: -kv[1])[:20]:
        print("     itemid %-7s %-42s %d" % (it, (di.get(it) or '')[:42], n), flush=True)

    fixes, details = {}, {}
    for (key, it, v, cat, sub, lab, row) in cands:
        arr = pools.get((key[0], it), array('d'))
        med = median_of(arr)
        if med is None:
            continue                      # 无可用读数 ⇒ 真值不可判定 ⇒ 不修
        fixes[key] = fmt(med)
        details[key] = (it, v, cat, sub, med, len(arr))
    print("  可修：%d / %d（不可修 %d 格 = 该患者该项一个可用读数都没有）"
          % (len(fixes), len(cands), len(cands) - len(fixes)), flush=True)

    ins_pos, key_patch, val_upd, n_log = scan_log(log_path, CE_FILE)
    print("  modification_log %d 条（chartevents insert/dup %d、key 列 patch %d、"
          "valuenum 注入 %d）" % (n_log, len(ins_pos), len(key_patch), len(val_upd)),
          flush=True)

    mode, (z, b) = probe_index_space(ins_pos, raw_path, log_path, CE_FILE)
    print("  索引空间探测：zero 命中 %d / bisect 命中 %d ⇒ %s" % (z, b, mode), flush=True)

    n_raw = sum(1 for _ in raw_path.open(encoding='utf-8')) - 1
    raw2log = build_raw2log(ins_pos, n_raw, mode)
    print("  raw 行数 %d" % n_raw, flush=True)

    want = set(fixes)
    val_upd_set = set(val_upd)
    cands_raw = defaultdict(list)
    with raw_path.open(newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            L = raw2log[i]
            if L < 0:
                continue
            patch = key_patch.get(L)
            if patch:
                ck = tuple(patch.get(c, row[c]) for c in CE_KEY)
            else:
                ck = (row['subject_id'], row['hadm_id'], row['stay_id'],
                      row['charttime'], row['itemid'])
            if ck in want:
                cands_raw[ck].append((i, L, row['valuenum'], row.get('itemid', '')))

    # 孪生行：同一 key 在 raw 里可能有多行 ⇒ 优先选 valuenum == 原生错值的那一行
    found, n_twin = {}, 0
    for ck, lst in cands_raw.items():
        if len(lst) > 1:
            n_twin += 1
        v_ref = details[ck][1]
        pick = None
        for hit in lst:
            v = num(hit[2])
            if v is not None and abs(v - float(v_ref)) < 1e-9:
                pick = hit
                break
        found[ck] = pick if pick is not None else lst[0]
    print("  定位到 raw 行：%d / %d（其中 %d 个 key 有孪生行）"
          % (len(found), len(want), n_twin), flush=True)

    entries, nat_rows, raw_restores = [], [], {}
    seq = 0
    n_inj = 0
    n_itemid_clash = 0
    excluded = set()
    n_fixable_total = len(fixes)
    drop_inj_rows = set()
    for key in sorted(fixes):
        hit = found.get(key)
        if hit is None:
            continue
        raw_idx, L, raw_val, raw_it = hit
        it, v, cat, sub, med, npeer = details[key]
        # 两层错误叠加（itemid 也被注入器改过）⇒ 不可判定 ⇒ 不进 native 层
        try:
            if raw_it != "" and int(raw_it) != it:
                n_itemid_clash += 1
                excluded.add(key)
                continue
        except (TypeError, ValueError):
            pass
        seq += 1
        had_inj = L in val_upd_set
        dirty = raw_val
        if had_inj:
            # 两层错误叠在一起无法判定 ⇒ 把 raw 恢复成 MIMIC 原生错值 ⇒ 一格只剩一个错误
            n_inj += 1
            raw_restores[raw_idx] = (fmt(v), fmt(v))
            dirty = v
            drop_inj_rows.add(L)
        entries.append(mk_entry(split, seq, L, sub, dirty, fixes[key]))
        rg = hard_range(di, it)
        nat_rows.append([
            "%s-mimic-%04d" % (split, seq), split, CE_FILE, str(L), 'valuenum',
            key[0], key[1], key[2], key[3], it, di.get(it, ''),
            v, fixes[key],
            'R2_patient_itemid_median_of_usable_readings',
            npeer,
            ("[%s,%s]" % rg) if rg else 'sentinel_code',
            dirty, 'yes' if had_inj else 'no', cat,
        ])
    print("  其中 %d 格同时被注入器叠了脏值 ⇒ raw 值已恢复成 MIMIC 原生错值，"
          "该格只剩一个错误" % n_inj, flush=True)
    if n_itemid_clash:
        print("  ⚠️ %d 格的 itemid 也被注入器改过（两层错误叠加、不可判定）"
              " ⇒ 不进 native 层，交给注入层评测" % n_itemid_clash, flush=True)

    if excluded:
        fixes = {k: v for k, v in fixes.items() if k not in excluded}
    targets = {}
    for key, nv in fixes.items():
        it, v, cat, sub, med, npeer = details[key]
        targets[(key, fmt(v))] = nv
    n_changed = rewrite_reference(ref_path, targets)
    n_raw_restored = rewrite_raw(raw_path, raw_restores) if raw_restores else 0
    n_added, n_drop = rewrite_log(log_path, entries, drop_rows=drop_inj_rows)
    write_native_log(dst_sp, nat_rows)
    print("  reference 改写 %d 行；raw 恢复 %d 格；log 追加 %d 条；"
          "丢弃与原生格重叠的注入条目 %d 条"
          % (n_changed, n_raw_restored, n_added, n_drop), flush=True)

    if n_changed != n_added:
        raise SystemExit("FATAL[%s]: 改了 %d 格但登记 %d 条" % (split, n_changed, n_added))
    if n_changed + n_itemid_clash != n_fixable_total:
        raise SystemExit("FATAL[%s]: 改了 %d + 排除 %d != 可修 %d"
                         % (split, n_changed, n_itemid_clash, n_fixable_total))
    if n_raw_restored != n_inj:
        raise SystemExit("FATAL[%s]: 恢复 %d 格但叠脏格是 %d" % (split, n_raw_restored, n_inj))
    print("  [%s] 自检 OK" % split, flush=True)
    return {
        'candidates': len(cands), 'fixable': n_fixable_total,
        'changed': n_changed, 'logged': n_added,
        'dropped_overlap_injections': n_drop,
        'raw_restored': n_raw_restored, 'also_injected': n_inj,
        'itemid_clash_excluded': n_itemid_clash,
        'twin_keys': n_twin, 'index_mode': mode,
        'by_cat': dict(by_cat),
    }


# ══════════════════ 阶段 B：往 train/validation 注入示例 ══════════════════
def inject_examples(di, dst_sp, split, n_target, native_repr, src_sp):
    """往 train/validation 的 raw 里注入**同形态**的孤立尖峰，让 agent 能自学规则。

    raw 该格 = 人造坏值 d
    reference 该格 = 该 (subject,itemid) 可用读数的中位数 m
    log 追加 (L, valuenum, dirty=d, clean=m)

    ⚠️⚠️ **池子必须从「源 reference」读**（src_sp），不能从产物 reference 读：
       阶段 A（process_existing）已经把越界的真实原生格**改成了修复值**，
       而修复值本身是**可用**的（落在区间内）⇒ 若从产物 reference 读池子，
       那格会混进池子 ⇒ 注入示例的中位数与「真实原生格的中位数」口径不一致
       （独立校验器用源 reference 重算就会对不上）。
       ⇒ 统一用源 reference 的可用读数池 ⇒ 两类格口径一致、可独立复算。

    ⚠️⚠️ **两种形态都要注入**（这是相对 icu5000 的改进，也是 i48 的教训）：
       · icu5000 版只注入 **T2 越界**形态 ⇒ agent 学到「越界值 ⇒ 用中位数」，
         但**哨兵码这一类完全没教** ⇒ i48 的 T1 检出率只有 4.3%。
       · N18 的 train/validation 里 T1 = **0 格**、T2 = 3 格（test 却有 66 + 701）
         ⇒ 不注入的话 agent **必然**学不到哨兵码 ⇒ T1 必然大面积漏检。
       ⇒ 这里按真实分布各造一半：
         · T2 形态：脏值 = 该 itemid 的真实越界代表值（NATIVE_REPR），偶数次再 ×10
                     （itemid 取**生命体征类**，与 test 真实分布一致）
         · T1 形态：脏值 = 该 itemid 的真实哨兵代表值（N18 里几乎全是 999999.0）
                     （itemid 取**化验类**，与 test 真实分布一致）
    """
    ref_path = dst_sp / 'reference' / CE_FILE
    src_ref_path = src_sp / 'reference' / CE_FILE
    raw_path = dst_sp / 'raw' / CE_FILE
    log_path = dst_sp / 'modification_log.csv'

    # NATIVE_REPR 按形态分成两个池子
    t2_items, t1_items = {}, {}
    for it, m in native_repr.items():
        (t1_items if m in SENTINELS else t2_items)[it] = m
    print("  NATIVE_REPR：T2 形态 %d 个 itemid / T1 形态 %d 个 itemid"
          % (len(t2_items), len(t1_items)), flush=True)

    ins_pos, key_patch, val_upd, n_log = scan_log(log_path, CE_FILE)
    mode, (z, b) = probe_index_space(ins_pos, raw_path, log_path, CE_FILE)
    n_raw = sum(1 for _ in raw_path.open(encoding='utf-8')) - 1
    raw2log = build_raw2log(ins_pos, n_raw, mode)
    print("  索引空间 %s（zero %d / bisect %d）；raw %d 行"
          % (mode, z, b, n_raw), flush=True)

    # 按 (subject, itemid) 分组 **源** reference 的行
    # ⚠️ 用源 reference（不是产物）：阶段 A 已把越界格改成修复值，修复值是「可用」的
    #    ⇒ 从产物读池子会把那格混进去 ⇒ 注入示例的中位数与真实原生格口径不一致。
    rows_by_group = defaultdict(list)
    row_meta = {}
    with src_ref_path.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for L, row in enumerate(rd):
            try:
                it = int(row['itemid'])
            except (ValueError, KeyError):
                continue
            v = num(row.get('valuenum'))
            if v is None or v == 0:
                continue
            g = (row['subject_id'], it)
            rows_by_group[g].append((L, v))
            # ⚠️ 存**原始字符串**的 valuenum 与**完整 5 段 key** ——
            #    rewrite_reference 按 (key, 原值) 定位（避开同 key 的重复行）；
            #    定位 raw 行也必须用完整 key（只用 (subject,itemid) 会改错行）。
            row_meta[L] = (it, v, g,
                           (row['subject_id'], row['hadm_id'], row['stay_id'],
                            row['charttime'], row['itemid']),
                           row['valuenum'])

    def usable(it, v):
        """与 R2 修复口径一致的「可用读数」：非哨兵、非 0、有硬区间则还要在区间内。"""
        if v == 0 or v in SENTINELS:
            return False
        rg = hard_range(di, it)
        return rg is None or (rg[0] <= v <= rg[1])

    # ── 第一步：扫 raw，建「可定位的完整 key」集合 ──
    # ⚠️⚠️ **必须先扫 raw 再筛候选**：N18 的 train/validation 里，
    #    同一个 (subject,itemid) 常常有好几行，其中**大部分的 key 列被注入器改过**
    #    ⇒ 定位不到。若先取候选、后定位，就会「取 30 个、只定位到 9 个」。
    #    正确顺序：先扫 raw 知道哪些完整 key 能用，再据此筛候选池。
    cand_pairs = set()
    for g in rows_by_group:
        cand_pairs.add((g[0], str(g[1])))
    raw_by_key = defaultdict(list)
    n_keydamaged = 0
    n_valdamaged = 0
    val_upd_set = set(val_upd)
    with raw_path.open(newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            L = raw2log[i]
            if L < 0:
                continue
            if L in key_patch:
                # ⚠️⚠️ **key 列被注入器改过的行一律不注入**：
                #    那样的行已经是「key 坏 + valuenum 坏」两层错误叠加 ⇒ 形态同构、
                #    不可判定 ⇒ 注入示例会**把 agent 教反**。
                #    原则与 icu5000 / i48 一致：**每格只有一个错误**。
                n_keydamaged += 1
                continue
            if L in val_upd_set:
                # ⚠️ valuenum 也被注入器改过的行 ⇒ 同样是两层错误叠加 ⇒ 跳过
                n_valdamaged += 1
                continue
            if (row['subject_id'], row['itemid']) not in cand_pairs:
                continue
            ck = (row['subject_id'], row['hadm_id'], row['stay_id'],
                  row['charttime'], row['itemid'])
            raw_by_key[ck].append((i, L, row['valuenum']))
    print("  raw 里 key 被注入器改过的行 %d 行、valuenum 被改过的行 %d 行 ⇒ 已跳过"
          "（避免两层错误叠加）；可定位的完整 key %d 个"
          % (n_keydamaged, n_valdamaged, len(raw_by_key)), flush=True)

    # ── 第二步：筛候选（组内可用读数 ≥ 2，且**该行在 raw 里能定位**）──
    pools = {'T2': defaultdict(list), 'T1': defaultdict(list)}
    n_unlocatable = 0
    for g, lst in rows_by_group.items():
        it = g[1]
        form = None
        if it in t2_items:
            form = 'T2'
        elif it in t1_items:
            form = 'T1'
        if form is None:
            continue
        us = [(L, v) for L, v in lst if usable(it, v)]
        if len(us) < 2:
            continue
        med_all = median_of([v for _, v in us])
        if med_all is None:
            continue
        L_bad, v_bad = max(us, key=lambda t: abs(t[1] - med_all))
        rest = [v for Lx, v in us if Lx != L_bad]
        med = median_of(rest)
        if med is None or abs(med - v_bad) < 1e-9:
            continue        # 中位数与原值相同 ⇒ 无教学价值
        # ⚠️ 该行在 raw 里必须能定位（完整 key 存在，且 valuenum 未被注入改过）
        kt = row_meta[L_bad][3]
        hits = raw_by_key.get(kt, [])
        if not hits:
            n_unlocatable += 1
            continue
        # 同一完整 key 可能有多行 ⇒ 优先选 valuenum == 原值（v_bad）的那一行
        pick = None
        for hit in hits:
            hv = num(hit[2])
            if hv is not None and abs(hv - float(v_bad)) < 1e-9:
                pick = hit
                break
        if pick is None:
            continue
        pools[form][g].append((L_bad, v_bad, med, len(rest), len(lst), pick))
    for form in ('T2', 'T1'):
        print("  %s 形态可注入的组 %d 个" % (form, len(pools[form])), flush=True)
    if n_unlocatable:
        print("  ⚠️ %d 个候选在 raw 里定位不到（key 被改或 valuenum 被改）⇒ 已排除"
              % n_unlocatable, flush=True)

    # 两种形态各占一半，按 itemid 轮转取
    n_half = max(1, n_target // 2)
    chosen = []
    for form, quota in (('T2', n_half), ('T1', n_target - n_half)):
        by_item = defaultdict(list)
        for g, lst in pools[form].items():
            lst.sort(key=lambda t: -t[4])
            by_item[g[1]].append((g, lst))
        order = sorted(by_item)
        k = 0
        got = 0
        while got < quota and any(by_item[i] for i in order):
            i = order[k % len(order)]
            if by_item[i]:
                g, lst = by_item[i].pop(0)
                chosen.append((form, g, lst[0][0], lst[0][1], lst[0][2],
                               lst[0][3], lst[0][4], lst[0][5]))
                got += 1
            k += 1
    if not chosen:
        print("  没有可注入的组，跳过", flush=True)
        return {'injected': 0}

    ref_fixes, raw_changes, entries, nat_rows = {}, {}, [], []
    n_skip = 0
    for seq, (form, g, L, v, med, npeer, ngrp, pick) in enumerate(chosen, 1):
        it2, v2, g2, key_tuple, val_str = row_meta[L]
        raw_idx, Llog, _ = pick
        # 脏值：T1 用哨兵代表值；T2 用越界代表值（偶数次再 ×10 造形态多样性）
        if form == 'T1':
            dv = t1_items.get(it2, 999999.0)
        else:
            dv = t2_items.get(it2, v * 100.0)
            if seq % 2 == 0:
                dv = dv * 10.0
        while dv == dv and usable(it2, dv) and abs(dv) < 1e12:
            dv *= 10.0
        dirty = fmt(dv)
        # ⚠️ rewrite_reference 按 **(key, 原值)** 定位（避开同 key 的重复行）
        ref_fixes[(key_tuple, val_str)] = fmt(med)
        raw_changes[raw_idx] = (dirty, dirty)
        entries.append(mk_entry(split, seq, Llog, 'mimic_native_injected_example',
                                dirty, fmt(med)))
        rg = hard_range(di, it2)
        # ⚠️ key 列**必须填**（icu5000 的注入版留空了 ⇒ 溯源表没法定位、
        #    原生格判定器按 (subject,hadm,stay,charttime,itemid) 查会找不到）
        nat_rows.append([
            "%s-mimic-%04d" % (split, seq), split, CE_FILE, str(Llog), 'valuenum',
            key_tuple[0], key_tuple[1], key_tuple[2], key_tuple[3], it2,
            di.get(it2, ''),
            dirty, fmt(med),
            'R2_patient_itemid_median_of_usable_readings',
            npeer, ("[%s,%s]" % rg) if rg else 'sentinel_code',
            dirty, 'no', ('T1_injected_example' if form == 'T1'
                          else 'T2_injected_example'),
        ])
    if n_skip:
        print("  ⚠️ %d 个候选在 raw 里定位不到，已跳过" % n_skip, flush=True)

    n_ref = rewrite_reference(ref_path, ref_fixes)
    n_raw = rewrite_raw(raw_path, raw_changes)
    n_added, n_drop = rewrite_log(log_path, entries, drop_rows=set())
    write_native_log(dst_sp, nat_rows, append=True)
    n_t1 = sum(1 for r in nat_rows if r[-1] == 'T1_injected_example')
    n_t2 = sum(1 for r in nat_rows if r[-1] == 'T2_injected_example')
    print("  注入 %d 格（T1 %d / T2 %d；reference 改 %d 行 / raw 改 %d 行 / log 加 %d 条）"
          % (len(nat_rows), n_t1, n_t2, n_ref, n_raw, n_added), flush=True)
    if not (n_ref == n_raw == n_added == len(nat_rows)):
        raise SystemExit("FATAL[%s]: 注入计数不一致" % split)
    return {'injected': len(nat_rows), 'T1': n_t1, 'T2': n_t2,
            'items': sorted({r[9] for r in nat_rows})}


# ───────────── 统计 NATIVE_REPR（--stat-repr） ─────────────
def stat_repr(di, ds, splits, out_path):
    """每个 itemid 的「原生错值」中位数 —— 注入 train/val 示例时造同形态的脏值用。"""
    per_item = defaultdict(list)
    for split in splits:
        ref = ds / split / 'reference' / CE_FILE
        if not ref.is_file():
            continue
        with ref.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    it = int(row['itemid'])
                except (ValueError, KeyError):
                    continue
                v = num(row['valuenum'])
                if v is None or v == 0:
                    continue
                cat, _ = classify(di, it, v)
                if cat is not None:
                    per_item[it].append(v)
    out = {}
    for it, vals in sorted(per_item.items()):
        med = median_of(array('d', vals))
        if med is not None:
            out[str(it)] = {"n": len(vals), "median": med, "label": di.get(it, "")}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding='utf-8')
    print("  NATIVE_REPR：%d 个 itemid -> %s" % (len(out), out_path), flush=True)
    return out


# ══════════════════════ 主流程 ══════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='train,validation,test')
    ap.add_argument('--src', default=None)
    ap.add_argument('--dst', default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--stat-repr', action='store_true')
    ap.add_argument('--inject-train', type=int, default=0,
                    help='往 train 注入多少个示例（0=不注入）')
    ap.add_argument('--inject-val', type=int, default=0)
    ap.add_argument('--repr-json', default=None,
                    help='NATIVE_REPR json（注入示例用）')
    args = ap.parse_args()
    splits = args.splits.split(',')

    global SRC, DST
    if args.src:
        SRC = Path(args.src)
    if args.dst:
        DST = Path(args.dst)
    print("SRC =", SRC)
    print("DST =", DST)

    di = load_di(D_ITEMS)

    if args.stat_repr:
        stat_repr(di, SRC, splits,
                  Path(__file__).resolve().parent / ("n18_native_repr_%s.json"
                                                     % tier_tag(DST.name)))
        return

    native_repr = {}
    if args.repr_json and Path(args.repr_json).is_file():
        raw_repr = json.loads(Path(args.repr_json).read_text(encoding='utf-8'))
        native_repr = {int(k): v['median'] for k, v in raw_repr.items()}
        print("  NATIVE_REPR 载入 %d 个 itemid" % len(native_repr))

    summary = {}
    for split in splits:
        src_sp = SRC / split
        dst_sp = DST / split
        print("\n" + "=" * 78)
        print("SPLIT:", split)
        print("=" * 78, flush=True)
        if not src_sp.is_dir():
            print("  源没有这个 split，跳过")
            continue

        if args.dry_run:
            ref_path = src_sp / 'reference' / CE_FILE
        else:
            if dst_sp.exists():
                if not args.force:
                    print("[skip] %s 已存在（要覆盖加 --force）" % dst_sp)
                    continue
                shutil.rmtree(dst_sp)
            print("  copy %s -> %s" % (src_sp, dst_sp), flush=True)
            shutil.copytree(src_sp, dst_sp)
            ref_path = dst_sp / 'reference' / CE_FILE

        if not ref_path.is_file():
            print("  无 reference/%s，跳过" % CE_FILE)
            continue

        if args.dry_run:
            cands, pools = scan_reference(di, ref_path)
            by_cat = defaultdict(int)
            for c in cands:
                by_cat[c[3]] += 1
            print("  原生错误候选：%d 格（T1 %d / T2 %d）"
                  % (len(cands), by_cat['T1_sentinel'], by_cat['T2_impossible']))
            summary[split] = {'n_candidates': len(cands)}
            continue

        summary[split] = process_existing(di, dst_sp, split)

        # 阶段 B：train/validation 注入示例（池子取**源** reference ⇒ 口径与真实原生格一致）
        if split == 'train' and args.inject_train:
            summary[split]['inject'] = inject_examples(
                di, dst_sp, split, args.inject_train, native_repr, src_sp)
        elif split == 'validation' and args.inject_val:
            summary[split]['inject'] = inject_examples(
                di, dst_sp, split, args.inject_val, native_repr, src_sp)

    if not args.dry_run:
        rp = DST / 'mimic_native_error_build_report.json'
        rp.write_text(json.dumps({
            'dataset': DST.name,
            'src': SRC.name,
            'rule': 'T1_sentinel + T2_impossible -> R2_patient_itemid_median_of_usable_readings',
            'sentinels': sorted(SENTINELS),
            'hard_by_label': {k: list(v) for k, v in HARD_BY_LABEL.items()},
            'splits': summary,
        }, ensure_ascii=False, indent=1), encoding='utf-8')
        print("\n构建报告:", rp)


if __name__ == '__main__':
    main()
