#!/usr/bin/env python3
"""构建 I48 的「MIMIC 原生错误」增强版数据集（v2 —— 对齐 icu5000 标准）。

默认 源: I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M
     新: I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M
（用 --src / --dst 可以换成别的注入预设，比如 4.7pct 版）

## 与上一版（mimicrawerror，Nov 2026-07/09 建的）的差别

上一版只做了 **T2（生理不可能区间）**，且缺 3 个 icu5000 上已修的坑。这版补齐：

| # | 上一版 | 这版 |
|---|---|---|
| 1 | 只有 T2（label 硬区间越界） | **T1 哨兵码类 + T2 越界类**（与 icu5000 同构） |
| 2 | 「itemid 也被注入器改过」的格**留在 native 层** | **排除**（两层错误叠加、形态同构、不可判定）⇒ 交给注入层评测 |
| 3 | 格上同时被注入器改过 valuenum 时，raw 里躺的是**注入层的脏值**（9999/-888） | **把 raw 恢复成 MIMIC 原生错值** ⇒ 一格只剩一个错误，目标唯一可判定 |
| 4 | 只查 train/validation | **train/validation/test 全查**，流式（test 的 log 2100 多万条不整表载入） |
| 5 | 溯源表 18 列 | **19 列**（加 `detection_mode`，与 icu5000 对齐） |
| 6 | 无构建报告 | 写 `mimic_native_error_build_report.json` |
| 7 | 无 NATIVE_REPR 统计 | `--stat-repr`：per-itemid 纯原生错值中位数（给 prompt 的对照例子用） |

## 检测规则

  * **T1 哨兵错误码**（对**所有** itemid 生效）：valuenum ∈ {-999,-998,-997,-990,
    9999,99999,999999,9999999,220000,220045}
  * **T2 生理不可能区间**：itemid 命中「硬不可能区间」且越界
    （区间按 PhysioNet d_items 的**权威 label 文本**匹配 ⇒ i48 自己的区间，不是照抄 K29）
  * **值 0 一律不算错**（MIMIC 里 0 = 未测到 / 呼吸机暂停，真值不可判定）

## 修复规则 R2（患者-itemid 中位数）

  把 MIMIC 原生错值替换为「该 subject、该 itemid 在 reference 里**可用**读数的中位数」。
  偶数个时取中间两个的平均（与 numpy/pandas/statistics.median 一致）。
  「可用」= 不是哨兵码、不是越界值、且 ≠ 0。

  为什么不用「最近的邻域读数」：等距时有歧义（例：CVP 261 前后各隔 15 分钟，
  29 还是 25 无法判定），agent 选错一边就会被判 b_wrong。中位数无歧义。

## 三条自检断言（任何一条不过就 SystemExit）

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

ROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集")
SRC = ROOT / "I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M"
DST = ROOT / "I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M"
D_ITEMS = Path(__file__).resolve().parent / "i48_native_error_fix/d_items.csv"

# ---- T2：i48 的「硬不可能区间」，按 d_items 的 label 文本匹配 ----
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
    """reference 里 valuenum 的格式：Python float 的 repr（'62.0' / '15.5'）"""
    return repr(float(v))


def median_of(arr):
    """arr: array('d') -> 中位数（偶数个取中间两个的平均，与 numpy 一致）"""
    n = len(arr)
    if n == 0:
        return None
    s = sorted(arr)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def classify(di, it, v):
    """→ (cat, subtype)；cat 为 None 表示这一格不是原生错误。"""
    if v in SENTINELS:
        return "T1_sentinel", "mimic_native_sentinel_value"
    rg = hard_range(di, it)
    if rg and not (rg[0] <= v <= rg[1]):
        return "T2_impossible", "mimic_native_impossible_value"
    return None, None


# ───────────── 扫 reference：候选格 + 可用读数池（两遍） ─────────────
def scan_reference(di, ref_path):
    """两遍扫 reference：

       第 1 遍：找原生错误候选（T1 哨兵码 / T2 越界），记下
                (key, itemid, 原生错值, 类别, 区间)
       第 2 遍：只对**含候选格的那些 (subject,itemid) 组**收集「可用读数」
                （不是哨兵码、不是越界、且 ≠ 0）⇒ 中位数的池子

       ⚠️ 为什么要两遍：test 分片的 reference 有 1277 万行，
          全量收集所有 itemid 的读数会吃掉几百 MB；
          而真正需要池子的只有「有候选格」的组（实测不到 2000 个）。

    返回 (cands, pools)
    """
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

    test 分片的 log 有 2100 多万条，整表读成 list（每行一个 dict）要 ~14 GB，
    16 GB 机器上必炸。所以这里只留四样：
      ins_pos  : insert/duplicate 的 raw_row_index（决定 raw->log 的错位）
      key_patch: 改过 **key 列**（subject_id/hadm_id/stay_id/charttime/itemid）
                  的 cell_update —— 只有这些会改变 raw 行匹配到的干净 key
      val_upd  : 改过 valuenum 的 cell_update 的 raw_row_index（用来判重）
      n_log    : log 总条数
    其余条目用完即弃。
    """
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
                    val_upd.append(int(e['raw_row_index']))
    ins_pos = array('i', sorted(ins_pos))
    val_upd = array('i', sorted(val_upd))
    return ins_pos, key_patch, val_upd, n_log


def build_raw2log(ins_pos, n_raw):
    """复刻评分器的 raw->log 索引表（Scheme A：bisect）。

    用 array('i') 而不是 list —— test 分片有 1700 多万行，list 会吃掉 ~600MB。
    """
    arr = array('i', [-1]) * n_raw if n_raw > 0 else array('i')
    used = bytearray(n_raw)
    for L in range(n_raw):
        r = L + bisect_left(ins_pos, L)
        if 0 <= r < n_raw and not used[r]:
            arr[r] = L
            used[r] = 1
    return arr


# ───────────── 改写 reference（按 key + 原值，避开同 key 的其它行） ─────────────
def rewrite_reference(ref_path, targets):
    """targets: {(key, old_value): new_value} —— 改写并返回改写行数。

    ⚠️ **为什么按 (key, 原值) 而不是只按 key**：i48 的 reference 里同一个
       (subject, hadm, stay, charttime, itemid) 可能出现多行（重复行）。
       只按 key 改会把**同一 key 的正常行也一起改掉** ⇒ 制造新的错误。
       加上「原值 == 原生错值」这一条，就只命中真正坏的那一行。
    """
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


# ───────────── 改写 raw 的指定行（恢复原生错值） ─────────────
def rewrite_raw(raw_path, changes):
    """changes: {raw_row_index: (new_valuenum, new_value_or_None)}

    ⚠️ **`value` 文本列必须与 `valuenum` 同步**（icu5000 实测出来的判据）：
       · MIMIC 原生错误 ⇒ raw 的 `value` 文本列记的**也是那个坏数**（== valuenum）
         ⇒ 没有任何线索能还原 ⇒ 只能用中位数；
       · 注入的量纲放大 ⇒ raw 的 `value` 列记的是**干净数**（!= valuenum）
         ⇒ 可以直接还原成 value。
       ⇒ 所以把「叠脏格」的 raw 恢复成原生错值时，`value` 必须写成同一个坏数，
          否则形态与测试集的真实分布相反，会把 agent 教反。
    """
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
    """流式重写 modification_log.csv：
         ① **丢弃** raw_file=CE_FILE 且 column=valuenum 且 raw_row_index ∈
            drop_rows 的 cell_update 条目 —— 那些格既是 MIMIC 原生错误格、
            又被注入器叠了一层脏值。原生层只要一条（clean=中位数），留着注入条目
            会让 agent 分不清该用「÷100」还是「中位数」。
         ② 末尾追加原生错误条目（评分器同格只认最后一条 ⇒ 追加即覆盖）。
       返回 (追加条数, 丢弃条数)
    """
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


def write_native_log(dst_split, rows):
    p = dst_split / 'mimic_native_error_log.csv'
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(NATIVE_LOG_COLS)
        for r in rows:
            w.writerow(r)


# ───────────── 统计 NATIVE_REPR（per-itemid 纯原生错值中位数） ─────────────
def stat_repr(ds, splits, out_path):
    """统计每个 itemid 的「纯原生错值」的中位数 —— 给 prompt 的对照例子用。

    「纯原生错值」= 在 reference 里被判为原生错误、且**没被注入器改过**的那些值。
    """
    di = load_di(D_ITEMS)
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
            out[str(it)] = {"n": len(vals), "median": med,
                            "label": di.get(it, "")}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding='utf-8')
    print("  NATIVE_REPR：%d 个 itemid -> %s" % (len(out), out_path), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='train,validation,test')
    ap.add_argument('--src', default=None, help='源数据集目录（默认 I48 1.6pct）')
    ap.add_argument('--dst', default=None, help='目标数据集目录')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='只统计，不复制不修改')
    ap.add_argument('--stat-repr', action='store_true',
                    help='只统计 NATIVE_REPR（per-itemid 原生错值中位数），不构建')
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
        stat_repr(SRC, splits,
                  Path(__file__).resolve().parent /
                  "i48_native_repr_%s.json" % DST.name.split('test')[-1].split('pct')[0])
        return

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

        ref_src = src_sp / 'reference/icu/chartevents.csv'
        if args.dry_run:
            ref_path = ref_src
        else:
            if dst_sp.exists():
                if not args.force:
                    print("[skip] %s 已存在（要覆盖加 --force）" % dst_sp)
                    continue
                shutil.rmtree(dst_sp)
            print("  copy %s -> %s" % (src_sp, dst_sp), flush=True)
            shutil.copytree(src_sp, dst_sp)
            ref_path = dst_sp / 'reference/icu/chartevents.csv'

        # ---------- 1. 扫 reference：候选格 + 可用读数池 ----------
        cands, pools = scan_reference(di, ref_path)
        by_cat = defaultdict(int)
        for c in cands:
            by_cat[c[3]] += 1
        print("  原生错误候选：%d 格（T1 哨兵码 %d + T2 越界 %d）"
              % (len(cands), by_cat['T1_sentinel'], by_cat['T2_impossible']),
              flush=True)
        byitem = defaultdict(int)
        for c in cands:
            byitem[c[1]] += 1
        for it, n in sorted(byitem.items(), key=lambda kv: -kv[1])[:20]:
            print("     itemid %-7s %-42s %d" % (it, (di.get(it) or '')[:42], n))

        # ---------- 2. 算修复值 ----------
        fixes = {}          # key -> 修复值
        details = {}        # key -> (it, v, cat, sub, med, npeer)
        for (key, it, v, cat, sub, lab, row) in cands:
            g = (key[0], it)
            arr = pools.get(g, array('d'))
            med = median_of(arr)
            if med is None:
                continue                      # 没有可用读数 ⇒ 真值不可判定 ⇒ 不修
            fixes[key] = fmt(med)
            details[key] = (it, v, cat, sub, med, len(arr))
        print("  可修：%d / %d（不可修 %d 格 = 该患者该项一个可用读数都没有）"
              % (len(fixes), len(cands), len(cands) - len(fixes)), flush=True)

        if args.dry_run:
            for key in list(fixes)[:15]:
                it, v, cat, sub, med, np_ = details[key]
                print("    itemid=%-7s %-38s %-10s -> %-8s (n_peer=%d, %s)"
                      % (it, (di.get(it) or '')[:38], v, fmt(med), np_, cat))
            summary[split] = {'n_candidates': len(cands), 'n_fixable': len(fixes)}
            continue

        # ---------- 3. 定位 raw 行 ----------
        log_path = dst_sp / 'modification_log.csv'
        raw_path = dst_sp / 'raw/icu/chartevents.csv'
        ins_pos, key_patch, val_upd, n_log = scan_log(log_path, CE_FILE)
        print("  modification_log %d 条（chartevents insert/dup %d、key 列 patch %d、"
              "valuenum 注入 %d）" % (n_log, len(ins_pos), len(key_patch), len(val_upd)),
              flush=True)
        n_raw = sum(1 for _ in raw_path.open(encoding='utf-8')) - 1
        raw2log = build_raw2log(ins_pos, n_raw)
        print("  raw 行数 %d" % n_raw, flush=True)

        want = set(fixes)
        found = {}
        with raw_path.open(newline='', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                L = raw2log[i]
                if L < 0:
                    continue
                # 干净 key：被注入破坏的列用日志里的 clean_value，其余用 raw 原值。
                # 注意 **没有注入的原始行也要参与匹配** —— 原生错误格大多没有注入，
                # 早先用 `if patch:` 把它们整行跳过了，导致 990 格改了 reference
                # 却没登记进 log（评分器就看不见，agent 改对了也只得 natural_noise）。
                patch = key_patch.get(L)
                if patch:
                    ck = tuple(patch.get(c, row[c]) for c in CE_KEY)
                else:
                    ck = (row['subject_id'], row['hadm_id'], row['stay_id'],
                          row['charttime'], row['itemid'])
                if ck in want and ck not in found:
                    found[ck] = (i, L, row['valuenum'], row.get('itemid', ''))
        print("  定位到 raw 行：%d / %d" % (len(found), len(want)), flush=True)

        # ---------- 4. 逐格判定：叠脏格 / itemid 叠脏格 ----------
        entries, nat_rows = [], []
        raw_restores = {}
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
            # ⚠️ **两层错误叠加**：注入器把这一格的 **itemid** 也改了
            #    （chart_itemid_alias_split），而它同时又是 MIMIC 原生错误格
            #    ⇒ 两层错误在数据上**不可判定**（形态同构）⇒ 按 icu5000 的原则，
            #    **这一格不进 native 层**（留给注入层评测），保持「每格只有一个错误」。
            try:
                if raw_it != "" and int(raw_it) != it:
                    n_itemid_clash += 1
                    excluded.add(key)
                    continue
            except (TypeError, ValueError):
                pass
            seq += 1
            # 该格是否也被注入器改过 valuenum？
            had_inj = L in set(val_upd)
            dirty = raw_val
            if had_inj:
                # 该格既是 MIMIC 原生错误格，又被注入器叠了一层脏值。
                # ⚠️ 两层错误叠在一起 ⇒ 从数据上无法判定 ⇒ **把 raw 的值恢复成
                #    MIMIC 原生错值**（= reference 的原值，权威来源），让这一格
                #    只剩**一个**错误（原生错误）⇒ 目标唯一、可判定、可评测。
                n_inj += 1
                raw_restores[raw_idx] = (fmt(v), fmt(v))
                dirty = v
                drop_inj_rows.add(L)
            entries.append({
                'source_error_id': "%s-mimic-%04d" % (split, seq),
                'canonical_stay_id': '',
                'error_class': 'class1_intra_table',
                'error_subtype': sub,
                'target_effect_contract': '(see raw_level_error_split)',
                'operation': 'cell_update',
                'raw_file': CE_FILE,
                'raw_row_index': str(L),
                'column': 'valuenum',
                'clean_value': fixes[key],
                'dirty_value': dirty,
                'matching_columns': 'subject_id|hadm_id|stay_id',
            })
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

        # ---------- 5. 改 reference / raw / log ----------
        # ⚠️ 被排除的格（itemid 也被注入器改过 ⇒ 两层错误叠加、不可判定）必须
        #    **同时**从 fixes 里剔掉 —— 否则 reference 被改了、log 却没登记 ⇒
        #    agent 修对了反而会被判错。
        if excluded:
            fixes = {k: v for k, v in fixes.items() if k not in excluded}
        # 按 (key, 原值) 定位改写（避开同 key 的正常行）
        targets = {}
        for key, nv in fixes.items():
            it, v, cat, sub, med, npeer = details[key]
            targets[(key, fmt(v))] = nv
        n_changed = rewrite_reference(ref_path, targets)
        n_raw_restored = rewrite_raw(raw_path, raw_restores) if raw_restores else 0
        n_added, n_drop = rewrite_log(log_path, entries,
                                      drop_rows=drop_inj_rows)
        write_native_log(dst_sp, nat_rows)
        print("  reference 改写 %d 行；raw 恢复 %d 格；log 追加 %d 条；"
              "丢弃与原生格重叠的注入条目 %d 条"
              % (n_changed, n_raw_restored, n_added, n_drop), flush=True)

        # ---------- 6. 三条断言 ----------
        if n_changed != n_added:
            raise SystemExit("FATAL[%s]: 改了 %d 格但登记 %d 条"
                             % (split, n_changed, n_added))
        if n_changed + n_itemid_clash != n_fixable_total:
            raise SystemExit("FATAL[%s]: 改了 %d + 排除 %d != 可修 %d"
                             % (split, n_changed, n_itemid_clash, n_fixable_total))
        if n_raw_restored != n_inj:
            raise SystemExit("FATAL[%s]: 恢复 %d 格但叠脏格是 %d"
                             % (split, n_raw_restored, n_inj))
        print("  [%s] 自检 OK" % split, flush=True)
        summary[split] = {
            'candidates': len(cands), 'fixable': n_fixable_total,
            'changed': n_changed, 'logged': n_added,
            'dropped_overlap_injections': n_drop,
            'raw_restored': n_raw_restored, 'also_injected': n_inj,
            'itemid_clash_excluded': n_itemid_clash,
            'by_cat': dict(by_cat),
        }

    # ---------- 7. 构建报告 ----------
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
