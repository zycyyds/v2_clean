#!/usr/bin/env python3
"""train/validation 原生错误的**精确修复值**计算 + 全表通用离群兜底扫描。

修复规则 R1（局部邻域插补）：
    对硬不可能格 v（subject s, itemid i, 时刻 t）：
      P1 = 同一 (s,i) 中 |t' - t| 最小且落在合理区间的读数（±24h 内）
      P2 = 同一 (s,i) 全部合理读数的中位数（回退）
    采用 P1；若 P1 不存在则 P2。

兜底扫描：对每个输出表的每个数值列，按 (subject, 列) 求中位数，
          标记 |v - med| / med > 50 的离群（相对该患者自身），
          用来发现 hard-range 规则没覆盖到的原生错误。
"""
import csv, json, sys, datetime
from collections import defaultdict, Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)
DS = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/"
          "I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M")
T = Path("/Users/mac/.claude/jobs/07ccbc7a/tmp")
OUT = sys.argv[1]
di = {int(r['itemid']): r['label'] for r in
      csv.DictReader((T / 'd_items.csv').open(encoding='utf-8', errors='replace'))}

HARD_BY_LABEL = {
    'heart rate': (20, 300), 'blood pressure systolic': (40, 400),
    'arterial bp systolic': (40, 400), 'blood pressure diastolic': (20, 250),
    'blood pressure mean': (20, 250), 'respiratory rate': (0, 100),
    'temperature celsius': (25, 45), 'ptt': (8, 200),
    'central venous pressure': (-20, 80),
    'pulmonary artery pressure systolic': (0, 150),
    'pulmonary artery pressure mean': (0, 100),
}


def hard_range(itemid):
    lab = di.get(itemid)
    if not lab:
        return None
    l = lab.lower()
    for b in ('loss', 'unintentional', 'note', 'comment', 'finding', 'change',
              'alarm'):
        if b in l:
            return None
    for k, v in HARD_BY_LABEL.items():
        if k in l:
            return v
    return None


def ts(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return None


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(x):
    if x is None:
        return None
    return ('%.10g' % x)


report = {}
for split in ('train', 'validation'):
    F = DS / split / 'reference'
    R = DS / split / 'raw'
    print("\n" + "=" * 90)
    print("### ", split)
    print("=" * 90)

    # ---------- chartevents 原生错误 + 修复值 ----------
    ce_ref = list(csv.DictReader((F / 'icu/chartevents.csv').open(encoding='utf-8')))
    ce_raw = {}
    for r in csv.DictReader((R / 'icu/chartevents.csv').open(encoding='utf-8')):
        ce_raw[(r['subject_id'], r['hadm_id'], r['stay_id'], r['charttime'], r['itemid'])] = r
    seq = defaultdict(list)
    for r in ce_ref:
        seq[(r['subject_id'], r['itemid'])].append(r)
    for k in seq:
        seq[k].sort(key=lambda r: r['charttime'])

    rows = []
    # 每个 (subject,itemid) 序列里 0 的占比 —— 用来判断 0 是「未测到约定」还是错误
    zerostat = {}
    for (s, i), lst in seq.items():
        vs = [num(x['valuenum']) for x in lst]
        vs = [v for v in vs if v is not None]
        if not vs:
            continue
        zerostat[(s, i)] = (sum(1 for v in vs if v == 0) / len(vs), len(vs))
    for r in ce_ref:
        it = int(r['itemid'])
        rg = hard_range(it)
        if not rg:
            continue
        v = num(r['valuenum'])
        if v is None:
            continue
        if v == 0:
            # 0 一律不算错：MIMIC 里 0 = 「未测到 / 呼吸机暂停」的记录约定，
            # 无法判定真值（224689 呼吸频率 42.6% 是 0、220045 心率只有 0.03% 是 0，
            # 但两者的 0 语义相同且都不可判）→ 排除，不进修复集
            continue
        if not (v < rg[0] or v > rg[1]):
            continue
        t = ts(r['charttime'])
        peers = []
        for p in seq[(r['subject_id'], str(it))]:
            if p['charttime'] == r['charttime']:
                continue
            pv = num(p['valuenum'])
            if pv is None or not (rg[0] <= pv <= rg[1]):
                continue
            peers.append((p['charttime'], pv))
        near = None
        for ct, pv in peers:
            tt = ts(ct)
            if tt is None or t is None:
                continue
            d = abs(tt - t)
            if d <= 24 * 3600 and (near is None or d < near[0]):
                near = (d, pv, ct)
        vs = sorted(pv for _, pv in peers)
        med = vs[len(vs) // 2] if vs else None
        pick = near[1] if near else med
        k = (r['subject_id'], r['hadm_id'], r['stay_id'], r['charttime'], r['itemid'])
        rawr = ce_raw.get(k)
        rawvn = rawr['valuenum'] if rawr else None
        rvt = rawr['value'] if rawr else None
        rows.append({
            'subject_id': r['subject_id'], 'hadm_id': r['hadm_id'],
            'stay_id': r['stay_id'], 'charttime': r['charttime'], 'itemid': it,
            'label': di.get(it), 'ref_valuenum': r['valuenum'], 'valueuom': r['valueuom'],
            'hard_range': list(rg),
            'raw_valuenum': rawvn, 'raw_value_text': rvt,
            'ref_equals_raw_valuenum': (rawvn is not None and num(rawvn) is not None
                                         and abs(num(rawvn) - v) < 1e-9),
            'repair_P1_nearest_24h': fmt(near[1]) if near else None,
            'repair_P1_gap_h': round(near[0] / 3600, 2) if near else None,
            'repair_P1_at': near[2] if near else None,
            'repair_P2_peer_median': fmt(med),
            'repair_used': fmt(pick),
            'repair_source': 'P1_nearest_24h' if near else ('P2_peer_median' if med else 'NONE'),
            'n_ok_peers': len(peers),
        })
    print("\n[icu/chartevents] 原生错误 %d 格 —— 逐格修复值" % len(rows))
    hdr = ("subject_id", "charttime", "itemid", "label", "ref_valuenum", "raw_valuenum",
           "raw_value_text", "P1", "P2", "used", "src")
    print("   %-10s %-20s %-7s %-32s %-9s %-9s %-8s %-7s %-7s %-7s %s" % hdr)
    for x in rows:
        print("   %-10s %-20s %-7d %-32s %-9s %-9s %-8s %-7s %-7s %-7s %s" % (
            x['subject_id'], x['charttime'], x['itemid'], (x['label'] or '')[:32],
            x['ref_valuenum'], str(x['raw_valuenum'])[:9], str(x['raw_value_text'])[:8],
            str(x['repair_P1_nearest_24h'])[:7], str(x['repair_P2_peer_median'])[:7],
            str(x['repair_used'])[:7], x['repair_source']))
    report.setdefault(split, {})['chartevents'] = rows

    # ---------- 全表通用离群兜底 ----------
    print("\n[兜底扫描] 按 (subject, itemid) 分组的「相对该患者自身该测量项」离群")
    sweep = []
    for rel in ['icu/chartevents.csv', 'labs.csv', 'icu/inputevents.csv',
                'icu/outputevents.csv', 'demographics.csv', 'icu/icustays.csv',
                'medications.csv']:
        p = F / rel
        if not p.is_file():
            continue
        rowsf = list(csv.DictReader(p.open(encoding='utf-8')))
        if not rowsf:
            continue
        # 分组键：同 itemid 的量纲才可比
        if 'itemid' in rowsf[0]:
            gkey = lambda r: (r['subject_id'], r['itemid'])
            gname = 'itemid'
        elif 'drug' in rowsf[0]:
            gkey = lambda r: (r['subject_id'], r['drug'])
            gname = 'drug'
        else:
            gkey = lambda r: (r['subject_id'],)
            gname = 'subject'
        for c in rowsf[0]:
            if c in ('subject_id', 'hadm_id', 'stay_id', 'charttime', 'starttime',
                     'endtime', 'stoptime', 'intime', 'outtime', 'itemid', 'drug',
                     'value', 'valueuom', 'sex', 'gender', 'race', 'insurance',
                     'language', 'marital_status', 'route', 'admission_type',
                     'cohort_type', 'cohort_label', 'icd_code', 'icd_version',
                     'icd_root', 'seq_num', 'dose_unit_rx', 'amountuom', 'totalamountuom'):
                continue
            by = defaultdict(list)
            for r in rowsf:
                v = num(r.get(c))
                if v is not None:
                    by[gkey(r)].append((v, r))
            for gk, items in by.items():
                if len(items) < 8:
                    continue
                vs = sorted(v for v, _ in items)
                med = vs[len(vs) // 2]
                if med == 0:
                    continue
                for v, r in items:
                    if v > 0 and abs(v - med) / abs(med) > 50:
                        sweep.append({'file': rel, 'column': c, 'group_by': gname,
                                      'group': str(gk), 'value': v,
                                      'patient_median': med, 'n': len(items),
                                      'itemid': r.get('itemid'),
                                      'label': di.get(int(r['itemid'])) if r.get('itemid', '').isdigit() else None})
    report.setdefault(split, {})['sweep'] = sweep
    if sweep:
        agg = Counter((x['file'], x['column']) for x in sweep)
        for (f, c), n in agg.most_common(20):
            print("   %-24s %-18s %d" % (f, c, n))
        for x in sweep[:12]:
            print("     ", x)
    else:
        print("   （无）")

json.dump(report, open(OUT, 'w'), ensure_ascii=False, indent=1)
print("\nwritten", OUT)
