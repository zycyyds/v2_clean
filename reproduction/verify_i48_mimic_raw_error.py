#!/usr/bin/env python3
"""校验数据集的 modification_log ↔ raw ↔ reference 三方自洽。

用法: python3 verify_i48_mimic_raw_error.py <数据集根目录> [<数据集根目录> ...]

对每个 cell_update 条目：
  1. raw 该行该列的实际值 == dirty_value      （评分器的 f_mismatch 检查）
  2. reference 里能定位到该行时，其该列的值 == clean_value
     （定位不到 = 该行不在 cohort 里，reference 有意丢弃，属正常）
"""
import csv, sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(2 ** 31)

REF_OF = {
    'hosp/patients.csv': ('demographics.csv', ['subject_id']),
    'hosp/admissions.csv': ('admissions.csv', ['hadm_id']),
    'hosp/diagnoses_icd.csv': ('diagnoses.csv',
                               ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version']),
    'icu/icustays.csv': ('icu/icustays.csv', ['stay_id']),
    'icu/chartevents.csv': ('icu/chartevents.csv',
                            ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid']),
    'icu/inputevents.csv': ('icu/inputevents.csv',
                             ['subject_id', 'hadm_id', 'stay_id', 'starttime', 'itemid']),
    'icu/outputevents.csv': ('icu/outputevents.csv',
                             ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid']),
}


def check(ds_root, split):
    st = defaultdict(int)
    problems = []
    log_rows = list(csv.DictReader((ds_root / split / 'modification_log.csv').open(encoding='utf-8')))
    ins = defaultdict(list)
    cells = defaultdict(lambda: defaultdict(dict))
    for e in log_rows:
        rf = e['raw_file']
        ri = int(e['raw_row_index'])
        if e['operation'] == 'cell_update':
            cells[rf][ri][e['column']] = (e['dirty_value'], e['clean_value'])
        else:
            ins[rf].append(ri)
    for k in ins:
        ins[k].sort()

    for rf, (rel, keys) in REF_OF.items():
        rawp = ds_root / split / 'raw' / rf
        refp = ds_root / split / 'reference' / rel
        if not (rawp.is_file() and refp.is_file()) or rf not in cells:
            continue
        posl = ins.get(rf, [])
        n_raw = sum(1 for _ in rawp.open(encoding='utf-8')) - 1
        arr = [-1] * n_raw
        used = bytearray(n_raw)
        for L in range(n_raw):
            r = L + bisect_left(posl, L)
            if 0 <= r < n_raw and not used[r]:
                arr[r] = L
                used[r] = 1
        want_log = set(cells[rf])
        refmap = {}
        with refp.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                k = tuple(row[c] for c in keys)
                if k not in refmap:
                    refmap[k] = row
        with rawp.open(newline='', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                L = arr[i]
                if L not in want_log:
                    continue
                patch = cells[rf][L]
                for col, (dirty, clean) in patch.items():
                    st['checked'] += 1
                    if row.get(col, '') != dirty:
                        st['dirty_mismatch'] += 1
                        if len(problems) < 8:
                            problems.append("dirty %s log_row=%d col=%s dirty=%r raw=%r"
                                            % (rf, L, col, dirty, row.get(col, '')))
                    k = tuple(patch.get(c, (None, row[c]))[1] for c in keys)
                    rrow = refmap.get(k)
                    if rrow is None:
                        st['ref_row_absent'] += 1     # 非 cohort 行，reference 有意丢弃
                        continue
                    st['ref_row_found'] += 1
                    if rrow.get(col, '') != clean:
                        st['clean_mismatch'] += 1
                        if len(problems) < 16:
                            problems.append("clean %s log_row=%d col=%s clean=%r ref=%r"
                                            % (rf, L, col, clean, rrow.get(col, '')))
    st['native_entries'] = sum(1 for e in log_rows
                               if e['error_subtype'] == 'mimic_native_impossible_value')
    st['log_entries'] = len(log_rows)
    return st, problems


def main():
    roots = sys.argv[1:] or ['/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/'
                              'I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M']
    for r in roots:
        root = Path(r)
        print("\n" + "=" * 84)
        print("DATASET:", root.name)
        print("=" * 84)
        for split in ('train', 'validation'):
            if not (root / split).is_dir():
                continue
            st, problems = check(root, split)
            print("  [%s] cell_update %d 条 | 核对 %d 格 | dirty不符 %d | ref找到 %d "
                  "(clean不符 %d) | ref未收录 %d | 原生错误条目 %d"
                  % (split, st['log_entries'], st['checked'], st['dirty_mismatch'],
                     st['ref_row_found'], st['clean_mismatch'], st['ref_row_absent'],
                     st['native_entries']))
            for p in problems:
                print("       ", p)


if __name__ == '__main__':
    main()
