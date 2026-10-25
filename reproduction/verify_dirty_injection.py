"""快速校验 4 个数据集的 mod_log 错误实际注入到 raw 的情况。
跳过 chartevents (太大)，按 file 随机抽样检查 cell_update 的 dirty 值。
"""
import csv
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path('/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集')
random.seed(42)


def stream_dirty_rows(path: Path, wanted_indices: set[int]):
    """只把 wanted_indices 里的行读出来"""
    found = {}
    with path.open(newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i in wanted_indices:
                found[i] = row
    return found


for ds in sorted(ROOT.iterdir()):
    if ds.name.startswith('.') or not ds.is_dir():
        continue
    print(f'\n=== {ds.name} ===')
    for sp in ('train', 'validation', 'test'):
        log = ds / sp / 'modification_log.csv'
        if not log.exists():
            print(f'  {sp}: no log'); continue
        rows = list(csv.DictReader(log.open()))
        cell_ops = [r for r in rows if r['operation'] == 'cell_update']
        # 按 raw_file 分组，限制每文件最多抽样 20 个 cell_update（避免大文件慢）
        by_file = defaultdict(list)
        for r in cell_ops:
            by_file[r['raw_file']].append(r)
        n_match = 0; n_total = 0
        for fpath, errs in by_file.items():
            # 跳过 chartevents/labs（太大）
            if 'chartevents' in fpath or 'labevents' in fpath:
                continue
            full = ds / sp / 'raw' / fpath
            if not full.exists():
                continue
            # 抽样最多 30 个
            sample = random.sample(errs, min(30, len(errs)))
            wanted = {int(e['raw_row_index']) for e in sample}
            raw_rows = stream_dirty_rows(full, wanted)
            for e in sample:
                i = int(e['raw_row_index'])
                row = raw_rows.get(i)
                if row is None:
                    n_total += 1
                    continue
                if row.get(e['column'], '') == e['dirty_value']:
                    n_match += 1
                n_total += 1
        pct = 100*n_match/n_total if n_total else 0
        n_dup = sum(1 for r in rows if r['operation'] == 'duplicate')
        n_ins = sum(1 for r in rows if r['operation'] == 'insert')
        print(f'  {sp:11s}: total={len(rows):>4}  sampled_match={n_match}/{n_total} ({pct:.0f}%)  duplicate={n_dup}  insert={n_ins}')
