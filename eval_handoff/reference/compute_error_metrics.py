"""v28: error evaluation with index-shift correction.

THE BUG THIS FIXES
------------------
modification_log.csv records raw_row_index in the PRE-INSERTION layout. The
`duplicate` and `insert` (label_leakage) operations then add rows to raw, so
every later row shifts. Naively using log index == raw index mismatches ~95%
of chartevents entries.

Verified correction on train (100% match, all 7 tables):
    raw_index = log_index + (number of insert/duplicate ops at index < log_index)
The `<` is strict: variant A (`<=`) gave 80.2%, variant B (`<`) gave 94.9%,
and comparing only the LAST dirty value of multi-injected cells gave 100%.
Cells with multiple injections keep only the final value in raw, so a cell's
dirty_value is the last log entry for that (ri, col).

Everything else follows v21: full accounting (no injection silently dropped),
partial key fallback only when the dirty column is itself part of the key,
duplicate benchmark = that key's row count in ref, natural noise excluded.
"""
from array import array
import csv
import json
import sqlite3
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
CACHE_DIR = Path("/Users/mac/.claude/jobs/910902cd/tmp/v31_idx_cache")
DS_KEY = "k29_1800_work"
DS = ROOT / "datasets/注入错误数据集/K29_v2_rawerror_test4.7pct_14.1M"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
OUT_JSON = ROOT / ".claude/worktrees/eval-movies-cleaning/tmp/k29_1800_work_v32_error_metrics.json"
LOG = DS / SPLIT / "modification_log.csv"
RAW = DS / SPLIT / "raw"

RAW_TO_OUTPUT = {
    "hosp/patients.csv":      ("demographics.csv",     ["subject_id"]),
    "hosp/admissions.csv":    ("admissions.csv",       ["hadm_id"]),
    "hosp/diagnoses_icd.csv": ("diagnoses.csv",        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]),
    "icu/icustays.csv":       ("icu/icustays.csv",     ["stay_id"]),
    "icu/chartevents.csv":    ("icu/chartevents.csv",  ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
    "icu/inputevents.csv":    ("icu/inputevents.csv",  ["subject_id", "hadm_id", "stay_id", "starttime", "itemid"]),
    "icu/outputevents.csv":   ("icu/outputevents.csv", ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
}
INMEM_MAX = 10_000_000

# 干净 key 还原：key 列被注入破坏的行，raw 的完整 key 在 agent 输出里不存在
# （agent 修好了 key）。不还原的话，这批行的非键格会被整片记成 b_missing。
# 默认开启；--no-recover-clean-key 可退回旧口径，与历史 A/B/C/D 对齐。
RECOVER_CLEAN_KEY = True


def cache_path(raw_file, kind):
    return CACHE_DIR / ("%s__%s__%s.db" % (DS_KEY, raw_file.replace("/", "_"), kind))


class DB:
    def __init__(self, path, keys, all_cols):
        self.keys = keys
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA cache_size=-200000")
        self.db_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(data)")]
        self.cidx = {c: i for i, c in enumerate(self.db_cols)}
        self.non_key_cols = [c for c in all_cols if c in self.cidx and c not in keys]
        self.n_rows = self.conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
        self.kidx = [self.cidx[k] for k in keys]
        wh = " AND ".join('"%s"=?' % k for k in keys)
        self._sql_full = "SELECT %s FROM data WHERE %s" % (
            ",".join('"%s"' % c for c in self.db_cols), wh)
        self._sql_count = "SELECT COUNT(*) FROM data WHERE %s" % wh
        self._sql_nonkey = ("SELECT %s FROM data WHERE %s" % (
            ",".join('"%s"' % c for c in self.non_key_cols), wh)
            if self.non_key_cols else None)
        self._sql_partial = {}
        self._keep_pos = {}
        for o in keys:
            keep = [k for k in keys if k != o]
            self._keep_pos[o] = [keys.index(k) for k in keep]
            self._sql_partial[o] = (
                "SELECT %s FROM data WHERE %s" % (
                    ",".join('"%s"' % c for c in self.db_cols),
                    " AND ".join('"%s"=?' % k for k in keep)), self._keep_pos[o])
        self._mem = None
        self._cnt = None
        self._sub = {}
        if self.n_rows <= INMEM_MAX:
            cur = self.conn.execute(
                "SELECT %s FROM data" % ",".join('"%s"' % c for c in self.db_cols))
            self._mem = {}
            for row in cur:
                self._mem[tuple(row[i] for i in self.kidx)] = row
            self._cnt = {k: 1 for k in self._mem}

    def _sub_index(self, omit_key):
        if omit_key in self._sub:
            return self._sub[omit_key]
        pos = self._keep_pos[omit_key]
        m = defaultdict(list)
        for k, row in self._mem.items():
            m[tuple(k[i] for i in pos)].append(row)
        self._sub[omit_key] = m
        return m

    def get_full(self, kt):
        if self._mem is not None:
            return self._mem.get(kt)
        return self.conn.execute(self._sql_full, kt).fetchone()

    def get_nonkey(self, kt):
        if self._mem is not None:
            r = self._mem.get(kt)
            return None if r is None else tuple(r[self.cidx[c]] for c in self.non_key_cols)
        return self.conn.execute(self._sql_nonkey, kt).fetchone()

    def count_key(self, kt):
        if self._cnt is not None:
            return self._cnt.get(kt, 0)
        return self.conn.execute(self._sql_count, kt).fetchone()[0]

    def partial(self, omit_key, kt):
        pos = self._keep_pos[omit_key]
        if self._mem is not None:
            return self._sub_index(omit_key).get(tuple(kt[i] for i in pos), [])
        sql, _ = self._sql_partial[omit_key]
        return self.conn.execute(sql, tuple(kt[i] for i in pos)).fetchall()

    def close(self):
        self.conn.close()


def main():
    t_all = time.time()
    print("Loading mod_log ...", flush=True)
    with LOG.open(newline='', encoding='utf-8') as f:
        mod_rows = list(csv.DictReader(f))

    ins_pos = defaultdict(list)          # raw_file -> ALL insert+dup positions (offset)
    ins_kind = defaultdict(list)         # raw_file -> 'dup'/'leak' parallel to ins_pos
    cell_by_ri = defaultdict(lambda: defaultdict(dict))  # file -> ri -> col -> err
    n_cell = n_dup = n_ins = 0
    for e in mod_rows:
        rf = e['raw_file']
        ri = int(e['raw_row_index'])
        op = e['operation']
        if op == 'cell_update':
            # last log entry for a cell wins: raw keeps only the final value
            cell_by_ri[rf][ri][e['column']] = e
            n_cell += 1
        elif op in ('insert', 'duplicate'):
            ins_pos[rf].append(ri)
            ins_kind[rf].append('dup' if op == 'duplicate' else 'leak')
            if op == 'duplicate':
                n_dup += 1
            else:
                n_ins += 1
    del mod_rows
    # ins_pos / ins_kind 是平行的两列（都按日志出现顺序 append）。
    # 排序必须成对做 —— 之前先把 ins_pos 单独 sort 掉，后面再按「已经有序的 posl」
    # 去排 kind，等于恒等排序，kinds 一直保持「日志顺序 = 按 subtype 分组」的形态：
    # 所有 duplicate 在前、所有 label_leak 在后。而 raw 里注入行是按
    # 「同一 raw_row_index 上 dup 先、leak 后」交错的，于是前 N 个注入行全被标成
    # dup，leak 标签整体后移 —— 实测标签正确率只有 76.6%（正好等于 dup 的占比），
    # 后果是「被清掉的泄露行」被记成 leak_kept（假漏检）、「真重复行」被记成
    # dup_removed（假命中），直接扭曲 Det/Rep 的 P 和 R。
    ins_pairs = {}
    for rf, pl in ins_pos.items():
        kl = ins_kind.get(rf, [])
        pairs = sorted(zip(pl, kl), key=lambda p: p[0])
        ins_pairs[rf] = ([p[0] for p in pairs], [p[1] for p in pairs])
    ins_pos = {rf: v[0] for rf, v in ins_pairs.items()}
    ins_kind = {rf: v[1] for rf, v in ins_pairs.items()}
    # pre-insertion index -> raw index (same correction as cell lookups)
    def _shift(rf, lst):
        p = ins_pos.get(rf, [])
        return {i + bisect_left(p, i) for i in lst}
    # Scheme A (validated on train at 100%): an original row with log index L
    # sits at raw index L + bisect_left(pos, L). Invert that into an explicit
    # raw -> log table so the streaming loop never has to invert it (the
    # inverse is NOT raw - bisect_left(pos, raw) once insertions share an index).
    # Raw rows not in the image of the map are the injected rows themselves.
    n_raw_of = {}
    for rf in RAW_TO_OUTPUT:
        rp = DS / SPLIT / "raw" / rf
        n_raw_of[rf] = sum(1 for _ in rp.open(
            newline='', encoding='utf-8')) - 1
    raw2log = {}
    ins_kind_by_raw = {}
    for rf in RAW_TO_OUTPUT:
        posl = ins_pos.get(rf, [])
        n = n_raw_of[rf]
        arr = array('i', [-1]) * n if n > 0 else array('i')
        kindlist = ins_kind.get(rf, [])
        # ins_pos / ins_kind 已经成对排好序（见上面的 ins_pairs）：
        # 同一 raw_row_index 上日志顺序 = dup 先、leak 后，
        # 跟 apply_modification_log.py 写 raw 的顺序一致
        n_ins_here = len(posl)
        kinds_sorted = kindlist
        used = bytearray(n)
        for L in range(n):
            r = L + bisect_left(posl, L)
            if 0 <= r < n and not used[r]:
                arr[r] = L
                used[r] = 1
        raw2log[rf] = arr
        # free slots are injected rows; order them by raw index
        ins_kind_by_raw[rf] = (used, kinds_sorted, posl, n_ins_here)
    print("  cell_update=%d duplicate=%d insert=%d total=%d"
          % (n_cell, n_dup, n_ins, n_cell + n_dup + n_ins), flush=True)

    T = defaultdict(int)
    per_file = {}
    raw_mismatch = 0

    for raw_file, (out_rel, keys) in RAW_TO_OUTPUT.items():
        raw_path = RAW / raw_file
        cp_codex = cache_path(raw_file, 'codex')
        cp_ref = cache_path(raw_file, 'ref')
        if not (raw_path.exists() and cp_codex.exists() and cp_ref.exists()):
            print("SKIP %s" % raw_file, flush=True)
            continue
        print("\n--- %s ---" % raw_file, flush=True)
        t0 = time.time()
        with raw_path.open(newline='', encoding='utf-8') as f:
            fields = csv.DictReader(f).fieldnames or []
        raw_non_key = [c for c in fields if c not in keys]

        codex = DB(cp_codex, keys, raw_non_key)
        ref = DB(cp_ref, keys, raw_non_key)
        print("  codex=%d(mem=%s) ref=%d(mem=%s)"
              % (codex.n_rows, codex._mem is not None,
                 ref.n_rows, ref._mem is not None), flush=True)

        pos = ins_pos.get(raw_file, [])
        cells = cell_by_ri.get(raw_file, {})
        jp = 0
        f_ = defaultdict(int)
        dup_keys = set()
        leak_keys = set()
        get_full = codex.get_full
        ref_nonkey = ref.get_nonkey
        nk_idx = [codex.cidx[c] for c in codex.non_key_cols]
        ref_nk_idx = [ref.cidx[c] for c in ref.non_key_cols]
        ki = {k: i for i, k in enumerate(keys)}
        log_arr = raw2log[raw_file]
        ins_kinds = ins_kind_by_raw[raw_file][1]
        n_ins_rows = len(ins_kinds)

        n_raw = 0
        f_mismatch = 0
        with raw_path.open(newline='', encoding='utf-8') as fh:
            for ri, raw_row in enumerate(csv.DictReader(fh)):
                n_raw += 1
                if n_raw % 3000000 == 0:
                    print("    raw row %d  %.0fs" % (n_raw, time.time() - t0),
                          flush=True)
                kt = tuple(raw_row[c] for c in keys)
                log_ri = log_arr[ri] if ri < len(log_arr) else -1
                if log_ri < 0:
                    # this row IS an injected duplicate / label-leak row:
                    # account for it, never evaluate it as an original row
                    if jp < n_ins_rows:
                        if ins_kinds[jp] == 'dup':
                            dup_keys.add(kt)
                        else:
                            leak_keys.add(kt)
                        jp += 1
                    continue
                cinj = cells.get(log_ri)
                if cinj is not None:
                    for col, e in cinj.items():
                        raw_v = raw_row.get(col, '')
                        if raw_v != e['dirty_value']:
                            f_mismatch += 1

                crow = get_full(kt)
                # ---- 干净 key 还原 ----
                # key 列被注入破坏时（如 stay_hadm_mismatch 改 hadm_id、
                # medication_itemid_alias_split 改 itemid），raw 这一行的完整 key
                # 在 agent 输出里根本不存在 —— agent 修好了 key，它的行用的是原值。
                # 严格查找失败 -> 这一行的非键格全被记成 b_missing（当成漏检），
                # 可 agent 其实修对了。日志里同时存着 clean_value，可以还原
                # 「干净 key」再查一次，把这批格捞回来。
                if crow is None and cinj and RECOVER_CLEAN_KEY:
                    kt_clean = list(kt)
                    dirty_key = False
                    for c, e in cinj.items():
                        i = ki.get(c)
                        if i is None:
                            continue
                        cv = e.get('clean_value')
                        if cv is not None and kt[i] != cv:
                            kt_clean[i] = cv
                            dirty_key = True
                    if dirty_key:
                        crow = get_full(tuple(kt_clean))
                        if crow is not None:
                            f_['crow_recovered'] += 1

                if not cinj:
                    if crow is not None:
                        rrow = ref_nonkey(kt)
                        if rrow is not None:
                            cn = [crow[i] for i in nk_idx]
                            rn = rrow
                            for j, c in enumerate(codex.non_key_cols):
                                rv = raw_row.get(c, '')
                                if cn[j] != rv:
                                    if rn[j] == rv:
                                        f_['fp_det'] += 1
                                    else:
                                        f_['natural_noise'] += 1
                    continue

                for col, e in cinj.items():
                    clean_v = e['clean_value']
                    dirty_v = e['dirty_value']
                    if col in codex.cidx:
                        if col in keys:
                            f_['kcol_inj'] += 1
                            if crow is not None:
                                f_['kcol_strict_hit'] += 1
                                cv = crow[codex.cidx[col]]
                            else:
                                cands = codex.partial(col, kt)
                                if not cands:
                                    f_['b_missing'] += 1
                                    continue
                                f_['kcol_partial_hit'] += 1
                                ci = codex.cidx[col]
                                cv = None
                                for r in cands:
                                    if r[ci] == clean_v:
                                        cv = r[ci]
                                        break
                                if cv is None:
                                    cv = cands[0][ci]
                        else:
                            if crow is not None:
                                f_['ncol_strict_hit'] += 1
                                cv = crow[codex.cidx[col]]
                            else:
                                f_['ncol_strict_miss'] += 1
                                f_['b_missing'] += 1
                                continue
                    else:
                        f_['b_missing'] += 1
                        continue

                    if cv == clean_v:
                        f_['b_clean'] += 1
                    elif cv == dirty_v:
                        f_['b_dirty'] += 1
                    else:
                        f_['b_wrong'] += 1

        # ---- duplicates, benchmarked against ref's count for that key ----
        ref_count = ref.count_key
        for kt in dup_keys:
            n_codex = codex.count_key(kt)
            n_ref = ref_count(kt)
            if n_codex == n_ref:
                f_['dup_removed'] += 1
            elif n_codex > n_ref:
                f_['dup_kept'] += 1
            elif n_codex == 0:
                f_['dup_row_gone'] += 1
            else:
                f_['dup_row_short'] += 1

        # ---- label-leakage inserted rows: removed if codex row absent ----
        for kt in leak_keys:
            if codex.count_key(kt) == 0:
                f_['leak_removed'] += 1
            else:
                f_['leak_kept'] += 1

        for k in ('b_clean', 'b_dirty', 'b_wrong', 'b_missing',
                  'dup_removed', 'dup_kept', 'dup_row_gone', 'dup_row_short',
                  'leak_removed', 'leak_kept',
                  'fp_det', 'natural_noise', 'kcol_inj', 'kcol_strict_hit',
                  'kcol_partial_hit', 'ncol_strict_hit', 'ncol_strict_miss',
                  'crow_recovered'):
            T[k] += f_[k]
        per_file[out_rel] = dict(f_)
        per_file[out_rel]['raw_rows'] = n_raw
        per_file[out_rel]['raw_dirty_mismatch'] = f_mismatch
        raw_mismatch += f_mismatch
        print("  cell: clean=%d dirty=%d wrong=%d missing=%d  raw_mismatch=%d"
              % (f_['b_clean'], f_['b_dirty'], f_['b_wrong'], f_['b_missing'],
                 f_mismatch), flush=True)
        print("  dup: removed=%d kept=%d gone=%d short=%d  %.0fs"
              % (f_['dup_removed'], f_['dup_kept'], f_['dup_row_gone'],
                 f_['dup_row_short'], time.time() - t0), flush=True)
        codex.close()
        ref.close()

    n_cell_eval = T['b_clean'] + T['b_dirty'] + T['b_wrong'] + T['b_missing']
    n_dup_eval = (T['dup_removed'] + T['dup_kept'] + T['dup_row_gone']
                  + T['dup_row_short'] + T['leak_removed'] + T['leak_kept'])

    tp_det = (T['b_clean'] + T['b_wrong'] + T['dup_removed'] + T['leak_removed'])
    fn_det = (T['b_dirty'] + T['b_missing'] + T['dup_kept'] + T['dup_row_gone']
              + T['dup_row_short'] + T['leak_kept'])
    tp_rep = T['b_clean'] + T['dup_removed'] + T['leak_removed']
    fp_rep = (T['b_wrong'] + T['dup_kept'] + T['dup_row_gone']
              + T['dup_row_short'] + T['leak_kept'])

    P_det = tp_det / (tp_det + T['fp_det']) if (tp_det + T['fp_det']) else 0.0
    R_det = tp_det / (tp_det + fn_det) if (tp_det + fn_det) else 0.0
    F1_det = 2 * P_det * R_det / (P_det + R_det) if (P_det + R_det) else 0.0
    cells_mod = tp_det + T['fp_det']
    P_rep = tp_rep / cells_mod if cells_mod else 0.0
    R_rep = tp_rep / (tp_rep + fp_rep + fn_det) if (tp_rep + fp_rep + fn_det) else 0.0
    F1_rep = 2 * P_rep * R_rep / (P_rep + R_rep) if (P_rep + R_rep) else 0.0

    print("\n" + "=" * 66)
    print("K29_v2_rawerror_test4.7pct_14.1M  --  v28 (index-shift corrected)")
    print("=" * 66)
    print("injected   : cell_update=%d duplicate=%d insert=%d"
          % (n_cell, n_dup, n_ins))
    print("evaluated  : cell=%d  dup=%d" % (n_cell_eval, n_dup_eval))
    print("  clean=%d  dirty=%d  wrong=%d  row_missing=%d"
          % (T['b_clean'], T['b_dirty'], T['b_wrong'], T['b_missing']))
    print("  dup_removed=%d  dup_kept=%d  dup_row_gone=%d  dup_row_short=%d"
          % (T['dup_removed'], T['dup_kept'], T['dup_row_gone'], T['dup_row_short']))
    print("  leak_removed=%d  leak_kept=%d" % (T['leak_removed'], T['leak_kept']))
    print("  FP_det=%d  natural_noise(excluded)=%d  raw_dirty_mismatch=%d"
          % (T['fp_det'], T['natural_noise'], raw_mismatch))
    print("DETECTION  TP=%d FN=%d FP=%d" % (tp_det, fn_det, T['fp_det']))
    print("           P=%.6f R=%.6f F1=%.6f" % (P_det, R_det, F1_det))
    print("REPAIR     TP=%d FN=%d FP=%d" % (tp_rep, fn_det, fp_rep))
    print("           P=%.6f R=%.6f F1=%.6f" % (P_rep, R_rep, F1_rep))
    print("total %.0fs" % (time.time() - t_all))

    out = {
        'dataset': DS.name,
        'note': 'v28: index-shift corrected (raw_idx = log_idx + |inserts<log_idx|); '
                'multi-injected cells use last dirty_value; full accounting; '
                'partial key fallback only for key-column injections; natural noise excluded',
        'recover_clean_key': RECOVER_CLEAN_KEY,
        'injected': {'cell_update': n_cell, 'duplicate': n_dup, 'insert': n_ins,
                     'total': n_cell + n_dup + n_ins},
        'evaluated': {'cell': n_cell_eval, 'dup': n_dup_eval},
        'buckets': dict(T),
        'raw_dirty_mismatch': raw_mismatch,
        'detection': {'TP': tp_det, 'FN': fn_det, 'FP': T['fp_det'],
                      'P': P_det, 'R': R_det, 'F1': F1_det},
        'repair': {'TP': tp_rep, 'FN': fn_det, 'FP': fp_rep,
                   'P': P_rep, 'R': R_rep, 'F1': F1_rep},
        'per_file': per_file,
    }
    p = OUT_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("saved: %s" % p)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="注入数据集目录（含 test/）")
    ap.add_argument("--ds-key", required=True, help="cache 命名空间")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, help="输出 json")
    ap.add_argument("--no-recover-clean-key", action="store_true",
                    help="退回旧口径（不还原干净 key），与历史 A/B/C/D 对齐")
    a = ap.parse_args()
    RECOVER_CLEAN_KEY = not a.no_recover_clean_key
    DS = Path(a.dataset_dir)
    DS_KEY = a.ds_key
    SPLIT = a.split
    OUT_JSON = Path(a.out)
    LOG = DS / SPLIT / "modification_log.csv"
    RAW = DS / SPLIT / "raw"
    main()