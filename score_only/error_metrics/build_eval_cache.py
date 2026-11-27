"""Build sqlite caches for the agent_work/k29_1800 run (DS_KEY k29_1800_work).

The gold-side indexes already exist under k29_test4p7 (built from the same
unchanged test/reference), so they are reused rather than rebuilt -- that skips
re-indexing 9.18M gold chartevents rows.
"""
import argparse
import csv
import shutil
import sqlite3
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path("/Users/mac/PycharmProjects/v2_clean")
CACHE_DIR = Path("/Users/mac/.claude/jobs/910902cd/tmp/v31_idx_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 运行参数（由 main() 覆盖）
DS_KEY = "k29_1800_work"
REF_KEY = ""                     # 不用了：ref 索引直接从数据集的 test/reference 建
OUT = ROOT / "agent_work/k29_1800/output/test"
REF_DIR = ROOT / "datasets/注入错误数据集/K29_v2_rawerror_test4.7pct_14.1M/test/reference"

RAW_TO_OUTPUT = {
    "hosp/patients.csv":      ("demographics.csv",     ["subject_id"]),
    "hosp/admissions.csv":    ("admissions.csv",       ["hadm_id"]),
    "hosp/diagnoses_icd.csv": ("diagnoses.csv",        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]),
    "icu/icustays.csv":       ("icu/icustays.csv",     ["stay_id"]),
    "icu/chartevents.csv":    ("icu/chartevents.csv",  ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
    "icu/inputevents.csv":    ("icu/inputevents.csv",  ["subject_id", "hadm_id", "stay_id", "starttime", "itemid"]),
    "icu/outputevents.csv":   ("icu/outputevents.csv", ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]),
}


def cache_path(key, raw_file, kind):
    return CACHE_DIR / ("%s__%s__%s.db" % (key, raw_file.replace('/', '_'), kind))


def build_one(raw_file, kind, src_path):
    dst = cache_path(DS_KEY, raw_file, kind)
    for suffix in ("", "-journal", "-wal", "-shm"):
        p = Path(str(dst) + suffix)
        if p.exists():
            p.unlink()

    with src_path.open(newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        keys = RAW_TO_OUTPUT[raw_file][1]
        use = [c for c in cols if c in keys]

        if not cols or len(use) < len(keys):
            # 【降级补丁 · worktree 本地】结构坏掉的表（空表头 / 缺 key 列 / 无可用 key）
            # ⇒ 记成「agent 没有交出这张表」：建一个 **0 行**的表（列 = 文件列 ∪ key 列）。
            # 不打这个补丁会出三种错：
            #   ① 空表头 ⇒ CREATE TABLE data () ⇒ SQL 语法错 near ")"
            #   ② 缺 key 列 ⇒ DB.__init__ 里 self.cidx[k] ⇒ KeyError
            #   ③ 整表 SKIP ⇒ 它名下的注入错误**凭空消失** ⇒ 算出来的 F1 偏乐观
            # 降级后：这���表的注入错误全部如实计成 b_missing（= 没修好），
            # 对 deepprep 这类「产物结构本身坏掉」的基线才是诚实的记账。
            cols2 = list(dict.fromkeys(list(cols) + list(keys)))
            con = sqlite3.connect(str(dst))
            con.execute("PRAGMA journal_mode=OFF")
            con.execute("PRAGMA synchronous=OFF")
            con.execute("CREATE TABLE data (%s)"
                        % ",".join(['"%s" TEXT' % c for c in cols2]))
            con.commit()
            con.close()
            print("built %-52s %10d rows  (结构不可用 ⇒ 记成没有输出)"
                  % (dst.name, 0), flush=True)
            return

        con = sqlite3.connect(str(dst))
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("CREATE TABLE data (%s)"
                    % ",".join(['"%s" TEXT' % c for c in cols]))
        insert = 'INSERT INTO data VALUES (%s)' % ",".join("?" * len(cols))
        con.execute("BEGIN")
        batch, n = [], 0
        for row in rd:
            batch.append([row.get(c, '') for c in cols])
            n += 1
            if len(batch) >= 50000:
                con.executemany(insert, batch)
                batch = []
        if batch:
            con.executemany(insert, batch)
        con.commit()
        con.execute("CREATE INDEX idx ON data (%s)"
                    % ",".join(['"%s"' % c for c in use]))
        con.commit()
        con.close()
    print("built %-52s %10d rows  %2d cols  (%s)"
          % (dst.name, n, len(cols), kind), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True, help="agent 输出的结果包目录")
    ap.add_argument("--ref-dir", required=True, help="数据集的 test/reference 目录")
    ap.add_argument("--ds-key", required=True, help="cache 命名空间")
    a = ap.parse_args()
    OUT = Path(a.result_dir)
    REF_DIR = Path(a.ref_dir)
    DS_KEY = a.ds_key
    print("result-dir = %s" % OUT)
    print("ref-dir    = %s" % REF_DIR)
    print("ds-key     = %s" % DS_KEY)
    print("=== agent output ===", flush=True)
    for raw_file, (out_rel, _k) in RAW_TO_OUTPUT.items():
        src = OUT / out_rel
        if not src.is_file():
            print("SKIP (missing) %s -> %s" % (raw_file, out_rel), flush=True)
            continue
        if src.stat().st_size == 0:
            print("built %-52s %10d rows  (empty file)" % (out_rel, 0), flush=True)
            continue
        build_one(raw_file, "codex", src)

    print("=== gold (from %s) ===" % REF_DIR, flush=True)
    for raw_file, (out_rel, _k) in RAW_TO_OUTPUT.items():
        src = REF_DIR / out_rel
        if not src.is_file():
            print("SKIP (missing gold) %s -> %s" % (raw_file, out_rel), flush=True)
            continue
        if src.stat().st_size == 0:
            print("built %-52s %10d rows  (empty file)" % (out_rel, 0), flush=True)
            continue
        build_one(raw_file, "ref", src)
    print("done", flush=True)
