#!/usr/bin/env python3
"""取证：1.6pct 的 valuenum 注入格里，agent 输出 / gold / log(改前) / log(改后) 四方关系。

**只输出聚合计数，不打印任何行级内容。**

回答三个问题：
  Q1 agent 写对了吗？（agent 输出 vs gold，逐字节相同 = 做对了）
  Q2 老口径（纯字符串）改前/改后各判多少 clean？
  Q3 老口径改后判 wrong 的那些格，其中有多少是「agent 写对了」的 ⇒ 误罚数

## 前置条件
1. 评分缓存已建（`score_only/error_metrics/build_eval_cache.py`）：
   `k29_mimicraw_16_agent__icu_chartevents.csv__{codex,ref}.db`
   —— `codex.db` = agent 输出，`ref.db` = gold（rowid 连续，rowid = 行号 + 1）。
2. log 改前的备份在 `/Users/mac/.claude/jobs/<job>/tmp/log_backup/1p6_test__modification_log.csv.pre_rawform.bak`。

## 用法
    python3 forensic_log_form.py [样本数]     # 默认 60000
"""
import csv
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(2 ** 31)

DS = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/k29/"
          "K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror")
BAK = Path("/Users/mac/.claude/jobs/448aad57/tmp/log_backup/"
           "1p6_test__modification_log.csv.pre_rawform.bak")
CACHE = Path("/Users/mac/.claude/jobs/910902cd/tmp/v31_idx_cache")
CODEX = CACHE / "k29_mimicraw_16_agent__icu_chartevents.csv__codex.db"
REF = CACHE / "k29_mimicraw_16_agent__icu_chartevents.csv__ref.db"

KEYS = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60000

# ---- 1. 取 log 的 valuenum 注入格（改后 / 改前按 raw_row_index 对齐）----
new_by_row = {}
with (DS / "test/modification_log.csv").open(newline="", encoding="utf-8") as f:
    for e in csv.DictReader(f):
        if (e.get("raw_file") == "icu/chartevents.csv" and e.get("column") == "valuenum"
                and e.get("operation") == "cell_update"):
            new_by_row[int(e["raw_row_index"])] = e.get("clean_value")
old_by_row = {}
with BAK.open(newline="", encoding="utf-8") as f:
    for e in csv.DictReader(f):
        if (e.get("raw_file") == "icu/chartevents.csv" and e.get("column") == "valuenum"
                and e.get("operation") == "cell_update"):
            old_by_row[int(e["raw_row_index"])] = e.get("clean_value")

rows = sorted(set(new_by_row) & set(old_by_row))
print("chartevents.valuenum 注入格总数: %d" % len(rows))
rng = random.Random(666)
sample = rng.sample(rows, min(N, len(rows)))

# ---- 2. gold（ref.db）按行号取 key + valuenum ----
ref = sqlite3.connect("file:%s?mode=ro" % REF, uri=True)
rcols = [r[1] for r in ref.execute("PRAGMA table_info(data)")]
ridx = {c: i for i, c in enumerate(rcols)}
ref_rows = {}
for L in sample:
    # ref.db 的 rowid 连续（1..N）⇒ rowid = 行号 + 1，直接定位（不能用 OFFSET，那会全表扫描）
    r = ref.execute("SELECT * FROM data WHERE rowid = ?", (L + 1,)).fetchone()
    if r:
        ref_rows[L] = r
print("gold 命中行: %d / %d" % (len(ref_rows), len(sample)))

# ---- 3. agent 输出（codex.db）按 key 取 valuenum ----
codex = sqlite3.connect("file:%s?mode=ro" % CODEX, uri=True)
ccols = [r[1] for r in codex.execute("PRAGMA table_info(data)")]
cidx = {c: i for i, c in enumerate(ccols)}
where = " AND ".join('"%s"=?' % k for k in KEYS)
sel = ",".join('"%s"' % c for c in ccols)
crows = {}
for L, r in ref_rows.items():
    kt = tuple(r[ridx[k]] for k in KEYS)
    c = codex.execute("SELECT %s FROM data WHERE %s" % (sel, where), kt).fetchone()
    if c:
        crows[L] = c
print("agent 输出命中行: %d / %d" % (len(crows), len(ref_rows)))

# ---- 4. 四方关系（聚合计数）----
stat = Counter()
for L in crows:
    g = ref_rows[L][ridx["valuenum"]]          # gold
    a = crows[L][cidx["valuenum"]]              # agent 输出
    cv_new = new_by_row[L]                      # log 改后
    cv_old = old_by_row[L]                      # log 改前
    stat["样本数"] += 1
    stat["agent==gold 逐字节相同（做对了）"] += (a == g)
    stat["agent==log改前clean（改前老口径判 clean）"] += (a == cv_old)
    stat["agent==log改后clean（改后老口径判 clean）"] += (a == cv_new)
    # 数值相等？
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    ga, gg = num(a), num(g)
    if ga is not None and gg is not None:
        stat["agent 与 gold 数值相等"] += (abs(ga - gg) < 1e-9)
        stat["agent 与 gold 数值相等但字符串不同"] += (abs(ga - gg) < 1e-9 and a != g)
    # 改后判 wrong 且 agent==gold ⇒ 误罚
    if a != cv_new and a == g:
        stat["⚠️ 改后判 wrong 且 agent 写对了（误罚）"] += 1
    if a != cv_new and a == g and num(a) is not None and num(cv_new) is not None:
        if abs(num(a) - num(cv_new)) < 1e-9:
            stat["⚠️ 其中：与 log 值数值相等（纯写法差异）"] += 1

print("\n=== 样本 %d 个 valuenum 注入格，四方关系（聚合）===" % stat["样本数"])
for k, v in stat.most_common():
    print("   %-46s %8d  (%5.1f%%)" % (k, v, 100.0 * v / stat["样本数"]))

# ---- 5. 全量外推（用全量 log 条目 + 命中率）----
tot_new = len(new_by_row)
print("\n=== 全量外推 ===")
print("   valuenum 注入格总数: %d" % tot_new)
ratio = stat["⚠️ 改后判 wrong 且 agent 写对了（误罚）"] / max(1, stat["样本数"])
print("   抽样里「改后判 wrong 且 agent 写对了」的比例: %.2f%%" % (100 * ratio))
print("   ⇒ 全量误罚估计: %d 格" % int(tot_new * ratio))
