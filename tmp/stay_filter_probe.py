"""为什么 cc 按 stay_id 过滤后只剩 586 万行，而 gold 有 660 万行？

假设：raw 里有 stay_id 被注入破坏（带 9 前缀 / 串位）的行，
cc 的 `isin(cohort_stays)` 把它们全过滤掉了 —— 而 gold 里这些行是好的。
只输出聚合数字。
"""
import csv
import re
from collections import Counter
from pathlib import Path

DS = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/"
          "N18_ckd_ED_v2_rawerror_test4.7pct_490M/test/raw/icu")
AG = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning/"
          "k29_agent_run/experiments/n18_4p7_test_seed666_v1/host/test_result_package/"
          "icu/chartevents.csv")

# 1) cohort stays（照 cc 的算法）
import pandas as pd
ics = pd.read_csv(DS / "icustays.csv", dtype=str)
cohort_stays = set(ics["stay_id"])
print(f"raw icustays 的 stay_id 数: {len(cohort_stays):,}")

# 2) 我们 agent 输出里的 stay_id 分布
ag_stays = Counter()
ag_rows = 0
with open(AG, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        ag_rows += 1
        ag_stays[r.get("stay_id")] += 1
print(f"我们 agent 输出: {ag_rows:,} 行, stay_id 数 {len(ag_stays):,}")
print(f"  agent 输出里 stay_id 不在 raw icustays 里的行: "
      f"{sum(v for k, v in ag_stays.items() if k not in cohort_stays):,}")

# 3) raw chartevents：stay_id 命中率 + 被过滤掉的行的形态
in_set = Counter()
out_stays = Counter()
stripped_hit = 0
out_rows = 0
n = 0
with open(DS / "chartevents.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        n += 1
        sid = r.get("stay_id") or ""
        if sid in cohort_stays:
            in_set["in"] += 1
        else:
            out_rows += 1
            out_stays[sid] += 1
            s = sid.lstrip("9")
            if s in cohort_stays:
                stripped_hit += 1
print(f"\nraw chartevents 共 {n:,} 行")
print(f"  stay_id ∈ cohort_stays: {in_set['in']:,}")
print(f"  stay_id ∉ cohort_stays: {out_rows:,}  (其中去掉 9 前缀后能命中: {stripped_hit:,})")
print(f"  被排除掉的 stay_id 唯一值数: {len(out_stays):,}")
print("  被排除行的 stay_id 长度分布:",
      dict(sorted(Counter(len(k) for k in out_stays).items())))
