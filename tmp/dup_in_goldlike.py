"""gold 口径的输出列里到底有没有重复行？

我们 agent 的输出行数 = gold 行数（6,597,012），所以它的输出是 gold 的行级近似。
数一下它的输出里有多少「完全重复的行」。
"""
import csv
from collections import Counter
from pathlib import Path

CASES = [
    ("N18 4.7pct · 我们 agent",
     "/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning/"
     "k29_agent_run/experiments/n18_4p7_test_seed666_v1/host/test_result_package/icu/chartevents.csv"),
    ("N18 4.7pct · cc",
     "/Users/mac/PycharmProjects/v2_clean/agent_work/n18_more_v1/output/icu/chartevents.csv"),
]
COLS = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"]

for tag, p in CASES:
    path = Path(p)
    if not path.is_file():
        print(f"\n{tag}: 缺文件")
        continue
    c = Counter()
    n = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            c[tuple(r.get(k) for k in COLS)] += 1
    dup_groups = sum(1 for v in c.values() if v > 1)
    dup_rows = sum(v - 1 for v in c.values() if v > 1)
    print(f"\n{tag}")
    print(f"  行数 {n:,}   唯一(7列)行数 {len(c):,}")
    print(f"  重复行组 {dup_groups:,}   多出来的重复行 {dup_rows:,}")
    print(f"  组大小分布: {Counter(c.values()).most_common(6)}")
