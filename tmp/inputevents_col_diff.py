"""逐列比对 cc 与我们 agent 的 icu/inputevents.csv，找出是哪一列全错（只出聚合数字）。"""
import csv
from collections import Counter
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
CC = PROJ / "agent_work/i48_less_cc_v1/output/test/icu/inputevents.csv"
AG = (PROJ / ".claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments/"
      "i48_1p6_test_seed666_v1/host/test_result_package/icu/inputevents.csv")


def cols(p):
    with open(p, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fs = rd.fieldnames
        acc = {f: Counter() for f in fs}
        n = 0
        for r in rd:
            n += 1
            for f in fs:
                acc[f][r.get(f)] += 1
        return fs, acc, n


cf, cc, cn = cols(CC)
af, ag, an = cols(AG)
print(f"cc {cn:,} 行 / agent {an:,} 行\n")
print(f"{'列':<12}{'cc 唯一值':>12}{'agent 唯一值':>14}{'交集':>12}{'cc 值在 agent 里找不到的':>26}")
for f in cf:
    inter = set(cc[f]) & set(ag[f])
    miss = sum(v for k, v in cc[f].items() if k not in ag[f])
    print(f"{f:<12}{len(cc[f]):>12,}{len(ag[f]):>14,}{len(inter):>12,}{miss:>26,}")
