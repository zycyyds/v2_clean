"""诊断 cc 的 icu/inputevents.csv 为什么一行都匹配不上。

只对比两个 agent 的产物（cc 的 vs 我们 agent 的），不碰 gold。
只输出聚合数字。
"""
import csv
from collections import Counter
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
CC = PROJ / "agent_work/i48_less_cc_v1/output/test/icu/inputevents.csv"
AG = (PROJ / ".claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments/"
      "i48_1p6_test_seed666_v1/host/test_result_package/icu/inputevents.csv")


def load(p):
    with open(p, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


cf, crows = load(CC)
af, arows = load(AG)
print("cc  列:", cf)
print("agent 列:", af)
print(f"cc 行 {len(crows):,} / agent 行 {len(arows):,}")

# 1) 键的每一列，cc 与 agent 是否一致（按 (subject_id, stay_id, itemid, starttime) 对齐后比）
key4 = ("subject_id", "stay_id", "itemid", "starttime")
amap = {}
for r in arows:
    amap.setdefault(tuple(r.get(k) for k in key4), r)
print(f"\nagent 在 4 列键 ({','.join(key4)}) 上唯一行数: {len(amap):,}")

hit = 0
diff = Counter()
checked = 0
for r in crows:
    a = amap.get(tuple(r.get(k) for k in key4))
    if a is None:
        continue
    checked += 1
    for col in ("hadm_id", "endtime", "amount"):
        if r.get(col) != a.get(col):
            diff[col] += 1
    if checked >= 200000:
        break
print(f"4 列键对上的 cc 行数: {checked:,} / {len(crows):,}")
if checked:
    for col, n in diff.most_common():
        print(f"  {col:10s} 与 agent 不一致: {n:,}  ({n/checked:.2%})")

# 2) cc 自己的 hadm_id 是不是「从 icustays 重建」的
ics_cc = PROJ / "agent_work/i48_less_cc_v1/output/test/icu/icustays.csv"
if ics_cc.is_file():
    with open(ics_cc, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        ics_cols = rd.fieldnames
        stay2hadm = {}
        for r in rd:
            stay2hadm.setdefault(r.get("stay_id"), set()).add(r.get("hadm_id"))
    print(f"\nicustays 列: {ics_cols}")
    multi = sum(1 for v in stay2hadm.values() if len(v) > 1)
    print(f"  stay_id 数 {len(stay2hadm):,}，其中对应多个 hadm_id 的: {multi:,}")
    n_rebuilt = same = tot = 0
    for r in crows:
        h = stay2hadm.get(r.get("stay_id"))
        if h is None:
            continue
        tot += 1
        if r.get("hadm_id") in h:
            same += 1
        else:
            n_rebuilt += 1
    if tot:
        print(f"  cc 的 inputevents.hadm_id 与其 icustays 映射一致: {same:,}/{tot:,} "
              f"({same/tot:.2%})，不一致 {n_rebuilt:,}")
