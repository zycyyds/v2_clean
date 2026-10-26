"""查 cc 被 `itemid == 999001` 过滤删掉的 118 万行：这些行在 gold 里本来是什么？

用我们 agent 的输出（行数与 gold 完全一致）做对照，只输出聚合数字。
"""
import csv
from collections import Counter
from pathlib import Path

RAW = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/"
           "N18_ckd_ED_v2_rawerror_test4.7pct_490M/test/raw/icu/chartevents.csv")
AG = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning/"
          "k29_agent_run/experiments/n18_4p7_test_seed666_v1/host/test_result_package/"
          "icu/chartevents.csv")

# 1) raw 里 itemid 原始形态的分布
raw_counter = Counter()
leak_rows = []
n = 0
with open(RAW, newline="", encoding="utf-8") as fh:
    rd = csv.DictReader(fh)
    print("raw 列:", rd.fieldnames)
    for r in rd:
        n += 1
        iid = r.get("itemid") or ""
        raw_counter[len(iid)] += 1
        if iid.endswith("999001") or iid == "999001":
            leak_rows.append({k: r.get(k) for k in
                              ("subject_id", "stay_id", "hadm_id", "charttime", "itemid",
                               "valuenum", "value", "valueuom", "storetime", "caregiver_id")})
        if n >= 4_000_000:
            break
print(f"\nraw 抽样 {n:,} 行，itemid 长度分布: {dict(sorted(raw_counter.items()))}")
print(f"其中 itemid 以 999001 结尾/等于 999001 的行: {len(leak_rows):,}")
print("  它们的 valuenum 分布 top5:", Counter(x["valuenum"] for x in leak_rows).most_common(5))
print("  它们的 value 分布 top5:", Counter((x["value"] or "")[:12] for x in leak_rows).most_common(5))
print("  它们的 charttime 空值数:", sum(1 for x in leak_rows if not x["charttime"]))
print("  它们的 storetime 空值数:", sum(1 for x in leak_rows if not x["storetime"]))
print("  它们的 caregiver_id 空值数:", sum(1 for x in leak_rows if not x["caregiver_id"]))

# 2) 这些行在「我们 agent 的输出」里有没有同 (stay_id, charttime, storetime, caregiver_id) 的对应行？
key4 = ("stay_id", "charttime", "storetime", "caregiver_id")
ag_map = {}
with open(AG, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        ag_map.setdefault(tuple(r.get(k) for k in key4), []).append(r.get("itemid"))

hit = 0
matched_item = Counter()
for x in leak_rows:
    cand = ag_map.get(tuple(x[k] for k in key4))
    if not cand:
        continue
    hit += 1
    matched_item[tuple(sorted(set(cand)))[:3]] += 1
print(f"\n在 agent 输出里能按 ({','.join(key4)}) 找到对应行的: {hit:,} / {len(leak_rows):,}")
print("  对应行的 itemid 组合 top8:", matched_item.most_common(8))
