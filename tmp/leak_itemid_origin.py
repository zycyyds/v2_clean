"""最后一步：raw 里被 cc 当成「泄漏行」删掉的那些行，
它们在 raw 里的原始 itemid 长什么样？

如果原始 itemid 是 8 位（99 前缀）⇒ 是「真实行被改写了 itemid」，删了就丢 gold 行；
如果原始 itemid 本来就是 6 位 999001 ⇒ 是注入的新行，删掉是对的。
只输出聚合数字。
"""
import csv
from collections import Counter
from pathlib import Path

RAW = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/"
           "N18_ckd_ED_v2_rawerror_test4.7pct_490M/test/raw/icu/chartevents.csv")
AG = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning/"
          "k29_agent_run/experiments/n18_4p7_test_seed666_v1/host/test_result_package/"
          "icu/chartevents.csv")

import re
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cc_n18_clean",
    "/Users/mac/PycharmProjects/v2_clean/agent_work/n18_more_v1/pipeline/n18_clean.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
print("ITEMID_99_RE =", cc.ITEMID_99_RE)
print("LEAK_ITEMID  =", cc.LEAK_ITEMID)

orig_when_leak = Counter()
n_leak = 0
n = 0
with open(RAW, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        n += 1
        raw_iid = r.get("itemid") or ""
        if not raw_iid.startswith("99"):
            continue
        stripped = cc.strip_itemid_alias(pd.Series([raw_iid])).iloc[0]
        if stripped == cc.LEAK_ITEMID:
            n_leak += 1
            orig_when_leak[(len(raw_iid), raw_iid == cc.LEAK_ITEMID)] += 1
print(f"\nraw 共 {n:,} 行；strip 后变成 '999001' 的行: {n_leak:,}")
print("  (原始长度, 原始是否就是 999001) 计数:", dict(orig_when_leak))

# 我们 agent 的输出里有没有对应行（按 stay+charttime+storetime+caregiver_id 找）
print("\n在 agent 输出里找这些行的对应行 …")
leak_rows = []
with open(RAW, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        raw_iid = r.get("itemid") or ""
        if not raw_iid.startswith("99"):
            continue
        stripped = cc.strip_itemid_alias(pd.Series([raw_iid])).iloc[0]
        if stripped == cc.LEAK_ITEMID:
            leak_rows.append((r.get("stay_id"), r.get("charttime"), r.get("storetime"),
                               r.get("caregiver_id"), raw_iid, r.get("valuenum")))
ag_keys = set()
with open(AG, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        ag_keys.add((r.get("stay_id"), r.get("charttime"), r.get("storetime"), r.get("caregiver_id")))
hit = sum(1 for x in leak_rows if x[:4] in ag_keys)
print(f"  agent 输出里能按 (stay_id, charttime, storetime, caregiver_id) 找到对应行: "
      f"{hit:,} / {len(leak_rows):,}")
