"""逐行复现 cc 的 repair_chartevents，定位 chartevents 少掉的 138 万行去哪了。

只读：import cc 的 pipeline，自己跑一遍并数行数。不改任何文件。
只输出聚合数字。
"""
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

CC_PIPELINE = Path("/Users/mac/PycharmProjects/v2_clean/agent_work/n18_more_v1/pipeline/n18_clean.py")
spec = importlib.util.spec_from_file_location("cc_n18_clean", CC_PIPELINE)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

DS = Path(sys.argv[1] if len(sys.argv) > 1 else
          "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_test4.7pct_490M")
RAW = DS / "test/raw"

hosp, icu, ed = RAW / "hosp", RAW / "icu", RAW / "ed"

# --- cohort（照抄 cc 的 run()）
diagnoses = cc.read_csv(os.path.join(hosp, "diagnoses_icd.csv"))
diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.replace(
    cc.ALIAS_PREFIX, "", regex=False)
icd = pd.concat([
    diagnoses[diagnoses["icd_code"].str.startswith(cc.ICD10_PREFIXES)]["hadm_id"],
    diagnoses[diagnoses["icd_code"].str.startswith(cc.ICD9_PREFIXES)]["hadm_id"],
])
n18_hadm = set(icd.dropna())
ed_stays_raw = cc.read_csv(os.path.join(ed, "edstays.csv"))
ed_hadm = set(ed_stays_raw["hadm_id"].dropna())
cohort_hadm = sorted(n18_hadm & ed_hadm)
print(f"cohort_hadm = {len(cohort_hadm):,}")

icustays = cc.read_csv(os.path.join(icu, "icustays.csv"))
icustays = cc.repair_icustays(icustays)
icustays = icustays[icustays["hadm_id"].isin(cohort_hadm)].copy()
stay_meta = icustays.set_index("stay_id")[["subject_id", "hadm_id", "outtime"]]
cohort_stays = set(icustays["stay_id"])
print(f"cohort_stays = {len(cohort_stays):,}")

# --- chartevents 逐阶段
ce = cc.read_csv_filtered(os.path.join(icu, "chartevents.csv"), ("stay_id", cohort_stays))
n_raw = len(ce)
print(f"\n[0] raw chartevents（按 cohort_stays 过滤后）: {n_raw:,}")

ce2 = ce.reset_index(drop=True)
ce2["itemid"] = cc.strip_itemid_alias(ce2["itemid"])
n_alias = len(ce2)
ce2 = ce2[ce2["itemid"] != cc.LEAK_ITEMID].reset_index(drop=True)
print(f"[1] strip_itemid_alias 后: {n_alias:,}")
print(f"[2] 去掉 LEAK_ITEMID({cc.LEAK_ITEMID}) 行后: {len(ce2):,}   （删了 {n_alias-len(ce2):,}）")

ce2["_raw_valuenum"] = ce2["valuenum"].fillna("~").astype(str)
ce2["_raw_valueuom"] = ce2["valueuom"].fillna("~").astype(str)

ct = pd.to_datetime(ce2["charttime"], errors="coerce")
outtime = pd.to_datetime(ce2["stay_id"].map(stay_meta["outtime"]), errors="coerce")
leaked = (ct - outtime) == pd.Timedelta(days=cc.WINDOW_LEAK_DAYS)
fixed = ce2["charttime"].copy()
for _, grp in ce2.groupby("stay_id", sort=False):
    prev = None
    for i in grp.index:
        if leaked.at[i]:
            fixed.at[i] = prev
        else:
            prev = ce2.at[i, "charttime"]
ce2["charttime"] = fixed
print(f"[3] charttime 修复（泄漏行 {int(leaked.sum()):,}）后: {len(ce2):,}")

ce2["valuenum"] = cc.repair_valuenum(ce2["value"], ce2["valuenum"], ce2["itemid"])
uom = ce2.groupby("itemid")["valueuom"].agg(
    lambda s: s.dropna().mode().iat[0] if len(s.dropna()) else pd.NA)
ce2["valueuom"] = ce2["itemid"].map(uom)
print(f"[4] valuenum / valueuom 修复后: {len(ce2):,}")

dup_key = [c for c in ["stay_id", "charttime", "itemid", "value", "_raw_valuenum",
                        "_raw_valueuom", "storetime", "caregiver_id"] if c in ce2.columns]
before = len(ce2)
ce3 = ce2.drop_duplicates(subset=dup_key, keep="first")
print(f"[5] 去重（key = {len(dup_key)} 列）后: {len(ce3):,}   （删了 {before-len(ce3):,}）")

# 被去重删掉的行，长什么样？
dropped = ce2.loc[~ce2.index.isin(ce3.index)]
print(f"\n被去重删掉的行: {len(dropped):,}")
for c in dup_key:
    nun = dropped[c].nunique()
    print(f"  {c:16s} 唯一值 {nun:,}  空值 {int(dropped[c].isna().sum() + (dropped[c]=='').sum()):,}")
# 这些行在 dedup key 上其实是「重复」的：看看同 key 的组有多大
sizes = ce2.groupby(dup_key, dropna=False).size()
print(f"\n  dedup key 分组数: {len(sizes):,}   其中组大小>1 的组: {int((sizes>1).sum()):,}"
      f"   被折叠掉的行: {int(sizes[sizes>1].sum()-1):,}")
print("  组大小分布:", sizes.value_counts().head(8).to_dict())
