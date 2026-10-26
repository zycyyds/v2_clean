"""量化 cc 去重的破坏性：用 cc 自己的 8 列 key 分组，
统计被折叠的行在 gold 口径（输出列集合）下是否本来可区分。

gold 的输出列 = subject_id, hadm_id, stay_id, charttime, itemid, valuenum, valueuom
cc 的去重 key = stay_id, charttime, itemid, value, _raw_valuenum, _raw_valueuom, storetime, caregiver_id
⇒ key 漏掉的列：subject_id, hadm_id, valuenum, valueuom
只输出聚合数字。
"""
import importlib.util
import os
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "cc_n18_clean",
    "/Users/mac/PycharmProjects/v2_clean/agent_work/n18_more_v1/pipeline/n18_clean.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

DS = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/"
          "N18_ckd_ED_v2_rawerror_test4.7pct_490M/test/raw")
icu, hosp, ed = DS / "icu", DS / "hosp", DS / "ed"

diag = cc.read_csv(os.path.join(hosp, "diagnoses_icd.csv"))
diag["icd_code"] = diag["icd_code"].astype(str).str.replace(cc.ALIAS_PREFIX, "", regex=False)
icd = pd.concat([diag[diag["icd_code"].str.startswith(cc.ICD10_PREFIXES)]["hadm_id"],
                 diag[diag["icd_code"].str.startswith(cc.ICD9_PREFIXES)]["hadm_id"]])
ed_hadm = set(cc.read_csv(os.path.join(ed, "edstays.csv"))["hadm_id"].dropna())
cohort_hadm = sorted(set(icd.dropna()) & ed_hadm)

icustays = cc.repair_icustays(cc.read_csv(os.path.join(icu, "icustays.csv")))
icustays = icustays[icustays["hadm_id"].isin(cohort_hadm)].copy()
stay_meta = icustays.set_index("stay_id")[["subject_id", "hadm_id", "outtime"]]
cohort_stays = set(icustays["stay_id"])

ce = cc.read_csv_filtered(os.path.join(icu, "chartevents.csv"), ("stay_id", cohort_stays))
ce = ce.reset_index(drop=True)
ce["itemid"] = cc.strip_itemid_alias(ce["itemid"])
ce = ce[ce["itemid"] != cc.LEAK_ITEMID].reset_index(drop=True)
print(f"去掉泄漏行后: {len(ce):,}")

ce["_raw_valuenum"] = ce["valuenum"].fillna("~").astype(str)
ce["_raw_valueuom"] = ce["valueuom"].fillna("~").astype(str)
ce["subject_id"] = ce["stay_id"].map(stay_meta["subject_id"])
ce["hadm_id"] = ce["stay_id"].map(stay_meta["hadm_id"])

ct = pd.to_datetime(ce["charttime"], errors="coerce")
ot = pd.to_datetime(ce["stay_id"].map(stay_meta["outtime"]), errors="coerce")
leaked = (ct - ot) == pd.Timedelta(days=cc.WINDOW_LEAK_DAYS)
fixed = ce["charttime"].copy()
for _, grp in ce.groupby("stay_id", sort=False):
    prev = None
    for i in grp.index:
        if leaked.at[i]:
            fixed.at[i] = prev
        else:
            prev = ce.at[i, "charttime"]
ce["charttime"] = fixed

ce["valuenum"] = cc.repair_valuenum(ce["value"], ce["valuenum"], ce["itemid"])
uom = ce.groupby("itemid")["valueuom"].agg(lambda s: s.dropna().mode().iat[0] if len(s.dropna()) else pd.NA)
ce["valueuom"] = ce["itemid"].map(uom)

dup_key = ["stay_id", "charttime", "itemid", "value", "_raw_valuenum", "_raw_valueuom",
            "storetime", "caregiver_id"]
dup_key = [c for c in dup_key if c in ce.columns]

sizes = ce.groupby(dup_key, dropna=False).size()
multi = sizes[sizes > 1]
print(f"去重 key 分组数 {len(sizes):,}；多行组 {len(multi):,}；被折叠掉的行 "
      f"{int(multi.sum()-len(multi)):,}")

# 在这些多行组里，成员在 gold 口径的列上是否可区分
gold_cols = ["subject_id", "hadm_id", "valuenum", "valueuom"]
g = ce.groupby(dup_key, dropna=False)[gold_cols].nunique(dropna=False)
diff_any = (g > 1).any(axis=1)
print(f"\n多行组里，至少有一列(gold 口径)在成员间不同: {int(diff_any.sum()):,}")
for c in gold_cols:
    print(f"  {c:12s} 在组内不同的组数: {int((g[c] > 1).sum()):,}")
# 这些组里，被折叠掉的行数
print(f"\n⇒ 这些「组内可区分」的组里被 cc 去重删掉的行数: "
      f"{int((multi * diff_any.reindex(multi.index, fill_value=False)).sum() - int(diff_any.sum())):,}")
