"""验证 cc 的 charttime 修复是否会「把连续泄漏行压成同一时间戳 → 被自己的去重删掉」。

前提：raw 的 future_window_leakage 行在同一个 stay 内是否成片出现。
只输出聚合数字。
"""
import sys
from pathlib import Path

import pandas as pd

WINDOW_LEAK_DAYS = 1  # cc 的 pipeline/n18_clean.py: WINDOW_LEAK_DAYS


def analyse(name, ds_dir):
    ds = Path(ds_dir)
    raw = ds / "test/raw"
    ics = pd.read_csv(raw / "icu/icustays.csv", dtype=str)
    outtime = pd.Series(ics["outtime"].values, index=ics["stay_id"].values)

    ce = pd.read_csv(raw / "icu/chartevents.csv", dtype=str, keep_default_na=False,
                     na_filter=False, usecols=["stay_id", "charttime"])
    ct = pd.to_datetime(ce["charttime"], errors="coerce")
    ot = pd.to_datetime(ce["stay_id"].map(outtime), errors="coerce")
    leaked = (ct - ot) == pd.Timedelta(days=WINDOW_LEAK_DAYS)

    # 按 stay 分组，看泄漏行在组内是否连成一片（cc 的修复会把同一片全赋成同一个 prev）
    df = pd.DataFrame({"stay_id": ce["stay_id"], "leaked": leaked.astype(int)})
    grp = df.groupby("stay_id", sort=False)["leaked"]
    prev = grp.shift(1)
    # 一片连续泄漏行（长度≥2）的判定：当前行泄漏 且 组内上一行也是泄漏
    run_start = leaked & prev.fillna(False).astype(bool)
    # 统计「连续泄漏段」的条数：段内除第一行外的行数
    consec = int(run_start.sum())
    n_leak = int(leaked.sum())

    # cc 修复后：连续泄漏段内除第一行外，其余行全被赋成同一个 prev 时间戳 → 与前一行重复 → 被去重删掉
    print(f"\n=== {name} ===")
    print(f"  raw chartevents 行数            : {len(ce):,}")
    print(f"  未来窗口泄漏行 (charttime=outtime+{WINDOW_LEAK_DAYS}d): {n_leak:,}")
    print(f"  其中「紧跟另一条泄漏行」的行数     : {consec:,}  ({consec/max(n_leak,1):.1%} of leaked)")
    print(f"  ⇒ 这些行在 cc 的修复后时间戳与前一行相同")
    print(f"  ⇒ cc 的去重 key 含 charttime ⇒ 这些行会被当成重复行删掉")
    print(f"  ⇒ 预计被误删的行数上限 ≈ {consec:,}")


DS = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集")
analyse("N18 4.7pct", DS / "N18_ckd_ED_v2_rawerror_test4.7pct_490M")
analyse("N18 1.6pct", DS / "N18_ckd_ED_v2_rawerror_test1.6pct_490M")
