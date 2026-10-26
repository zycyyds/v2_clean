"""Probe: how to tell a categorical value text from a redacted measurement."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
RAW = f"{BASE}/raw"

for name, path, itemcol in [
    ("labs", f"{RAW}/hosp/labevents.csv", "itemid"),
    ("chartevents", f"{RAW}/icu/chartevents.csv", "itemid"),
]:
    df = pd.read_csv(path, dtype=str)
    df["vnum"] = pd.to_numeric(df.value, errors="coerce")
    df["vn"] = pd.to_numeric(df.valuenum, errors="coerce")
    text = df[df.vnum.isna()]
    rows = []
    for (it, vl), g in text.groupby([itemcol, "value"], dropna=False):
        vals = g["vn"]
        nonnan = vals.dropna().unique()
        rows.append(
            {
                "itemid": it,
                "value": vl,
                "n": len(g),
                "n_nonnull": int(vals.notna().sum()),
                "n_distinct": len(nonnan),
                "nan_frac": float(vals.isna().mean()),
            }
        )
    d = pd.DataFrame(rows)
    print(f"===== {name}: {len(text)} text rows, {len(d)} (itemid,value) groups")
    print("  groups where all non-null valuenums are identical:",
          int((d.n_distinct <= 1).sum()))
    print("  groups with >1 distinct non-null valuenum:", int((d.n_distinct > 1).sum()))
    print("  value texts of the multi-valued groups:",
          d[d.n_distinct > 1]["value"].value_counts().head(10).to_dict())
    print("  value texts of the single-valued groups:",
          d[d.n_distinct <= 1]["value"].value_counts().head(10).to_dict())
    print("  sample multi-valued groups:")
    print(d[d.n_distinct > 1].head(6).to_string())
    print()
