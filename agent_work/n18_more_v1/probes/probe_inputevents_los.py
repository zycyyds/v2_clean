import pandas as pd
import numpy as np

BASE = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/n18_more/train"
pd.set_option("display.width", 400)
pd.set_option("display.max_rows", 60)

ri = pd.read_csv(f"{BASE}/reference/icu/inputevents.csv", dtype=str)
raw = pd.read_csv(f"{BASE}/raw/icu/inputevents.csv", dtype=str)
ri["delta"] = (pd.to_datetime(ri.endtime) - pd.to_datetime(ri.starttime)).dt.total_seconds()
print(ri[ri.delta != 60].to_string())
print("\n--- raw rows for those ---")
for _, x in ri[ri.delta != 60].iterrows():
    m = raw[(raw.stay_id == x.stay_id) & (raw.starttime == x.starttime)]
    print(x.stay_id, x.starttime, "ref_end", x.endtime, "delta", x.delta)
    print(
        m[
            [
                "starttime",
                "endtime",
                "storetime",
                "itemid",
                "amount",
                "totalamount",
                "originalamount",
                "patientweight",
                "statusdescription",
            ]
        ].to_string()
    )

print("\n=== los formula search ===")
ics = pd.read_csv(f"{BASE}/raw/icu/icustays.csv", dtype=str)
refic = pd.read_csv(f"{BASE}/reference/icu/icustays.csv", dtype=str)
m = refic.merge(ics, on="stay_id", suffixes=("_ref", "_raw"))


def cands(it, ot, los):
    d = ot - it
    return {
        "total/86400": d.total_seconds() / 86400,
        "total/3600/24": d.total_seconds() / 3600 / 24,
        "days+sec/86400": d.days + d.seconds / 86400,
        "td/1day": d / pd.Timedelta(days=1),
        "td/86400s": d / pd.Timedelta(seconds=86400),
        "raw_los": los,
        "min/1440": (d.total_seconds() / 60) / 1440,
        "days+sec/86400+us": d.days
        + d.seconds / 86400
        + d.microseconds / 86400e6,
    }


for _, x in m.iterrows():
    it = pd.to_datetime(x.intime_ref)
    los = float(x.los_raw)
    ot = it + pd.Timedelta(seconds=los * 86400)
    print(x.stay_id, "ref_los", x.los_ref, "raw_los", x.los_raw)
    for k, v in cands(it, ot, los).items():
        print(f"    {k:26s} {v!r}  match={repr(v) == x.los_ref}")
