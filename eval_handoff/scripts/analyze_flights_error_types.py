"""flights：把每一个脏 cell 的「破坏形态」分类，回答「有几类错误」。"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
SRC = PROJ / "datasets/_source/flights"
TIME = re.compile(r"(\d{1,2}):(\d{2})")
DATE_PREFIX = re.compile(r"^\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]{2}\s+\d{1,2})")


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def minutes(v):
    m = TIME.search(v or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


_, drows = load(SRC / "dirty.csv")
_, crows = load(SRC / "clean.csv")
COLS = ("sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time")


def classify(dv, cv):
    """返回 (形态, 偏移分钟)。"""
    if dv == cv:
        return None
    a, b = minutes(dv), minutes(cv)
    if dv.strip() == "":
        return ("① 被清空（空串）", None)
    if a is None and b is None:
        if DATE_PREFIX.match(dv):
            return ("② 加了日期前缀", None)
        return ("③ 换成标记文本（Not Available 等）", None)
    if a is None or b is None:
        return ("④ 一边可解析一边不可解析", None)
    d = (a - b) % 1440
    if d == 0:
        return ("⑤ 时间没变、脏在字符串形态", 0)
    if d <= 30:
        return ("⑥ 时间被往后挪 1~30 分钟", d)
    if d >= 1410:
        return ("⑦ 时间被往前挪 1~30 分钟", 1440 - d)
    return ("⑧ 时间被挪了别的量", d)


print("=== flights 的错误类型（按列 × 形态）===")
table = defaultdict(Counter)
shift_detail = defaultdict(Counter)
total = 0
for d, c in zip(drows, crows):
    for col in COLS:
        dv, cv = d.get(col, ""), c.get(col, "")
        if dv == cv:
            continue
        total += 1
        kind, off = classify(dv, cv)
        table[col][kind] += 1
        if off is not None and off:
            shift_detail[col][off] += 1

kinds = sorted({k for col in COLS for k in table[col]})
print(f"{'形态':32s}" + "".join(f"{col:>18s}" for col in COLS) + f"{'合计':>10s}")
for k in kinds:
    vals = [table[col].get(k, 0) for col in COLS]
    print(f"{k:32s}" + "".join(f"{v:>18,}" for v in vals) + f"{sum(vals):>10,}")
print(f"{'—— 合计脏 cell':32s}" + "".join(f"{sum(table[col].values()):>18,}" for col in COLS)
      + f"{total:>10,}")

print("\n=== 形态⑥⑦ 的偏移分钟分布（前 12）===")
for col in COLS:
    if shift_detail[col]:
        print(f"  {col:16s} {shift_detail[col].most_common(12)}")

print("\n=== 脏值样本：每种形态各取 2 个 ===")
seen = defaultdict(list)
for d, c in zip(drows, crows):
    for col in COLS:
        dv, cv = d.get(col, ""), c.get(col, "")
        res = classify(dv, cv)
        if res is None:
            continue
        kind, off = res
        if len(seen[(col, kind)]) < 2:
            seen[(col, kind)].append((dv, cv))
for (col, kind), items in seen.items():
    for dv, cv in items:
        print(f"  {col:16s} {kind:26s} 脏 {dv!r:26s} → 干净 {cv!r}")
