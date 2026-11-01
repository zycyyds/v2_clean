"""flights：按「错误形态」拆开，codex 修对多少 / 组标准值规则能修对多少。

形态定义（见 flights_error_types.py）：
  ① 被清空（空串）          ④ 换成标记文本(Not Available)
  ② 加了日期前缀            ⑤ 时间没变、脏在字符串形态（后缀/前缀）
  ⑥ 时间被往后挪 1~30 分    ⑦ 时间被往前挪 1~30 分
  ⑧ 时间被挪了别的量（含跨小时/跨天、格式换掉）
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
SRC = PROJ / "datasets/_source/flights"
SPL = PROJ / "datasets/correction/flights_curated10_v1"
RES = PROJ / "agent_work/flights_correction/output/flights.csv"
TIME = re.compile(r"(\d{1,2}):(\d{2})")
COLS = ("sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time")


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def minutes(v):
    m = TIME.search(v or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def classify(dv, cv):
    a, b = minutes(dv), minutes(cv)
    if dv.strip() == "":
        return "① 被清空（空串）"
    if a is None and b is None:
        return "⑤ 时间没变、脏在字符串形态"
    if a is None or b is None:
        return "④ 换成标记文本"
    d = (a - b) % 1440
    if d == 0:
        return "⑤ 时间没变、脏在字符串形态"
    if d <= 30:
        return "⑥ 时间被往后挪 1~30 分"
    if d >= 1410:
        return "⑦ 时间被往前挪 1~30 分"
    return "⑧ 时间被挪了别的量"


dcols, drows = load(SRC / "dirty.csv")
ccols, crows = load(SRC / "clean.csv")
_, rrows = load(RES)
rmap = defaultdict(list)
for r in rrows:
    rmap[r.get("tuple_id", "")].append(r)
idx = []
for d in drows:
    b = rmap.get(d.get("tuple_id", ""))
    idx.append(b.pop(0) if b else None)

# 组估计器：每列的组内众数 / 组内最早时间
groups = defaultdict(list)
for d in drows:
    groups[d.get("flight", "")].append(d)
mode_map, early_map = {}, {}
for col in COLS:
    mode_map[col], early_map[col] = {}, {}
    for fl, rows in groups.items():
        mode_map[col][fl] = Counter(x.get(col, "") for x in rows).most_common(1)[0][0]
        cand = [(minutes(x.get(col, "")), x.get(col, ""))
                for x in rows if minutes(x.get(col, "")) is not None]
        early_map[col][fl] = min(cand)[1] if cand else None

stat = defaultdict(lambda: Counter())
for i, (d, c) in enumerate(zip(drows, crows)):
    r = idx[i]
    if r is None:
        continue
    for dc, cc in zip(dcols, ccols):
        dv, cv = d.get(dc, ""), c.get(cc, "")
        if dv == cv:
            continue
        k = classify(dv, cv)
        out = r.get(dc, "") or ""
        stat[k]["n"] += 1
        if out == cv:
            stat[k]["agent"] += 1
        elif out == dv:
            stat[k]["untouched"] += 1
        else:
            stat[k]["wrong"] += 1
        # 组标准值规则：**按列各用最合适的估计器**
        #   act_dep（偏移往后挪）→ 组内最早时间；act_arr（偏移往前挪）→ 组内众数
        #   sched_* → codex 本来就比这些规则好，保留它的输出
        if dc == "act_dep_time":
            cand = early_map[dc].get(d.get("flight", ""))
        elif dc == "act_arr_time":
            cand = mode_map[dc].get(d.get("flight", ""))
        else:
            cand = out
        stat[k]["rule"] += (cand == cv)

print("=== 按错误形态拆开（test 切分口径的同款逐 cell 对比，全表）===")
print(f"{'形态':26s} {'脏cell':>8s} {'codex修对':>10s} {'修复率':>8s} {'没动':>8s} {'改错':>7s}"
      f" {'组规则修对':>11s} {'修复率':>8s}")
tn = ta = tu = tw = tr = 0
for k in sorted(stat):
    s = stat[k]
    tn += s["n"]; ta += s["agent"]; tu += s["untouched"]; tw += s["wrong"]; tr += s["rule"]
    print(f"{k:26s} {s['n']:8,} {s['agent']:10,} {s['agent']/s['n']*100:7.2f}% "
          f"{s['untouched']:8,} {s['wrong']:7,} {s['rule']:11,} {s['rule']/s['n']*100:7.2f}%")
print(f"{'—— 合计':26s} {tn:8,} {ta:10,} {ta/tn*100:7.2f}% {tu:8,} {tw:7,} "
      f"{tr:11,} {tr/tn*100:7.2f}%")

print("\n=== 逐列 × 形态 的脏 cell 分布 ===")
print(f"{'形态':26s}" + "".join(f"{c:>18s}" for c in COLS))
dist = defaultdict(Counter)
for d, c in zip(drows, crows):
    for col in COLS:
        dv, cv = d.get(col, ""), c.get(col, "")
        if dv != cv:
            dist[classify(dv, cv)][col] += 1
for k in sorted(dist):
    print(f"{k:26s}" + "".join(f"{dist[k].get(col,0):>18,}" for col in COLS))
print(f"{'—— 合计':26s}" + "".join(f"{sum(dist[k].get(col,0) for k in dist):>18,}" for col in COLS))
