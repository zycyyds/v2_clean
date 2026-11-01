"""flights 失败根因的结构解剖（台账 §2.14 的证据链，可复现）。

只用 `datasets/_source/flights/{dirty,clean}.csv`（原版脏/净表）+ 切分 keys，
不读 host_private、不碰 gold 作为输入。

产出：
  ① 每列的污染形态（时间解析不出来 / 时间被挪几分钟）+ 偏移分布
  ② 每列干净值的不同取值数、众数占比 ⇒ 「列级众数」假设为什么注定不成立
  ③ **干净值在每个 flight 组内是否恒定** ⇒ 正确规则是「组标准值」
  ④ 组内污染率 ⇒ 为什么「组内众数」也不是直接可用
  ⑤ 四种估计器（组众数 / 组最早 / 平移众数(learned) / 平移众数(0~25分 先验)）各能修多少
  ⑥ 投影：在 agent 输出上只对 act_* 套用组标准值规则
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


dcols, drows = load(SRC / "dirty.csv")
ccols, crows = load(SRC / "clean.csv")
_, rrows = load(RES)

with open(SPL / "correction/keys.csv", newline="", encoding="utf-8") as f:
    rd = csv.reader(f)
    next(rd)
    test_keys = {r[0] for r in rd if r and r[0]}

# 干净侧：每个 flight 组的「标准值」
std = {}
for c in crows:
    std.setdefault(c.get("flight", ""), {col: c.get(col, "") for col in COLS})

print("=== ① 每列的污染形态与时间偏移 ===")
for col in COLS:
    unparse = 0
    deltas = Counter()
    n = 0
    for d, c in zip(drows, crows):
        dv, cv = d.get(col, ""), c.get(col, "")
        if dv == cv:
            continue
        n += 1
        a, b = minutes(dv), minutes(cv)
        if a is None or b is None:
            unparse += 1
        else:
            deltas[(a - b) % 1440] += 1
    tot = sum(deltas.values())
    print(f"  {col:16s} 脏 cell {n:5,}  解析不出时间 {unparse:5,} ({unparse/n*100:5.1f}%)"
          f"  偏移众数 {deltas.most_common(1)[0]}  前 6 {deltas.most_common(6)}")

print("\n=== ② 「列级众数」假设为什么注定不成立 ===")
for col in COLS:
    vals = Counter(c.get(col, "") for c in crows)
    print(f"  {col:16s} 行数 {len(crows):,}  干净侧不同取值 {len(vals):,}  "
          f"众数占比 {vals.most_common(1)[0][1]/len(crows)*100:.2f}%")

print("\n=== ③ 干净值在 flight 组内是否恒定 ===")
for col in COLS:
    g = defaultdict(set)
    for c in crows:
        g[c.get("flight", "")].add(c.get(col, ""))
    uniq = sum(1 for v in g.values() if len(v) == 1)
    print(f"  {col:16s} 组 {len(g):,}  组内干净值唯一的组 {uniq:,} ({uniq/len(g)*100:.1f}%)")

print("\n=== ④ 组内污染率（为什么组内众数也不是直接可用）===")
groups = defaultdict(list)
for d in drows:
    groups[d.get("flight", "")].append(d)
for col in COLS:
    fr = []
    for fl, rows in groups.items():
        cnt = Counter(x.get(col, "") for x in rows)
        fr.append(1 - cnt.get(std[fl][col], 0) / len(rows))
    print(f"  {col:16s} 组内被污染比例 均值 {sum(fr)/len(fr)*100:5.1f}%  最大 {max(fr)*100:5.1f}%")

print("\n=== ⑤ 四种估计器各能修好多少 act_* ===")
r2, r1 = {}, {}
mode_map = {col: {} for col in COLS}
r2_map = {col: {} for col in COLS}          # 每列各自的「组内最早时间」
for fl, rows in groups.items():
    cand = [(minutes(x.get("act_dep_time", "")), x.get("act_dep_time", ""))
            for x in rows if minutes(x.get("act_dep_time", "")) is not None]
    if cand:
        r2[fl] = min(cand)[1]
    r1[fl] = Counter(x.get("act_arr_time", "") for x in rows).most_common(1)[0][0]
    for col in COLS:                       # 每列各自的组内众数 / 组内最早时间
        mode_map[col][fl] = Counter(x.get(col, "") for x in rows).most_common(1)[0][0]
        cand = [(minutes(x.get(col, "")), x.get(col, ""))
                for x in rows if minutes(x.get(col, "")) is not None]
        if cand:
            r2_map[col][fl] = min(cand)[1]


def shift_est(col, S):
    """平移众数估计器：argmax_c |{v : (v-c) mod 1440 ∈ S}|"""
    g = defaultdict(list)
    for d in drows:
        m = minutes(d.get(col, ""))
        if m is not None:
            g[d.get("flight", "")].append(m)
    out = {}
    for fl, vals in g.items():
        cnt = Counter()
        for v in vals:
            for s in S:
                cnt[(v - s) % 1440] += 1
        if cnt:
            out[fl] = cnt.most_common(1)[0][0]
    return out


# train 学到的偏移集合
with open(SPL / "train/keys.csv", newline="", encoding="utf-8") as f:
    rd = csv.reader(f)
    next(rd)
    train_keys = {r[0] for r in rd if r and r[0]}
learned = {}
for col in COLS:
    s = Counter()
    for d, c in zip(drows, crows):
        if d.get("tuple_id", "") not in train_keys:
            continue
        dv, cv = d.get(col, ""), c.get(col, "")
        if dv == cv:
            continue
        a, b = minutes(dv), minutes(cv)
        if a is not None and b is not None:
            s[(a - b) % 1440] += 1
    learned[col] = set(s)
    print(f"  train 学到 {col:16s} 偏移集合 {sorted(learned[col])}")

est = {}
for col in ("act_dep_time", "act_arr_time"):
    tot = sum(1 for d, c in zip(drows, crows) if d.get(col, "") != c.get(col, ""))
    print(f"\n  --- {col}（脏 {tot:,}）")
    # R1：组内众数（原始字符串）；R2：组内最早时间（原始字符串）
    for name, guess in (("R1 组内众数", mode_map[col]), ("R2 组内最早时间", r2_map[col])):
        hit = sum(1 for d, c in zip(drows, crows)
                  if d.get(col, "") != c.get(col, "")
                  and guess.get(d.get("flight", "")) == c.get(col, ""))
        hit_t = sum(1 for d, c in zip(drows, crows)
                    if d.get(col, "") != c.get(col, "")
                    and minutes(guess.get(d.get("flight", ""), "")) == minutes(c.get(col, "")))
        print(f"     {name:20s} 字符串全对 {hit:5,}/{tot:,} ({hit/tot*100:6.2f}%)"
              f"   时间点对 {hit_t:5,} ({hit_t/tot*100:6.2f}%)")
    # R3 / R3'：平移众数 —— 只能给出「时间点」，回写还要还原原格式
    for name, S in (("R3 平移众数(train S)", learned[col]),
                    ("R3' 平移众数(0~25分)", set(range(26)))):
        g = shift_est(col, S)
        hit_t = sum(1 for d, c in zip(drows, crows)
                    if d.get(col, "") != c.get(col, "")
                    and g.get(d.get("flight", "")) == minutes(c.get(col, "")))
        print(f"     {name:20s} 时间点对 {hit_t:5,}/{tot:,} ({hit_t/tot*100:6.2f}%)"
              f"   （回写成原格式还需额外一步，字符串口径更低）")
    hit = sum(1 for d, c in zip(drows, crows)
              if d.get(col, "") != c.get(col, "") and std.get(d.get("flight", ""), {}).get(col) == c.get(col, ""))
    print(f"     {'上限(组标准值已知)':20s} 字符串全对 {hit:5,}/{tot:,} ({hit/tot*100:6.2f}%)")

print("\n=== ⑥ 投影：agent 现状 vs 只对 act_* 套用组标准值规则（test 切分）===")
rmap = defaultdict(list)
for r in rrows:
    rmap[r.get("tuple_id", "")].append(r)
idx = []
for d in drows:
    b = rmap.get(d.get("tuple_id", ""))
    idx.append(b.pop(0) if b else None)


def project(use_rule):
    tot = ex = ms = inc = coll = 0
    per = defaultdict(lambda: [0, 0])
    for i, (d, c) in enumerate(zip(drows, crows)):
        if d.get("tuple_id", "") not in test_keys:
            continue
        r = idx[i]
        if r is None:
            continue
        for dc, cc in zip(dcols, ccols):
            dv, cv = d.get(dc, ""), c.get(cc, "")
            rv = r.get(dc, "") or ""
            if use_rule and dv != cv:
                if dc == "act_dep_time":
                    rv = r2.get(d.get("flight", ""))
                elif dc == "act_arr_time":
                    rv = r1.get(d.get("flight", ""))
            if dv != cv:
                tot += 1
                per[dc][0] += 1
                if rv == cv:
                    ex += 1
                    per[dc][1] += 1
                elif rv == dv:
                    ms += 1
                else:
                    inc += 1
            elif rv != dv:
                coll += 1
    return tot, ex, ms, inc, coll, per


for tag, flag in (("A 现状(agent)", False), ("B +组标准值规则", True)):
    tot, ex, ms, inc, coll, per = project(flag)
    print(f"  {tag:14s} 脏 {tot:,} 修对 {ex:,} ({ex/tot*100:6.2f}%) 未检出 {ms:,} 改错 {inc:,} 误伤 {coll:,}")
print("  逐列：")
_, _, _, _, _, perA = project(False)
_, _, _, _, _, perB = project(True)
for col in COLS:
    t0, f0 = perA.get(col, [0, 0])
    t1, f1 = perB.get(col, [0, 0])
    print(f"    {col:16s} 脏 {t0:5,}  agent {f0:5,} ({f0/t0*100:6.2f}%)  →  规则后 {f1:5,} ({f1/t0*100:6.2f}%)")
