#!/usr/bin/env python3
"""核 `校正类/gold/` 里的表是不是**错误全部修好**的干净表。

口径：用注入器留下的**逐 cell 差分档案** `host_private/cell_diff.jsonl`（host 专用，
每条记一行：脏值 → 干净值 + key + 列名 + split），
对 test 段（split=correction）的每一条坏 cell：
  ① gold 的对应格是否 == clean_value   （⇒ 该格已修好）
  ② gold 的对应格是否 != dirty_value   （⇒ 没残留脏值）
  ③ 顺带核 raw 的对应格是否 == dirty_value（⇒ 档案与脏输入对得上）
只出聚合计数，不出行级数据。
"""
import csv
import json
from pathlib import Path

csv.field_size_limit(2 ** 31)
C = Path("/Users/mac/PycharmProjects/v2_clean/datasets/correction")
GOLD = Path("/Users/mac/Downloads/ceshi1/校正类/gold/reference")

DS = {"hospital": "raha_hospital_curated10_val20_dense_v1",
      "beers": "beers_curated10_val20_v1",
      "rayyan": "rayyan_curated10_val20_v1",
      "tax": "tax_curated10_val20_v1"}


def read(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


print("%-9s %8s %10s %10s %10s %10s" % ("数据集", "坏cell", "gold=干净", "gold≠脏值", "raw=脏值", "key 未命中"))
allok = True
for d, ds in DS.items():
    diff_p = C / ds / "host_private/cell_diff.jsonl"
    raw_p = C / ds / "correction/raw" / ("%s.csv" % d)
    gold_p = GOLD / ("%s.csv" % d)
    if not diff_p.exists():
        print("%-9s (无 cell_diff 档案)" % d)
        continue
    entries = []
    for line in diff_p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("split") == "correction":
            entries.append(o)
    # key 列名 = 除了 column/clean_value/dirty_value/split 之外的第一个字段
    keycol = None
    for k in entries[0]:
        if k not in ("column", "clean_value", "dirty_value", "split"):
            keycol = k
            break
    gh, grows = read(gold_p)
    rh, rrows = read(raw_p)
    gmap = {r.get(keycol): r for r in grows}
    rmap = {r.get(keycol): r for r in rrows}
    n = len(entries)
    n_clean = n_dirty = n_raw = n_miss = 0
    for e in entries:
        k = e.get(keycol)
        col = e.get("column")
        g = gmap.get(k)
        r = rmap.get(k)
        if g is None:
            n_miss += 1
            continue
        gv = g.get(col, "")
        if gv == e.get("clean_value", ""):
            n_clean += 1
        if gv != e.get("dirty_value", ""):
            n_dirty += 1
        if r is not None and r.get(col, "") == e.get("dirty_value", ""):
            n_raw += 1
    ok = (n_clean == n) and (n_dirty == n) and (n_miss == 0)
    allok &= ok
    print("%-9s %8d %10d %10d %10d %10d%s"
          % (d, n, n_clean, n_dirty, n_raw, n_miss, "" if ok else "   ⚠️"))
print("\n结论：", "四档 gold 的 test 段坏 cell **全部已修好，且无残留脏值** ✅" if allok else "存在未修好的格 ❌")
