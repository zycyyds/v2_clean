#!/usr/bin/env python3
"""查 `对比结果/` 那批旧基线产出与当前 test gold 的可比性。

只出聚合（行数、命中数），不出行级数据。
"""
import csv
from pathlib import Path

csv.field_size_limit(2 ** 31)
C = Path("/Users/mac/PycharmProjects/v2_clean/datasets/correction")
BASE = Path("/Users/mac/PycharmProjects/v2_clean/对比结果")

DS = {"hospital": ("raha_hospital_curated10_val20_dense_v1", "index"),
      "beers": ("beers_curated10_val20_v1", "index"),
      "rayyan": ("rayyan_curated10_val20_v1", "id"),
      "tax": ("tax_curated10_val20_v1", "__row_id")}
BASEFILE = {"hospital": ("hospital", "result_package/cleaned.csv"),
            "beers": ("beers_correction_v1", "result_package/beers.csv"),
            "rayyan": ("rayyan_clean_v1", "result_package/cleaned.csv"),
            "tax": ("tax_correction_v1", "result_package/tax.csv")}


def read(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        rows = list(rd)
    return hdr, rows


for d, (ds, key) in DS.items():
    gold_p = C / ds / "correction/reference_private" / ("%s.csv" % d)
    bdir, brel = BASEFILE[d]
    base_p = BASE / bdir / brel
    gh, grows = read(gold_p)
    bh, brows = read(base_p)
    gk = gh.index(key) if key in gh else None
    bk = bh.index(key) if key in bh else None
    print("=== %s" % d)
    print("   gold: %d 行, key 列 %s (%s)" % (len(grows), key, "有" if gk is not None else "无"))
    print("   base: %d 行, key 列 %s (%s), 列数 %d vs gold %d"
          % (len(brows), key, "有" if bk is not None else "无", len(bh), len(gh)))
    if gk is None or bk is None:
        # tax：靠位置对齐（__row_id 当作原表行号）
        rid = [r[gk] for r in grows] if gk is not None else []
        print("   ⇒ 无共同 key 列；gold 前 3 个 __row_id: %s / 末 3 个: %s"
              % (rid[:3], rid[-3:]))
        if all(x.isdigit() for x in rid):
            nums = sorted(int(x) for x in rid)
            print("   ⇒ __row_id 全为数字，最小 %d 最大 %d，base 行数 %d ⇒ 可按位置取子集"
                  % (nums[0], nums[-1], len(brows)))
            # 抽查：按位置取 base 的这些行，与 gold 的第一列比
            idx = [int(x) for x in rid]
            same_cols = [c for c in gh if c in bh]
            gi = [gh.index(c) for c in same_cols]
            bi = [bh.index(c) for c in same_cols]
            hit = sum(1 for k, r in zip(idx, grows)
                      if k < len(brows) and [brows[k][j] for j in bi] == [r[j] for j in gi])
            print("   ⇒ 按位置取子集后逐格完全一致的行: %d / %d (%.1f%%)"
                  % (hit, len(grows), 100.0 * hit / len(grows)))
        continue
    gset = {r[gk] for r in grows}
    bset = {r[bk] for r in brows}
    print("   ⇒ gold 的 key 有 %d 个出现在 base 里 (%.1f%%); base 多出的 key %d 个"
          % (len(gset & bset), 100.0 * len(gset & bset) / len(gset), len(bset - gset)))
    # 抽查按 key 对齐后逐格一致的行
    common = [c for c in gh if c in bh]
    gi = [gh.index(c) for c in common]
    bi = [bh.index(c) for c in common]
    bmap = {}
    for r in brows:
        bmap.setdefault(r[bk], r)
    hit = sum(1 for r in grows if r[gk] in bmap
              and [bmap[r[gk]][j] for j in bi] == [r[j] for j in gi])
    print("   ⇒ 按 key 对齐后逐格完全一致的行: %d / %d (%.1f%%)"
          % (hit, len(grows), 100.0 * hit / len(grows)))
