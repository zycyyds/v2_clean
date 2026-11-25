#!/usr/bin/env python3
"""把校正类（hospital / beers / rayyan / tax）的跑批结果拷进 ceshi1 包。

拷两类：
  ① 我们的 agent 的 test 结果包（correction_agent_run/.../host/test_result_package）
  ② test gold（datasets/correction/<ds>/correction/reference_private）

逐文件 md5 回验；目标已存在则直接退出（不覆盖）。
"""
import hashlib
import shutil
import sys
from pathlib import Path

W = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
C = Path("/Users/mac/PycharmProjects/v2_clean/datasets/correction")
DST = Path("/Users/mac/Downloads/ceshi1/校正类")

DS = {"hospital": "raha_hospital_curated10_val20_dense_v1",
      "beers": "beers_curated10_val20_v1",
      "rayyan": "rayyan_curated10_val20_v1",
      "tax": "tax_curated10_val20_v1"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if DST.exists():
    print("FATAL: 目标已存在，不覆盖:", DST)
    sys.exit(2)
(DST / "gold" / "reference").mkdir(parents=True)

pairs = []
for d, ds in DS.items():
    src = (W / "correction_agent_run/experiments" /
           ("%s_correction_test_seed666_v1" % d) / "host/test_result_package" / ("%s.csv" % d))
    gold = C / ds / "correction/reference_private" / ("%s.csv" % d)
    if not src.exists():
        print("FATAL: 源不存在:", src)
        sys.exit(2)
    dst = DST / d / "结果包" / ("%s.csv" % d)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    gdst = DST / "gold/reference" / ("%s.csv" % d)
    shutil.copy2(gold, gdst)
    pairs.append((d, src, dst, gold, gdst))

ok = True
for d, src, dst, gold, gdst in pairs:
    m1 = md5(src) == md5(dst)
    m2 = md5(gold) == md5(gdst)
    ok &= m1 and m2
    print("  %-9s 结果包 md5 %s ; gold md5 %s" % (d, "✓" if m1 else "✗", "✓" if m2 else "✗"))
print("md5 全量回验：", "全部一致 ✅" if ok else "有差异 ❌")
sys.exit(0 if ok else 1)
