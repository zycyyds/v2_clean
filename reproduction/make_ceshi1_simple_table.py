#!/usr/bin/env python3
"""按「结果目录 / composite / Det f1 / Rep f1 / mimic 原生错误」的版式，出 n18 与 I48 两档的简表。

版式照用户给的示例（每个档位内三行：our / DeepPrep / 基线，**每列在该档内最优值加粗**）。

口径（与示例图可能不同，示例图里 Det/Rep 两档用了不同口径、基线行的数字与仓库对不上）：
  · Det f1 / Rep f1 = **goldv29 口径**（仓库 ㉘ 读表提醒 #1：只有 goldv29 可跨档比，
    str 口径会把「值对但写法不同」判成改错）
  · mimic 原生错误列 = **原生错误格的修复率**（修对格数 / 候选可修格数）
  · composite = harness 打分器 `equal_file_row_aligned_v1`

数据从包内合并后的 `00_三维度横向对比.md` 解析（与包内其它表同源）。
"""
import re
from pathlib import Path

DL = Path("/Users/mac/Downloads/ceshi1")
REPO = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw/reproduction")
SRC = DL / "00_三维度横向对比.md"

text = SRC.read_text(encoding="utf-8")
secs = {"一": [], "二": [], "三": []}
cur = None
for line in text.split("\n"):
    m = re.match(r"^## 维度(.)", line)
    if m:
        cur = secs[m.group(1)]
        continue
    if line.startswith("## ") or line.startswith("# "):
        cur = None
        continue
    if cur is not None and line.startswith("|"):
        cur.append(line)


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


comp, err, nat = {}, {}, {}
for line in secs["一"]:
    c = cells(line)
    if len(c) == 9 and not c[0].startswith("---") and "结果目录" not in c[0]:
        comp[c[0].strip("`")] = {"runner": c[1], "tier": c[2], "composite": c[4].strip("*")}
for line in secs["二"]:
    c = cells(line)
    if len(c) == 9 and not c[0].startswith("---") and "结果目录" not in c[0]:
        err.setdefault(c[0].strip("`"), {})[c[2].strip("*")] = (c[3].strip("*"), c[4].strip("*"))
for line in secs["三"]:
    c = cells(line)
    if len(c) == 8 and not c[0].startswith("---") and "结果目录" not in c[0]:
        nat[c[0].strip("`")] = c[7].strip("*")

TIERS = ["n18", "I48"]
LEVELS = [("less 1.6%", "1.6pct", "deepprep-less"),
          ("more 4.7%", "4.7pct", "deepprep-more")]


def pick(tier, lvl, who, dp):
    """找到那一行：who ∈ 我们/基线/deepprep"""
    for d in comp:
        t = d.split("/")[0]
        if t != tier:
            continue
        if who.lower() == "deepprep":
            if d.endswith(dp):
                return d
        else:
            r = comp[d]["runner"]
            ok = (who == "我们" and r == "我们") or (who == "基线" and r in ("codex", "cc"))
            if ok and (("1.6pct" in d and lvl == "1.6pct") or ("4.7pct" in d and lvl == "4.7pct")):
                return d
    return None


LABEL = {"我们": "our", "DeepPrep": "DeepPrep", "基线": "cc"}
ORDER = ["我们", "DeepPrep", "基线"]


def num(x):
    return float(x)


def fmt_best(vals, i, fmt="%.4f"):
    """vals = [(行标签, 数值串)]；该列最大值加粗。"""
    nums = []
    for _, v in vals:
        try:
            nums.append(num(v))
        except Exception:
            nums.append(None)
    best = max([n for n in nums if n is not None], default=None)
    out = []
    for (lab, v), n in zip(vals, nums):
        s = "—" if n is None else (fmt % n if fmt else v)
        if n is not None and best is not None and n == best:
            s = "**%s**" % s
        out.append((lab, s))
    return out


L = []
L.append("# ceshi1 简表 —— **n18 / I48**（每档：our · DeepPrep · cc × 1.6pct · 4.7pct）\n")
L.append("> **三个错误级口径并排**，供挑选：**str**（纯字符串比对，log 的 clean_value 当判据）、"
         "**oldsw**（数值比对 + 不还原 key + 不排除 gold 外行）、"
         "**goldv29**（gold 的真实值当判据，**唯一可跨档比**的口径 —— str 会把「值对但写法不同」判成改错）。")
L.append("> **mimic 原生错误 = 原生错误格的修复率**（修对格数 / 候选可修格数）；"
         "composite = harness 打分器 `equal_file_row_aligned_v1`。")
L.append("> **粗体 = 该档（less / more）内三行里的最优值**。"
         "DeepPrep 四行的错误级算不出来（产物有空表 / 缺 key 列）⇒ 记「—」，不是 0。\n")

for tier in TIERS:
    L.append("## %s\n" % tier)
    L.append("| 结果目录 | composite | str Det f1 | str Rep f1 | oldsw Det f1 | oldsw Rep f1 "
             "| goldv29 Det f1 | goldv29 Rep f1 | mimic原生错误修复率 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lv_label, lvl, dp in LEVELS:
        rows = []
        for who in ORDER:
            d = pick(tier, lvl, who, dp)
            if not d:
                continue
            e = err.get(d, {})
            rows.append((who, d, comp[d]["composite"],
                         e.get("str", ("—", "—"))[0], e.get("str", ("—", "—"))[1],
                         e.get("oldsw", ("—", "—"))[0], e.get("oldsw", ("—", "—"))[1],
                         e.get("goldv29", ("—", "—"))[0], e.get("goldv29", ("—", "—"))[1],
                         nat.get(d, "—")))
        if not rows:
            continue
        # 每列在该 level 组内取最优（粗体）
        cols = []
        for idx, fmt in ((2, "%.6f"), (3, "%.4f"), (4, "%.4f"), (5, "%.4f"),
                         (6, "%.4f"), (7, "%.4f"), (8, "%.4f"), (9, "%.4f")):
            cols.append(fmt_best([(r[0], r[idx]) for r in rows], idx, fmt))
        for i, r in enumerate(rows):
            vals = [c[i][1] for c in cols]
            L.append("| %s (%s) | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (LABEL[r[0]], lv_label, *vals))
        L.append("| | | | | | | | | |")
    while L and L[-1] == "| | | | | | | | | |":
        L.pop()
    L.append("")

out = "\n".join(L) + "\n"
(REPO / "ceshi1_简表_n18与I48.md").write_text(out, encoding="utf-8")
print("写出仓库留档: reproduction/ceshi1_简表_n18与I48.md (%d 字符)" % len(out))
try:
    (DL / "00_简表_n18与I48.md").write_text(out, encoding="utf-8")
    print("写出包内: %s/00_简表_n18与I48.md" % DL)
except Exception as e:
    print("包内写入被拒（%s）—— 内容已在仓库留档" % e)
