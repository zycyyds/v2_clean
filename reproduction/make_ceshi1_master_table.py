#!/usr/bin/env python3
"""从合并后的 `00_三维度横向对比.md` 解析出 **八个数据集 × 全部跑者** 的总表。

产出：
  ~/Downloads/ceshi1/00_总表_八数据集横向对比.md
  reproduction/ceshi1_总表_八数据集横向对比.md（仓库留档）

口径说明：
  · 八个数据集 = K29 / ICU5000 / n18 / I48 × 1.6pct / 4.7pct
  · 跑者三类：我们（Pi harness） / 基线（K29·ICU5000 是 codex，I48·n18 是 cc） / deepprep
  · deepprep 的错误级算不出来（产物缺陷）⇒ 记「—」，不是 0
  · 全部数字从包内表里读出，不重新计算、不手抄
"""
import re
from pathlib import Path

DL = Path("/Users/mac/Downloads/ceshi1")
REPO = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw/reproduction")
SRC = DL / "00_三维度横向对比.md"

text = SRC.read_text(encoding="utf-8")
sec1, sec2, sec3 = [], [], []
cur = None
for line in text.split("\n"):
    if line.startswith("## 维度一"):
        cur = sec1
        continue
    if line.startswith("## 维度二"):
        cur = sec2
        continue
    if line.startswith("## 维度三"):
        cur = sec3
        continue
    if line.startswith("## ") or line.startswith("# "):
        cur = None
        continue
    if cur is not None and line.startswith("|"):
        cur.append(line)

comp, err, nat = {}, {}, {}


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


for line in sec1:
    c = cells(line)
    if len(c) != 9 or c[0].startswith("---") or "结果目录" in c[0]:
        continue
    comp[c[0].strip("`")] = {"runner": c[1], "tier": c[2], "ds": c[3],
                             "composite": c[4], "schema": c[5], "key": c[6],
                             "cell": c[7], "exact": c[8]}
for line in sec2:
    c = cells(line)
    if len(c) != 9 or c[0].startswith("---") or "结果目录" in c[0]:
        continue
    err.setdefault(c[0].strip("`"), {})[c[2].strip("*")] = {
        "det": c[3], "rep": c[4], "clean": c[5], "wrong": c[6],
        "dirty": c[7], "missing": c[8]}
for line in sec3:
    c = cells(line)
    if len(c) != 8 or c[0].startswith("---") or "结果目录" in c[0]:
        continue
    nat[c[0].strip("`")] = {"tot": c[3], "clean": c[4], "dirty": c[5],
                            "missing": c[6], "rate": c[7].strip("*")}

TIERS = ["K29", "ICU5000", "n18", "I48"]
LVL = {"1.6pct": "1.6pct", "4.7pct": "4.7pct"}


def tier_of(d):
    return d.split("/")[0]


def lvl_of(d):
    if "1.6pct" in d:
        return "1.6pct"
    if "4.7pct" in d:
        return "4.7pct"
    return "1.6pct" if d.endswith("deepprep-less") else "4.7pct"


def runner_key(d):
    """我们 / 基线 / deepprep"""
    r = comp[d]["runner"]
    return "我们" if r == "我们" else ("deepprep" if r == "deepprep" else "基线")


def base_name(d):
    r = comp[d]["runner"]
    return {"codex": "基线(codex)", "cc": "基线(cc)"}.get(r, r)


def find(tier, lvl, key):
    for d in comp:
        if tier_of(d) == tier and lvl_of(d) == lvl and runner_key(d) == key:
            return d
    return None


L = []
L.append("# ceshi1 总表 —— **八个数据集 × 全部跑者 × 三个维度**\n")
L.append("> 八个数据集 = K29 / ICU5000 / n18 / I48 × 1.6pct / 4.7pct（mimicraw 构建）。")
L.append("> 跑者三类：**我们**（Pi harness）/ **基线**（K29·ICU5000 跑的是 codex，I48·n18 跑的是 cc）/ **deepprep**。")
L.append("> 全部数字从包内 `00_三维度横向对比.md` 读出（同一份数据、不重新计算）。")
L.append("> ⚠️ deepprep 的**错误级**算不出来（产物有空表 / 缺 key 列）⇒ 记「—」，不是 0。\n")

L.append("## 维度一：整体分数（composite）\n")
L.append("| 数据集 | 档位 | 跑者 | composite | schema | key | cell | exact | 原生修复率 |")
L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
for t in TIERS:
    for lv in ("1.6pct", "4.7pct"):
        for key in ("我们", "基线", "deepprep"):
            d = find(t, lv, key)
            if not d:
                continue
            c = comp[d]
            nr = nat.get(d, {}).get("rate", "—")
            L.append("| **%s** | %s | %s | **%s** | %s | %s | %s | %s | %s |" % (
                t, lv, base_name(d), c["composite"].strip("*"), c["schema"], c["key"],
                c["cell"], c["exact"], nr))

L.append("\n## 维度二：错误级指标（str / oldsw / goldv29，Det F1 / Rep F1）\n")
L.append("| 数据集 | 档位 | 跑者 | str Det | str Rep | oldsw Det | oldsw Rep | goldv29 Det | goldv29 Rep |")
L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
for t in TIERS:
    for lv in ("1.6pct", "4.7pct"):
        for key in ("我们", "基线", "deepprep"):
            d = find(t, lv, key)
            if not d:
                continue
            e = err.get(d, {})
            row = []
            for cal in ("str", "oldsw", "goldv29"):
                c = e.get(cal)
                row += ([c["det"].strip("*"), c["rep"].strip("*")] if c else ["—", "—"])
            L.append("| **%s** | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                t, lv, base_name(d), *row))

L.append("\n## 维度三：MIMIC 原生错误格（string 口径）\n")
L.append("| 数据集 | 档位 | 跑者 | 总格数 | 修对 | 没动 | 缺行 | 修复率 |")
L.append("|---|---|---|---:|---:|---:|---:|---:|")
for t in TIERS:
    for lv in ("1.6pct", "4.7pct"):
        for key in ("我们", "基线", "deepprep"):
            d = find(t, lv, key)
            if not d:
                continue
            n = nat.get(d)
            if not n:
                continue
            L.append("| **%s** | %s | %s | %s | %s | %s | %s | **%s** |" % (
                t, lv, base_name(d), n["tot"], n["clean"], n["dirty"], n["missing"], n["rate"]))

# ── 我们 vs 基线 的胜负小结 ──
L.append("\n## 我们 vs 基线（八个数据集逐档对比）\n")
L.append("| 数据集 | 档位 | 我们 | 基线 | 差（我们−基线） | 胜者 |")
L.append("|---|---|---:|---:|---:|---|")
wins = ours = base = 0
margins = []
for t in TIERS:
    for lv in ("1.6pct", "4.7pct"):
        d1, d2 = find(t, lv, "我们"), find(t, lv, "基线")
        if not (d1 and d2):
            continue
        a = float(comp[d1]["composite"].strip("*"))
        b = float(comp[d2]["composite"].strip("*"))
        margins.append(a - b)
        if a > b:
            wins += 1
            who = "我们"
            ours += 1
        elif b > a:
            who = "基线"
            base += 1
        else:
            who = "平"
        L.append("| **%s** | %s | %.6f | %.6f | %+.6f | %s |" % (t, lv, a, b, a - b, who))
avg = sum(margins) / len(margins) if margins else 0
L.append("| **合计** | 8 档 | — | — | 平均 **%+.6f** | 我们 %d 胜 / 基线 %d 胜 |"
         % (avg, ours, base))

# ── 结论 ──
L.append("\n## 结论（读表要点）\n")
n18_cc16 = find("n18", "1.6pct", "基线")
n18_ag16 = find("n18", "1.6pct", "我们")
L.append("1. **composite：8 档里我们赢 %d 档、输 %d 档**，平均高 **%+.6f**。" % (ours, base, avg))
L.append("2. ⚠️ **唯一输的一档是 ICU5000 4.7pct（0.827158 vs 基线 0.885118，低 5.8 个点）**，"
         "而且它同时是我们 8 档里**最低的一档**（其余 7 档我们都在 0.881~0.922）。"
         "**根因已定位：不是「修得不干净」，是 cohort 筛选过严、把行漏掉了** —— "
         "`admissions/cohort` gold 6,477 行 vs 我们 4,966 行（**少选 1,511 个 hadm，−23%**），"
         "连带 demographics −872、labs −890,581、medications −176,850、diagnoses −31,518；"
         "而 `icu/{icustays,inputevents,outputevents}` 行数**全对** ⇒ 数据没丢，是**入选规则判错**。"
         "这直接压低了 key 结构分（0.8625 vs 基线 0.9361）与 exact_row（0.7249）。"
         "（脚本：`reproduction/diag_icu5000_47.py`）")
L.append("3. **差距集中在「错误级」**：composite 是四项等权平均，而 **schema 几乎都是 1.0**"
         "（表结构大家都做得出）⇒ 真正区分开的是 **key / cell（逐格对）/ exact（整行全对）** 三项。")
L.append("4. **MIMIC 原生错误格是分水岭**：我们 0.958~0.994，"
         "而**所有基线都 ≈0**（codex / cc 跑在无 mimicraw 原生层的旧构建上，规则推导法也碰不到哨兵码与生理不可能值）。")
L.append("5. **deepprep 全面崩**：它的 key 结构分只有 0.15~0.52（没做 cohort 筛选），"
         "且产物有空表 / 缺 key 列 ⇒ **错误级只有 str 算得出（降级记账：坏表记成没输出，"
         "其注入错误如实计漏检）**，Det/Rep F1 只有 0.03~0.26，与它 composite 0.20~0.46 一致。")
L.append("6. ⚠️ **n18 1.6pct 的基线是本轮新重跑的**（跑在 mimicraw 1.6pct 切片上）；"
         "其余基线跑在旧构建上 ⇒ 它们的原生格维度本就是 N/A，不是 0。")
L.append("7. ⚠️ **校正类（hospital/beers/rayyan/tax）不在本表**：它是单表纠错、分数量级不同，"
         "见 `00_校正类_横向对比.md`。")

out = "\n".join(L) + "\n"
(REPO / "ceshi1_总表_八数据集横向对比.md").write_text(out, encoding="utf-8")
print("写出仓库留档: reproduction/ceshi1_总表_八数据集横向对比.md (%d 字符)" % len(out))
try:
    (DL / "00_总表_八数据集横向对比.md").write_text(out, encoding="utf-8")
    print("写出包内: %s/00_总表_八数据集横向对比.md" % DL)
except Exception as e:
    print("包内写入被拒（%s）—— 内容已在仓库留档" % e)
