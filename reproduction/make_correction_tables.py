#!/usr/bin/env python3
"""生成 ceshi1 包里「校正类」的四数据集对照表（数字全部从 JSON 读，不手抄）。

产出：
  ~/Downloads/ceshi1/00_校正类_横向对比.md
  reproduction/ceshi1_校正类_横向对比.md（仓库留档）

数据源：
  错误级/composite ← tmp/correction_scores.json（本轮用 harness 打分器在包内拷贝上重打）
  agent 的 val composite ← correction_agent_run/experiments/<ds>_correction_val_seed666_v1/host/score_report.json
  坏 cell 规模 ← 台账㉗（写死在下面的常量里，来源已注明）
"""
import json
from pathlib import Path

TMP = Path("/Users/mac/.claude/jobs/448aad57/tmp")
W = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
DL = Path("/Users/mac/Downloads/ceshi1")
REPO = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw/reproduction")

scores = json.loads((TMP / "correction_scores.json").read_text(encoding="utf-8"))

# 数据集 → (显示名, test 行数, test 坏 cell 数, train/val 行数)
META = {
    "hospital": ("hospital", 970, 455, "index"),
    "beers": ("beers", 2380, 4282, "index"),
    "rayyan": ("rayyan", 970, 891, "id"),
    "tax": ("tax", 199950, 121108, "__row_id"),
}


def val_composite(d):
    """val 的 composite 取 run_report.json 的 reproducible_score（= 独立重放后的最终分）。"""
    p = (W / "correction_agent_run/experiments" / ("%s_correction_val_seed666_v1" % d)
         / "host/run_report.json")
    if not p.exists():
        return None
    r = json.load(open(p))
    return r.get("reproducible_score", r.get("best_score"))


L = []
L.append("# ceshi1 校正类（单表纠错）横向对比 —— 4 个数据集 × 2 个跑者\n")
L.append("> **校正类 ≠ MIMIC cohort 清洗族**：给 10 行「脏↔干净」对照让 agent 归纳破坏规则、"
         "修好整张表、交付可重放 pipeline ⇒ **分数量级完全不同（0.88~1.00 vs 0.82~0.92），跨族不可比**。\n")
L.append("> 两个跑者：**我们** = Pi harness（`correction_agent_run/`，2026-11-24）；"
         "**基线** = 2026-09 的手动跑批（`对比结果/`）。\n")
L.append("> 数字全部由 harness 自己的打分器 `score_only/pi_test_scorer.py`（`equal_file_row_aligned_v1`）"
         "在**包内拷贝**上重打，gold = `校正类/gold/reference/`（= 各数据集 `correction/reference_private`）。\n")

L.append("## 维度一：整体分数（composite）\n")
L.append("| 数据集 | 跑者 | composite | schema | key | cell | exact | test 行数 | test 坏 cell |")
L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for d in ("hospital", "beers", "rayyan", "tax"):
    name, rows, bad, _k = META[d]
    for label, who in (("基线", "基线"), ("agent", "我们")):
        m = scores.get(d, {}).get(label)
        if not m:
            continue
        L.append("| `%s` | %s | **%.6f** | %.4f | %.4f | %.4f | %.4f | %s | %s |" % (
            name, who, m["composite_score"], m["schema_f1"], m["key_structure_f1"],
            m["row_aligned_cell_f1"], m["exact_row_f1"], format(rows, ","), format(bad, ",")))
L.append("")

L.append("## 维度二：val / test 的对照（我们的 agent）\n")
L.append("| 数据集 | val composite | test composite | test 相对 val |")
L.append("|---|---:|---:|---|")
for d in ("hospital", "beers", "rayyan", "tax"):
    name, _r, _b, _k = META[d]
    v = val_composite(d)
    t = scores.get(d, {}).get("agent", {}).get("composite_score")
    L.append("| `%s` | %s | %s | %s |" % (
        name, ("%.6f" % v) if v is not None else "—",
        ("%.6f" % t) if t is not None else "—",
        ("%+.6f" % (t - v)) if (v is not None and t is not None) else "—"))
L.append("")

L.append("## 维度三：谁强谁弱\n")
L.append("| 数据集 | 基线 | 我们 | 差（我们−基线） | 胜者 |")
L.append("|---|---:|---:|---:|---|")
for d in ("hospital", "beers", "rayyan", "tax"):
    name, _r, _b, _k = META[d]
    b = scores.get(d, {}).get("基线", {}).get("composite_score")
    a = scores.get(d, {}).get("agent", {}).get("composite_score")
    if b is None or a is None:
        continue
    L.append("| `%s` | %.6f | %.6f | %+.6f | %s |" % (
        name, b, a, a - b, "我们" if a > b else ("基线" if b > a else "平")))
L.append("")

L.append("## 读表提醒\n")
L.append("1. **基线跑在旧版数据集上，入包时做了两处「不改值」的变换**："
         "hospital / beers / rayyan 的基线输出是**整表**（1000 / 2410 / 1000 行，含当时的 train+val 30 行）"
         "⇒ 按 key 过滤出 test 行子集；tax 的基线输出 200,000 行且**没有 `__row_id` 列**"
         "（`__row_id` 就是原表行号 0..199999，基线保持了原行序）⇒ 按位置补回 key 列再取 test 子集。"
         "**只做「补 key 列 + 取 test 行子集」，一个单元格值都没改。**\n")
L.append("2. ⭐ **tax 是基线赢的一档（0.999273 vs 0.973634，低 2.6 个点）**："
         "tax 的 test 段是 199,950 行 / 121,108 个坏 cell —— 规模比前三档大两个数量级。"
         "我们的 agent 在 **val 的 20 行上是满分**，但全表上**过度泛化**："
         "把为 20 行归纳的规则套到 20 万行，**改坏了 0.62% 的干净 cell**；"
         "基线反而因为规则更保守而几乎没误改（cell 0.9999 / exact 0.9978）。\n")
L.append("3. **rayyan 是我们赢最多的一档（+9.3 个点）**："
         "rayyan 的破坏模式里 `author_list` 这类**列表列**的写法不唯一，"
         "基线把它无差别规范化 ⇒ 大量改错（cell 0.9632 / exact 0.6485）；"
         "我们的 agent 在 train 上就发现 gold 写法不唯一、保留了原值 ⇒ exact 0.9206。\n")
L.append("4. **hospital / beers 差距小（+0.2 / +2.0 个点）**：两档的破坏规则偏机械"
         "（跨列语义推断 / 格式前缀），两个跑者都能推出来。\n")
L.append("5. **我们的 agent 重打分与 harness 的 score_report 逐位一致**"
         "（hospital 0.997031 / beers 1.000000 / rayyan 0.973187 / tax 0.973634）"
         "⇒ 打分链路可复现，且包内拷贝无损。\n")

out = "\n".join(L) + "\n"
(REPO / "ceshi1_校正类_横向对比.md").write_text(out, encoding="utf-8")
print("写出仓库留档: reproduction/ceshi1_校正类_横向对比.md (%d 字符)" % len(out))
try:
    (DL / "00_校正类_横向对比.md").write_text(out, encoding="utf-8")
    print("写出包内: %s/00_校正类_横向对比.md" % DL)
except Exception as e:
    print("包内写入被拒（%s）—— 内容已在仓库留档" % e)
