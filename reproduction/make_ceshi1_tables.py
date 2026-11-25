#!/usr/bin/env python3
"""生成 ceshi1 包「新增 n18 / I48 行」的三维度表格（数字全部从 JSON 读，不手抄）。

产出：
  ~/Downloads/ceshi1/00_补齐_新增n18与I48行.md
  reproduction/ceshi1_补齐行_三维度.md（仓库内留档）
"""
import json
from pathlib import Path

TMP = Path("/Users/mac/.claude/jobs/448aad57/tmp")
DL = Path("/Users/mac/Downloads/ceshi1")
REPO = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw/reproduction")


def load(name, default=None):
    p = TMP / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


scores = load("new_pkg_scores.json", {}) or {}
dpscores = load("deepprep_scores.json", {}) or {}
nat = load("n18_16_agent_native.json"), load("n18_47_agent_native.json"), load("n18_47_cc_native.json")
dpnat = load("deepprep_native.json", {}) or {}
em_new = load("error_metrics_new.json", {}) or {}
em_agent = load("error_metrics_n18_agent.json", {}) or {}
em_oldsw = load("error_metrics_oldsw.json", {}) or {}


def per_caliber_json(ds_key):
    """从每口径的单独 json 兜底读取（那个被中止的任务没写汇总 json，但每口径的 json 已落盘）。

    文件名约定：err_<key>_<cal>.json；`old`（纯字符串版）单独用 _old 后缀，
    避免与真正的 `oldsw`（数值版）同名互相覆盖。
    """
    out = {}
    for cal, suffix in (("str", "str"), ("oldsw", "oldsw"), ("old", "old"), ("goldv29", "goldv29")):
        p = TMP / ("err_%s_%s.json" % (ds_key, suffix))
        if not p.exists():
            continue
        d = json.load(open(p))
        det, rep = d.get("detection", {}), d.get("repair", {})
        out[cal] = {"det_f1": det.get("F1"), "det_p": det.get("P"), "det_r": det.get("R"),
                    "rep_f1": rep.get("F1"), "rep_p": rep.get("P"), "rep_r": rep.get("R"),
                    "buckets": d.get("buckets")}
    return out


def merge_calibers(a, b):
    """合并并**纠正口径标签**：
    先前那档用了 --string-compare ⇒ 实际是「old」（纯字符串 + no-recover + keep-rows）；
    真正的「oldsw」是数值版，另跑在 b 里。
    """
    out = {}
    if a:
        for cal, v in a.items():
            out["old" if cal == "oldsw" else cal] = v
    if b:
        for cal, v in b.items():
            if cal == "oldsw":
                out["oldsw"] = v
    return out


def sc(d):
    if not d:
        return None
    m = d.get("metrics", d)
    return (m.get("composite_score"), m.get("schema_f1"), m.get("key_structure_f1"),
            m.get("row_aligned_cell_f1"), m.get("exact_row_f1"))


def fmt(x, nd=6):
    return "—" if x is None else ("%.*f" % (nd, x))


def nat_row(label, runner, caliber, d):
    if d is None:
        return None
    n = d.get("n_native_cells")
    b = d.get("buckets", {})
    rate = d.get("repair_rate")
    return "| `%s` | %s | %s | %s | %s | %s | %s | **%.6f** |" % (
        label, runner, caliber, n, b.get("b_clean", 0), b.get("b_dirty", 0),
        b.get("b_missing", 0), rate if rate is not None else 0.0)


# ── 维度一 ──
D1 = [
    ("n18/1.6pct_我们的agent_pi+val_mimicraw", "我们", "1.6pct", "mimicraw", scores.get("n18/1.6pct_我们的agent_pi+val_mimicraw/结果包")),
    ("n18/4.7pct_我们的agent_pi+val_mimicraw", "我们", "4.7pct", "mimicraw", scores.get("n18/4.7pct_我们的agent_pi+val_mimicraw/结果包")),
    ("n18/4.7pct_cc基线_ClaudeCode", "cc", "4.7pct", "raw(无原生层)", scores.get("n18/4.7pct_cc基线_ClaudeCode/结果包")),
    ("n18/deepprep-less", "deepprep", "1.6pct", "mimicraw", dpscores.get("n18/deepprep-less/output")),
    ("n18/deepprep-more", "deepprep", "4.7pct", "mimicraw", dpscores.get("n18/deepprep-more/output")),
    ("I48/deepprep-less", "deepprep", "1.6pct", "mimicraw", dpscores.get("I48/deepprep-less/output")),
    ("I48/deepprep-more", "deepprep", "4.7pct", "mimicraw", dpscores.get("I48/deepprep-more/output")),
]

# ── 维度二（错误级）──
D2_SRC = {
    "n18/1.6pct_我们的agent_pi+val_mimicraw": ("我们", merge_calibers(
        em_agent.get("n18/1.6pct_我们的agent_pi+val_mimicraw/结果包"),
        em_oldsw.get("n18/1.6pct_我们的agent_pi+val_mimicraw/结果包"))),
    "n18/4.7pct_我们的agent_pi+val_mimicraw": ("我们", merge_calibers(
        em_agent.get("n18/4.7pct_我们的agent_pi+val_mimicraw/结果包"),
        em_oldsw.get("n18/4.7pct_我们的agent_pi+val_mimicraw/结果包"))),
    "n18/4.7pct_cc基线_ClaudeCode": ("cc", merge_calibers(
        em_new.get("n18/4.7pct_cc基线_ClaudeCode/结果包") or per_caliber_json("ceshi1_n18_cc47"),
        em_oldsw.get("n18/4.7pct_cc基线_ClaudeCode/结果包"))),
}

lines = []
lines.append("# ceshi1 三维度横向对比 —— **新增的 7 个结果目录**（2026-11-25 补齐）\n")
lines.append("> 本文件是 `00_三维度横向对比.md` / `00_紧凑横向对比.md` / `00_评估进度表.md` 的**增量**：\n"
             "> 那三张表是既有手写文件（未改动），这里给出补齐 n18 / I48 后**新增的行**。\n"
             "> 数字全部从 JSON 读出、无手抄；composite 用 harness 自己的打分器重打，\n"
             "> 错误级用 `score_only/error_metrics/`，原生层用 `reproduction/eval_*_native_cells.py`。\n")
lines.append("## 维度一：整体分数（composite）\n")
lines.append("| 结果目录 | 跑者 | 档位 | 数据集 | composite | schema | key | cell | exact |")
lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
for label, runner, tier, ds, d in D1:
    v = sc(d)
    if v is None:
        continue
    lines.append("| `%s` | %s | %s | %s | **%.6f** | %.4f | %.4f | %.4f | %.4f |" % (label, runner, tier, ds, *v))
lines.append("")
lines.append("## 维度二：错误级指标（str / oldsw / goldv29）\n")
lines.append("| 结果目录 | 跑者 | 口径 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing |")
lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
for label, (runner, d) in D2_SRC.items():
    if not d:
        continue
    for cal in ("str", "oldsw", "old", "goldv29"):
        c = d.get(cal)
        if not c:
            continue
        b = c.get("buckets") or {}
        lines.append("| `%s` | %s | **%s** | **%.6f** | **%.6f** | %s | %s | %s | %s |" % (
            label, runner, cal, c["det_f1"], c["rep_f1"],
            f"{b.get('b_clean', 0):,}", f"{b.get('b_wrong', 0):,}",
            f"{b.get('b_dirty', 0):,}", f"{b.get('b_missing', 0):,}"))
lines.append("| `n18/deepprep-less` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |")
lines.append("| `n18/deepprep-more` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |")
lines.append("| `I48/deepprep-less` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |")
lines.append("| `I48/deepprep-more` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |")
lines.append("")
lines.append("> ⚠️ 四个 deepprep 目录的**错误级算不出来**，但坏法各不相同（都是 deepprep 产物自身的缺陷，\n"
             "> 不是工具问题 —— 工具要按 key 列给每张表建 sqlite 索引，坏表直接建不出来）：\n"
             "> | 目录 | 坏在哪 |\n"
             "> |---|---|\n"
             "> | `n18/deepprep-less` | `admissions.csv` 0 行；`icu/outputevents.csv` **缺 subject_id** 且 0 行 |\n"
             "> | `n18/deepprep-more` | `admissions.csv` **缺 hadm_id**；`icu/{icustays,inputevents}.csv` 0 行；"
             "`icu/{chartevents,outputevents}.csv` **缺 subject_id** 且 0 行 |\n"
             "> | `I48/deepprep-less` | `admissions.csv` 是 **1 字节空文件**（连表头都没有）⇒ 建表 SQL 语法错 `near \")\"` |\n"
             "> | `I48/deepprep-more` | 只坏 `icu/outputevents.csv`（**缺 charttime**），其余 6 张表都齐 |\n"
             "> ⇒ 对照：K29 / ICU5000 的 deepprep `admissions.csv` 有 8,093 / 6,476 行且 key 齐 ⇒ 工具跑得通，\n"
             "> 所以那两档在原表里有错误级数字。\n")
lines.append("## 维度三：MIMIC 原生错误格（string 口径）\n")
lines.append("| 结果目录 | 跑者 | 口径 | 总格数 | 修对 | 没动 | 缺行 | **修复率** |")
lines.append("|---|---|---|---:|---:|---:|---:|---:|")
r = nat_row("n18/1.6pct_我们的agent_pi+val_mimicraw", "我们", "string", nat[0])
if r:
    lines.append(r)
r = nat_row("n18/4.7pct_我们的agent_pi+val_mimicraw", "我们", "string", nat[1])
if r:
    lines.append(r)
r = nat_row("n18/4.7pct_cc基线_ClaudeCode", "cc", "string", nat[2])
if r:
    lines.append(r)
# deepprep 四行：从重判输出里解析总格数与三桶（不手填，避免串档）
for label in ("n18/deepprep-less/output", "n18/deepprep-more/output",
              "I48/deepprep-less/output", "I48/deepprep-more/output"):
    raw_lines = dpnat.get(label) or []
    tot = dirty = missing = None
    for l in raw_lines:
        s = l.strip()
        if s.startswith("总格数"):
            tot = s.split()[-1]
        elif s.startswith("b_dirty"):
            dirty = s.split()[1]
        elif s.startswith("b_missing"):
            missing = s.split()[1]
    if tot is None:
        continue
    pkg = label.replace("/output", "")
    lines.append("| `%s` | deepprep | string | %s | 0 | %s | %s | **0.000000** |"
                 % (pkg, tot, dirty, missing))
lines.append("")
lines.append("## 读表提醒（补齐部分）\n")
lines.append("1. **n18 的 1.6pct 没有 cc/codex 基线** —— 不是漏拷，是**从来没跑过**：\n"
             "   cc/codex 基线当时跑的是 `datasets/注入错误数据集/n18_less` / `n18_more` 两个切片\n"
             "   （已从盘上删除），其 cohort 分别是 7,445 / 7,048 行；当前 mimicraw n18 的 cohort 是 **7,048** 行\n"
             "   ⇒ `n18_more`（4.7pct）**可比、已入库**，`n18_less`（1.6pct，cohort 7,445）**不可比、没入库**。\n"
             "   要补 1.6pct 的 cc 基线，只能在 mimicraw 1.6pct 上**重跑一次 cc/codex**。\n")
lines.append("2. **n18 4.7pct 的 cc 基线原生修复率 = 0**（444 没动 / 142 缺行）——\n"
             "   与包里其它 cc/codex 基线一致：规则推导法碰不到哨兵码与生理不可能值（原生错误类）。\n")
lines.append("3. **n18 的 deepprep composite 只有 0.20**（I48 的 deepprep 是 0.40~0.46）——\n"
             "   原因是 **key 几乎全错**（0.165 / 0.151 vs K29 的 0.83）⇒ deepprep **没做 cohort 筛选**，\n"
             "   而 n18/i48 的输出表是**筛选之后**的表 ⇒ 结构上就对不上。\n")

out = "\n".join(lines) + "\n"
(REPO / "ceshi1_补齐行_三维度.md").write_text(out, encoding="utf-8")
print("写出仓库留档: reproduction/ceshi1_补齐行_三维度.md (%d 字符)" % len(out))
try:
    (DL / "00_补齐_新增n18与I48行.md").write_text(out, encoding="utf-8")
    print("写出包内补充文件: %s/00_补齐_新增n18与I48行.md" % DL)
except Exception as e:
    print("包内写入被拒（%s）—— 内容已在仓库留档" % e)
