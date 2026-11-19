#!/usr/bin/env python3
"""给打包好的目录生成 README（顶层 + 每个子目录）+ 横向对比总表。"""
import json
from pathlib import Path

OUT = Path("/Users/mac/Downloads/ceshi1")

CAL_DESC = {
    "v29": "数值 + key还原 + 排除gold外行 + log判据",
    "str": "纯字符串 + key还原 + 排除gold外行 + log判据",
    "oldsw": "数值 + 关key还原 + 不排除gold外行 + log判据",
    "old": "纯字符串 + 关key还原 + 不排除gold外行 + log判据",
    "goldv29": "数值 + key还原 + 排除gold外行 + gold判据",
    "goldstr": "纯字符串 + key还原 + 排除gold外行 + gold判据",
    "goldold": "纯字符串 + 关key还原 + 不排除gold外行 + gold判据",
}
ORDER = ["v29", "str", "oldsw", "old", "goldv29", "goldstr", "goldold"]


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else ("" if n is None else str(n))


def write_folder_readme(d):
    s = load(d / "summary.json")
    if not s:
        return
    L = ["# %s" % d.name, "",
         "| | |", "|---|---|",
         "| **数据集** | %s |" % s.get("dataset", ""),
         "| **谁跑的** | %s |" % s.get("who", ""),
         "| **说明** | %s |" % s.get("note", ""), ""]
    if s.get("val"):
        v = s["val"]
        L += ["| **val composite** | %s（status=%s，%s 轮） |" % (
            v.get("best_score"), v.get("status"), v.get("rounds"))]
        L += ["| **val 独立重放** | %s |" % v.get("reproducible_score")]
    if s.get("test_composite") is not None:
        L += ["| **test composite** | %s |" % s["test_composite"]]
    L.append("")
    if s.get("calibers"):
        L += ["## 七口径错误级指标（test）", "",
              "| 口径 | 开关组合 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for cal in ORDER:
            c = s["calibers"].get(cal)
            if not c:
                continue
            L.append("| **%s** | %s | **%.6f** | **%.6f** | %s | %s | %s | %s |"
                     % (cal, CAL_DESC[cal], c["det_f1"], c["rep_f1"],
                        fmt(c["b_clean"]), fmt(c["b_wrong"]), fmt(c["b_dirty"]),
                        fmt(c["b_missing"])))
        L.append("")
    if s.get("native_cells"):
        n = s["native_cells"]
        L += ["## MIMIC 原生错误格专项", "",
              "| 项 | 值 |", "|---|---:|",
              "| 总格数 | %s |" % fmt(n.get("总格数")),
              "| 修对 b_clean | %s |" % fmt(n.get("修对(b_clean)")),
              "| 改错 b_wrong | %s |" % fmt(n.get("改错(b_wrong)")),
              "| 没动 b_dirty | %s |" % fmt(n.get("没动(b_dirty)")),
              "| 缺行 b_missing | %s |" % fmt(n.get("缺行(b_missing)")),
              "| 修复率 | %s |" % n.get("修复率"),
              "| 按检测类型 | %s |" % json.dumps(n.get("按类型"), ensure_ascii=False), ""]
    if s.get("files"):
        L += ["## 基线结果", ""]
        for k, v in s["files"].items():
            L.append("- **%s**：%s" % (k, json.dumps(v, ensure_ascii=False)))
        L.append("")
    L += ["## 文件说明", "",
          "- `error_metrics_<口径>.json` —— 错误级指标（逐文件明细 + 全局汇总）",
          "- `native_cells_*.json` —— MIMIC 原生错误格专项判定",
          "- `score_report.json` / `run_report.json` / `val_run_report.json` —— harness 的评分与运行报告",
          "- `summary.json` —— 本目录关键数字的汇总（机器可读）", ""]
    (d / "README.md").write_text("\n".join(L), encoding="utf-8")


def main():
    for d in sorted(OUT.iterdir()):
        if d.is_dir() and d.name != "Z_附加基线":
            write_folder_readme(d)
        elif d.is_dir():
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    write_folder_readme(sub)
    # ── 横向对比总表 ──
    rows = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir() or d.name == "Z_附加基线":
            continue
        s = load(d / "summary.json") or {}
        cals = s.get("calibers", {})
        base = s.get("files", {})
        rows.append((d.name, s.get("dataset", ""), s.get("who", ""),
                     s.get("test_composite"), s.get("val", {}).get("best_score"),
                     cals.get("v29"), cals.get("goldv29"), cals.get("goldstr"),
                     cals.get("old"), base))
    L = ["# 横向对比总表（10 个）", "",
         "> 数字全部来自各目录里的 JSON（不是手抄）。",
         "> A* = 我们的 agent（pi + val harness，跑在 mimicraw 系列数据上）；",
         "> B* = 基线（codex / Claude Code，**跑在原始注入错误数据集上**，与 A* 不是同一份数据）。", ""]
    L += ["## 一、我们的 5 档 vs 对应基线", "",
          "| 文件夹 | 数据集 | 谁跑的 | test composite | val composite | v29 Det/Rep | goldv29 Det/Rep | goldstr Det/Rep | old Det/Rep |",
          "|---|---|---|---:|---:|---|---|---|---|"]
    for name, ds, who, tc, vc, v29, gv, gs, old, base in rows:
        if not name.startswith("A"):
            continue
        f = lambda c: ("%.6f / %.6f" % (c["det_f1"], c["rep_f1"])) if c else "—"
        L.append("| **%s** | %s | %s | %s | %s | %s | %s | %s | %s |"
                 % (name, ds, who, tc or "—", vc or "—", f(v29), f(gv), f(gs), f(old)))
    L += ["", "## 二、基线（codex / Claude Code）", "",
          "| 文件夹 | 数据集 | 谁跑的 | 错误级 Det/Rep（它自己的口径） | composite |",
          "|---|---|---|---|---:|"]
    for name, ds, who, tc, vc, v29, gv, gs, old, base in rows:
        if not name.startswith("B"):
            continue
        txt = []
        for k, v in base.items():
            if isinstance(v, dict) and "det_f1" in v:
                txt.append("%s: %.6f / %.6f" % (k.replace("codex_error_metrics_", "")
                                               .replace("cc_error_metrics_", "")
                                               .replace(".json", ""),
                                               v["det_f1"], v["rep_f1"]))
            elif isinstance(v, dict) and "composite" in v:
                txt.append("%s composite=%.6f" % (k.replace("_overall_score", ""), v["composite"]))
        L.append("| **%s** | %s | %s | %s | %s |"
                 % (name, ds, who, "；".join(txt) or "见目录内 JSON",
                    next((v["composite"] for v in base.values()
                          if isinstance(v, dict) and "composite" in v), "—")))
    L += ["", "## 三、读表提醒", "",
          "1. **A 与 B 不是同一份数据**：A 跑在 mimicraw 系列（MIMIC 原生错误层已修好），"
          "B 跑在原始注入错误数据集 ⇒ **composite 可比（同一套评分器），错误级格子数不可直接比**。",
          "2. **跨档比错误级指标只能用 goldv29 / goldstr**：K29 1.6pct 的 log 被按比例改过，"
          "它的 str/old 是调出来的值。",
          "3. **I48 的 old/oldsw 有结构性偏差**：它的任务含 cohort 筛选，"
          "agent 正确删掉的行上的注入格会被记成假漏检（44.6%）⇒ I48 的 old 不能跟别的档直接比。",
          "4. **各口径怎么算**：见顶层 `README.md` 的「口径是怎么算的」一节。", ""]
    (OUT / "00_横向对比总表.md").write_text("\n".join(L), encoding="utf-8")
    print("README 生成完成")


if __name__ == "__main__":
    main()
