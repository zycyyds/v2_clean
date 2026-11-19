#!/usr/bin/env python3
"""生成 README：按疾病分组（一个疾病一个文件夹，档位/基线在里面）。"""
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
DISEASES = ["K29", "ICU5000", "I48", "N18"]


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else ("" if n is None else str(n))


def entries():
    """[(疾病, 目录名, summary)]"""
    out = []
    for dz in DISEASES:
        d = OUT / dz
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            s = load(sub / "评分" / "summary.json") or load(sub / "summary.json")
            if s:
                out.append((dz, sub.name, s))
    return out


def write_entry_readme(d, s):
    L = ["# %s" % d.name, "",
         "| | |", "|---|---|",
         "| **数据集** | %s |" % s.get("dataset", ""),
         "| **谁跑的** | %s |" % s.get("who", ""),
         "| **说明** | %s |" % s.get("note", ""), ""]
    if s.get("val"):
        v = s["val"]
        L += ["| **val composite** | %s（status=%s，%s 轮） |"
              % (v.get("best_score"), v.get("status"), v.get("rounds")),
              "| **val 独立重放** | %s |" % v.get("reproducible_score")]
    if s.get("test_composite") is not None:
        L += ["| **test composite** | %s |" % s["test_composite"]]
    L.append("")
    # 结果包清单
    pkg = d / "结果包"
    if pkg.is_dir():
        fs = sorted(f for f in pkg.rglob("*") if f.is_file())
        total = sum(f.stat().st_size for f in fs)
        L += ["## 结果包（agent 实际产出的清洗结果）", "",
              "共 **%d 个文件 / %.1f MB**，目录结构原样保留（可原样重放）。" % (len(fs), total / 1e6),
              "", "| 文件 | 大小 |", "|---|---:|"]
        for f in fs[:40]:
            L.append("| `%s` | %.1f MB |" % (f.relative_to(pkg), f.stat().st_size / 1e6))
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
    sc = d / "评分"
    if sc.is_dir():
        L += ["## `评分/` 里有什么", ""]
        for f in sorted(sc.iterdir()):
            if f.is_file():
                L.append("- `%s`" % f.name)
        L.append("")
    L += ["## 说明", "",
          "- `结果包/` —— **agent 产出的清洗结果**（csv + submission.json），目录结构原样保留；",
          "- `评分/` —— 评分与统计产物（错误级七口径、原生格判定、harness 报告），不参与重放。", ""]
    (d / "README.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ents = entries()
    for dz, name, s in ents:
        write_entry_readme(OUT / dz / name, s)
    # ── 总表（按疾病分组）──
    L = ["# 横向对比总表（按疾病分组）", "",
         "> 数字全部来自各目录里的 JSON（不是手抄）。",
         "> 「我们的」= pi + val harness，跑在 **mimicraw 系列**数据集上；",
         "> 「基线」= codex / Claude Code，跑在**原始注入错误数据集**上 ⇒ composite 可比，错误级格子数不可直接比。", ""]
    for dz in DISEASES:
        rows = [e for e in ents if e[0] == dz]
        if not rows:
            continue
        L += ["## %s" % dz, "",
              "| 目录 | 谁跑的 | test composite | val composite | v29 Det/Rep | goldv29 Det/Rep | goldstr Det/Rep | old Det/Rep |",
              "|---|---|---:|---:|---|---|---|---|"]
        for _dz, name, s in rows:
            c = s.get("calibers", {})
            f = lambda k: ("%.6f / %.6f" % (c[k]["det_f1"], c[k]["rep_f1"])) if k in c else "—"
            base = s.get("files", {})
            comp = next((v["composite"] for v in base.values()
                         if isinstance(v, dict) and "composite" in v), None)
            # 基线只有一个口径（v28 或 v29）⇒ 把它放进 v29 列并标注
            bdet = next((v for v in base.values()
                         if isinstance(v, dict) and "det_f1" in v), None)
            tc = s.get("test_composite") if s.get("calibers") else comp
            vc = s.get("val", {}).get("best_score") if s.get("calibers") else None
            if bdet:
                v29c = "%.6f / %.6f" % (bdet["det_f1"], bdet["rep_f1"])
                gv29c = gsc = oldc = "同左（只有一档）"
            else:
                v29c, gv29c, gsc, oldc = f("v29"), f("goldv29"), f("goldstr"), f("old")
            L.append("| **%s** | %s | %s | %s | %s | %s | %s | %s |"
                     % (name, s.get("who", ""), tc if tc is not None else "—",
                        vc if vc is not None else "—", v29c, gv29c, gsc, oldc))
        L.append("")
    L += ["## 读表提醒", "",
          "1. **「我们的」与「基线」不是同一份数据** ⇒ composite 可比，错误级格子数不可直接比。",
          "2. **跨档比错误级指标只能用 goldv29 / goldstr** —— K29 1.6pct 的 log 被按比例改过，"
          "它的 str/old 是调出来的目标值。",
          "3. **I48 的 old/oldsw 有结构性偏差** —— 它的任务含 cohort 筛选，agent 正确删掉的行上的"
          "注入格会被记成假漏检（44.6%）⇒ 不能跟别的档直接比。",
          "4. **各口径怎么算** —— 见顶层 `README.md`。", ""]
    (OUT / "00_横向对比总表.md").write_text("\n".join(L), encoding="utf-8")
    print("README 生成完成（%d 个目录）" % len(ents))


if __name__ == "__main__":
    main()
