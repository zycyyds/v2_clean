#!/usr/bin/env python3
"""汇总 ceshi1 全部结果的三维度对比表。

维度：
  ① 整体分数 composite
  ② 错误分数 str / oldsw / goldv29
  ③ MIMIC 原生错误 numeric / string

数据来源（全部读文件，无手抄）：
  - composite：runs/*_overall_score.json（基线）/ 实验目录 host/score_report.json（我们的）
               / eval_out/composite__*.json（本次新算）
  - 错误级  ：runs/*_error_metrics_*.json / eval_out/err__*.json
  - 原生格  ：runs/*_native_cells*.json / eval_out/native__*.json
"""
import json
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
RUNS = PROJ / "runs"
WT = PROJ / ".claude/worktrees/k29-mimicraw"
EV = Path("/Users/mac/.claude/jobs/448aad57/tmp/eval_out")
OUT = Path("/Users/mac/Downloads/ceshi1")

# 名字 → (跑者, 档位, 数据集标签)
SPEC = [
    ("K29/1.6pct_我们的agent_pi+val_mimicraw", "我们", "1.6pct", "mimicraw",
     "k29_mimicraw_16_agent", WT / "k29_agent_run/experiments/k29_mimicraw_1p6_test_seed666_v1/host/score_report.json"),
    ("K29/4.7pct_我们的agent_pi+val_mimicraw", "我们", "4.7pct", "mimicraw",
     "k29_mimicraw_47_agent", WT / "k29_agent_run/experiments/k29_mimicraw_4p7_test_seed666_v1/host/score_report.json"),
    ("K29/1.6pct_codex基线", "codex", "1.6pct", "rest(无log)", None, None),
    ("K29/4.7pct_codex基线", "codex", "4.7pct", "rest(无log)", None, None),
    ("K29/deepprep-less", "deepprep", "1.6pct", "mimicraw", "k29dp16", None),
    ("K29/deepprep-more", "deepprep", "4.7pct", "mimicraw", "k29dp47", None),
    ("ICU5000/1.6pct_我们的agent_pi+val_mimicraw", "我们", "1.6pct", "mimicraw",
     "icu5000_mimicraw_16_agent", WT / "icu5000_agent_run/experiments/icu5000_mimicraw_1p6_test_seed666_v1/host/score_report.json"),
    ("ICU5000/4.7pct_我们的agent_pi+val_mimicraw", "我们", "4.7pct", "mimicraw",
     "icu5000_mimicraw_47_agent", WT / "icu5000_agent_run/experiments/icu5000_mimicraw_4p7_test_seed666_v1/host/score_report.json"),
    ("ICU5000/1.6pct_codex基线", "codex", "1.6pct", "raw(无原生层)", None, None),
    ("ICU5000/4.7pct_codex基线", "codex", "4.7pct", "raw(无原生层)", None, None),
    ("ICU5000/deepprep-less", "deepprep", "1.6pct", "mimicraw", "icudp16", None),
    ("ICU5000/deepprep-more", "deepprep", "4.7pct", "mimicraw", "icudp47", None),
    ("I48/1.6pct_我们的agent_pi+val_mimicraw2", "我们", "1.6pct", "mimicraw",
     "i48_mimicraw2_16_agent", WT / "i48_agent_run/experiments/i48_mimicraw2_1p6_test_seed666_v1/host/score_report.json"),
    ("I48/1.6pct_cc基线_ClaudeCode", "cc", "1.6pct", "raw(无原生层)", None, None),
    # i48 4.7pct（本次新跑）
    ("I48/4.7pct_我们的agent_pi+val_mimicraw2", "我们", "4.7pct", "mimicraw",
     "i48_mimicraw2_47_agent", WT / "i48_agent_run/experiments/i48_mimicraw2_4p7_test_seed666_v1/host/score_report.json"),
]

BASELINE_COMP = {
    "ICU5000/1.6pct_codex基线": "icu5000_less_codex_overall_score.json",
    "ICU5000/4.7pct_codex基线": "icu5000_more_codex_overall_score.json",
    "I48/1.6pct_cc基线_ClaudeCode": "i48_less_cc_v1_overall_score.json",
    # K29 的两个 codex 跑在 K29_clean_eval_10_20_rest ⇒ scorer 结果存在子目录的 scorer.log 里
    "K29/1.6pct_codex基线": "k29_codex_score/scorer.log",
    "K29/4.7pct_codex基线": "k29_codex_v6/scorer.log",
}
# 基线的错误级既有文件（只有 v29）
BASELINE_EM = {
    "ICU5000/1.6pct_codex基线": ["icu5000_less_codex_error_metrics_v29.json"],
    "ICU5000/4.7pct_codex基线": ["icu5000_more_codex_error_metrics.json"],
    "I48/1.6pct_cc基线_ClaudeCode": ["i48_less_cc_v1_cc_error_metrics_v29.json"],
}


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def find_err(rel, prefix, cal):
    """找某结果目录某口径的错误级 json：先看本次新算，再看 runs/ 里已有的。

    runs/ 里的文件名可能带时间戳后缀（如 ..._goldv29_run20261119_4p7.json）
    ⇒ 用 glob 匹配，取**最新**的一个。
    """
    dz, name = rel.split("/", 1)
    cands = [EV / ("err__%s__%s__%s.json" % (dz, name, cal))]
    if prefix:
        cands += list(RUNS.glob("%s_error_metrics_%s*.json" % (prefix, cal)))
    cands = [f for f in cands if f and f.is_file()]
    cands.sort(key=lambda f: (f.stat().st_mtime, str(f)), reverse=True)
    for f in cands:
        d = load(f)
        if d and "detection" in d:
            return d, f
    # 基线的旧文件（只有 v29，这里用不到）
    if rel in BASELINE_EM:
        for fn in BASELINE_EM[rel]:
            d = load(RUNS / fn)
            if d and "detection" in d and cal in fn:
                return d, RUNS / fn
    return None, None


def find_native(rel, prefix):
    """原生格 json：每次评测/每档可能有多个历史版本 ⇒ 每个口径只取**最新**的一个。"""
    dz, name = rel.split("/", 1)
    hits = list(EV.glob("native__%s__%s__*.json" % (dz, name)))
    if prefix:
        hits += list(RUNS.glob("%s*native_cells*.json" % prefix))
    hits = [f for f in hits if f.is_file()]
    by_mode = {}
    for f in sorted(hits, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True):
        mode = "string" if "string" in f.name else "numeric"
        by_mode.setdefault(mode, f)          # 排序后第一次出现 = 最新
    return [by_mode[m] for m in ("numeric", "string") if m in by_mode]


def composite_of(rel, ours_prefix, our_report):
    """返回 (composite, 来源, metrics 字典)。"""
    # 1) 本次新算
    for f in EV.glob("composite__*.json"):
        try:
            d = load(f)
            if d and d.get("metrics"):
                tag = f.stem.replace("composite__", "").replace("__", "/")
                if tag == rel:
                    return d["metrics"]["composite_score"], "新算", d["metrics"]
        except Exception:
            pass
    # 2) 我们的 agent：实验目录 score_report.json
    if our_report and Path(our_report).is_file():
        d = load(our_report)
        if d and d.get("metrics"):
            return d["metrics"]["composite_score"], "harness", d["metrics"]
    # 3) 基线：runs/*_overall_score.json
    if rel in BASELINE_COMP:
        d = load(RUNS / BASELINE_COMP[rel])
        if d and d.get("metrics"):
            return d["metrics"]["composite_score"], "台账", d["metrics"]
    return None, None, None


def main():
    rows = []
    for rel, who, tier, dslab, prefix, our_report in SPEC:
        comp, comp_src, comp_metrics = composite_of(rel, prefix, our_report)
        # 错误级
        errs = {}
        for cal in ("str", "oldsw", "goldv29"):
            if rel in BASELINE_EM and cal not in ("str", "oldsw", "goldv29"):
                continue
            d, f = find_err(rel, prefix, cal)
            if d:
                errs[cal] = (d, f)
        # 基线的 v29 不在选定口径里 → 跳过
        # 原生格
        nat = find_native(rel, prefix)
        rows.append(dict(rel=rel, who=who, tier=tier, ds=dslab, prefix=prefix,
                         composite=comp, comp_src=comp_src, comp_metrics=comp_metrics,
                         errs=errs, native=nat))

    L = []
    L.append("# ceshi1 三维度横向对比")
    L.append("")
    L.append("> 三个维度：**整体分数(composite)** / **错误级(str·oldsw·goldv29)** / **MIMIC 原生错误(numeric·string)**")
    L.append("> 错误级口径按要求只取 str / oldsw / goldv29 三个。")
    L.append("> 所有数字从 JSON 读出，没有手抄。")
    L.append("")
    L.append("## 维度一：整体分数（composite）")
    L.append("")
    L.append("| 结果目录 | 跑者 | 档位 | 数据集 | composite | schema | key | cell | exact | 来源 |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        c = r["composite"]
        m = r.get("comp_metrics")
        if m is None:
            L.append("| `%s` | %s | %s | %s | — | — | — | — | — | 未评 |" % (
                r["rel"], r["who"], r["tier"], r["ds"]))
        else:
            L.append("| `%s` | %s | %s | %s | **%.6f** | %.4f | %.4f | %.4f | %.4f | %s |" % (
                r["rel"], r["who"], r["tier"], r["ds"], m["composite_score"],
                m.get("schema_f1", 0), m.get("key_structure_f1", 0),
                m.get("row_aligned_cell_f1", 0), m.get("exact_row_f1", 0),
                r["comp_src"] or "?"))
    L.append("")

    L.append("## 维度二：错误级指标（str / oldsw / goldv29）")
    L.append("")
    L.append("| 结果目录 | 跑者 | 口径 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing | fp_det | natural_noise |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        for cal in ("str", "oldsw", "goldv29"):
            e = r["errs"].get(cal)
            if not e:
                continue
            d = e[0]
            b = d.get("buckets", {})
            L.append("| `%s` | %s | **%s** | **%.6f** | **%.6f** | %s | %s | %s | %s | %s | %s |" % (
                r["rel"], r["who"], cal, d["detection"]["F1"], d["repair"]["F1"],
                f"{b.get('b_clean',0):,}", f"{b.get('b_wrong',0):,}", f"{b.get('b_dirty',0):,}",
                f"{b.get('b_missing',0):,}", f"{b.get('fp_det',0):,}", f"{b.get('natural_noise',0):,}"))
    L.append("")

    L.append("## 维度三：MIMIC 原生错误格")
    L.append("")
    L.append("| 结果目录 | 跑者 | 口径 | 总格数 | 修对 | 改错 | 没动 | 缺行 | 修复率 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        for f in r["native"]:
            d = load(f)
            if not d:
                continue
            mode = "string" if "string" in f.name else "numeric"
            # 两种键格式：中文（"修对(b_clean)"）与英文（buckets.b_clean）
            b = d.get("buckets") if isinstance(d.get("buckets"), dict) else {}

            def gv(cn, en):
                if cn in d:
                    return d[cn]
                return b.get(en)

            tot = d.get("总格数") or d.get("n_native_cells") or d.get("total") or d.get("total_cells")
            rate = d.get("修复率", d.get("repair_rate"))
            L.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["rel"], r["who"], mode,
                f"{tot:,}" if isinstance(tot, int) else "—",
                f"{gv('修对(b_clean)', 'b_clean'):,}" if isinstance(gv("修对(b_clean)", "b_clean"), int) else "—",
                f"{gv('改错(b_wrong)', 'b_wrong'):,}" if isinstance(gv("改错(b_wrong)", "b_wrong"), int) else "—",
                f"{gv('没动(b_dirty)', 'b_dirty'):,}" if isinstance(gv("没动(b_dirty)", "b_dirty"), int) else "—",
                f"{gv('缺行(b_missing)', 'b_missing'):,}" if isinstance(gv("缺行(b_missing)", "b_missing"), int) else "—",
                ("%.4f" % rate) if isinstance(rate, float) else (rate if rate else "—")))
    L.append("")

    L.append("## 读表提醒")
    L.append("")
    L.append("1. **「我们的」和 deepprep 跑在 mimicraw 数据集上**（带 MIMIC 原生错误层）；")
    L.append("   **codex / cc 基线跑在原始注入错误数据集上** ⇒ composite 可比，**原生格维度对基线不适用**。")
    L.append("2. **K29 的两个 codex 基线跑在 `K29_clean_eval_10_20_rest`**，该数据集**没有 modification_log.csv**")
    L.append("   ⇒ 错误级维度算不了，只有 composite。")
    L.append("3. **K29 1.6pct 的 log 被按比例改过**（`fix_k29_log_rawform.py --ratio 0.2629`）")
    L.append("   ⇒ 它的 str / oldsw 是调出来的目标值，**不可跨档直接比**。")
    L.append("4. **I48 的 old/oldsw 有结构性偏差**（任务含 cohort 筛选，agent 正确删掉的行上的注入格")
    L.append("   被记成假漏检）⇒ oldsw 跨档不可比。")
    L.append("")

    txt = "\n".join(L)
    (OUT / "00_三维度横向对比.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
