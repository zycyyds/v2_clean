"""icu5000 最终并排表：1.6pct (less) 与 4.7pct (more) 两档 × codex vs 我们的 agent。

口径：
  Raha composite —— 两边完全同口径（同一 pi_test_scorer、同一 icu_5000_v1.json、
                    同一份 test raw/gold）。这是唯一能直接比总分的地方。
  错误级 Det/Rep F1 —— 全部为**新口径**（recover_clean_key=true），由同一份
                    reproduction/compute_error_metrics.py 算出。

换路径：用环境变量覆盖，或直接改下面的常量。
  PROJ   项目根
  RUN    agent run 目录（含 experiments/）
  RUNS   分数输出目录
"""
import json
import os
from pathlib import Path

PROJ = Path(os.environ.get("PROJ", "/Users/mac/PycharmProjects/v2_clean"))
WT = Path(os.environ.get("WT", str(PROJ)))
RUN = Path(os.environ.get("RUN", str(PROJ / "k29_agent_run")))
RUNS = Path(os.environ.get("RUNS", str(PROJ / "runs")))

SCORE_16 = RUN / "experiments/icu5000_1p6_test_seed666_v3/host/score_report.json"
SCORE_47 = RUN / "experiments/icu5000_4p7_test_seed666_v1/host/score_report.json"
CODEX_16 = RUNS / "icu5000_less_codex_overall_score.json"
CODEX_47 = Path(os.environ.get(
    "CODEX_47", "/Users/mac/.claude/jobs/910902cd/tmp/raha_icu5000_more.json"))

ERR = {
    ("agent", "16"): RUNS / "icu5000_less_agent_error_metrics.json",
    ("codex", "16"): RUNS / "icu5000_less_codex_error_metrics.json",
    ("agent", "47"): RUNS / "icu5000_more_agent_error_metrics.json",
    ("codex", "47"): RUNS / "icu5000_more_codex_error_metrics.json",
}


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def score(p):
    d = jload(p)
    return d["metrics"]["composite_score"] if d else None


def files(p):
    d = jload(p)
    return {m["relative_path"]: m["metrics"]["file_score"] for m in d["file_reports"]} if d else {}


def err(p):
    d = jload(p)
    if not d:
        return None
    return {"det": d["detection"], "rep": d["repair"], "b": d["buckets"],
            "rc": d.get("recover_clean_key")}


E = {k: err(v) for k, v in ERR.items()}
S = {
    ("agent", "16"): score(SCORE_16), ("codex", "16"): score(CODEX_16),
    ("agent", "47"): score(SCORE_47), ("codex", "47"): score(CODEX_47),
}
F = {
    ("agent", "16"): files(SCORE_16), ("codex", "16"): files(CODEX_16),
    ("agent", "47"): files(SCORE_47), ("codex", "47"): files(CODEX_47),
}

print("=" * 84)
print("icu5000 · codex  vs  我们的 agent（pi worker + val 循环，全程没见过 test）")
print("=" * 84)

print("\n## 一、Raha composite（test）\n")
print("| 档位 | 错误率 | codex | 我们的 agent | 差 |")
print("|---|---|---:|---:|---:|")
for tag, name in (("16", "1.6pct (less)"), ("47", "4.7pct (more)")):
    c, a = S[("codex", tag)], S[("agent", tag)]
    rate = {"16": "1.58%", "47": "4.72%"}[tag]
    print(f"| {name} | {rate} | {c:.6f} | **{a:.6f}** | **{a-c:+.6f}** |")

print("\n## 二、错误级指标（新口径，双方同一份脚本）\n")
print("| 档位 | 谁 | Det P | Det R | **Det F1** | Rep P | Rep R | **Rep F1** |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for tag, name in (("16", "1.6pct"), ("47", "4.7pct")):
    for who in ("codex", "agent"):
        m = E[(who, tag)]
        if not m:
            continue
        d, r = m["det"], m["rep"]
        w = "**" if who == "agent" else ""
        print(f"| {name} | {w}{who}{w} | {d['P']:.6f} | {d['R']:.6f} | {w}{d['F1']:.6f}{w} | "
              f"{r['P']:.6f} | {r['R']:.6f} | {w}{r['F1']:.6f}{w} |")

print("\n## 三、错误分桶\n")
print("| 档位 | 谁 | b_clean 修对 | b_dirty 没动 | b_wrong 修错 | b_missing 丢行 | 误改干净格 |")
print("|---|---|---:|---:|---:|---:|---:|")
for tag, name in (("16", "1.6pct"), ("47", "4.7pct")):
    for who in ("codex", "agent"):
        m = E[(who, tag)]
        if not m:
            continue
        b = m["b"]
        print(f"| {name} | {who} | {b['b_clean']:,} | {b['b_dirty']:,} | {b['b_wrong']:,} | "
              f"{b['b_missing']:,} | {b['fp_det']:,} |")

print("\n## 四、逐文件 Raha composite\n")
for tag, name in (("16", "1.6pct (less)"), ("47", "4.7pct (more)")):
    fa, fc = F[("agent", tag)], F[("codex", tag)]
    print(f"\n### {name}\n")
    print("| 文件 | codex | 我们的 agent | 差 |")
    print("|---|---:|---:|---:|")
    for k in sorted(fa, key=lambda k: fa[k]):
        c = fc.get(k, 0)
        print(f"| {k} | {c:.4f} | {fa[k]:.4f} | {fa[k]-c:+.4f} |")

print("\n## 五、val → test 落差（泛化能力）\n")
print("| 档位 | val composite | test composite | 落差 |")
print("|---|---:|---:|---:|")
for tag, name, v in (("16", "1.6pct", 0.903785), ("47", "4.7pct", 0.915858)):
    t = S[("agent", tag)]
    print(f"| {name} | {v:.6f} | {t:.6f} | **{t-v:+.6f}** |")
