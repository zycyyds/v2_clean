"""核对：9 条结果的 composite / Det / Rep 是否都真的写进了各个汇总文件。"""
import json
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
WT = PROJ / ".claude/worktrees/eval-movies-cleaning"
RUNS = PROJ / "runs"
AGENT_RUN = WT / "k29_agent_run"
AW = PROJ / "agent_work"

# tag -> (composite 来源, 错误分来源)
TRUTH = {
    "icu5000-1.6/agent": (
        RUNS / "icu5000_less_agent_overall_score.json",
        RUNS / "icu5000_less_agent_error_metrics_v29.json"),
    "icu5000-1.6/codex": (
        RUNS / "icu5000_less_codex_overall_score.json",
        RUNS / "icu5000_less_codex_error_metrics_v29.json"),
    "icu5000-1.6/cc_v1": (
        RUNS / "icu5000_less_cc_v1_overall_score.json",
        RUNS / "icu5000_less_cc_v1_error_metrics_v29.json"),
    "icu5000-4.7/cc": (
        RUNS / "icu5000_more_cc_overall_score.json",
        RUNS / "icu5000_more_cc_error_metrics_v29.json"),
    "icu5000-4.7/codex": (
        RUNS / "icu5000_more_codex_overall_score.json",
        RUNS / "icu5000_more_codex_error_metrics_v29.json"),
    "icu5000-4.7/agent": (
        AGENT_RUN / "experiments/icu5000_4p7_test_seed666_v1/host/score_report.json",
        RUNS / "icu5000_more_agent_error_metrics.json"),
    "n18-1.6/agent": (
        AGENT_RUN / "experiments/n18_1p6_test_seed666_v1/host/score_report.json",
        RUNS / "n18_agent_error_metrics_v29.json"),
    "n18-1.6/cc": (
        RUNS / "n18_less_cc_v1_overall_score.json",
        RUNS / "n18_less_cc_v1_error_metrics_v29.json"),
    "n18-4.7/agent": (
        AGENT_RUN / "experiments/n18_4p7_test_seed666_v1/host/score_report.json",
        RUNS / "n18_more_agent_error_metrics_v29.json"),
    "i48-1.6/agent": (
        AGENT_RUN / "experiments/i48_1p6_test_seed666_v1/host/score_report.json",
        RUNS / "i48_less_agent_error_metrics_v29.json"),
    "i48-4.7/agent": (
        AGENT_RUN / "experiments/i48_4p7_test_seed666_v2/host/score_report.json",
        RUNS / "i48_more_agent_error_metrics_v29.json"),
    "n18-4.7/cc": (
        RUNS / "n18_more_v1_overall_score.json",
        RUNS / "n18_more_v1_cc_error_metrics_v29.json"),
    "i48-1.6/cc": (
        RUNS / "i48_less_cc_v1_overall_score.json",
        RUNS / "i48_less_cc_v1_cc_error_metrics_v29.json"),
}

truth = {}
for tag, (sp, ep) in TRUTH.items():
    s = json.loads(Path(sp).read_text(encoding="utf-8"))
    c = s["metrics"]["composite_score"]
    d = r = None
    if Path(ep).is_file():
        e = json.loads(Path(ep).read_text(encoding="utf-8"))
        d, r = e["detection"]["F1"], e["repair"]["F1"]
    truth[tag] = (c, d, r)

FILES = {
    "评测总表.md (台账)": WT / "评测总表.md",
    "score_only/总表.md": WT / "score_only/总表.md",
    "score_only/评测规范.md": WT / "score_only/评测规范.md",
}

for name, fp in FILES.items():
    if not fp.is_file():
        print(f"\n{name}: 文件不存在")
        continue
    txt = fp.read_text(encoding="utf-8")
    miss = []
    for tag, (c, d, r) in truth.items():
        if f"{c:.6f}" not in txt:
            miss.append(f"{tag}  composite {c:.6f}")
            continue
        if d is not None and f"{d:.6f}" not in txt:
            miss.append(f"{tag}  Det {d:.6f}")
        if r is not None and f"{r:.6f}" not in txt:
            miss.append(f"{tag}  Rep {r:.6f}")
    print(f"\n=== {name}  ({len(txt.splitlines())} 行) ===")
    if miss:
        for m in miss:
            print("   缺:", m)
    else:
        print(f"   OK  {len(truth)} 条记录的 composite + Det + Rep 全部命中")
