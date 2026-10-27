"""从 composite（评分器产物）里取逐文件的行数对账：reference vs result。

不自己读 gold —— 行数差由评分器算好后写在 file_reports 里。
"""
import json
from pathlib import Path

RUNS = Path("/Users/mac/PycharmProjects/v2_clean/runs")
WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning")

JOBS = [
    ("cc N18 4.7pct", RUNS / "n18_more_v1_overall_score.json"),
    ("cc N18 4.7pct · agent", WT / "k29_agent_run/experiments/n18_4p7_test_seed666_v1/host/score_report.json"),
    ("cc I48 1.6pct", RUNS / "i48_less_cc_v1_overall_score.json"),
    ("cc I48 1.6pct · agent", WT / "k29_agent_run/experiments/i48_1p6_test_seed666_v1/host/score_report.json"),
    ("cc I48 4.7pct", RUNS / "i48_more_cc_v1_overall_score.json"),
    ("cc I48 4.7pct · agent", WT / "k29_agent_run/experiments/i48_4p7_test_seed666_v2/host/score_report.json"),
]

for tag, p in JOBS:
    if not p.is_file():
        print(f"\n=== {tag} ===\n  (缺 {p.name})")
        continue
    d = json.loads(p.read_text())
    print(f"\n=== {tag} ===  composite={d['metrics']['composite_score']:.6f}")
    print(f"  {'文件':<30}{'ref 行':>12}{'result 行':>12}{'缺行':>12}{'多行':>10}{'文件分':>9}")
    for m in sorted(d["file_reports"], key=lambda x: x["metrics"]["file_score"]):
        print(f"  {m['relative_path']:<30}{m['reference_rows']:>12,}{m['result_rows']:>12,}"
              f"{m['missing_reference_rows']:>12,}{m['extra_result_rows']:>10,}"
              f"{m['metrics']['file_score']:>9.4f}")
