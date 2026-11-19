#!/usr/bin/env python3
"""把每个 agent 实际产出的结果包（清洗后的 csv）复制进对应目录，
并把评分/日志文件收进「评分/」子文件夹 —— 让结果包成为主角。"""
import shutil
from pathlib import Path

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
CODEX = Path("/Users/mac/Documents/Codex/2026-09-19")
OUT = Path("/Users/mac/Downloads/ceshi1")

# 目录 → 结果包来源
SRC = {
    "K29/1.6pct_我们的agent_pi+val_mimicraw":
        WT / "k29_agent_run/experiments/k29_mimicraw_1p6_test_seed666_v1/host/test_result_package",
    "K29/4.7pct_我们的agent_pi+val_mimicraw":
        WT / "k29_agent_run/experiments/k29_mimicraw_4p7_test_seed666_v1/host/test_result_package",
    "ICU5000/1.6pct_我们的agent_pi+val_mimicraw":
        WT / "icu5000_agent_run/experiments/icu5000_mimicraw_1p6_test_seed666_v1/host/test_result_package",
    "ICU5000/4.7pct_我们的agent_pi+val_mimicraw":
        WT / "icu5000_agent_run/experiments/icu5000_mimicraw_4p7_test_seed666_v1/host/test_result_package",
    "I48/1.6pct_我们的agent_pi+val_mimicraw2":
        WT / "i48_agent_run/experiments/i48_mimicraw2_1p6_test_seed666_v1/host/test_result_package",
    "K29/1.6pct_codex基线":
        CODEX / "data-cleaning-agent-mimic-iv-k29/outputs/K29",
    "K29/4.7pct_codex基线":
        CODEX / "data-cleaning-agent-mimic-iv-k29-6/outputs",
    "ICU5000/1.6pct_codex基线":
        PROJ / "agent_work/icu5000_less_codex/output/test_final3/result_root",
    "ICU5000/4.7pct_codex基线":
        PROJ / "agent_work/icu5000_more/output/test",
    "I48/1.6pct_cc基线_ClaudeCode":
        PROJ / "agent_work/i48_less_cc_v1/output/test",
}

# 归到「评分/」的文件类型
SCORE_GLOBS = ("error_metrics_*.json", "native_cells_*.json", "*_native_cells_*.json",
               "score_report.json", "run_report.json", "val_run_report.json",
               "*_overall_score.json", "codex_scorer.log", "*_scorer.log",
               "summary.json")


def main():
    for rel, src in SRC.items():
        d = OUT / rel
        if not d.is_dir():
            print("  跳过（目录不存在）:", rel)
            continue
        if not src.is_dir():
            print("  缺结果包:", rel, src)
            continue
        # ① 评分文件 → 评分/
        score_dir = d / "评分"
        score_dir.mkdir(exist_ok=True)
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            if any(f.match(g) for g in SCORE_GLOBS):
                shutil.move(str(f), str(score_dir / f.name))
        # ② 结果包 → 结果包/
        dst = d / "结果包"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=False)
        n = sum(1 for _ in dst.rglob("*") if _.is_file())
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        print("  ✓ %-46s 结果包 %2d 个文件 / %7.1f MB" % (rel, n, size / 1e6))


if __name__ == "__main__":
    main()
