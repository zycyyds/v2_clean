#!/usr/bin/env python3
"""核对打包后的 结果包/ 与源目录一一对应（相对路径 + md5 全量）。"""
import hashlib
from pathlib import Path

OUT = Path("/Users/mac/Downloads/ceshi1")
WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
CODEX = Path("/Users/mac/Documents/Codex/2026-09-19")

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
    "K29/1.6pct_codex基线": CODEX / "data-cleaning-agent-mimic-iv-k29/outputs/K29",
    "K29/4.7pct_codex基线": CODEX / "data-cleaning-agent-mimic-iv-k29-6/outputs",
    "ICU5000/1.6pct_codex基线": PROJ / "agent_work/icu5000_less_codex/output/test_final3/result_root",
    "ICU5000/4.7pct_codex基线": PROJ / "agent_work/icu5000_more/output/test",
    "I48/1.6pct_cc基线_ClaudeCode": PROJ / "agent_work/i48_less_cc_v1/output/test",
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def csv_map(root: Path):
    """{相对路径: 文件}；若所有 csv 都在同一个顶层子目录下，去掉这层 wrapper。"""
    m = {str(f.relative_to(root)): f for f in root.rglob("*.csv") if f.is_file()}
    if not m:
        return m
    tops = {k.split("/")[0] for k in m}
    if len(tops) == 1 and all("/" in k for k in m):
        t = tops.pop()
        m = {k[len(t) + 1:]: f for k, f in m.items()}
    return m


ok = True
for name, src in SRC.items():
    pkg = OUT / name / "结果包"
    if not src.is_dir():
        print("  ? %-46s 源不存在: %s" % (name, src))
        ok = False
        continue
    s = csv_map(src)
    d = {str(f.relative_to(pkg)): f for f in pkg.rglob("*.csv") if f.is_file()}
    only_s = sorted(set(s) - set(d))
    only_d = sorted(set(d) - set(s))
    mism = [k for k in sorted(set(s) & set(d)) if md5(s[k]) != md5(d[k])]
    flag = "✓" if not (only_s or only_d or mism) else "✗"
    if flag == "✗":
        ok = False
    print("  %s %-46s 源 %2d / 包 %2d  %s" % (
        flag, name, len(s), len(d),
        "md5 全一致" if not mism else "%d 个内容不同" % len(mism)))
    if only_s:
        print("      源有包无:", only_s)
    if only_d:
        print("      包有源无:", only_d)
    if mism:
        print("      内容不同:", mism[:5])

print("\n=== 结论：10 个结果包与源逐一对应、内容完全一致 ===" if ok else "\n=== 结论：有差异 ===")
