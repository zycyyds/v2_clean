#!/usr/bin/env python3
"""核对打包后的 结果包/ 与源目录一一对应（相对路径 + md5 全量）。"""
import hashlib
import json
from pathlib import Path

OUT = Path("/Users/mac/Downloads/ceshi1")
WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
CODEX = Path("/Users/mac/Documents/Codex/2026-09-19")

# ⚠️ 老 harness（2026-11-17 前后那批）**没有** `host/test_result_package`：
#    test 的产物写进 `host/replay_runtime/test_replay-*/frozen/result/`，跑完就被清理 ⇒
#    盘上只剩 `host/frozen_snapshot/result`，而那是**train 快照**（K29 1.6pct 只有 20 行，
#    而包里那份是 test 产物 8,093 行）⇒ **两者不可比**。
#    ⇒ 这 5 个旧 agent 包无法再逐一比对（不是内容不同，是源已不可得）。
#    对它们只能核对「打分记录」（score_report）与目录/表数结构。
OLD_SNAPSHOT_ONLY = {
    "K29/1.6pct_我们的agent_pi+val_mimicraw":
        WT / "k29_agent_run/experiments/k29_mimicraw_1p6_test_seed666_v1/host/frozen_snapshot/result",
    "K29/4.7pct_我们的agent_pi+val_mimicraw":
        WT / "k29_agent_run/experiments/k29_mimicraw_4p7_test_seed666_v1/host/frozen_snapshot/results",
    "ICU5000/1.6pct_我们的agent_pi+val_mimicraw":
        WT / "icu5000_agent_run/experiments/icu5000_mimicraw_1p6_test_seed666_v1/host/frozen_snapshot/result",
    "ICU5000/4.7pct_我们的agent_pi+val_mimicraw":
        WT / "icu5000_agent_run/experiments/icu5000_mimicraw_4p7_test_seed666_v1/host/frozen_snapshot/result",
    "I48/1.6pct_我们的agent_pi+val_mimicraw2":
        WT / "i48_agent_run/experiments/i48_mimicraw2_1p6_test_seed666_v1/host/frozen_snapshot/result",
}

SRC = {
    "K29/1.6pct_codex基线": CODEX / "data-cleaning-agent-mimic-iv-k29/outputs/K29",
    "K29/4.7pct_codex基线": CODEX / "data-cleaning-agent-mimic-iv-k29-6/outputs",
    "ICU5000/1.6pct_codex基线": PROJ / "agent_work/icu5000_less_codex/output/test_final3/result_root",
    "ICU5000/4.7pct_codex基线": PROJ / "agent_work/icu5000_more/output/test",
    "I48/1.6pct_cc基线_ClaudeCode": PROJ / "agent_work/i48_less_cc_v1/output/test",
    # ── 2026-11-25 补齐的 n18 / I48 ──
    "I48/4.7pct_我们的agent_pi+val_mimicraw2":
        WT / "i48_agent_run/experiments/i48_mimicraw2_4p7_test_seed666_v1/host/test_result_package",
    "n18/1.6pct_我们的agent_pi+val_mimicraw":
        WT / "n18_agent_run/experiments/n18_mimicraw_1p6_test_seed666_v1/host/test_result_package",
    "n18/4.7pct_我们的agent_pi+val_mimicraw":
        WT / "n18_agent_run/experiments/n18_mimicraw_4p7_test_seed666_v1/host/test_result_package",
    "n18/4.7pct_cc基线_ClaudeCode": PROJ / "agent_work/n18_more_v1/output",
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

print("\n=== 旧 harness 档（test 产物已清理，只能核对结构与评估记录）===")
for name in OLD_SNAPSHOT_ONLY:
    pkg = OUT / name / "结果包"
    n_pkg = len(list(pkg.rglob("*.csv"))) if pkg.exists() else 0
    tier, rest = name.split("/", 1)
    var = "1p6" if "1.6pct" in rest else "4p7"
    base = {"K29": "k29_mimicraw", "ICU5000": "icu5000_mimicraw",
            "I48": "i48_mimicraw2", "n18": "n18_mimicraw"}[tier]
    sr = WT / ("%s_agent_run/experiments/%s_%s_test_seed666_v1/host/score_report.json"
               % (tier.lower(), base, var))
    comp = json.load(open(sr)).get("metrics", {}).get("composite_score") if sr.exists() else None
    print("  ~ %-46s 包 %2d 表   该档 test 评估记录 composite=%s"
          % (name, n_pkg, comp if comp is not None else "?"))
print("    （这 5 个的 test 产物写进 replay_runtime 后被清理，盘上只剩 train 快照 ⇒ 不可逐一比对）")

print("\n=== 结论：%d 个结果包与源逐一对应、内容完全一致 ===" % len(SRC)
      if ok else "\n=== 结论：有差异 ===")
