#!/usr/bin/env python3
"""校正类四档：轮次曲线 / 修复轮 / 用量 / 断点（只出聚合）。"""
import glob
import json
import os
import time
from pathlib import Path

RUN = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw/correction_agent_run")
TIERS = ["hospital", "beers", "rayyan", "tax"]

print("=" * 92)
print("【轮次曲线】每个 harness round 的分数（round_*.json）")
print("=" * 92)
for t in TIERS:
    vh = RUN / "experiments" / f"{t}_correction_val_seed666_v1/host"
    if not vh.exists():
        print("%-9s （无）" % t)
        continue
    curves = []
    for p in sorted(glob.glob(str(vh / "rounds/round_*.json"))):
        d = json.load(open(p))
        m = d.get("metrics", {})
        curves.append((os.path.basename(p), m.get("composite_score")))
    print("%-9s %s" % (t, " → ".join("%s=%s" % (n.replace("round_", "r").replace(".json", ""), s)
                                     for n, s in curves)))

print()
print("=" * 92)
print("【每档汇总】收尾原因 / 轮数 / 修复轮 / 用量")
print("=" * 92)
hdr = "%-9s | %-22s | %-8s | %-6s | %-6s | %-8s | %-10s | %s" % (
    "档", "val 状态 / 收尾", "轮数", "修复", "模型调用", "in token", "工具调用", "工具错误")
print(hdr)
print("-" * 118)
for t in TIERS:
    vh = RUN / "experiments" / f"{t}_correction_val_seed666_v1/host"
    th = RUN / "experiments" / f"{t}_correction_test_seed666_v1/host"
    vr = json.load(open(vh / "run_report.json"))
    tr = json.load(open(th / "run_report.json"))
    u = vr.get("validation_usage", {})
    print("%-9s | %-22s | %-8s | %-6s | %-6s | %-8s | %-10s | %s" % (
        t,
        "%s / %s" % (vr.get("status"), vr.get("stop_reason")),
        vr.get("rounds"), vr.get("repair_rounds"),
        u.get("model_calls"), "%.1fM" % (u.get("input_tokens", 0) / 1e6),
        u.get("tool_calls"), u.get("tool_errors")))
    print("%-9s |   test: %s %.6f  replay=%s  轮=%s" % (
        "", tr.get("status"), tr.get("score", -1), tr.get("replay_status"),
        tr.get("test_execution_count")))

print()
print("=" * 92)
print("【断点】每档的失败/修复事件（从日志里取 harness 层的关键行）")
print("=" * 92)
KEYS = ("EXECUTION_FAILED", "replay repair", "worker_crash", "patience",
        "target_score", "independent replay", "round ", "Traceback")
for t in TIERS:
    lg = RUN / "logs" / f"{t}_correction_val_seed666_v1.log"
    if not lg.exists():
        continue
    hits = []
    with open(lg, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if line.startswith("[Harness]"):
                hits.append((i, line.rstrip()[:120]))
    print("\n--- %s (%d 条 harness 行) ---" % (t, len(hits)))
    for i, l in hits:
        print("   L%-7d %s" % (i, l))

print()
print("=" * 92)
print("【时间线】实验目录的创建/修改时间（看是否一个一个串行）")
print("=" * 92)
rows = []
for t in TIERS:
    for kind, sub in (("val", f"{t}_correction_val_seed666_v1"), ("test", f"{t}_correction_test_seed666_v1")):
        d = RUN / "experiments" / sub
        rp = d / "host/run_report.json"
        if rp.exists():
            rows.append((os.path.getmtime(rp), t, kind, sub))
for m, t, kind, sub in sorted(rows):
    print("   %s  %-9s %-5s  %s" % (time.strftime("%m-%d %H:%M:%S", time.localtime(m)), t, kind, sub))
