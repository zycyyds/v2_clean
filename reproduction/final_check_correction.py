#!/usr/bin/env python3
"""校正类四档收官自检：状态 / 分数 / 冻结产物一致性 / 重放 rc。"""
import json
import sys
from pathlib import Path

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
sys.path.insert(0, "/Users/mac/PycharmProjects/v2_clean")
from agent.pi_harness import directory_sha256  # noqa: E402

RUN = WT / "correction_agent_run/experiments"
TIERS = ["hospital", "beers", "rayyan", "tax"]
ok_all = True

print("%-9s | %-28s | %-26s | %s" % ("档", "VAL", "TEST", "冻结产物 sha 一致"))
print("-" * 100)
for t in TIERS:
    vh = RUN / f"{t}_correction_val_seed666_v1/host"
    th = RUN / f"{t}_correction_test_seed666_v1/host"
    if not vh.exists() or not th.exists():
        print("%-9s | ❌ 缺实验目录" % t)
        ok_all = False
        continue
    vr = json.load(open(vh / "run_report.json"))
    tr = json.load(open(th / "run_report.json"))
    vs = vh / "reproducible_snapshot"
    ts = th / "frozen_snapshot"
    h1 = directory_sha256(vs) if vs.exists() else None
    h2 = directory_sha256(ts) if ts.exists() else None
    same = (h1 == h2 == tr.get("frozen_snapshot_sha256"))
    ok_all = ok_all and same
    v_ok = vr.get("status") == "SUCCESS_REPRODUCIBLE"
    t_ok = tr.get("status") == "SUCCESS" and tr.get("replay_status") == "SUCCESS"
    ok_all = ok_all and v_ok and t_ok
    print("%-9s | %s %.6f %-18s | %s %.6f %-16s | %s" % (
        t,
        "✅" if v_ok else "❌", vr.get("best_score", -1), vr.get("stop_reason", "-"),
        "✅" if t_ok else "❌", tr.get("score", -1), tr.get("replay_status", "-"),
        "✅ 一致" if same else "❌ 不一致"))

print("-" * 100)
print("⇒ 四档收官自检：%s" % ("全部通过 ✅" if ok_all else "有问题 ❌"))
