"""提取每次 val 实验的逐轮 composite 分数（用于「同种子不可复现」证据）。"""
import json
from pathlib import Path

EXP = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments")

for d in sorted(EXP.iterdir()):
    rounds = d / "host" / "rounds"
    if not rounds.is_dir():
        continue
    rows = []
    for f in sorted(rounds.glob("round_*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        sc = r.get("score", r.get("composite_score"))
        if sc is None and isinstance(r.get("metrics"), dict):
            sc = r["metrics"].get("composite_score")
        if sc is None:
            sc = r.get("private_feedback", {}).get("score")
        rows.append((f.stem, sc, r.get("submission_status"), r.get("promoted")))
    if not rows:
        continue
    print(f"\n=== {d.name} ===")
    for name, sc, st, pr in rows:
        print(f"  {name}  score={sc}  submission_status={st}  promoted={pr}")
