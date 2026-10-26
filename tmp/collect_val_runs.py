"""收集所有 val / test 实验的 run_report + score_report，用于「同种子不可复现」的证据。"""
import json
from pathlib import Path

EXP = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments")

for d in sorted(EXP.iterdir()):
    if not d.is_dir():
        continue
    rep = d / "host" / "run_report.json"
    man = d / "host" / "run_manifest.json"
    if not man.is_file():
        continue
    m = json.loads(man.read_text())
    row = {
        "exp": d.name,
        "prompt_sha256": (m.get("prompt_sha256") or "")[:12],
        "max_rounds": m.get("max_rounds"),
    }
    if rep.is_file():
        r = json.loads(rep.read_text())
        row.update(
            status=r.get("status"),
            rounds=r.get("rounds"),
            best=r.get("best_score"),
            repro=r.get("reproducible_score"),
            stop=r.get("stop_reason"),
        )
    else:
        row.update(status="(no run_report)", rounds=None, best=None, repro=None, stop=None)
    sr = d / "host" / "score_report.json"
    if sr.is_file():
        s = json.loads(sr.read_text())
        row["test_composite"] = s.get("metrics", {}).get("composite_score")
    print(json.dumps(row, ensure_ascii=False))
