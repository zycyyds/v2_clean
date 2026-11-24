#!/usr/bin/env python3
"""tax train 自检：我独立复跑冻结产物的 pipeline 到 train raw，再与 train/reference 比对。

用法：python3 tax_train_selfcheck.py <tier>
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
DS = Path("/Users/mac/PycharmProjects/v2_clean/datasets/correction/tax_curated10_val20_v1")
KEY = "__row_id"

tier = sys.argv[1] if len(sys.argv) > 1 else "tax"
snap = WT / ("correction_agent_run/experiments/%s_correction_val_seed666_v1/host/reproducible_snapshot" % tier)
entry = snap / "scripts/clean.py"
if not entry.exists():
    print("  ❌ 缺 %s" % entry)
    raise SystemExit(2)

out = Path(tempfile.mkdtemp(prefix="tax_train_selfcheck_"))
cmd = [sys.executable, str(entry),
       "--raw_root", str(DS / "train/raw"),
       "--output_dir", str(out),
       "--train_reference", str(DS / "train/reference")]
print("  $ %s" % " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
print("  rc=%d" % r.returncode)
if r.stderr.strip():
    print("  stderr(尾部 800 字): %s" % r.stderr.strip()[-800:])
produced = out / "tax.csv"
if not produced.exists():
    print("  ❌ 没产出 tax.csv")
    raise SystemExit(3)
print("  产出: %s (%d 字节)" % (produced, produced.stat().st_size))

subprocess.run([sys.executable, str(WT / "reproduction/correction_cell_diff.py"),
                "--result", str(produced),
                "--gold", str(DS / "train/reference/tax.csv"),
                "--key", KEY])
subprocess.run([sys.executable, str(WT / "reproduction/correction_error_metrics.py"),
                "--diff", str(DS / "host_private/cell_diff.jsonl"),
                "--result", str(produced),
                "--gold", str(DS / "train/reference/tax.csv"),
                "--split", "train", "--key", KEY, "--label", "tax train(独立复跑)"])
shutil.rmtree(out, ignore_errors=True)
