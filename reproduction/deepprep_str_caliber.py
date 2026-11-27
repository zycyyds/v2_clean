#!/usr/bin/env python3
"""deepprep 四档的 **str** 口径错误级（配合 build_eval_cache 的降级补丁）。

① 先**回归验证**补丁：重跑一个已知档位（n18 1.6pct 我们的 agent，期望 str 0.997021 / 0.770292）
   ⇒ 补丁不影响已有数字。
② 再算四个 deepprep 目录的 str 口径。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
EM = WT / "score_only/error_metrics"
TMP = Path("/Users/mac/.claude/jobs/448aad57/tmp")
PKG = Path("/Users/mac/Downloads/ceshi1")
DSROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集")

# (名字, 结果目录, 数据集, ds-key)
JOBS = [
    ("【回归】n18/1.6pct_我们的agent", PKG / "n18/1.6pct_我们的agent_pi+val_mimicraw/结果包",
     DSROOT / "n18/N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M", "ceshi1_n18_agent16_reg"),
    ("n18/deepprep-less", PKG / "n18/deepprep-less/output",
     DSROOT / "n18/N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M", "ceshi1_n18_dp16"),
    ("n18/deepprep-more", PKG / "n18/deepprep-more/output",
     DSROOT / "n18/N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M", "ceshi1_n18_dp47"),
    ("I48/deepprep-less", PKG / "I48/deepprep-less/output",
     DSROOT / "i48/I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M", "ceshi1_i48_dp16"),
    ("I48/deepprep-more", PKG / "I48/deepprep-more/output",
     DSROOT / "i48/I48_atrial_fibrillation_v2_rawerror_mimicraw2_test4.7pct_411M", "ceshi1_i48_dp47"),
]


def run(res, ds, dskey, oj):
    c = subprocess.run([sys.executable, str(EM / "build_eval_cache.py"),
                        "--result-dir", str(res), "--ref-dir", str(ds / "test/reference"),
                        "--ds-key", dskey], capture_output=True, text=True, timeout=28800)
    if c.returncode != 0:
        return None, "cache rc=%d %s" % (c.returncode, (c.stderr or c.stdout)[-500:])
    r = subprocess.run([sys.executable, str(EM / "compute_error_metrics.py"),
                        "--dataset-dir", str(ds), "--ds-key", dskey, "--split", "test",
                        "--out", str(oj), "--string-compare"],
                       capture_output=True, text=True, timeout=28800)
    if r.returncode != 0:
        return None, "metrics rc=%d %s" % (r.returncode, (r.stderr or r.stdout)[-500:])
    return oj, ""


out = {}
for name, res, ds, dskey in JOBS:
    print("=" * 78, flush=True)
    print("=== %s" % name, flush=True)
    t0 = time.time()
    oj = TMP / ("err_%s_str.json" % dskey)
    p, err = run(res, ds, dskey, oj)
    if not p:
        print("   ❌ %s" % err.strip()[:600], flush=True)
        continue
    d = json.load(open(p))
    det, rep, b = d["detection"], d["repair"], d["buckets"]
    print("   str Det F1=%.6f Rep F1=%.6f (P/R: det %.4f/%.4f, rep %.4f/%.4f)"
          % (det.get("F1"), rep.get("F1"), det.get("P"), det.get("R"),
             rep.get("P"), rep.get("R")), flush=True)
    print("   b_clean=%s b_wrong=%s b_dirty=%s b_missing=%s fp_det=%s"
          % (f"{b.get('b_clean',0):,}", f"{b.get('b_wrong',0):,}", f"{b.get('b_dirty',0):,}",
             f"{b.get('b_missing',0):,}", f"{b.get('fp_det',0):,}"), flush=True)
    print("   (%.0fs)" % (time.time() - t0), flush=True)
    out[name] = {"det_f1": det.get("F1"), "rep_f1": rep.get("F1"),
                 "buckets": b}

(TMP / "deepprep_str_metrics.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n写出 deepprep_str_metrics.json", flush=True)
