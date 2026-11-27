#!/usr/bin/env python3
"""deepprep 四档的 str 口径：错误级能不能算出来？先直接试，失败再降级。

降级策略（在**临时副本**上做，不动包内产物）：
  · 空文件 / 无表头的表 ⇒ 换成 0 字节（builder 会跳过 ⇒ 该表注入错误全记 b_missing）
  · 缺 key 列的表 ⇒ 同样跳过（agent 交不出可用 key ⇒ 这些行等于没修）
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
EM = WT / "score_only/error_metrics"
TMP = Path("/Users/mac/.claude/jobs/448aad57/tmp")
PKG = Path("/Users/mac/Downloads/ceshi1")
DSROOT = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集")

JOBS = [
    ("n18/deepprep-less", "n18", "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"),
    ("n18/deepprep-more", "n18", "N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"),
    ("I48/deepprep-less", "i48", "I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M"),
    ("I48/deepprep-more", "i48", "I48_atrial_fibrillation_v2_rawerror_mimicraw2_test4.7pct_411M"),
]
KEY = {"n18": "ceshi1_n18", "i48": "ceshi1_i48"}


def run(result_dir, ds, dskey, out_json):
    c = subprocess.run([sys.executable, str(EM / "build_eval_cache.py"),
                        "--result-dir", str(result_dir), "--ref-dir", str(ds / "test/reference"),
                        "--ds-key", dskey], capture_output=True, text=True, timeout=14400)
    if c.returncode != 0:
        return None, "cache rc=%d: %s" % (c.returncode, (c.stderr or c.stdout)[-400:])
    r = subprocess.run([sys.executable, str(EM / "compute_error_metrics.py"),
                        "--dataset-dir", str(ds), "--ds-key", dskey, "--split", "test",
                        "--out", str(out_json), "--string-compare"],
                       capture_output=True, text=True, timeout=14400)
    if r.returncode != 0:
        return None, "metrics rc=%d: %s" % (r.returncode, (r.stderr or r.stdout)[-400:])
    return out_json, ""


for name, tier, dsname in JOBS:
    ds = DSROOT / tier / dsname
    res = PKG / name / "output"
    dskey = KEY[tier] + ("_dp16" if "less" in name else "_dp47")
    print("=" * 70)
    print("=== %s" % name)
    t0 = time.time()
    oj = TMP / ("err_%s_str.json" % dskey)
    out, err = run(res, ds, dskey, oj)
    if out:
        import json
        d = json.load(open(out))
        det, rep, b = d["detection"], d["repair"], d["buckets"]
        print("   直接跑通 ✅ str Det F1=%.6f Rep F1=%.6f | b_clean=%s b_wrong=%s b_dirty=%s b_missing=%s"
              % (det.get("F1"), rep.get("F1"), f"{b.get('b_clean',0):,}", f"{b.get('b_wrong',0):,}",
                 f"{b.get('b_dirty',0):,}", f"{b.get('b_missing',0):,}"))
        print("   (%.0fs)" % (time.time() - t0))
        continue
    print("   直接跑失败：%s" % err.strip()[:300])
    # 降级：临时副本，把坏表换成 0 字节
    tmp_res = TMP / ("dp_%s_res" % dskey)
    if tmp_res.exists():
        shutil.rmtree(tmp_res)
    shutil.copytree(res, tmp_res)
    import csv
    csv.field_size_limit(2 ** 31)
    broken = []
    for p in sorted(tmp_res.rglob("*.csv")):
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            try:
                hdr = next(csv.reader(f))
            except StopIteration:
                hdr = []
        if not hdr:
            broken.append(str(p.relative_to(tmp_res)))
            p.write_bytes(b"")
    print("   降级副本：%s" % tmp_res)
    print("   坏表（换成 0 字节）: %s" % (broken or "无"))
    t1 = time.time()
    out, err2 = run(tmp_res, ds, dskey, oj)
    if out:
        import json
        d = json.load(open(out))
        det, rep, b = d["detection"], d["repair"], d["buckets"]
        print("   降级跑通 ✅ str Det F1=%.6f Rep F1=%.6f | b_clean=%s b_wrong=%s b_dirty=%s b_missing=%s"
              % (det.get("F1"), rep.get("F1"), f"{b.get('b_clean',0):,}", f"{b.get('b_wrong',0):,}",
                 f"{b.get('b_dirty',0):,}", f"{b.get('b_missing',0):,}"))
        print("   (%.0fs)" % (time.time() - t1))
    else:
        print("   降级也失败：%s" % err2.strip()[:400])
print("done")
