#!/usr/bin/env python3
"""把 `对比结果/` 那批旧基线产出整成「与当前 test gold 可比」的表，拷进 ceshi1，并打分。

背景：`对比结果/` 是 2026-09 那批手动跑批的基线，跑在**旧版数据集**上：
  - hospital / beers / rayyan：输出是**整表**（1000 / 2410 / 1000 行），含当时的 train+val 30 行；
    当前 test gold 只剩 970 / 2380 / 970 行 ⇒ 按 key 过滤即可。
  - tax：输出 200,000 行且**没有 `__row_id` 列**；gold 的 `__row_id` 就是原表行号（0..199999），
    基线输出保持了原行序 ⇒ 按位置补回 `__row_id` 并取 test 子集。
⇒ 变换只做「补 key 列 + 取 test 行子集」，不改任何单元格值。

打分：`score_only/pi_test_scorer.py`（harness 自己的打分器），gold = correction/reference_private，
manifest = evaluation_manifests/raha_correction_<ds>_v1.json
⇒ agent 的结果包也用**同一个打分器在同一份 gold 上重打**，保证两张表同口径可比。
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

csv.field_size_limit(2 ** 31)
REPO = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")
C = Path("/Users/mac/PycharmProjects/v2_clean/datasets/correction")
BASE = Path("/Users/mac/PycharmProjects/v2_clean/对比结果")
DST = Path("/Users/mac/Downloads/ceshi1/校正类")
TMP = Path("/Users/mac/.claude/jobs/448aad57/tmp")

DS = {"hospital": ("raha_hospital_curated10_val20_dense_v1", "index"),
      "beers": ("beers_curated10_val20_v1", "index"),
      "rayyan": ("rayyan_curated10_val20_v1", "id"),
      "tax": ("tax_curated10_val20_v1", "__row_id")}
BASEFILE = {"hospital": ("hospital", "result_package/cleaned.csv"),
            "beers": ("beers_correction_v1", "result_package/beers.csv"),
            "rayyan": ("rayyan_clean_v1", "result_package/cleaned.csv"),
            "tax": ("tax_correction_v1", "result_package/tax.csv")}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        return hdr, list(rd)


def write(p, hdr, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)


scores = {}
for d, (ds, key) in DS.items():
    gold_p = C / ds / "correction/reference_private" / ("%s.csv" % d)
    gh, grows = read(gold_p)
    bdir, brel = BASEFILE[d]
    bh, brows = read(BASE / bdir / brel)

    if key in bh:
        # 按 key 取 test 子集
        gk = gh.index(key)
        bk = bh.index(key)
        keep = {r[gk] for r in grows}
        bmap = {}
        for r in brows:
            bmap.setdefault(r[bk], r)
        sel = [bmap[r[gk]] for r in grows if r[gk] in bmap]
        out_hdr = list(bh)
        mode = "按 key(%s) 取 test 行子集" % key
    else:
        # tax：基线无 key 列 ⇒ 按位置补回 gold 的 __row_id，再取 test 行
        gk = gh.index(key)
        sel = []
        for r in grows:
            k = int(r[gk])
            sel.append([r[gk]] + list(brows[k]))
        out_hdr = [key] + list(bh)
        mode = "按原表行号补回 `__row_id` 并取 test 行子集（基线保持了原行序）"

    out_dir = DST / ("%s_基线_ClaudeCode" % d) / "结果包"
    out_p = out_dir / ("%s.csv" % d)
    if out_p.exists():
        print("FATAL: 目标已存在，不覆盖:", out_p)
        sys.exit(2)
    write(out_p, out_hdr, sel)
    print("=== %s" % d)
    print("   变换：%s" % mode)
    print("   基线 %d 行 → 包内 %d 行（gold %d 行）；列 %d → %d"
          % (len(brows), len(sel), len(grows), len(bh), len(out_hdr)))
    assert len(sel) == len(grows), "行数与 gold 不一致：%d vs %d" % (len(sel), len(grows))

    # 打分：基线 vs agent（同一打分器、同一 gold）
    for label, root in (("基线", out_dir),
                        ("agent", DST / d / "结果包")):
        man = REPO / ("evaluation_manifests/raha_correction_%s_v1.json" % d)
        oj = TMP / ("score_correction_%s_%s.json" % (d, label))
        r = subprocess.run([sys.executable, str(REPO / "score_only/pi_test_scorer.py"),
                            "--result-root", str(root), "--gold-root", str(gold_p.parent),
                            "--evaluation-manifest", str(man)],
                           capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print("   %-6s ❌ rc=%d %s" % (label, r.returncode, (r.stderr or r.stdout)[-300:]))
            continue
        j = json.loads(r.stdout)
        m = j["metrics"]
        scores.setdefault(d, {})[label] = m
        print("   %-6s composite=%.6f  schema=%.4f key=%.4f cell=%.4f exact=%.4f"
              % (label, m["composite_score"], m["schema_f1"], m["key_structure_f1"],
                 m["row_aligned_cell_f1"], m["exact_row_f1"]))

(TMP / "correction_scores.json").write_text(
    json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n写出 correction_scores.json")
