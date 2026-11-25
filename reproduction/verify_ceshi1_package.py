#!/usr/bin/env python3
"""核对打包后的 结果包/ 与源目录一一对应（相对路径 + md5 全量）。"""
import csv
import hashlib
import json
from pathlib import Path

csv.field_size_limit(2 ** 31)

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
    "n18/1.6pct_cc基线_ClaudeCode":
        Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/n18-codex-pipeline/"
             "agent_work/n18_codex_16/work/test_out"),
    "I48/4.7pct_cc基线_ClaudeCode":
        PROJ / "agent_work/i48_more_v1/output/test",
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

print("\n=== 校正类（单表纠错）四数据集 ===")
CORR = {
    "hospital": ("raha_hospital_curated10_val20_dense_v1", "index"),
    "beers": ("beers_curated10_val20_v1", "index"),
    "rayyan": ("rayyan_curated10_val20_v1", "id"),
    "tax": ("tax_curated10_val20_v1", "__row_id"),
}
BASE_DIR = PROJ / "对比结果"
BASE_FILE = {"hospital": ("hospital", "result_package/cleaned.csv"),
             "beers": ("beers_correction_v1", "result_package/beers.csv"),
             "rayyan": ("rayyan_clean_v1", "result_package/cleaned.csv"),
             "tax": ("tax_correction_v1", "result_package/tax.csv")}
CORR_DS = PROJ / "datasets/correction"
corr_ok = True
for d, (ds, key) in CORR.items():
    # ① agent 结果包：与跑批源逐一 md5
    src = WT / ("correction_agent_run/experiments/%s_correction_test_seed666_v1"
                "/host/test_result_package" % d)
    pkg = OUT / "校正类" / d / "结果包" / ("%s.csv" % d)
    m_agent = pkg.exists() and md5(src / ("%s.csv" % d)) == md5(pkg)
    # ② gold：与数据集的 reference_private 逐一 md5
    gsrc = CORR_DS / ds / "correction/reference_private" / ("%s.csv" % d)
    gpkg = OUT / "校正类/gold/reference" / ("%s.csv" % d)
    m_gold = gpkg.exists() and md5(gsrc) == md5(gpkg)
    # ③ 基线：包内是**变换后**的表（补 key 列 + 取 test 行子集）⇒ 重新推导再比 md5
    bdir, brel = BASE_FILE[d]
    bsrc = BASE_DIR / bdir / brel
    bpkg = OUT / "校正类" / ("%s_基线_ClaudeCode" % d) / "结果包" / ("%s.csv" % d)
    m_base = False
    if bsrc.exists() and bpkg.exists():
        with open(bsrc, newline="", encoding="utf-8", errors="replace") as f:
            rd = csv.reader(f)
            bh = next(rd)
            brows = list(rd)
        with open(bpkg, newline="", encoding="utf-8", errors="replace") as f:
            rd = csv.reader(f)
            ph = next(rd)
            prows = list(rd)
        if key in bh:
            with open(gsrc, newline="", encoding="utf-8", errors="replace") as f:
                rd = csv.reader(f)
                gh = next(rd)
                grows = list(rd)
            bk, gk = bh.index(key), gh.index(key)
            keep = {r[gk] for r in grows}
            bmap = {}
            for r in brows:
                bmap.setdefault(r[bk], r)
            exp_hdr, exp_rows = list(bh), [bmap[r[gk]] for r in grows if r[gk] in bmap]
        else:  # tax：按原表行号补 key 列
            with open(gsrc, newline="", encoding="utf-8", errors="replace") as f:
                rd = csv.reader(f)
                gh = next(rd)
                grows = list(rd)
            gk = gh.index(key)
            exp_hdr = [key] + list(bh)
            exp_rows = [[r[gk]] + list(brows[int(r[gk])]) for r in grows]
        m_base = (ph == exp_hdr and len(prows) == len(exp_rows)
                  and all(a == b for a, b in zip(prows, exp_rows)))
    corr_ok &= m_agent and m_gold and m_base
    print("  %s %-26s agent结果包 %s | gold %s | 基线(变换后重推) %s"
          % ("✓" if (m_agent and m_gold and m_base) else "✗", d,
             "md5 一致" if m_agent else "不一致",
             "md5 一致" if m_gold else "不一致",
             "逐行一致" if m_base else "不一致"))
ok &= corr_ok

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
