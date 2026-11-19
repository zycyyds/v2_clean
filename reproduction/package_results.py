#!/usr/bin/env python3
"""把我们跑出来的 5 个数据集 + 对应基线（codex / cc）打包到 /Users/mac/Downloads/ceshi1。

命名规则：
  A<N>_<数据集>_<档位>_<跑的人>      ← 我们的 agent（pi + val harness，mimicraw 系列）
  B<N>_<数据集>_<档位>_基线_<跑的人>  ← 基线（codex / cc Claude Code）
  Z_<附加>                          ← 找到但不属于那 10 个的基线

每个文件夹里：
  error_metrics_<口径>.json   错误级指标（几个口径就几个）
  native_cells_*.json         MIMIC 原生错误格专项（有原生层的才有）
  score_report.json           harness 的 test 评分（我们的）
  overall_score.json          评分器的 composite（基线的）
  run_report.json             harness 的 run 报告（我们的）
  summary.json                关键数字 + 出处（路径/数据集/gold）
  README.md                   人读的说明
"""
import json
import shutil
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
WTK = PROJ / ".claude/worktrees/k29-mimicraw"
WTE = PROJ / ".claude/worktrees/eval-movies-cleaning"
RUNS = PROJ / "runs"
OUT = Path("/Users/mac/Downloads/ceshi1")

CAL_NAMES = {
    "v29": "v29（数值 + key还原 + 排除gold外行 + log判据）",
    "str": "str（纯字符串 + key还原 + 排除gold外行 + log判据）",
    "oldsw": "oldsw（数值 + 关key还原 + 不排除gold外行 + log判据）",
    "old": "old（纯字符串 + 关key还原 + 不排除gold外行 + log判据）",
    "goldv29": "goldv29（数值 + key还原 + 排除gold外行 + gold判据）",
    "goldstr": "goldstr（纯字符串 + key还原 + 排除gold外行 + gold判据）",
    "goldold": "goldold（纯字符串 + 关key还原 + 不排除gold外行 + gold判据）",
}

# ─────────────────────────── 我们的 5 档 ───────────────────────────
OURS = [
    {
        "dir": "A1_K29_mimicraw_1.6pct_我们的agent_pi+val",
        "dataset": "K29 mimicraw 1.6pct",
        "who": "我们的 agent（pi + val harness，seed 666）",
        "note": "数据 = K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror（MIMIC 原生错误层已修好）",
        "cal": {
            "v29": RUNS / "k29_mimicraw_16_agent_error_metrics_v29.json",
            "str": RUNS / "k29_mimicraw_16_agent_error_metrics_ratio0p2629_str.json",
            "oldsw": RUNS / "k29_mimicraw_16_agent_error_metrics_v29_oldsw.json",
            "old": RUNS / "k29_mimicraw_16_agent_error_metrics_ratio0p2629_old.json",
            "goldv29": RUNS / "k29_mimicraw_16_agent_error_metrics_goldv29.json",
            "goldstr": RUNS / "k29_mimicraw_16_agent_error_metrics_goldstr.json",
            "goldold": RUNS / "k29_mimicraw_16_agent_error_metrics_goldold.json",
        },
        "native": [RUNS / "k29_mimicraw_16_agent_native_cells_ratio0p2629_numeric.json",
                   RUNS / "k29_mimicraw_16_agent_native_cells_ratio0p2629_string.json"],
        "host": WTK / "k29_agent_run/experiments/k29_mimicraw_1p6_test_seed666_v1/host",
        "val_host": WTK / "k29_agent_run/experiments/k29_mimicraw_1p6_val_seed666_v1/host",
    },
    {
        "dir": "A2_K29_mimicraw_4.7pct_我们的agent_pi+val",
        "dataset": "K29 mimicraw 4.7pct",
        "who": "我们的 agent（pi + val harness，seed 666）",
        "note": "数据 = K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror",
        "cal": {
            "v29": RUNS / "k29_mimicraw_47_agent_error_metrics_v29.json",
            "str": RUNS / "k29_mimicraw_47_agent_error_metrics_v29_str.json",
            "oldsw": RUNS / "k29_mimicraw_47_agent_error_metrics_v29_oldsw.json",
            "old": RUNS / "k29_mimicraw_47_agent_error_metrics_v29_old.json",
            "goldv29": RUNS / "k29_mimicraw_47_agent_error_metrics_goldv29.json",
            "goldstr": RUNS / "k29_mimicraw_47_agent_error_metrics_goldstr.json",
            "goldold": RUNS / "k29_mimicraw_47_agent_error_metrics_goldold.json",
        },
        "native": [RUNS / "k29_mimicraw_47_agent_native_cells_restore_numeric.json",
                   RUNS / "k29_mimicraw_47_agent_native_cells_restore_string.json"],
        "host": WTK / "k29_agent_run/experiments/k29_mimicraw_4p7_test_seed666_v1/host",
        "val_host": WTK / "k29_agent_run/experiments/k29_mimicraw_4p7_val_seed666_v1/host",
    },
    {
        "dir": "A3_ICU5000_mimicraw_1.6pct_我们的agent_pi+val",
        "dataset": "ICU5000 mimicraw 1.6pct",
        "who": "我们的 agent（pi + val harness，seed 666）",
        "note": "数据 = icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror",
        "cal": {c: RUNS / ("icu5000_mimicraw_16_agent_error_metrics_%s_run20261118_021004.json" % c)
                for c in ("v29", "str", "oldsw", "old", "goldv29", "goldstr", "goldold")},
        "native": [RUNS / "icu5000_mimicraw_16_agent_native_cells_run20261118_021004_numeric.json",
                   RUNS / "icu5000_mimicraw_16_agent_native_cells_run20261118_021004_string.json"],
        "host": WTK / "icu5000_agent_run/experiments/icu5000_mimicraw_1p6_test_seed666_v1/host",
        "val_host": WTK / "icu5000_agent_run/experiments/icu5000_mimicraw_1p6_val_seed666_v1/host",
    },
    {
        "dir": "A4_ICU5000_mimicraw_4.7pct_我们的agent_pi+val",
        "dataset": "ICU5000 mimicraw 4.7pct",
        "who": "我们的 agent（pi + val harness，seed 666）",
        "note": "数据 = icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror",
        "cal": {c: RUNS / ("icu5000_mimicraw_47_agent_error_metrics_%s_run20261118_4p7.json" % c)
                for c in ("v29", "str", "oldsw", "old", "goldv29", "goldstr")},
        "native": [RUNS / "icu5000_mimicraw_47_agent_native_cells_run20261118_4p7_numeric.json",
                   RUNS / "icu5000_mimicraw_47_agent_native_cells_run20261118_4p7_string.json"],
        "host": WTK / "icu5000_agent_run/experiments/icu5000_mimicraw_4p7_test_seed666_v1/host",
        "val_host": WTK / "icu5000_agent_run/experiments/icu5000_mimicraw_4p7_val_seed666_v1/host",
    },
    {
        "dir": "A5_I48_mimicraw2_1.6pct_我们的agent_pi+val",
        "dataset": "I48 mimicraw2 1.6pct",
        "who": "我们的 agent（pi + val harness，seed 666）",
        "note": "数据 = I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M"
                "（本档 val 因 provider 流式错误只跑到第 5 轮，用驱动自带 finalize() 收尾）",
        "cal": {c: RUNS / ("i48_mimicraw2_16_agent_error_metrics_%s_run20261119_1p6.json" % c)
                for c in ("v29", "str", "oldsw", "old", "goldv29", "goldstr", "goldold")},
        "native": [RUNS / "i48_mimicraw2_16_agent_native_cells_run20261119_1p6_numeric.json",
                   RUNS / "i48_mimicraw2_16_agent_native_cells_run20261119_1p6_string.json"],
        "host": WTK / "i48_agent_run/experiments/i48_mimicraw2_1p6_test_seed666_v1/host",
        "val_host": WTK / "i48_agent_run/experiments/i48_mimicraw2_1p6_val_seed666_v1/host",
    },
]

# ─────────────────────────── 基线（codex / cc） ───────────────────────────
BASELINES = [
    {
        "dir": "B1_K29_codex基线_无同档mimicraw",
        "dataset": "K29（eval-rest 数据集，非 mimicraw）",
        "who": "codex（OpenAI Codex CLI，2026-09-19）",
        "note": "⚠️ K29 的 mimicraw 两档**没有**同档基线；这里放的是 K29 家族 codex 在"
                "「原始 eval-rest 数据集」上的两个尝试（K29_clean_eval_10_20_rest）",
        "files": [("codex_尝试A_scorer.log", RUNS / "k29_codex_score/scorer.log"),
                  ("codex_尝试B_scorer.log", RUNS / "k29_codex_v6/scorer.log")],
    },
    {
        "dir": "B2_ICU5000_1.6pct_codex基线",
        "dataset": "ICU5000 1.6pct（原始注入错误数据集）",
        "who": "codex（OpenAI Codex CLI）",
        "note": "gold = icu_5000_v2_rawerror_test1.6pct_614M/test/reference",
        "files": [("codex_error_metrics_v29.json", RUNS / "icu5000_less_codex_error_metrics_v29.json"),
                  ("codex_overall_score.json", RUNS / "icu5000_less_codex_overall_score.json")],
    },
    {
        "dir": "B3_ICU5000_4.7pct_codex基线",
        "dataset": "ICU5000 4.7pct（原始注入错误数据集）",
        "who": "codex（OpenAI Codex CLI）",
        "note": "gold = icu_5000_v2_rawerror_test4.7pct_614M/test/reference",
        "files": [("codex_error_metrics_v28.json", RUNS / "icu5000_more_codex_error_metrics.json"),
                  ("codex_overall_score.json", RUNS / "icu5000_more_codex_overall_score.json")],
    },
    {
        "dir": "B4_ICU5000_4.7pct_cc基线_ClaudeCode",
        "dataset": "ICU5000 4.7pct（原始注入错误数据集）",
        "who": "cc（Claude Code，2026-10-25）",
        "note": "gold = icu_5000_v2_rawerror_test4.7pct_614M/test/reference",
        "files": [("cc_error_metrics_v29.json", RUNS / "icu5000_more_cc_error_metrics_v29.json"),
                  ("cc_overall_score.json", RUNS / "icu5000_more_cc_overall_score.json")],
    },
    {
        "dir": "B5_I48_1.6pct_cc基线_ClaudeCode",
        "dataset": "I48 1.6pct（原始注入错误数据集，非 mimicraw2）",
        "who": "cc（Claude Code）",
        "note": "gold = I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M/test/reference",
        "files": [("cc_error_metrics_v29.json", RUNS / "i48_less_cc_v1_cc_error_metrics_v29.json"),
                  ("cc_overall_score.json", RUNS / "i48_less_cc_v1_overall_score.json")],
    },
]

# ─────────────────────────── 附加基线（找到但不属那 10 个） ───────────────────────────
EXTRA = [
    {
        "dir": "Z1_ICU5000_1.6pct_cc基线_ClaudeCode",
        "dataset": "ICU5000 1.6pct（原始注入错误数据集）", "who": "cc（Claude Code）",
        "note": "gold = icu_5000_v2_rawerror_test1.6pct_614M/test/reference",
        "files": [("cc_error_metrics_v29.json", RUNS / "icu5000_less_cc_v1_error_metrics_v29.json"),
                  ("cc_overall_score.json", RUNS / "icu5000_less_cc_v1_overall_score.json")],
    },
    {
        "dir": "Z2_N18_1.6pct_cc基线_ClaudeCode",
        "dataset": "N18 1.6pct", "who": "cc（Claude Code，2026-10-25）", "note": "非本次 5 档",
        "files": [("cc_error_metrics_v29.json", RUNS / "n18_less_cc_v1_error_metrics_v29.json"),
                  ("cc_overall_score.json", RUNS / "n18_less_cc_v1_overall_score.json")],
    },
    {
        "dir": "Z3_N18_4.7pct_cc基线_ClaudeCode",
        "dataset": "N18 4.7pct", "who": "cc（Claude Code）", "note": "非本次 5 档",
        "files": [("cc_error_metrics_v29.json", RUNS / "n18_more_v1_cc_error_metrics_v29.json"),
                  ("cc_overall_score.json", RUNS / "n18_more_v1_overall_score.json")],
    },
    {
        "dir": "Z4_K29_1800档_cc基线_ClaudeCode",
        "dataset": "K29 1800 低密度档", "who": "cc（Claude Code）", "note": "非本次 5 档",
        "files": [("cc_error_metrics_v29.json", RUNS / "k29_1800_cc_v3_error_metrics_v29.json")],
    },
    {
        "dir": "Z5_I48_codex基线_evalrest",
        "dataset": "I48（eval-rest 数据集）", "who": "codex（2026-09-19）",
        "note": "非 mimicraw2 数据集",
        "files": [("codex_scorer.log", RUNS / "i48_codex/scorer.log")],
    },
    {
        "dir": "Z6_ICU5000_codex基线_evalrest",
        "dataset": "ICU5000（eval-rest 数据集）", "who": "codex（2026-09-19）",
        "note": "两个尝试；非 mimicraw 数据集",
        "files": [("codex_尝试A_scorer.log", RUNS / "icu5000_codex/scorer.log"),
                  ("codex_尝试B_scorer.log", RUNS / "icu5000_codex_v2/scorer.log")],
    },
]


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_metrics(d):
    if not d or "detection" not in d:
        return None
    b = d.get("buckets", {})
    return {
        "det_f1": round(d["detection"]["F1"], 6), "rep_f1": round(d["repair"]["F1"], 6),
        "det_p": round(d["detection"]["P"], 6), "det_r": round(d["detection"]["R"], 6),
        "rep_p": round(d["repair"]["P"], 6), "rep_r": round(d["repair"]["R"], 6),
        "b_clean": b.get("b_clean", 0), "b_wrong": b.get("b_wrong", 0),
        "b_dirty": b.get("b_dirty", 0), "b_missing": b.get("b_missing", 0),
        "fp_det": b.get("fp_det", 0), "natural_noise": b.get("natural_noise", 0),
    }


def write_ours(e, dst):
    dst.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": e["dataset"], "who": e["who"], "note": e["note"], "calibers": {}}
    for cal, src in e["cal"].items():
        if not Path(src).is_file():
            continue
        shutil.copy2(src, dst / ("error_metrics_%s.json" % cal))
        d = read_json(src)
        m = extract_metrics(d)
        if m:
            summary["calibers"][cal] = {"定义": CAL_NAMES.get(cal, cal), **m,
                                        "源文件": str(src)}
    for p in e["native"]:
        if Path(p).is_file():
            shutil.copy2(p, dst / Path(p).name)
            d = read_json(p)
            if d:
                summary["native_cells"] = {
                    "总格数": d.get("n_native_cells"),
                    "修对(b_clean)": d.get("buckets", {}).get("b_clean"),
                    "改错(b_wrong)": d.get("buckets", {}).get("b_wrong"),
                    "没动(b_dirty)": d.get("buckets", {}).get("b_dirty"),
                    "缺行(b_missing)": d.get("buckets", {}).get("b_missing"),
                    "修复率": d.get("repair_rate"),
                    "按类型": d.get("per_detection_mode"),
                }
    for name, p in (("score_report.json", e["host"] / "score_report.json"),
                    ("run_report.json", e["host"] / "run_report.json"),
                    ("val_run_report.json", e["val_host"] / "run_report.json")):
        if Path(p).is_file():
            shutil.copy2(p, dst / name)
            d = read_json(p)
            if name == "score_report.json" and d:
                summary["test_composite"] = (d.get("metrics") or {}).get("composite_score")
            if name == "val_run_report.json" and d:
                summary["val"] = {"status": d.get("status"), "best_score": d.get("best_score"),
                                  "reproducible_score": d.get("reproducible_score"),
                                  "rounds": d.get("rounds")}
    (dst / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return summary


def write_baseline(e, dst):
    dst.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": e["dataset"], "who": e["who"], "note": e["note"], "files": {}}
    for name, src in e["files"]:
        if not Path(src).is_file():
            continue
        shutil.copy2(src, dst / name)
        d = read_json(src)
        m = extract_metrics(d) if d else None
        if m:
            summary["files"][name] = m
        elif d and "metrics" in d:
            summary["files"][name] = {"composite": d["metrics"].get("composite_score"),
                                      "schema_f1": d["metrics"].get("schema_f1"),
                                      "key_structure_f1": d["metrics"].get("key_structure_f1"),
                                      "row_aligned_cell_f1": d["metrics"].get("row_aligned_cell_f1"),
                                      "exact_row_f1": d["metrics"].get("exact_row_f1")}
        elif d:
            summary["files"][name] = {"note": "scorer.log（完整 JSON，含逐文件明细）"}
    (dst / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return summary


def main():
    # ⚠️ 不删已有内容（目标目录由用户指定）；只创建/覆盖同名文件
    OUT.mkdir(parents=True, exist_ok=True)
    ours_sum = []
    for e in OURS:
        ours_sum.append((e, write_ours(e, OUT / e["dir"])))
    for e in BASELINES + EXTRA:
        write_baseline(e, OUT / ("Z_附加基线" if e in EXTRA else e["dir"]) / e["dir"]
                       if e in EXTRA else OUT / e["dir"])
    print("打包完成 →", OUT)
    for e, s in ours_sum:
        print("  %-46s v29 Det/Rep = %s / %s" % (
            e["dir"],
            s["calibers"].get("v29", {}).get("det_f1"),
            s["calibers"].get("v29", {}).get("rep_f1")))


if __name__ == "__main__":
    main()
