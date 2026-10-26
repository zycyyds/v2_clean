"""评测总表 v29 口径：icu5000(1.6/4.7) + N18(1.6)，含每个结果包路径 + 新旧口径对照。"""
import json
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
RUNS = PROJ / "runs"
WT = PROJ / ".claude/worktrees/eval-movies-cleaning"
AW = PROJ / "agent_work"
AGENT_RUN = WT / "k29_agent_run"

DS16 = PROJ / "datasets/注入错误数据集/icu_5000_v2_rawerror_test1.6pct_614M"
DS47 = PROJ / "datasets/注入错误数据集/icu_5000_v2_rawerror_test4.7pct_614M"
DSN18 = PROJ / "datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_test1.6pct_490M"
DSN18M = PROJ / "datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_test4.7pct_490M"
DSI48 = PROJ / "datasets/注入错误数据集/I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M"
DSI48M = PROJ / "datasets/注入错误数据集/I48_atrial_fibrillation_v2_rawerror_test4.7pct_411M"

# 档位 → 谁 → (composite 文件, 错误分 v29 文件, 错误分 v28 文件, 结果包)
DATA = {
    "icu5000 1.6pct": {
        "gold": DS16 / "test/reference",
        "players": {
            "我们的 agent (pi+val)": (
                RUNS / "icu5000_less_agent_overall_score.json",
                RUNS / "icu5000_less_agent_error_metrics_v29.json",
                RUNS / "icu5000_less_agent_error_metrics.json",
                AGENT_RUN / "experiments/icu5000_1p6_test_seed666_v3/host/test_result_package"),
            "codex": (
                RUNS / "icu5000_less_codex_overall_score.json",
                RUNS / "icu5000_less_codex_error_metrics_v29.json",
                RUNS / "icu5000_less_codex_error_metrics.json",
                AW / "icu5000_less_codex/output/test_final3/result_root"),
            "cc_v1 (Claude Code)": (
                RUNS / "icu5000_less_cc_v1_overall_score.json",
                RUNS / "icu5000_less_cc_v1_error_metrics_v29.json",
                RUNS / "icu5000_less_cc_v1_error_metrics.json",
                AW / "icu5000_less_cc_v1/output/test_v3/result_root"),
        },
    },
    "icu5000 4.7pct": {
        "gold": DS47 / "test/reference",
        "players": {
            "我们的 agent (pi+val)": (
                AGENT_RUN / "experiments/icu5000_4p7_test_seed666_v1/host/score_report.json",
                RUNS / "icu5000_more_agent_error_metrics_v29.json",
                RUNS / "icu5000_more_agent_error_metrics.json",
                AGENT_RUN / "experiments/icu5000_4p7_test_seed666_v1/host/test_result_package"),
            "codex": (
                RUNS / "icu5000_more_codex_overall_score.json",
                RUNS / "icu5000_more_codex_error_metrics_v29.json",
                RUNS / "icu5000_more_codex_error_metrics.json",
                AW / "icu5000_more/output/test"),
            # cc 2026-10-25 重跑的一版（跟上面 codex 的 output/test 不是同一个包）
            "cc (Claude Code) 10-25": (
                RUNS / "icu5000_more_cc_overall_score.json",
                RUNS / "icu5000_more_cc_error_metrics_v29.json",
                RUNS / "icu5000_more_cc_error_metrics_v29.json",
                AW / "icu5000_more/result_root"),
        },
    },
    "N18 1.6pct": {
        "gold": DSN18 / "test/reference",
        "players": {
            "我们的 agent (pi+val)": (
                AGENT_RUN / "experiments/n18_1p6_test_seed666_v1/host/score_report.json",
                RUNS / "n18_agent_error_metrics_v29.json",
                RUNS / "n18_agent_error_metrics.json",
                AGENT_RUN / "experiments/n18_1p6_test_seed666_v1/host/test_result_package"),
            # cc 2026-10-25：一次性跑，无 val 阶段
            "cc (Claude Code) 10-25": (
                RUNS / "n18_less_cc_v1_overall_score.json",
                RUNS / "n18_less_cc_v1_error_metrics_v29.json",
                RUNS / "n18_less_cc_v1_error_metrics_v29.json",
                AW / "n18_less_cc_v1/output/result"),
        },
    },
    "N18 4.7pct": {
        "gold": DSN18M / "test/reference",
        "players": {
            "我们的 agent (pi+val)": (
                AGENT_RUN / "experiments/n18_4p7_test_seed666_v1/host/score_report.json",
                RUNS / "n18_more_agent_error_metrics_v29.json",
                RUNS / "n18_more_agent_error_metrics.json",
                AGENT_RUN / "experiments/n18_4p7_test_seed666_v1/host/test_result_package"),
        },
    },
    "I48 1.6pct": {
        "gold": DSI48 / "test/reference",
        "players": {
            "我们的 agent (pi+val)": (
                AGENT_RUN / "experiments/i48_1p6_test_seed666_v1/host/score_report.json",
                RUNS / "i48_less_agent_error_metrics_v29.json",
                RUNS / "i48_less_agent_error_metrics.json",
                AGENT_RUN / "experiments/i48_1p6_test_seed666_v1/host/test_result_package"),
        },
    },
    "I48 4.7pct": {
        "gold": DSI48M / "test/reference",
        "players": {
            "我们的 agent (pi+val)": (
                AGENT_RUN / "experiments/i48_4p7_test_seed666_v2/host/score_report.json",
                RUNS / "i48_more_agent_error_metrics_v29.json",
                RUNS / "i48_more_agent_error_metrics.json",
                AGENT_RUN / "experiments/i48_4p7_test_seed666_v2/host/test_result_package"),
        },
    },
}

VAL = {  # val composite（只有 pi+val 有 val 阶段）
    "icu5000 1.6pct": 0.903785,
    "icu5000 4.7pct": 0.915858,
    "N18 1.6pct": 0.949407,
    "N18 4.7pct": 0.947905,
    "I48 1.6pct": 0.769646,
    "I48 4.7pct": 0.915444,
}


def jl(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def err_v29(v29_path, v28_path):
    """优先用 v29；没有就回落 v28（icu5000 实测 v28≡v29，excluded=0 行）。"""
    d = jl(v29_path)
    if d:
        return d, "v29"
    d = jl(v28_path)
    return (d, "v28") if d else (None, "—")


def short(p):
    s = str(p)
    for pre, tag in ((str(PROJ), "$PROJ"), (str(WT), "$WT")):
        if s.startswith(pre):
            return tag + s[len(pre):]
    return s


def err_cells(er):
    if not er:
        return None
    b = er["buckets"]
    return (er["detection"]["F1"], er["repair"]["F1"],
            b.get("b_dirty", 0), b.get("b_wrong", 0), b.get("b_missing", 0),
            b.get("fp_det", 0))


print("=" * 104)
print("评测总表（v29 口径：recover_clean_key=true + exclude_rows_not_in_gold=true）")
print("=" * 104)

print("\n## 一、总分 + 错误级指标\n")
print("| 数据集 | 谁 | composite | 口径 | Det F1 | Rep F1 | b_dirty 没动 | b_wrong 修错 | b_missing 丢行 | 误改干净格 |")
print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
for ds, cfg in DATA.items():
    for who, (sc_p, e29, e28, _pkg) in cfg["players"].items():
        sc = jl(sc_p)
        er, cal = err_v29(e29, e28)
        if not sc:
            print(f"| {ds} | {who} | — | — | — | — | — | — | — | — |")
            continue
        c = sc["metrics"]["composite_score"]
        ec = err_cells(er)
        mark = "**" if "我们的" in who else ""
        if ec:
            print(f"| {ds} | {mark}{who}{mark} | {mark}{c:.6f}{mark} | {cal} | "
                  f"{ec[0]:.6f} | {ec[1]:.6f} | {ec[2]:,} | {ec[3]:,} | {ec[4]:,} | {ec[5]:,} |")
        else:
            print(f"| {ds} | {mark}{who}{mark} | {mark}{c:.6f}{mark} | {cal} | "
                  f"— | — | — | — | — | — |")

print("\n## 二、新旧口径对照（v28 vs v29，只看错误级）\n")
print("| 数据集 | 谁 | 口径 | Det F1 | Rep F1 | b_missing | 排除掉的格子 |")
print("|---|---|---|---:|---:|---:|---:|")
for ds, cfg in DATA.items():
    for who, (_sc, e29, e28, _pkg) in cfg["players"].items():
        for tag, path in (("v28", e28), ("v29", e29)):
            er = jl(path)
            if not er:
                continue
            ec = err_cells(er)
            if not ec:
                continue
            excl = er["buckets"].get("row_not_in_gold_cells", 0)
            print(f"| {ds} | {who} | {tag} | {ec[0]:.6f} | {ec[1]:.6f} | {ec[4]:,} | {excl:,} |")

print("\n## 三、每个结果包的路径（被评对象）\n")
for ds, cfg in DATA.items():
    print(f"### {ds}\n")
    for who, (sc_p, e29, _e28, pkg) in cfg["players"].items():
        p = Path(pkg)
        n = len(list(p.rglob("*.csv"))) if p.is_dir() else 0
        print(f"**{who}**  ({n} csv)")
        print(f"  结果包 : {short(p)}")
        print(f"  gold   : {short(cfg['gold'])}")
        print(f"  分数   : {short(sc_p)}")
        print(f"  错误分 : {short(e29)}")
        print()

print("## 四、逐文件 composite\n")
for ds, cfg in DATA.items():
    players = cfg["players"]
    perf = {}
    for who, (sc_p, _e29, _e28, _pkg) in players.items():
        sc = jl(sc_p)
        if sc:
            perf[who] = {m["relative_path"]: m["metrics"]["file_score"] for m in sc["file_reports"]}
    if not perf:
        continue
    who_list = list(players)
    print(f"### {ds}\n")
    print("| 文件 | " + " | ".join(who_list) + " | 最好 |")
    print("|---|" + "---:|" * len(who_list) + "---|")
    keys = sorted(perf[who_list[0]], key=lambda k: perf[who_list[0]][k])
    for k in keys:
        vals = [perf[w].get(k) for w in who_list]
        best = max((w for w, v in zip(who_list, vals) if v is not None),
                   key=lambda w: perf[w][k])
        cells = " | ".join("—" if v is None else f"{v:.4f}" for v in vals)
        print(f"| {k} | {cells} | {best} |")
    print()

print("## 五、val → test 落差（泛化能力，只有我们的 agent 有 val 阶段）\n")
print("| 数据集 | val composite | test composite | 落差 |")
print("|---|---:|---:|---:|")
for ds, cfg in DATA.items():
    sc = jl(cfg["players"]["我们的 agent (pi+val)"][0])
    if not sc:
        continue
    t = sc["metrics"]["composite_score"]
    v = VAL.get(ds)
    if v is None:
        print(f"| {ds} | — | {t:.6f} | — |")
    else:
        print(f"| {ds} | {v:.6f} | {t:.6f} | **{t - v:+.6f}** |")
