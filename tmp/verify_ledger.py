"""核对：16 条结果的 composite / Det / Rep 是否都真的写进了各个汇总文件。"""
import json
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
WT = PROJ / ".claude/worktrees/eval-movies-cleaning"
RUNS = PROJ / "runs"
AGENT_RUN = WT / "k29_agent_run"
AW = PROJ / "agent_work"

# tag -> (composite 来源, 错误分来源)
TRUTH = {
    "icu5000-1.6/agent": (
        RUNS / "icu5000_less_agent_overall_score.json",
        RUNS / "icu5000_less_agent_error_metrics_v29.json"),
    "icu5000-1.6/codex": (
        RUNS / "icu5000_less_codex_overall_score.json",
        RUNS / "icu5000_less_codex_error_metrics_v29.json"),
    "icu5000-1.6/cc_v1": (
        RUNS / "icu5000_less_cc_v1_overall_score.json",
        RUNS / "icu5000_less_cc_v1_error_metrics_v29.json"),
    "icu5000-4.7/cc": (
        RUNS / "icu5000_more_cc_overall_score.json",
        RUNS / "icu5000_more_cc_error_metrics_v29.json"),
    "icu5000-4.7/codex": (
        RUNS / "icu5000_more_codex_overall_score.json",
        RUNS / "icu5000_more_codex_error_metrics_v29.json"),
    "icu5000-4.7/agent": (
        AGENT_RUN / "experiments/icu5000_4p7_test_seed666_v1/host/score_report.json",
        RUNS / "icu5000_more_agent_error_metrics.json"),
    "n18-1.6/agent": (
        AGENT_RUN / "experiments/n18_1p6_test_seed666_v1/host/score_report.json",
        RUNS / "n18_agent_error_metrics_v29.json"),
    "n18-1.6/cc": (
        RUNS / "n18_less_cc_v1_overall_score.json",
        RUNS / "n18_less_cc_v1_error_metrics_v29.json"),
    "n18-4.7/agent": (
        AGENT_RUN / "experiments/n18_4p7_test_seed666_v1/host/score_report.json",
        RUNS / "n18_more_agent_error_metrics_v29.json"),
    "i48-1.6/agent": (
        AGENT_RUN / "experiments/i48_1p6_test_seed666_v1/host/score_report.json",
        RUNS / "i48_less_agent_error_metrics_v29.json"),
    "i48-4.7/agent": (
        AGENT_RUN / "experiments/i48_4p7_test_seed666_v2/host/score_report.json",
        RUNS / "i48_more_agent_error_metrics_v29.json"),
    "n18-4.7/cc": (
        RUNS / "n18_more_v1_overall_score.json",
        RUNS / "n18_more_v1_cc_error_metrics_v29.json"),
    "i48-1.6/cc": (
        RUNS / "i48_less_cc_v1_overall_score.json",
        RUNS / "i48_less_cc_v1_cc_error_metrics_v29.json"),
    "i48-4.7/cc": (
        RUNS / "i48_more_cc_v1_overall_score.json",
        RUNS / "i48_more_cc_v1_cc_error_metrics_v29.json"),
    "k29-1800/agent": (
        RUNS / "k29_1800_agent_overall_score.json",
        RUNS / "k29_1800_agent_error_metrics_v29.json"),
    "k29-1800/cc_v3": (
        RUNS / "k29_1800_cc_v3_overall_score.json",
        RUNS / "k29_1800_cc_v3_error_metrics_v29.json"),
}

truth = {}
for tag, (sp, ep) in TRUTH.items():
    s = json.loads(Path(sp).read_text(encoding="utf-8"))
    c = s["metrics"]["composite_score"]
    d = r = None
    if Path(ep).is_file():
        e = json.loads(Path(ep).read_text(encoding="utf-8"))
        d, r = e["detection"]["F1"], e["repair"]["F1"]
    truth[tag] = (c, d, r)

FILES = {
    "评测总表.md (台账)": WT / "评测总表.md",
    "score_only/总表.md": WT / "score_only/总表.md",
    "score_only/评测规范.md": WT / "score_only/评测规范.md",
}

# 校正类（单表纠错）这一族没有 composite / Det / Rep，
# 它的指标是「错误修复率」—— 单独核对行级 + cell 级两个率。
CORRECTION = {
    "tax": RUNS / "correction_tax_score.json",
    "beers": RUNS / "correction_beers_score.json",
    "rayyan": RUNS / "correction_rayyan_score.json",
    "flights": RUNS / "correction_flights_score.json",
}

for name, fp in FILES.items():
    if not fp.is_file():
        print(f"\n{name}: 文件不存在")
        continue
    txt = fp.read_text(encoding="utf-8")
    miss = []
    for tag, (c, d, r) in truth.items():
        if f"{c:.6f}" not in txt:
            miss.append(f"{tag}  composite {c:.6f}")
            continue
        if d is not None and f"{d:.6f}" not in txt:
            miss.append(f"{tag}  Det {d:.6f}")
        if r is not None and f"{r:.6f}" not in txt:
            miss.append(f"{tag}  Rep {r:.6f}")
    print(f"\n=== {name}  ({len(txt.splitlines())} 行) ===")
    if miss:
        for m in miss:
            print("   缺:", m)
    else:
        print(f"   OK  {len(truth)} 条记录的 composite + Det + Rep 全部命中")

# ---- 校正类（单表纠错）：核对「错误修复率」写进了台账 §1.4 / §2 ----
ledger = (WT / "评测总表.md").read_text(encoding="utf-8")
miss = []
for name, path in CORRECTION.items():
    if not Path(path).is_file():
        miss.append(f"{name}  分数文件缺失: {path}")
        continue
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for label, rate in (("行级", d["row_level_exact_rate"]),
                        ("cell级", d["cell_level_exact_rate"])):
        pct = f"{rate * 100:.2f}%"
        if pct not in ledger:
            miss.append(f"{name}  {label}修复率 {pct}")
print("\n=== 校正类（单表纠错）错误修复率 ===")
if miss:
    for m in miss:
        print("   缺:", m)
else:
    print(f"   OK  {len(CORRECTION)} 个数据集的行级 + cell 级修复率全部命中")

# ---- 校正类：按切分的检测 / 修复率也必须写进台账 §1.4 / §2 ----
SPLIT_NUMBERS = {
    "tax":     [("test 行级", "99.58%"), ("test cell级", "99.65%")],
    "beers":   [("test 行级", "94.17%"), ("test cell级", "96.75%")],
    "rayyan":  [("test 行级", "73.30%"), ("test cell级", "93.00%")],
    "flights": [("test 行级", "8.39%"), ("test cell级", "38.26%"),
                ("train 行级", "0.00%"), ("train cell级", "50.00%")],
}
miss = []
for name, items in SPLIT_NUMBERS.items():
    path = RUNS / f"correction_{name}_score_by_split.json"
    if not path.is_file():
        miss.append(f"{name}  切分分数文件缺失: {path}")
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    te = d["test(correction)"]
    for label, expect in items:
        if label.startswith("test"):
            want = te["row_repair_rate"] if "行级" in label else te["cell_repair_rate"]
        else:
            tr = d["train"]
            want = tr["row_repair_rate"] if "行级" in label else tr["cell_repair_rate"]
        if f"{want * 100:.2f}%" != expect:
            miss.append(f"{name}  {label} 分数文件={want*100:.2f}% 与台账写的 {expect} 不一致")
        elif expect not in ledger:
            miss.append(f"{name}  {label} 修复率 {expect}（台账里找不到）")
print("\n=== 校正类：train / test 切分的检测+修复率 ===")
if miss:
    for m in miss:
        print("   缺:", m)
else:
    print(f"   OK  {len(SPLIT_NUMBERS)} 个数据集的切分口径数字与分数文件一致、且都写进了台账")
