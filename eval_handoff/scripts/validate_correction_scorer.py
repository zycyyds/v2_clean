"""校正类统一评分器的**已知答案校验**：拿 codex 那四份已知结果跑一遍，
数字必须和台账 §1.4 记的一致（否则评分器本身有问题）。"""
import importlib.util
import sys
from pathlib import Path

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning")
PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
spec = importlib.util.spec_from_file_location(
    "scr", WT / "eval_handoff/scripts/score_correction_result.py")
scr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scr)

CASES = [
    ("tax", PROJ / "agent_work/tax_correction_v1/result_package/tax.csv",
     "tax_curated10_v1", 0.9958, 0.9965, 0.999794, 0.996516),
    ("beers", PROJ / "agent_work/beers_correction_v1/result_package/beers.csv",
     "beers_curated10_v1", 0.9417, 0.9675, 0.983498, 0.983498),
    ("rayyan", PROJ / "rayyan_clean_v1/result_package/cleaned.csv",
     "rayyan_curated10_v1", 0.7330, 0.9300, 0.842691, 0.788863),
    ("flights", PROJ / "agent_work/flights_correction/output/flights.csv",
     "flights_curated10_v1", 0.0839, 0.3826, 0.582426, 0.542338),
]

print("=== 已知答案校验（台账 §1.4 记录的数字）===")
allok = True
for name, res, split, row, cell, det, rep in CASES:
    m = scr.score(name, res, split)["test(correction)"]
    got = (m["row_repair_rate"], m["cell_repair_rate"], m["det_F1"], m["rep_F1"])
    want = (row, cell, det, rep)
    ok = (abs(got[0] - want[0]) < 5e-5 and abs(got[1] - want[1]) < 5e-5
          and abs(got[2] - want[2]) < 5e-6 and abs(got[3] - want[3]) < 5e-6)
    allok &= ok
    print(f"  {'✅' if ok else '❌'} {name:8s} 行级 {got[0]*100:6.2f}% (期望 {want[0]*100:.2f}%)"
          f"  cell级 {got[1]*100:6.2f}% (期望 {want[1]*100:.2f}%)"
          f"  DetF1 {got[2]:.6f} (期望 {want[2]:.6f})"
          f"  RepF1 {got[3]:.6f} (期望 {want[3]:.6f})")

print("\n=== flights 在**新切分**（10:20:2346）上重算 codex 结果 —— 与 agent 同口径的基线 ===")
m = scr.score("flights", PROJ / "agent_work/flights_correction/output/flights.csv",
              "flights_curated10_val20_v1")["test(correction)"]
print(f"  codex 在新 test 切片（2,346 行 / {m['dirty_cells']:,} 脏 cell）上：")
print(f"    行级 exact {m['row_repair_rate']*100:6.2f}%   cell 级 exact {m['cell_repair_rate']*100:6.2f}%"
      f"   cell 级检测 {m['cell_detect_rate']*100:6.2f}%")
print(f"    Det F1 {m['det_F1']:.6f}   Rep F1 {m['rep_F1']:.6f}   "
      f"误伤 {m['collateral']:,}/{m['clean_cells']:,}")

print("\n" + ("✅ 评分器通过已知答案校验" if allok else "❌ 评分器与台账数字不一致"))
