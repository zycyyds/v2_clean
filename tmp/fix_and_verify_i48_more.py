"""宿主侧最小修复 + 语义等价性证明。

问题：agent 的 scripts/pipeline.py:repair_charttime_sentinel 里
    right = good[np.searchsorted(good, bad_idx, side="right")]
当某个坏行的 charttime 大于该 stay 全部"好"行时，searchsorted 返回 len(good)，
good[len(good)] 越界 -> IndexError。validation 上不触发，test 的 3,279 万行上触发。

作者本来的意图就写在下一行：right_ok = (right < len(good)) & ...
（right 是 good 的下标，永远 < len(good)，所以这个守卫是空写的；
他真正想判的是 searchsorted 的**位置**是否越界）。
=> 修法 = 先取位置、夹住再索引、守卫改用位置判断。

本脚本做三件事：
  1. 在副本上打补丁
  2. 用打过补丁的 pipeline 重跑 validation raw，输出与 agent 交付的包逐字节比对
  3. 逐字节一致才允许继续（不一致就中止）
"""
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning")
SNAP = WT / "k29_agent_run/experiments/i48_4p7_val_seed666_v1/host/reproducible_snapshot"
VAL_RAW = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/i48_more_1/validation/raw")
WORK = Path("/Users/mac/.claude/jobs/910902cd/tmp/i48_more_fix")

OLD = """    # nearest untouched row before / after every corrupted row
    left = good[np.searchsorted(good, bad_idx, side="left") - 1]
    right = good[np.searchsorted(good, bad_idx, side="right")]
    left_ok = (left >= 0) & (stay[np.clip(left, 0, None)] == stay[bad_idx])
    right_ok = (right < len(good)) & (stay[np.clip(right, 0, None)] == stay[bad_idx])"""

NEW = """    # nearest untouched row before / after every corrupted row
    # searchsorted 可能返回 0 或 len(good)：先取位置、夹住再索引，否则越界。
    # （守卫要判的是**位置**是否越界，而不是夹住后的下标——后者永远不越界。）
    left_pos = np.searchsorted(good, bad_idx, side="left")
    right_pos = np.searchsorted(good, bad_idx, side="right")
    left = good[np.clip(left_pos - 1, 0, len(good) - 1)]
    right = good[np.clip(right_pos, 0, len(good) - 1)]
    left_ok = (left_pos >= 1) & (stay[left] == stay[bad_idx])
    right_ok = (right_pos < len(good)) & (stay[right] == stay[bad_idx])"""


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    snap = WORK / "snapshot"
    shutil.copytree(SNAP, snap)

    p = snap / "scripts" / "pipeline.py"
    src = p.read_text(encoding="utf-8")
    if OLD not in src:
        print("!! 找不到待修的代码段 —— agent 改了 pipeline，补丁不适用", file=sys.stderr)
        return 2
    p.write_text(src.replace(OLD, NEW), encoding="utf-8")
    print("补丁已应用（只改 repair_charttime_sentinel 的边界处理）")

    out = WORK / "val_out"
    out.mkdir()
    print(f"\n用打过补丁的 pipeline 重跑 validation raw ...", flush=True)
    r = subprocess.run(
        ["python3", "scripts/pipeline.py", "--raw-root", str(VAL_RAW), "--output-dir", str(out)],
        cwd=snap, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print("!! 重跑失败：", r.stderr[-3000:], file=sys.stderr)
        return 3

    ref = snap / "result"
    cmp_ = filecmp.dircmp(str(out), str(ref))
    diffs = []
    for name in sorted(cmp_.common_files):
        if not filecmp.cmp(str(out / name), str(ref / name), shallow=False):
            diffs.append(name)
    only_out = sorted(cmp_.left_only)
    only_ref = sorted(cmp_.right_only)
    sub = []
    for name, sub_cmp in cmp_.subdirs.items():
        for f in sorted(sub_cmp.common_files):
            if not filecmp.cmp(str(out / name / f), str(ref / name / f), shallow=False):
                sub.append(f"{name}/{f}")
        sub.extend(f"{name}/{f}" for f in sub_cmp.left_only)
        sub.extend(f"{name}/{f}" for f in sub_cmp.right_only)

    n_ref = sum(1 for _ in ref.rglob("*.csv"))
    print(f"\n=== 语义等价性检查 ===")
    print(f"  reference 里的 csv 数 : {n_ref}")
    print(f"  逐字节不一致的文件    : {diffs if diffs else '无'}")
    print(f"  子目录里的差异        : {sub if sub else '无'}")
    print(f"  只在新输出里          : {only_out if only_out else '无'}")
    print(f"  只在 reference 里     : {only_ref if only_ref else '无'}")
    if diffs or sub or only_out or only_ref:
        print("\n!! 不一致 —— 补丁改变了 agent 的行为，中止", file=sys.stderr)
        return 4
    print("\n==> 补丁**逐字节不改变** validation 上的输出，")
    print("    说明它只阻止了崩溃，没有改动 agent 的清洗规则。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
