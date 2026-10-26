"""把已验证的边界修复落到 I48 4.7pct 的冻结快照里，并记录补丁前后的目录 hash。

前置：tmp/fix_and_verify_i48_more.py 已经证明补丁在 validation 上**逐字节不改变**输出。
这里只做「落盘 + 留痕」：
  - 改 best_snapshot/scripts/pipeline.py（val 的最佳轮）
  - 改 reproducible_snapshot/scripts/pipeline.py（test harness 真正读的那份）
两处原本应当完全一致，所以打完补丁仍然一致。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/mac/PycharmProjects/v2_clean")
from agent.pi_harness import directory_sha256  # noqa: E402

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning")
EXP = WT / "k29_agent_run/experiments/i48_4p7_val_seed666_v1"
SNAPS = [EXP / "host/best_snapshot", EXP / "host/reproducible_snapshot"]

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
    before = {}
    for s in SNAPS:
        before[s.name] = directory_sha256(s)
    print("补丁前目录 hash：")
    for k, v in before.items():
        print(f"  {k:24s} {v}")

    for s in SNAPS:
        p = s / "scripts" / "pipeline.py"
        src = p.read_text(encoding="utf-8")
        if NEW in src:
            print(f"  {s.name}: 已经是打过补丁的版本，跳过")
            continue
        if OLD not in src:
            print(f"!! {s.name}: 找不到待修代码段，中止", file=sys.stderr)
            return 2
        p.write_text(src.replace(OLD, NEW), encoding="utf-8")
        print(f"  {s.name}: 已打补丁")

    after = {}
    for s in SNAPS:
        after[s.name] = directory_sha256(s)
    print("\n补丁后目录 hash：")
    for k, v in after.items():
        print(f"  {k:24s} {v}")

    # best_snapshot 与 reproducible_snapshot 的 pipeline.py 必须仍然逐字相同
    a = (SNAPS[0] / "scripts/pipeline.py").read_bytes()
    b = (SNAPS[1] / "scripts/pipeline.py").read_bytes()
    print(f"\nbest_snapshot 与 reproducible_snapshot 的 pipeline.py 一致: {a == b}")

    # 留痕：写一份补丁说明到 val 实验的 host 目录
    note = {
        "kind": "host_side_bounds_fix",
        "reason": "agent 的 repair_charttime_sentinel 在 searchsorted 返回 len(good) 时越界崩溃；"
                  "validation 不触发，test 的 3279 万行触发 IndexError",
        "author_intent": "作者已写 right_ok = (right < len(good)) 守卫，但索引先执行、守卫后判断，"
                         "且 right 是 good 的下标永远不越界——真正要判的是 searchsorted 的位置",
        "change": "先取 left_pos/right_pos，clip 后再索引，守卫改用位置判断",
        "semantics_preserving_evidence":
            "打补丁后重跑 validation raw，13 个输出 csv 与 agent 交付的 reproducible_snapshot/result "
            "逐字节相同（tmp/fix_and_verify_i48_more.py，输出：逐字节不一致的文件=无）",
        "not_a_rule_change": "只阻止越界崩溃，未改动任何清洗规则、阈值、cohort 逻辑",
        "patched_files": [str(s / "scripts/pipeline.py") for s in SNAPS],
        "directory_sha256_before": before,
        "directory_sha256_after": after,
    }
    note_path = EXP / "host/host_side_fix.json"
    note_path.write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n补丁留痕已写入：{note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
