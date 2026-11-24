#!/usr/bin/env python3
"""核对：test 段重放用的 frozen_snapshot 与 val 的 reproducible_snapshot 是同一份。

用 harness 自己的 directory_sha256 算，别自己另编一套。
"""
import sys
from pathlib import Path

PROJ = "/Users/mac/PycharmProjects/v2_clean"
sys.path.insert(0, PROJ)
from agent.pi_harness import directory_sha256  # noqa: E402

RUN = Path("correction_agent_run/experiments")


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "rayyan"
    suffix = sys.argv[2] if len(sys.argv) > 2 else "v1"
    val = RUN / f"{tier}_correction_val_seed666_{suffix}" / "host" / "reproducible_snapshot"
    test = RUN / f"{tier}_correction_test_seed666_{suffix}" / "host" / "frozen_snapshot"
    import json
    tr = json.load(open(test.parent / "run_report.json"))
    reported = tr.get("frozen_snapshot_sha256")

    h_val = directory_sha256(val) if val.exists() else None
    h_test = directory_sha256(test) if test.exists() else None
    print("[%s]" % tier)
    print("  val  reproducible_snapshot sha256 = %s" % h_val)
    print("  test frozen_snapshot       sha256 = %s" % h_test)
    print("  test 报告里的 frozen_snapshot_sha256 = %s" % reported)
    ok = (h_val == h_test == reported)
    print("  ⇒ %s" % ("一致 ✅（test 重放的就是 val 冻结的那份产物）" if ok else "不一致 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
