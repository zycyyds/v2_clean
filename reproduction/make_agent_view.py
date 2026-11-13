#!/usr/bin/env python3
"""make_agent_view.py —— 从完整数据集（A 类）生成 agent 视图（B 类）。

视图规则（与 make_agent_views.py 一致）：
  · 每个 split 复制 {keys.csv, raw/**, reference/**}
  · **剥掉答案文件**：modification_log.csv / mimic_native_error_log.csv / verify_report.json
  · **不带 test**（val 阶段不需要；test 的 reference 本身就是答案）

用法：
  python3 make_agent_view.py --src <完整数据集> --dst <视图目录> [--splits train,validation]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ANSWER_FILES = {"modification_log.csv", "mimic_native_error_log.csv", "verify_report.json"}


def build(src: Path, dst: Path, splits):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n_files = 0
    for s in splits:
        ss = src / s
        if not ss.is_dir():
            print("  跳过（无此 split）:", s)
            continue
        (dst / s).mkdir(parents=True, exist_ok=True)
        for item in sorted(ss.iterdir()):
            if item.name in ANSWER_FILES or item.name == ".DS_Store":
                continue
            if item.name == "reference":
                shutil.copytree(item, dst / s / "reference")
            elif item.is_dir():
                shutil.copytree(item, dst / s / item.name)
            else:
                shutil.copy2(item, dst / s / item.name)
        n_files += sum(1 for _ in (dst / s).rglob("*") if _.is_file())
        print("  %-12s → %d 个文件" % (s, sum(1 for _ in (dst / s).rglob("*") if _.is_file())))
    # 泄漏自检
    leaked = [p.name for p in dst.rglob("*") if p.name in ANSWER_FILES]
    if leaked:
        sys.exit("FATAL: 视图里有答案文件: %s" % leaked)
    print("  泄漏自检: ✅ 无答案文件")
    print("  合计 %d 个文件 → %s" % (n_files, dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", default="train,validation")
    args = ap.parse_args()
    src = Path(args.src)
    if not src.is_dir():
        sys.exit("源不存在: %s" % src)
    print("构建 agent 视图: %s → %s" % (src, args.dst))
    build(src, Path(args.dst), args.splits.split(","))


if __name__ == "__main__":
    main()
