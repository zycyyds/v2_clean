#!/usr/bin/env python3
"""按 n18_more 的样子补建扣留视图（真拷贝，不用 symlink）。

结构（与用户手建的 n18_more / i48_more / i48_less 一致）：
    <view>/train/keys.csv      拷贝
    <view>/train/raw/          拷贝（注入版，带错）
    <view>/train/reference/    拷贝（干净金标准，agent 自检用）
    <view>/test/keys.csv       拷贝
    <view>/test/raw/           拷贝（注入版，带错）
    <view>/test/reference/     不放 —— 那是 gold
    <view>/validation/         不放 —— codex 没有 val
    modification_log.csv        不放 —— 那是答案

用法:
    python3 build_agent_view_copy.py <view_name> <源数据集目录名>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

INJ = Path("/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集")


def build(view: str, source: str) -> int:
    src_root = INJ / source
    if not src_root.is_dir():
        print(f"[FAIL] 源目录不存在: {src_root}")
        return 1
    dst = INJ / view
    if dst.exists() and any(dst.iterdir()):
        print(f"[SKIP] {view} 已存在且非空，保留不动")
        return 0
    if dst.exists():
        shutil.rmtree(dst)

    for split in ("train", "test"):
        s = src_root / split
        if not s.is_dir():
            print(f"[WARN] 源缺 {split}，跳过")
            continue
        d = dst / split
        d.mkdir(parents=True)
        for name in ("keys.csv", "raw", "reference"):
            p = s / name
            if not p.exists():
                continue
            # test 的 reference 是 gold —— 扣住
            if split == "test" and name == "reference":
                continue
            print(f"  拷贝 {view}/{split}/{name} ...", flush=True)
            if p.is_dir():
                shutil.copytree(p, d / name, symlinks=False)
            else:
                shutil.copy2(p, d / name)

    # 复核：不该有答案
    leaks = [str(p.relative_to(dst)) for p in dst.rglob("modification_log.csv")]
    if (dst / "test" / "reference").exists():
        leaks.append("test/reference")
    if leaks:
        print(f"[FAIL] {view} 答案泄漏: {leaks}")
        return 1
    print(f"[OK] {view}  建好（无 modification_log.csv、无 test/reference）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(build(sys.argv[1], sys.argv[2]))
