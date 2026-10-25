#!/usr/bin/env python3
"""建 more1 的**密封 val 视图**：datasets/val_views/icu5000_more1/

为什么必须是 val_views/ 下的独立父目录，而不是直接用 datasets/注入错误数据集/icu5000_more1：

  1. more1 的父目录 datasets/注入错误数据集/ 里躺着 4.7pct **全量**数据集，
     它的 test/modification_log.csv 就是 test 的完整答案（每个被注入 cell 的
     clean_value 真值 + dirty_value + error_subtype）。
  2. harness 的 sandbox 虽然只放行 train/ 和 validation/raw/ 两个递归根，
     但它额外放行了 `(allow file-read* (literal "<项目根>"))` —— agent 能
     列出项目根目录。datasets/注入错误数据集/ 这一层的可枚举性一旦出现，
     就等于把答案目录摆在旁边。
  3. 上一轮 less1 就是因为指向了完整数据集目录，agent 第一轮就去读
     train/modification_log.csv 被我 kill 掉（见 memory: val-harness-sealed-view）。

  val_views/ 这个父目录下只放密封视图（less1 / more1），两个视图内部都
  只有 train+validation，没有任何 modification_log，也都跟全量数据集分属
  不同的父目录。

结构（跟 less1 一致）：
    <view>/train/keys.csv       拷贝
    <view>/train/raw/           拷贝（注入版，带错）
    <view>/train/reference/     拷贝（干净金标准，agent 自检用）
    <view>/validation/keys.csv  拷贝
    <view>/validation/raw/      拷贝（注入版，带错）
    <view>/validation/reference/ 拷贝（干净金标准，harness 打分用，路径会被 redact）
    <view>/test/                **不放** —— val 阶段不需要，test 阶段直接指向全量
    modification_log.csv        **不放** —— 那是答案

用法:
    python3 build_sealed_view.py --source <源目录> --dest <目标视图目录> [--force]

例:
    python3 build_sealed_view.py \
      --source "$PROJ/datasets/注入错误数据集/icu5000_more1" \
      --dest   "$PROJ/datasets/val_views/icu5000_more1"
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
SRC: Path
DST: Path


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_split(split: str) -> int:
    s = SRC / split
    d = DST / split
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in ("keys.csv", "raw", "reference"):
        p = s / name
        if not p.exists():
            print(f"  [WARN] 源缺 {split}/{name}")
            continue
        print(f"  拷贝 {split}/{name} ...", flush=True)
        if p.is_dir():
            shutil.copytree(p, d / name, symlinks=False)
        else:
            shutil.copy2(p, d / name)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="建密封 val 视图")
    ap.add_argument("--source", required=True, help="源数据集目录（含 train/ validation/）")
    ap.add_argument("--dest", required=True, help="目标视图目录（父目录应只放密封视图）")
    ap.add_argument("--force", action="store_true", help="已存在时删除重建")
    args = ap.parse_args()

    global SRC, DST
    SRC = Path(args.source).expanduser().resolve()
    DST = Path(args.dest).expanduser().resolve()

    if not SRC.is_dir():
        print(f"[FAIL] 源目录不存在: {SRC}")
        return 1
    if DST.exists() and any(DST.iterdir()):
        if not args.force:
            print(f"[SKIP] {DST} 已存在且非空（要重建加 --force）")
            return 0
        print(f"[FORCE] 删除并重建 {DST}")
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    for split in ("train", "validation"):
        copy_split(split)

    # ---- 复核 1：不能有答案 ----
    leaks = [str(p.relative_to(DST)) for p in DST.rglob("modification_log*")]
    if (DST / "test").exists():
        leaks.append("test/")
    if leaks:
        print(f"[FAIL] 答案泄漏: {leaks}")
        return 1

    # ---- 复核 2：与源逐字节一致 ----
    print("\n[校验] 逐字节比对 ...")
    bad = 0
    checked = 0
    for split in ("train", "validation"):
        if not (SRC / split).is_dir():
            continue
        for p in sorted((SRC / split).rglob("*")):
            if not p.is_file() or p.name == ".DS_Store":
                continue
            rel = p.relative_to(SRC)
            q = DST / rel
            checked += 1
            if not q.is_file():
                print(f"  [MISS] {rel}")
                bad += 1
            elif md5(p) != md5(q):
                print(f"  [DIFF] {rel}")
                bad += 1
    print(f"  比对 {checked} 个文件，差异 {bad} 个")
    if bad:
        return 1

    # ---- 复核 3：父目录里没有别的东西能翻到答案 ----
    siblings = sorted(x.name for x in DST.parent.iterdir() if x.is_dir())
    print(f"\n[OK] {DST}")
    print(f"  文件数 {checked}，全部逐字节一致")
    print(f"  父目录 {DST.parent} 下的目录: {siblings}")
    for s in siblings:
        logs = list((DST.parent / s).rglob("modification_log*"))
        if logs:
            print(f"  [FAIL] {s} 里有 modification_log: {logs}")
            return 1
    print("  父目录下所有视图均无 modification_log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
