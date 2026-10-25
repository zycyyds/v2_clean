#!/usr/bin/env python3
"""给 agent 侧建「扣留视图」—— 只暴露该给的目录。

按 K29 的 k29_1800/ 做法：
    <view>/<split>/keys.csv      真实拷贝
    <view>/<split>/raw           symlink → 注入版 raw
    <view>/<split>/reference     symlink → 注入版 reference（**test 不给**）
    <view>/v2_rawerror_*.json    报告拷贝

test 的 reference 是 gold，扣住不给 agent 看。modification_log.csv 一律
不进视图 —— 那是答案。

用法:
    python3 build_agent_views.py [--rates 4.7pct] [dataset_key ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_rawerror_by_ratio import DATASETS, OUT_ROOT, RATE_PRESETS  # noqa: E402

VIEW_ROOT = OUT_ROOT
SPLITS = ("train", "validation", "test")


def view_name(key: str, tag: str) -> str:
    return f"{key}_{tag}"


def build_view(key: str, tag: str) -> Path | None:
    cfg = DATASETS[key]
    out_name = cfg["out_base"].format(tag=tag)
    src_root = OUT_ROOT / out_name
    if not src_root.is_dir():
        print(f"  [SKIP] {key}: {src_root} 不存在")
        return None

    dst = VIEW_ROOT / view_name(key, tag)
    if dst.exists():
        # 已经有视图了 —— 可能已经建好、也可能 agent 正在用。
        # 绝不能 rmtree 掉，那会毁掉别人正在跑的工作。
        print(f"  [SKIP] {dst} 已存在，保留不动")
        return dst
    dst.mkdir(parents=True)

    for split in SPLITS:
        src = src_root / split
        if not src.is_dir():
            continue
        d = dst / split
        d.mkdir(parents=True)

        keys = src / "keys.csv"
        if keys.is_file():
            shutil.copy2(keys, d / "keys.csv")

        raw = src / "raw"
        if raw.is_dir():
            (d / "raw").symlink_to(raw.resolve())

        # test 的 reference 是 gold —— 扣住不给
        ref = src / "reference"
        if ref.is_dir() and split != "test":
            (d / "reference").symlink_to(ref.resolve())

    rep = src_root / f"v2_rawerror_{key}_final_report.json"
    if rep.is_file():
        shutil.copy2(rep, dst / rep.name)

    print(f"  [OK] {dst}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--rates", choices=sorted(RATE_PRESETS), default="4.7pct")
    args = ap.parse_args()
    keys = args.datasets or sorted(DATASETS)
    tag = RATE_PRESETS[args.rates]["tag"]
    for k in keys:
        build_view(k, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
