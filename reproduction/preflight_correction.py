#!/usr/bin/env python3
"""preflight_correction.py —— 校正类跑批前的**预检**（不起跑批、不调 API）。

对每一档检查：
  ① driver profile 引用的文件/目录**是否都在**（prompt / evaluation manifest / split_manifest /
     train/{raw,reference} / validation/{raw,gold}）
  ② **用 harness 自己的 manifest loader** 跑一遍（`load_evaluation_manifest`）
     ⇒ 这一步会真的校验 key_columns 存在、gold 目录里每个文件都在 manifest 里、
        manifest 不引用多余文件 —— **和开跑时用的是同一个函数**
  ③ 答案隔离：validation/gold 不在 validation/raw 里面；correction/reference_private 与
     host_private 不在 agent 可见的三个根里
  ④ test 段（correction/{raw,reference_private}）也在
  ⑤ prompt 里已含两条通用坑（replay.argv 契约 / 续跑摘要 10 字段）

只输出聚合与路径级结论。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/Users/mac/PycharmProjects/v2_clean")
from workflow.pi_harness_evaluation import load_evaluation_manifest  # noqa: E402

PROJ = Path("/Users/mac/PycharmProjects/v2_clean")
WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/k29-mimicraw")

TIERS = {
    "hospital": ("raha_hospital_curated10_val20_dense_v1",
                 "hospital_correction_val_prompt_v1.md",
                 "raha_correction_hospital_v1.json"),
    "beers": ("beers_curated10_val20_v1",
              "beers_correction_val_prompt_v1.md",
              "raha_correction_beers_v1.json"),
    "rayyan": ("rayyan_curated10_val20_v1",
               "rayyan_correction_val_prompt_v1.md",
               "raha_correction_rayyan_v1.json"),
    "tax": ("tax_curated10_val20_v1",
            "tax_correction_val_prompt_v1.md",
            "raha_correction_tax_v1.json"),
}

PROMPT_MUST_CONTAIN = [
    ("replay.argv 契约", "`argv[0]` 必须是"),
    ("静态绝对路径禁令", "cannot contain static absolute paths"),
    ("续跑摘要 10 字段", "verified_findings_and_evidence"),
    ("上下文压缩阈值", "49.6 万 token"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiers", nargs="*", default=list(TIERS), help="要检查的档")
    args = ap.parse_args()

    bad = 0
    for tier in (args.tiers or list(TIERS)):
        splitdir, prompt_name, manifest_name = TIERS[tier]
        ds = PROJ / "datasets" / "correction" / splitdir
        prompt = WT / "eval_handoff" / "prompts" / prompt_name
        manifest = WT / "evaluation_manifests" / manifest_name

        print("=" * 72)
        print("【%s】%s" % (tier, splitdir))
        ok = True

        # ① 路径齐不齐
        for p, label in [(prompt, "prompt"), (manifest, "evaluation manifest"),
                         (ds / "split_manifest.json", "split manifest")]:
            if not p.is_file():
                print("  ❌ 缺 %s: %s" % (label, p)); ok = False
        for p, label in [(ds / "train" / "raw", "train/raw"),
                         (ds / "train" / "reference", "train/reference"),
                         (ds / "validation" / "raw", "validation/raw"),
                         (ds / "validation" / "gold", "validation/gold"),
                         (ds / "correction" / "raw", "correction/raw(test)"),
                         (ds / "correction" / "reference_private", "correction/reference_private(test gold)")]:
            if not p.is_dir():
                print("  ❌ 缺目录 %s: %s" % (label, p)); ok = False

        # ② harness 自己的 manifest loader
        if manifest.is_file() and (ds / "validation" / "gold").is_dir():
            try:
                keys = load_evaluation_manifest(manifest, ds / "validation" / "gold")
                print("  ✅ manifest loader 通过（val gold）: %s"
                      % {k: list(v) for k, v in keys.items()})
            except Exception as exc:
                print("  ❌ manifest loader 失败: %s: %s" % (type(exc).__name__, exc))
                ok = False
        if manifest.is_file() and (ds / "correction" / "reference_private").is_dir():
            try:
                keys = load_evaluation_manifest(
                    manifest, ds / "correction" / "reference_private")
                print("  ✅ manifest loader 通过（test gold）: %s"
                      % {k: list(v) for k, v in keys.items()})
            except Exception as exc:
                print("  ❌ manifest loader 失败(test gold): %s: %s"
                      % (type(exc).__name__, exc))
                ok = False

        # ③ 答案隔离
        raw = (ds / "validation" / "raw").resolve()
        gold = (ds / "validation" / "gold").resolve()
        if gold == raw or raw in gold.parents:
            print("  ❌ validation/gold 落在 validation/raw 里面 ⇒ 漏答案"); ok = False
        else:
            print("  ✅ 答案隔离: validation/gold 与 validation/raw 不重叠")
        for hidden in [ds / "correction" / "reference_private", ds / "host_private"]:
            h = hidden.resolve()
            for visible in [(ds / "train" / "raw").resolve(),
                            (ds / "train" / "reference").resolve(), raw]:
                if h == visible or visible in h.parents:
                    print("  ❌ %s 落在 agent 可见根 %s 里面" % (hidden.name, visible))
                    ok = False

        # ⑤ prompt 里两条通用坑在不在
        if prompt.is_file():
            text = prompt.read_text(encoding="utf-8")
            for label, needle in PROMPT_MUST_CONTAIN:
                if needle not in text:
                    print("  ❌ prompt 里缺「%s」" % label); ok = False
            if ok:
                print("  ✅ prompt 含两条通用坑（replay.argv 契约 / 续跑摘要 10 字段 / 压缩阈值）")

        print("  ⇒ %s" % ("✅ 全部通过" if ok else "❌ 有问题，先别跑"))
        bad += 0 if ok else 1

    print("=" * 72)
    print("有问题的档数: %d" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
