#!/usr/bin/env python3
"""生成 K29 1800（低密度档，fixed-v1）的 codex 任务书。

**不新写任务书**：直接 import 同目录的 `gen_prompts.py`，复用它的模板
(`_template_codex_prompt.md`)、骨架 (`SKELETON`) 和替换机制 (`apply`)，
只给 k29 加一个 "1800" 口径：

  - err["1800"]     三个 split 的真实注入构成（按操作类型拆开）
  - overlap["1800"] 这一档特有的两条提醒（每 cell 只注入一次 / 错误极其稀疏）

这样已有的 8 份任务书**一个字节都不会变**（本脚本不重写它们），
而模板仍然只有一份（改模板 = 改 `gen_prompts.py` 那一份）。

⚠️ **生成出来的任务书不进 git**：里面有 MIMIC 的 train hadm_id，
   而 origin / public-origin 都是公开仓库（同 k29_agent_run/prompts/ 的处理）。
   但**这个生成器本身可以进 git** —— 它不含任何 identifier，
   任务书正文从共享的 `gen_prompts.py` 里 import 进来（任务书原文仍然单一来源）。

用法:
    python3 gen_codex_prompt_k29_1800.py            # 写 k29_1800_codex_prompt.md
    python3 gen_codex_prompt_k29_1800.py --check    # 只打印将要写的内容的 sha256
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path("/Users/mac/PycharmProjects/v2_clean/prompts/最新注入错误prompt")


def load_shared():
    """import 同目录的 gen_prompts.py（不执行它的 main）。"""
    spec = importlib.util.spec_from_file_location(
        "shared_gen_prompts", HERE / "gen_prompts.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# --- 这一档的真实数字（全部实测，来源见台账 §3.1）-----------------------
# 注入构成：train 343 cell_update + 29 duplicate + 28 insert = 400
#           test  1673 cell_update + 73 duplicate + 54 insert = 1800
# cell 总数：train 113,611 / validation 729,368 / test 265,585,384
# cell 错误率：train 0.3521% / validation 0.1097% / test 0.00068%
# 每个被改动的 cell 恰好 1 个错误（实测 0 个 cell 被重复注入）
# 错误型覆盖 18 类
ERR_1800 = (
    "    train: 约 343 cell_update + 29 duplicate_row + 28 label_leakage 插入行 = "
    "400 个错误分布在 10 个 hadm 上（40 错误/hadm，cell-level 0.352%，113,611 个 cell）\n"
    "    test: 约 1,673 cell_update + 73 duplicate_row + 54 label_leakage 插入行 = "
    "1,800 个错误分布在 8,092 个 hadm 上（0.22 错误/hadm，cell-level 0.00068%，"
    "265,585,384 个 cell）"
)

OVERLAP_1800 = (
    "    注意：每个被改动的 cell 恰好 1 个错误（这一档的注入日志由 fixed-v1 引擎生成，"
    "不做多错误叠加）；错误型覆盖 18 类，但 train 只有 10 个 hadm，"
    "**每类在 train 上只有十几个样本，有些类只有 1~2 个**"
    "（train 切分里只有 2 个 ICU stay，所以「ICU 表的时间戳类错误」几乎没有样本）"
    "——必须靠领域常识补齐判定规则，不要因为某类错误在 train 里只出现一两次就判定它不重要，"
    "也不要因为它在 train 里出现得多就以为 test 里也占多数\n"
    "    **这一档的错误极其稀疏**：test 的 1,800 个错误摊在 2.65 亿个 cell 上（0.00068%），"
    "平均每个 hadm 才 0.22 个 —— **绝大多数 hadm 完全干净**。"
    "不要对每个 hadm 都套一遍修正逻辑，也不要用重复行/截断行/填充空值去拟合 test 的规模"
)

# 别的口径的数字一个都不许残留
FORBIDDEN = [
    ("2,013 cell_update", "1.58pct 旧数字"),
    ("10,651,244", "1.58pct 旧数字"),
    ("2,907 个错误", "1.58pct 旧数字"),
    ("13.7M", "1.58pct 旧数字"),
    ("1.42%", "1.58pct 旧率"),
    ("1.66%", "1.58pct 旧率"),
    ("2,816", "4.7pct 旧数字"),
    ("6,580", "4.7pct 旧数字"),
    ("14,123,558", "4.7pct 旧数字"),
    ("4.72%", "4.7pct 旧率"),
    ("1.30%", "1.6pct 旧率"),
    ("1.58%", "1.6pct 旧率"),
    ("2.33%", "4.7pct 旧率"),
    ("k29_less", "别的档的视图名"),
    ("k29_more", "别的档的视图名"),
    ("validation/raw", "codex 没有 val 阶段"),
]


def main() -> int:
    gp = load_shared()
    gp.SKELETON["k29"]["err"]["1800"] = ERR_1800
    gp.SKELETON["k29"]["overlap"]["1800"] = OVERLAP_1800

    out = gp.build("k29", "1800")

    for bad, why in FORBIDDEN:
        if bad in out:
            raise SystemExit(f"残留未替换：{bad}（{why}）")

    # 视图路径必须指向刚建好的扣留视图
    want = "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/k29_1800/"
    if out.count(want) != 3:
        raise SystemExit(f"三个路径没都对上 {want}（实际 {out.count(want)} 个）")
    for p in ("train/raw", "train/reference", "test/raw"):
        if f"{want}{p}" not in out:
            raise SystemExit(f"路径缺失：{p}")

    # 扣留物只能以「已剔除」的方式被提到一次（跟别的档一样），不能出现路径
    if out.count("modification_log.csv") != 1:
        raise SystemExit(f"modification_log.csv 出现 {out.count('modification_log.csv')} 次（应为 1）")
    if out.count("test/reference") != 1:
        raise SystemExit(f"test/reference 出现 {out.count('test/reference')} 次（应为 1）")
    if "k29_1800/test/reference" in out:
        raise SystemExit("任务书里出现了 test/reference 的路径")
    if "k29_1800/modification_log" in out:
        raise SystemExit("任务书里出现了 modification_log 的路径")

    sha = hashlib.sha256(out.encode("utf-8")).hexdigest()
    name = "k29_1800_codex_prompt.md"
    print(f"sha256 = {sha}")
    print(f"字符数 = {len(out)}  行数 = {len(out.splitlines())}")
    if "--check" in sys.argv:
        return 0
    (HERE / name).write_text(out, encoding="utf-8")
    print(f"→ 已写 {HERE / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
