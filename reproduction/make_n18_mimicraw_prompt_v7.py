#!/usr/bin/env python3
"""make_n18_mimicraw_prompt_v7.py —— 由 **v6** prompt 生成 **v7**（N18 mimicraw）。

v7 相对 v6 只多**一节**：`CONTEXT_BUDGET_SECTION`（上下文预算 + 续跑摘要格式）。

## 为什么加这一节

**这一族最大的收尾杀手不是配额，是「上下文压缩摘要」的 schema 校验失败。**

实测（`n18_agent_run/logs/n18_mimicraw_*_val_*.log`）：

| run | 压缩事件数 | 结果 |
|---|---:|---|
| 4.7pct `_v1` | 1 | **连挂 4 次全失败 ⇒ worker 崩 ⇒ 整个 run 报废** |
| 4.7pct `_v2` | 1 | 一次就过 |
| 1.6pct `_v2` | 3 | 1 次失败后重试成功，2 次一次就过 |

⇒ **5 个压缩事件里 2 个先失败一次（40%）**，失败形态全是同一个：
`Pi worker ValidationError: ... is not of type 'string'`（把字段写成数组/对象）
或 `'current_state' is a required property`（漏字段）
⇒ 连挂 4 次就 fatal。

机制：上下文上限 512,000 token，**到 495,616 就强制压缩**，压缩时模型要交一份
**10 个纯字符串字段**的 JSON（`PiCompressionSummary`）。模型把「3 条发现」写成
`["a","b","c"]` ⇒ 校验失败 ⇒ 本轮失败 ⇒ 重试；连挂 4 次 ⇒ worker 崩。

⇒ prompt 侧能做的两件事：
1. **别把上下文撑爆**（少打大块 stdout）⇒ 少触发压缩 = 少踩雷；
2. **被要求写续跑摘要时，10 个字段各写一个纯字符串**。

⚠️ 真正的根治在 harness 侧（`agent/pi_runtime.py` 的 `PiCompressionSummary` 改成
容错 / 给压缩调用加回退）—— 那是 Pi harness 的代码，不在本仓这条线上，
这里只在台账里留建议，不去动别人的文件。

## 用法

    python3 make_n18_mimicraw_prompt_v7.py --in <v6 prompt> --out <v7 prompt>
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 插入锚点：prompt 末尾的「限制：」块之前（位置要靠后 —— 第 2 轮起 prompt 只含
# feedback，位置越靠后越可能在写代码时被想起来）
A_LIMITS = "\n限制：\n"

CONTEXT_BUDGET_SECTION = """
### 🧠 上下文预算（**这一条会让你整个 run 报废**）

你的上下文上限是 **512,000 token**，**到 495,616 会被 harness 强制压缩**——
压缩时你要交一份「续跑摘要」，它必须是 **10 个纯字符串字段**的 JSON。

⚠️ **压缩是这一族最大的收尾杀手**（实测，不是理论）：
模型一旦把某个字段写成**数组或对象**（例如把「3 条发现」写成 `["a","b","c"]`），
harness 就报 `ValidationError: ... is not of type 'string'`；
漏掉一个字段就报 `'current_state' is a required property`
⇒ 整轮失败 ⇒ 重试；**连挂 4 次 ⇒ worker 崩 ⇒ 整个 run 报废**。
（N18 4.7pct 的第 1 遍 run 就这样死在第 4 轮；
三个 run 的 5 次压缩事件里 **2 次先失败一次**。）

⇒ **三件事必须做**：

**① 不要把大块内容打到 stdout。** 要看的东西**一律写进文件**，命令只回显：
- 前 20 行 + 总行数（`wc -l`），或
- 聚合统计（`groupby(...).size()` / `value_counts()`）。

⚠️ validation 有 16.7 万行、test 有 3615 万行 ——
一次 `head -2000` 或 `cat` 几万行就是几万 token，几十次就撑到压缩线。

**② 被要求写「续跑摘要 / continuation summary」时，10 个字段每个都写纯字符串。**
字段就是这 10 个（一个都不能漏）：

```
task_and_constraints
current_state
verified_findings_and_evidence
failed_attempts_and_counterexamples
files_read
files_created_or_modified
key_commands
artifacts_and_paths
unresolved_work
next_steps
```

⇒ 每个字段写**一段纯文本**（多条目用换行或编号串起来），
**不要写 JSON 数组、JSON 对象，也不要漏字段**。

**③ 把这一节写进你自己的产物**（比如 workdir 里的 `NOTES.md`，
或 `train_analysis.json` 的一节），**每轮开工前先读它**。
⚠️ 后续回合的任务书里可能**没有**这一节（host 只回灌分数和逐文件反馈）
⇒ 规则只留在上下文里，下一回合就没了 ⇒ **写进文件才留得住**。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="v6 prompt")
    ap.add_argument("--out", dest="dst", required=True, help="v7 prompt 输出")
    args = ap.parse_args()

    text = Path(args.src).read_text(encoding="utf-8")
    if "上下文预算" in text:
        raise SystemExit("FATAL: 输入里已经有「上下文预算」一节了，别重复生成")
    if A_LIMITS not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_LIMITS)

    text = text.replace(A_LIMITS, "\n" + CONTEXT_BUDGET_SECTION + A_LIMITS, 1)
    Path(args.dst).write_text(text, encoding="utf-8")
    print("已写出:", args.dst, "(%d bytes)" % len(text.encode("utf-8")))
    print("  新增一节：上下文预算 + 续跑摘要格式（%d 字符）"
          % len(CONTEXT_BUDGET_SECTION))


if __name__ == "__main__":
    main()
