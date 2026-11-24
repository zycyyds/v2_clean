#!/usr/bin/env python3
"""make_n18_mimicraw_prompt_v8.py —— 由 **v7** prompt 生成 **v8**（N18 mimicraw）。

v8 相对 v7 只多**一节**：`TIME_ANCHOR_SECTION`（时间锚点：泄漏会把原生错误格的行「推走」）。

## 为什么加这一节

**实测发现（`reproduction/n18_native_key_vs_reference.py` + `n18_charttime_injection_profile.py`）**：

1. 注入器对 `icu/chartevents.csv` 的 **charttime** 列做了 **1,745,517** 处改动
   （subtype = `future_window_leakage`），其中 **99.96% 是往后推 30~48 小时**
   （偏移量有 4.5 万种、**不固定** ⇒ 写不成固定规则）。
2. 原生错误格按完整 5 段 key `(subject_id, hadm_id, stay_id, charttime, itemid)`
   去 raw 里找：
   - 1.6pct test：**671 / 715** 找得到（93.8%）
   - 4.7pct test：**401 / 586** 找得到（68.4%）
   ⇒ **剩下那些格，它的行被泄漏推到未来了** ⇒ 按精确 key 定位**必然找不到**。
3. reference 里这 715 / 586 个 key **一个不缺** ⇒ 不是登记口径的问题，是行真的被挪了。
4. agent 自己发明了「charttime > outtime 就取最近的有效时刻」这条规则，
   在被挪过的格子里救回 **41/44（93.2%）** 与 **178/185（96.2%）** ⇒
   **方向是对的，只是 prompt 没告诉它这件事**。

⇒ v8 补的正是：让 agent **知道**会被推、知道要按值定位、知道要把时间轴拉回来。

## 用法

    python3 make_n18_mimicraw_prompt_v8.py --in <v7 prompt> --out <v8 prompt>
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 插入锚点：prompt 末尾的「限制：」块之前
A_LIMITS = "\n限制：\n"

TIME_ANCHOR_SECTION = """
### ⏱ 时间锚点：**任务窗口泄漏会把原生错误格的行「推到未来」**

⚠️ **这一条会让原生错误格「找不到」⇒ 必然漏检。实测（test 分片）：**

| 实测项 | 1.6pct | 4.7pct |
|---|---:|---:|
| 原生错误格按 5 段 key 在 raw 里找得到 | **671 / 715（93.8%）** | **401 / 586（68.4%）** |
| 找得到的那部分，agent 的漏检 | **0 格** | **0 格** |
| 找不到的那部分 = 行被泄漏挪走的 | 44 格 | 185 格 |

⇒ **注入器把 `icu/chartevents.csv` 的 `charttime` 往后推了 30~48 小时**
（subtype = `future_window_leakage`，test 分片实测 1,745,517 处，99.96% 是往后推；
偏移量有 4.5 万种、**不固定** ⇒ **写不成固定偏移规则**），
⇒ 被挪走的那一行，在 raw 里**已经不在 reference 那个时刻上了**
⇒ 你如果按 `(stay_id, charttime, itemid)` 精确匹配去定位原生错误格，
**这些格一个都找不到**（会被记成 `b_missing` = 漏检）。

⇒ **三条做法**：

**① 定位原生错误格要靠「值」，不能只靠「key」。**
在整个 stay 的该 itemid 的**全部读数**里找那个**值越界 / 值是哨兵码**的，
不要只找 charttime 恰好对上的那一个。

**② 识别被泄漏推走的行：它的 charttime 落在该 stay 的时间轴之外。**
判据（train 上可验）：`charttime > 该 stay 的 outtime`，
或它比同一 stay 内同一 itemid 的其它读数**晚了一天以上**。

**③ 把时间轴拉回来。**
train 上实测有效的一条规则：
**charttime 落在 stay 时间轴之外时，取该 stay 内该 itemid 的「最近的前一个有效 charttime」**
（train 命中率约 90%）。
⚠️ **改了 charttime 之后，key 才对得上** ——
输出行的 charttime 必须回到 reference 那个时刻，否则 host 按 key 对齐时
照样把这一格记成 `b_missing` ⇒ **值修对了也等于没修**。

**④ 顺带提醒**：去重必须在「修好 key（itemid 别名）」**之后**做，
否则被别名污染的行会和别的行撞成重复、或者反过来把该留的行删掉。
"""

NATIVE_KEY_BULLET = """
- ⚠️ **修原生错误时，「这一行的 charttime」也可能是错的**：
  「任务窗口泄漏」会把 `chartevents` 的 charttime **往后推 30~48 小时**
  ⇒ 原生错误格按 `(stay_id, charttime, itemid)` 精确匹配**会找不到**（实测 test 分片：
  1.6pct 有 44/715、4.7pct 有 185/586 的原生格在 raw 里对不上这个 key）。
  ⇒ **两件事一起做**：① **按值定位**（在整个 stay 的该 itemid 读数里找越界/哨兵的那个，
  别只认 charttime 对得上的）；② **把 charttime 拉回该 stay 的正常时间轴**
  （`charttime > outtime` 时取该 stay 内该 itemid 最近的前一个有效 charttime，train 上命中约 90%）。
  ⇒ **只修值、不修时间 ⇒ 这一格评分时仍是 `b_missing`（漏检）**。
  （细节与实测数据见 prompt 末尾的「⏱ 时间锚点」一节。）
"""

# 「### 注意」这一段的最后一条 bullet 后面（= 原生错误节的末尾）
A_NOTES_END = ("先按注入错误的规则修好，再走上面的中位数填补 —— "
               "否则被注入错误弄坏的读数会污染中位数。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="v7 prompt")
    ap.add_argument("--out", dest="dst", required=True, help="v8 prompt 输出")
    args = ap.parse_args()

    text = Path(args.src).read_text(encoding="utf-8")
    # ⚠️ 基线 prompt 里本来就有「时间锚点」这个词（train_analysis.json 的字段要求），
    #    所以要用**带图标的标题**来判重
    if "⏱ 时间锚点" in text:
        raise SystemExit("FATAL: 输入里已经有「⏱ 时间锚点」一节了，别重复生成")
    if A_LIMITS not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_LIMITS)
    if A_NOTES_END not in text:
        raise SystemExit("FATAL: 找不到「### 注意」段末的锚点")

    # ① 原生错误节的「### 注意」清单里补一条（agent 写代码前正好会读这一段）
    text = text.replace(A_NOTES_END, A_NOTES_END + "\n" + NATIVE_KEY_BULLET, 1)
    # ② prompt 末尾补完整的「时间锚点」一节（含实测数据表）
    text = text.replace(A_LIMITS, "\n" + TIME_ANCHOR_SECTION + A_LIMITS, 1)

    Path(args.dst).write_text(text, encoding="utf-8")
    print("已写出:", args.dst, "(%d bytes)" % len(text.encode("utf-8")))
    print("  新增：原生错误节「注意」清单 1 条 + 末尾「时间锚点」一节（%d 字符）"
          % len(TIME_ANCHOR_SECTION))


if __name__ == "__main__":
    main()
