#!/usr/bin/env python3
"""make_correction_prompt.py —— 由「校正类」任务书模板生成我们给 agent 用的版本。

**不引入任何特定数据集的内容**（这一族是单表纠错：10 行脏↔干净对照 → 修整张表，
和 MIMIC cohort 那一族没有任何关系）。

做三件事：

  ① **修一处事实错误**（只有 hospital 模板有）：
     它写 train 对照样例是「同一个表的前 10 行切片」，而实际切分
     （`split_manifest.json` 的 `selection = curated_greedy_changed_column_coverage_v1`）
     是**按 index 散布挑出来的 10 行**（实测 index = 158/214/229/…/699，
     而整表前几行是 1/2/3/…）。另外三个模板（beers/rayyan/tax）写的就是
     「从整张表里挑出来的 N 行，不是前 N 行」⇒ 统一成这个说法。
     ⚠️ 这条必须对：agent 若以为「train 是前 10 行」，会推出只在前缀成立的规则。

  ② **加「replay.argv 契约」**（放在「输出与复现要求」里，紧跟着原有那条 replay 契约）：
     `argv[0]` 必须是 `"python3"`、脚本路径相对或 `{workdir}/…`、不许静态绝对路径、
     必须含 `{raw_root}` 与 `{output_dir}`、`argv[0]` 不能是 shell。
     ⇒ 写错的代价：**host 的「独立重放」这一关直接判 0**，而且 harness 的重放修复循环
     **没有上限**（修不好就一轮接一轮重跑，每轮都烧一大笔额度）。

  ③ **加「上下文预算 / 续跑摘要格式」**（放在 prompt 末尾的「限制：」之前）：
     上下文到 495,616 token 会触发压缩，压缩时要交一份 **10 个纯字符串字段**的摘要；
     字段写成数组/对象、或漏字段 ⇒ 本轮失败 ⇒ 连挂 4 次 ⇒ worker 崩 ⇒ 整个跑批零产出。
     ⇒ 同时要求把这条写进自己的产物（`train_analysis.json` / `NOTES.md`），每轮开工前先读。

用法:
    python3 reproduction/make_correction_prompt.py \
        --dataset <切分目录> --in <模板 prompt> --out <输出 prompt>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- 锚点（必须与模板原文**逐字**一致，否则报错）----
# ⚠️ 锚点取的是**整条 bullet**（含句末），不能只取前半段 ——
#    否则新内容会插进句子中间，把后半句甩到后面去。
A_CONSTRAINT_REPLAY = (
    "- **replay 契约**：harness 重放时只注入 `{raw_root}`、`{train_reference}`、"
    "`{output_dir}` 和可选 `{workdir}` 四个参数。`{raw_root}` 在训练时指向 train raw、"
    "在验证时指向 validation raw、在最终测试时指向测试集 raw。"
    "**不得**使用 `{train_raw}` 或 `{validation_raw}`——重放时它们根本不存在。")
A_OUTPUT_REPLAY = (
    "- **replay 契约**：只依赖 `{raw_root}`、`{train_reference}`、`{output_dir}` "
    "和可选 `{workdir}`；**不得**使用 `{train_raw}` 或 `{validation_raw}`。")
A_LIMITS = "\n限制：\n"

# ① 事实错误修正：老模板写「（同一个表的前 N 行切片）」
OLD_SLICE = "（同一个表的前 {n} 行切片）"
NEW_SLICE = "（**从整张表里挑出来的 {n} 行，不是前 {n} 行**）"

# ② replay.argv 契约（放在「输出与复现要求」里）
REPLAY_CONTRACT_BLOCK = """
- ⚠️ **`replay.argv` 的四条硬规则**（写错 ⇒ host 的「独立重放」这一关**直接判 0 分**，
  而 harness 的重放修复循环**没有上限** ⇒ 修不好会一轮接一轮地重跑，每轮都烧一大笔额度）：
  1. **`argv[0]` 必须是 `"python3"`** —— harness 会把它换成它自己那个 python，
     **你的脚本是作为参数传进去的（不解析 shebang）**。
     ⚠️ **不要把启动脚本当 `argv[0]`**：那样会按脚本的 shebang 启动，
     而重放环境的 PATH 只有 `/usr/bin:/bin:/usr/sbin:/sbin`，
     `#!/usr/bin/env python3` 会解析到 macOS 自带的 `/usr/bin/python3`（一个壳程序），
     在重放环境里起不来 ⇒ 报 `execvp() ... failed: No such file or directory`
     （**看着像文件不存在，其实是解释器起不来**）。
  2. **脚本路径要么相对**（`scripts/clean.py`，重放的 cwd 就是 workdir）、
     **要么写成 `{workdir}/scripts/clean.py`**；
     ⚠️ **不许出现「静态绝对路径」**（任何不含占位符的项都必须是相对路径），
     否则校验器直接拒：`submission replay.argv cannot contain static absolute paths`。
  3. **必须含 `{raw_root}` 和 `{output_dir}` 两个占位符**（缺一个直接被拒）；
     可选 `{train_reference}`、`{workdir}`。
  4. **`argv[0]` 不能是 shell 命令字符串**（`sh` / `bash` / `zsh` / `env` / `dash` …）
     ⇒ 校验器直接拒：`submission replay.argv cannot execute shell command strings`。
  ⇒ **照抄这个形状**（只改脚本名和你自己的参数名）：
  ```json
  "replay": {"argv": ["python3", "{workdir}/scripts/clean.py",
                      "--raw_root", "{raw_root}",
                      "--output_dir", "{output_dir}",
                      "--train_reference", "{train_reference}",
                      "--workdir", "{workdir}"]}
  ```
  ⇒ **交付前自己验一遍**（比事后猜省一整轮）：
  ```bash
  cd <workdir> && python3 -c "import json;print(json.load(open('submission.json'))['replay']['argv'])"
  # 再用「同样的 cwd + 最小 PATH」实跑一次（模拟重放环境）
  ( cd <workdir> && env -i PATH=/usr/bin:/bin HOME=$HOME TMPDIR=/tmp \\
      python3 scripts/clean.py --raw_root <raw> --output_dir <out> )
  ```
  ⚠️ 这一行**必须写成 `python3 <脚本路径>`**（不要靠 shebang 启动），这才是 harness 真正的执行形态。"""

# ③ 上下文预算 + 续跑摘要（放在 prompt 末尾「限制：」之前）
CONTEXT_BUDGET_BLOCK = """
### 🧠 上下文预算：别把大块内容打到 stdout，续跑摘要别写错格式

你的上下文有上限（**约 49.6 万 token 会触发一次「上下文压缩」**）：
压缩时 harness 要你交一份「续跑摘要」，它必须是 **10 个纯字符串字段**的 JSON。
⚠️ **这一关写错 ⇒ 本轮失败 ⇒ 连挂 4 次 ⇒ worker 崩 ⇒ 整个跑批零产出**
（实测：另一个规模的数据集上就是这么报废的 —— agent 把某个字段写成了数组而不是字符串）。

⇒ **① 别把大块内容打到 stdout**：要看的都**写进文件**，命令只回显前 20 行 + 总行数，
或者聚合统计。（这张表本身不大，但**最终测试集 970 行**，把整表 print 出来没有必要。）

⇒ **② 被要求写续跑摘要时，10 个字段每个都写纯字符串**（多条目用换行或编号串起来），
**不要写 JSON 数组、JSON 对象，也不要漏字段**。字段是这 10 个：

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

⇒ **③ 把这条写进你的产物**（`train_analysis.json` 里加一节，或在 workdir 里放一个 `NOTES.md`），
**每轮开工前先读它**。
⚠️ 后续回合的任务书里可能**没有**这一节（host 只会回灌分数和逐文件反馈）
⇒ 规则只留在上下文里，下一回合就没了 ⇒ **写进文件才留得住**。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="切分目录（含 split_manifest.json）")
    ap.add_argument("--in", dest="src", required=True, help="任务书模板")
    ap.add_argument("--out", dest="dst", required=True, help="输出 prompt")
    args = ap.parse_args()

    ds = Path(args.dataset)
    text = Path(args.src).read_text(encoding="utf-8")

    manifest = json.loads((ds / "split_manifest.json").read_text(encoding="utf-8"))
    counts = manifest.get("counts", {})
    n_train = int(counts.get("train", 0))
    n_val = int(counts.get("validation", 0))
    n_test = int(counts.get("correction", 0))
    if not n_train:
        raise SystemExit("FATAL: split_manifest 里没有 counts.train")

    # ---- ① 修「前 N 行切片」的事实错误 ----
    fixed_slice = False
    old = OLD_SLICE.format(n=n_train)
    if old in text:
        text = text.replace(old, NEW_SLICE.format(n=n_train), 1)
        fixed_slice = True

    # ---- ② replay.argv 契约 ----
    if A_OUTPUT_REPLAY not in text:
        raise SystemExit("FATAL: 找不到「输出与复现要求」里的 replay 契约锚点")
    text = text.replace(A_OUTPUT_REPLAY, A_OUTPUT_REPLAY + "\n" + REPLAY_CONTRACT_BLOCK, 1)

    # 「重要约束」里加一行指针（离「工作流契约」更近，agent 写 submission.json 前会看到）
    if A_CONSTRAINT_REPLAY not in text:
        raise SystemExit("FATAL: 找不到「重要约束」里的 replay 契约锚点")
    text = text.replace(
        A_CONSTRAINT_REPLAY,
        A_CONSTRAINT_REPLAY
        + "\n- ⚠️ **`submission.json` 的 `replay.argv` 写法错了 ⇒ 独立重放这一关直接判 0**"
          "（且 harness 的重放修复循环没有上限 ⇒ 会一轮接一轮烧额度）"
          "⇒ 具体四条见「输出与复现要求」里的 `replay.argv` 契约。",
        1)

    # ---- ③ 上下文预算 / 续跑摘要 ----
    if A_LIMITS not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_LIMITS)
    text = text.replace(A_LIMITS, "\n" + CONTEXT_BUDGET_BLOCK + A_LIMITS, 1)

    Path(args.dst).write_text(text, encoding="utf-8")
    print("已写出:", args.dst, "(%d bytes)" % len(text.encode("utf-8")))
    print("  train=%s 行 / validation=%s 行 / test(correction)=%s 行" %
          (n_train, n_val, n_test))
    print("  修正「前 N 行切片」事实错误: %s" % ("是" if fixed_slice else "无（模板本来就对）"))
    print("  新增：replay.argv 契约（1 块 + 重要约束 1 行指针）+ 上下文预算一节")


if __name__ == "__main__":
    main()
