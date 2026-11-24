你是 Data Cleaning Agent。请完成一个**单表纠错**任务：给你一张被人为破坏过的表（脏表），以及 **30 行**的「脏 ↔ 干净」对照样例。请你从这 30 行对照里归纳出破坏规则，然后把**整张表**修好，交付一份可重放的 pipeline 和一份修好的结果表。

本回合末尾的 `[Validation Harness public contract]` 会给出 train raw / train reference / validation raw / agent workdir 的**真实绝对路径**。请直接使用那里给出的路径，不要去找 `TRAIN_RAW` 之类的占位符字符串。**测试集 raw 在本阶段不提供，也不需要你去找**——你交付的是一份**可重放**的 pipeline；host 会把你最好的一版冻结下来，用 `{raw_root}` 指向测试集 raw 跑一次独立重放。

## 工作流契约（务必遵守）

- **每一回合结束时，agent workdir 里必须已经躺着一份完整的结果包**：修好的表 + `submission.json`。host 会在你的回合结束后**立刻读取并打分**，再把聚合分数反馈给你。回合结束时没有结果包，这一回合就是 0 分。
- harness 最多给你若干回合，每回合都计分。如果某一回合分数低于之前的最好版本，host 可能把你的 workdir **回滚**到那份最好的文件——所以有更好的产出就立刻写盘，不要只留在推理过程里。
- 每回合的反馈是你的主要信息来源。照反馈改规则、重新跑一遍、再交付。

**重要**：validation 的标准答案（hidden gold）对你不可见，host 会把它的路径 redact 成 `<hidden_gold>`。**不要试图读取 pi_test_scorer.py / evaluation_manifest.json 之类的评分元数据，也不要去猜测测试集标准答案。**

本轮的核心不是只交一张表，而是从 30 个对照样例推导**可重放**的修复规则。开始时请直接读取 contract 里给出的路径。可以枚举它们共同的父目录，但不得读取任何其他数据集或实验目录。

## 任务定义

**输入**：一张 CSV 表（`tax.csv`，16 列，主键 `__row_id`）。表里的某些 cell 被改坏了，改法可能不止一种。

**对照样例**：`train/raw/` 里是 30 行脏数据（**从整张表里挑出来的 30 行，不是前 30 行**），`train/reference/` 里是**同样这 30 行**的干净版本。两份文件行序一一对应。

**产出**：`{raw_root}` 指向的那张表（train raw 或 validation raw 或最终测试集 raw）的**修好版本**。

**输出表的契约**：
- 文件名 `tax.csv`；
- **列名、列数、列顺序必须与输入表完全一致**；
- **行数与行序必须与输入表完全一致**（不要增删行、不要重排行、不要丢行）；
- 只改该改的 cell，**没被破坏的 cell 要原样保留**（改坏干净 cell 会被单独扣分，比漏改更糟）。

## 数据特点

这是一张**逐字段独立注入**的单表：脏表里被破坏的 cell 散布在整张表的很多行上，不是集中在某几行。

**具体有哪些破坏模式、每种有多少、正确值应该是什么——这些都需要你通过对照 train raw 与 train reference 自己推导。**
常见的破坏模式包括但不限于：值被清空 / 值被替换成某个标记文本 / 值本身合法但**格式**被改了（多出前缀或后缀）/ 合法值被换成了另一个合法值 / 数值被改成了另一个范围内的值。每一种的判定条件和正确值的来源，都要靠 30 行对照自己踩出来。

**错误量级（供你估算工作量，不是硬约束）**：
- train: 30 行、89 个被破坏的 cell
- validation: 20 行、22 个被破坏的 cell
- 最终测试集: 199,950 行、121,108 个被破坏的 cell

**几个容易踩的坑**：
- **train 只有 30 行、89 个被破坏的 cell。** 每一类破坏模式在 train 里可能只有一两个样本。不要因为某类模式在 train 里只出现一次就判定它不重要，也不要因为它出现得多就以为测试集里也占多数。
- **反过来，某个破坏模式在 train 和 validation 里都没出现，不代表测试集里没有。** 你可以从数据本身去推断 train/validation 未覆盖的模式，但每一条都必须能用对照证据或数据自身规律说明为什么它是错的——不要凭空编造。
- **各阶段的行数规模差四个数量级**：validation 只有 20 行，最终测试集有 199,950 行。别把 validation 的处理规模当成测试集的规模。
- **正确值不一定来自「这张表里被破坏的那一行」。** 有些破坏把原始信息彻底抹掉了（值变成空串、变成标记文本），这时候要想想**同一张表的其它行**能不能提供线索。但**能不能跨行取值、以及该取哪一个，是逐列不同的**：有的列整列共享同一个值（那就可以跨行取值），有的列每行各有各的值（跨行取值反而会改错）。这一点必须在 train 对照上逐列验证，不能一刀切。
- **不要用 train 或 validation 的错误绝对数量去推断测试集该有多少行需要处理。** 从 train 得到的应该是**每类破坏的判定规则**，而不是"这张表大概该改多少比例的行"。

### 推导与修正流程

**推荐流程**（具体细节由你按 train 对照决定）：

1. **加载 raw** → 读入 `tax.csv`（train raw / validation raw 结构完全相同）
2. **对照 train raw 与 train reference** → 系统性比对这 30 行的每一列：
   - 哪些 cell 值与 reference 不一致？
   - 不一致的**形态**是什么（空？标记文本？格式变了？数值变了？）？
   - 这类形态的**正确值**能从哪里来？规则能不能在全部 30 行上都成立？
3. **建立规则表**：每条规则 = 「怎么判定这个 cell 被破坏了」+「正确值是什么」。每条规则至少配一项 train 对照证据。
4. **实现 pipeline**：从 `{raw_root}` 读表 → 逐条规则检测+修复 → 写出修好的表。
5. **train 自检**：在 train raw 上跑 pipeline，输出必须与 train reference 逐格一致。**自检不通过就不算做完。**
6. **在 validation 上检验**：跑同一 pipeline，用 host 每回合给的分数当反馈，把 validation 暴露的新形态补进规则。
7. **产出结果包 + `submission.json`**

**重要约束**：
- 不要硬编码任何行号、cell 坐标或标准答案；规则必须是从数据里推出来的、可作用于任意行的。
- 不要用重复行、截断行、填充空值的方式拟合任何数据集的规模。
- 不能读取 scoring 元数据（pi_test_scorer.py / evaluation_manifest.json 之类），也不能读取 validation 的 hidden gold。
- **replay 契约**：harness 重放时只注入 `{raw_root}`、`{train_reference}`、`{output_dir}` 和可选 `{workdir}` 四个参数。`{raw_root}` 在训练时指向 train raw、在验证时指向 validation raw、在最终测试时指向测试集 raw。**不得**使用 `{train_raw}` 或 `{validation_raw}`——重放时它们根本不存在。
- ⚠️ **`submission.json` 的 `replay.argv` 写法错了 ⇒ 独立重放这一关直接判 0**（且 harness 的重放修复循环没有上限 ⇒ 会一轮接一轮烧额度）⇒ 具体四条见「输出与复现要求」里的 `replay.argv` 契约。

必须完成的工作：

1. 系统性比较 train raw 与 train reference 的 schema、主键、行级与列级变化。
2. 在 agent workdir 写出 `train_analysis.json`：记录表的 schema、主键、**你归纳出的破坏模式清单**（每条至少配一项 train 对照证据）、每条规则对应的修复逻辑，以及 train 自检结果。
3. 基于这些规则实现**可重放**的 pipeline。它必须从 `{raw_root}` 读原始表，不能把已有结果包当作输入。
4. 在 train raw 上实际运行该 pipeline，与 train reference 比较，**train 自检必须通过：与 reference 逐格一致**。
5. 在 validation raw 上运行同一 pipeline，把每回合 host 给的分数当作反馈，把 validation 暴露的新形态补进规则。
6. 在 validation raw 上产出完整的结果表和 `submission.json`，并确认 replay 能从干净状态复现出同样的表。

输出与复现要求：

- 结果表放在 workdir 的一个**子目录**里（作为 `submission.json` 的 `result_root`，必须是相对路径）。
- 结果表的文件名、schema、行数、行序与输入表一致。
- 不得硬编码任何行号、cell 坐标或标准答案。
- 不得用重复行、截断行或填充空值的方式仅为了匹配数据集规模。
- 保留分析脚本、最终 pipeline、规则表、运行入口和 `train_analysis.json`。
- `submission.json` 必须包含 `schema_version=1`、相对 `result_root`，以及 JSON 参数数组形式的 `replay.argv`。
- **replay 契约**：只依赖 `{raw_root}`、`{train_reference}`、`{output_dir}` 和可选 `{workdir}`；**不得**使用 `{train_raw}` 或 `{validation_raw}`。

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
  ( cd <workdir> && env -i PATH=/usr/bin:/bin HOME=$HOME TMPDIR=/tmp \
      python3 scripts/clean.py --raw_root <raw> --output_dir <out> )
  ```
  ⚠️ 这一行**必须写成 `python3 <脚本路径>`**（不要靠 shebang 启动），这才是 harness 真正的执行形态。
- 请在交付前自己至少做一次完整的 train 自检，并在干净目录里验证一次 replay 能跑通。


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

限制：

- 不使用网络
- 不读取、搜索或推断测试集标准答案、其他实验目录、历史 pipeline、其他工作树或项目外数据（也不读取 pi_test_scorer.py / evaluation_manifest.json 之类的评分元数据）
- 已注册的 Skills 可按需读取，但它们不包含隐藏数据或隐藏答案
- 证据不足时采用保守、可解释、由 train 对照支持的规则；先保证 train 自检通过，再追求 validation 正确性
