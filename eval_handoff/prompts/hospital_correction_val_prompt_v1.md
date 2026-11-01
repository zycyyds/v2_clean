你是 Data Cleaning Agent。请完成一个**单表纠错**任务：给你一张被人为破坏过的表（脏表），以及 **10 行**的「脏 ↔ 干净」对照样例。请你从这 10 行对照里归纳出破坏规则，然后把**整张表**修好，交付一份可重放的 pipeline 和一份修好的结果表。

本回合末尾的 `[Validation Harness public contract]` 会给出 train raw / train reference / validation raw / agent workdir 的**真实绝对路径**。请直接使用那里给出的路径，不要去找 `TRAIN_RAW` 之类的占位符字符串。**测试集 raw 在本阶段不提供，也不需要你去找**——你交付的是一份**可重放**的 pipeline；host 会把你最好的一版冻结下来，用 `{raw_root}` 指向测试集 raw 跑一次独立重放。

## 工作流契约（务必遵守）

- **每一回合结束时，agent workdir 里必须已经躺着一份完整的结果包**：修好的表 + `submission.json`。host 会在你的回合结束后**立刻读取并打分**，再把聚合分数反馈给你。回合结束时没有结果包，这一回合就是 0 分。
- harness 最多给你若干回合，每回合都计分。如果某一回合分数低于之前的最好版本，host 可能把你的 workdir **回滚**到那份最好的文件——所以有更好的产出就立刻写盘，不要只留在推理过程里。
- 每回合的反馈是你的主要信息来源。照反馈改规则、重新跑一遍、再交付。

**重要**：validation 的标准答案（hidden gold）对你不可见，host 会把它的路径 redact 成 `<hidden_gold>`。**不要试图读取 pi_test_scorer.py / evaluation_manifest.json 之类的评分元数据，也不要去猜测测试集标准答案。**

本轮的核心不是只交一张表，而是从 10 个对照样例推导**可重放**的修复规则。开始时请直接读取 contract 里给出的路径。可以枚举它们共同的父目录，但不得读取任何其他数据集或实验目录。

## 任务定义

**输入**：一张 CSV 表（`hospital.csv`，20 列，主键 `index`）。表里的某些 cell 被改坏了，改法可能不止一种。

**对照样例**：`train/raw/` 里是 10 行脏数据（同一个表的前 10 行切片），`train/reference/` 里是**同样这 10 行**的干净版本。两份文件行序一一对应。

**产出**：`{raw_root}` 指向的那张表（train raw 或 validation raw 或最终测试集 raw）的**修好版本**。

**输出表的契约**：
- 文件名 `hospital.csv`；
- **列名、列数、列顺序必须与输入表完全一致**（用 train reference 的表头，不要改列名大小写或命名风格）；
- **行数与行序必须与输入表完全一致**（不要增删行、不要重排行、不要丢行）；
- 只改该改的 cell，**没被破坏的 cell 要原样保留**（改坏干净 cell 会被单独扣分，比漏改更糟）。

## 数据特点

这是一张**逐字段独立注入**的单表：脏表里被破坏的 cell 散布在整张表的每一行，不是集中在某几行。20 列里绝大多数列都带了错误。

**具体有哪些破坏模式、每种有多少、正确值应该是什么——这些都需要你通过对照 train raw 与 train reference 自己推导。**
常见的破坏模式包括但不限于：值被清空 / 值被替换成某个标记文本 / 值本身合法但**格式**被改了（多出前缀或后缀）/ 值被换成了别的合法值 / 数值型字段的数值被改过。每一种的判定条件和正确值的来源，都要靠 10 行对照自己踩出来。

**错误量级（供你估算工作量，不是硬约束）**：
- train: 10 行、30 个被破坏的 cell（这 10 行覆盖了 17 个有错的列）
- validation: 20 行、24 个被破坏的 cell
- 最终测试集: 970 行、455 个被破坏的 cell（分布在约 407 行上）

**几个容易踩的坑**：
- **train 只有 10 行、30 个被破坏的 cell。** 每一列平均只有不到两个样本。同一列里如果有多种破坏模式，train 上你可能只看到其中一种。不要因为某类模式在 train 里只出现一次就判定它不重要，也不要因为它出现得多就以为测试集里也占多数。
- **反过来，某个破坏模式在 train 和 validation 里都没出现，不代表测试集里没有。** 你可以从数据本身去推断 train/validation 未覆盖的模式，但每一条都必须能用对照证据或数据自身规律说明为什么它是错的——不要凭空编造。
- **各阶段的行数规模差两个数量级**：validation 只有 20 行，最终测试集有 970 行。别把 validation 的处理规模当成测试集的规模。
- **正确值不一定来自「这张表里被破坏的那一行」。** 有些破坏把原始信息彻底抹掉了（值变成空串、变成标记文本），这时候要想想**同一张表的其它行**能不能提供线索。但**能不能跨行取值、以及该取哪一个，是逐列不同的**：有的列整列共享同一个值（那就可以跨行取值），有的列每行各有各的值（跨行取值反而会改错）。这一点必须在 train 对照上逐列验证，不能一刀切。
- **不要用 train 或 validation 的错误绝对数量去推断测试集该有多少行需要处理。** 从 train 得到的应该是**每类破坏的判定规则**，而不是"这张表大概该改多少比例的行"。

### 推导与修正流程

**推荐流程**（具体细节由你按 train 对照决定）：

1. **加载 raw** → 读入 `hospital.csv`（train raw / validation raw 结构完全相同）
2. **对照 train raw 与 train reference** → 系统性比对这 10 行的每一列：
   - 哪些 cell 值与 reference 不一致？
   - 不一致的**形态**是什么（空？标记文本？格式变了？值被换了？）？
   - 这类形态的**正确值**能从哪里来？规则能不能在全部 10 行上都成立？
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

必须完成的工作：

1. 系统性比较 train raw 与 train reference 的 schema、主键、行级与列级变化。
2. 在 agent workdir 写出 `train_analysis.json`：记录表的 schema、主键、**你归纳出的破坏模式清单**（每条至少配一项 train 对照证据）、每条规则对应的修复逻辑、每列各自的取值域/跨行可取性判断，以及 train 自检结果。
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
- 请在交付前自己至少做一次完整的 train 自检，并在干净目录里验证一次 replay 能跑通。

限制：

- 不使用网络
- 不读取、搜索或推断测试集标准答案、其他实验目录、历史 pipeline、其他工作树或项目外数据（也不读取 pi_test_scorer.py / evaluation_manifest.json 之类的评分元数据）
- 已注册的 Skills 可按需读取，但它们不包含隐藏数据或隐藏答案
- 证据不足时采用保守、可解释、由 train 对照支持的规则；先保证 train 自检通过，再追求 validation 正确性
