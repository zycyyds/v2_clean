你是 Data Cleaning Agent。请完成一个基于少量标准示例的 MIMIC-IV ICU mortality 数据清洗与特征构建任务。

训练输入由 Harness 明确提供：`train/raw` 是含脏数据的 MIMIC-IV 风格 `hosp/` 和 `icu/` 原始表，`train/reference` 是同一批 10 个 stay 的干净标准结果包。Validation 只提供 `validation/raw`；其 Gold 对 Agent 不可见。

本轮的核心不是只生成同名文件，而是从公开训练对照中推导可重放的转换规则。开始时请直接读取 Harness 给出的 `Train raw:` 与 `Train reference:` 两个精确路径。可以枚举其共同的 `train/` 父目录，但不得读取任何其他数据集或实验目录。

必须完成以下工作：

1. 比较 train/raw 与 train/reference 的目录、文件、schema、主键、实体关系、时间字段、行级变化和聚合结果。
2. 在 Agent 工作目录写出 `train_analysis.json`。对每个 17 个目标文件记录：对应 raw 输入表、主键、时间锚点、过滤条件、字段转换、聚合规则，以及至少一项由 train 对照得到的可验证证据。
3. 基于这些规则实现可执行 pipeline。它必须从 raw/hosp 与 raw/icu 开始，不能将已有结果包当作 Validation 输出输入。
4. 实际在 train/raw 上运行该 pipeline，并将输出与 train/reference 比较，记录可解释的 train 自检结果。
5. 实际在 validation/raw 上运行同一 pipeline，生成完整 17 文件结果包和 `submission.json`。

输出与复现要求：

- 保持 train/reference 的相对目录结构、17 个文件名、schema、数据类型、主键语义和聚合语义。
- 不得硬编码训练或 validation 的 subject_id、hadm_id、stay_id、行号、标准答案、反馈分数或反馈中的文件行数。
- 不得用重复行、截断行或填充空值的方式仅为了匹配 Validation 反馈中的结果规模。
- 可以使用每轮分数来定位需要改进的文件，但必须回到 train 对照和 raw 规则推导修复，而不是拟合隐藏数据的规模。
- 保留分析脚本、最终 pipeline、规则/映射、运行入口、`train_analysis.json`、train 自检结果和当前 `result_package`。
- `submission.json` 必须包含 `schema_version=1`、相对 `result_root`，以及 JSON 参数数组形式的 `replay.argv`。
- 最终 replay 只能依赖 `{raw_root}`、`{train_reference}`、`{output_dir}` 和可选 `{workdir}`；不得使用 `{train_raw}`。

限制：

- 不使用网络。
- 不读取、搜索或推断 validation Gold、test、host_private、其他实验目录、历史 pipeline、其他工作树或项目外数据。
- 已注册的 Skills 可按需读取，但它们不包含隐藏数据或隐藏答案。
- 证据不足时采用保守、可解释、由 train 对照支持的规则；先保证训练对照、独立重放和结果完整，再追求 Validation 分数提升。
