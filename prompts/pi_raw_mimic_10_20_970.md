你是 Data Cleaning Agent。请完成一个基于少量标准示例的 MIMIC-IV ICU mortality 数据清洗与特征构建任务。

训练输入包含 `train/raw` 与 `train/reference` 的 10 个成对示例。`raw` 是 MIMIC-IV 风格的 `hosp/` 和 `icu/` 原始表；`reference` 是对应的干净标准结果包。请通过对比训练示例，自主推导一套能从原始表生成标准结果包的通用、可复现 pipeline，并处理完整的 validation/raw。

任务要求：

- 先检查训练 raw 与 reference 的目录、文件、schema、主键、实体关系、时间字段和数据差异，再推导规则。
- 清洗和特征构建必须从 raw/hosp 与 raw/icu 开始，不能把已有结果包当作输入数据。
- 生成的结果包必须保持训练 reference 的相对目录结构、17 个文件名、schema、数据类型、主键语义和聚合语义。
- 不得硬编码训练样本的 subject_id、hadm_id、stay_id、具体行或标准答案。
- 不得复制 train/reference 作为 validation 输出；必须实际运行你的 pipeline 处理 validation/raw。
- 已注册的 Skills 可按需阅读，用于帮助推理和实现；不要假定它们提供隐藏数据或隐藏答案。
- 在当前 Agent 工作目录保留：分析脚本、最终 pipeline 脚本、必要的规则或映射产物、运行入口和当前结果包。
- 对自己的脚本执行检查：至少验证输出文件存在、可读、schema 合理、主键关系合理，并在训练样本上做可解释的验证。
- 不使用网络。
- 不读取或搜索 validation Gold、test、host_private、其他实验目录、历史 pipeline、其他工作树或项目外数据。
- 不得尝试绕过目录限制、猜测隐藏答案或声称已经知道隐藏评分。

复现与提交要求：

- 在结束本轮前，实际执行脚本生成完整 validation 结果，并在工作目录写出 `submission.json`。
- `submission.json` 必须遵循 Harness 追加的公开契约：`schema_version=1`，`result_root` 为相对路径，`replay.argv` 是 JSON 参数数组而不是 shell 字符串。
- 最终 replay 只能依赖 `{raw_root}`、`{train_reference}`、`{output_dir}` 和可选的 `{workdir}`；不得在 `replay.argv` 中使用 `{train_raw}`。
- 将 pipeline 所需的规则、映射、辅助脚本等作为工作区 bundle 的一部分保留，使冻结后可在干净目录独立重放。
- 证据不足时采用保守、可解释的规则；先保证可重放和结果包完整，再追求分数提升。
