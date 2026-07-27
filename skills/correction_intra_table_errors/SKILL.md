---
name: correction_intra_table_errors
description: Use when Data Cleaning Agent must detect and conservatively repair data-quality problems that can be proven from one table without joins or external answer keys.
---

# 单表错误纠正

## 适用条件

- 异常可以只依赖一张表的 schema、枚举、取值分布、缺失模式、行内时序或重复关系证明。
- 不需要借助另一张业务表或任务标签才能判断记录是否异常。
- 如果证据依赖实体映射、跨表关联或任务时间窗，继续读取对应的其他错误 View Skill。

## 证据要求

- 使用 `InspectDataFile` 检查 schema、类型、空值、唯一值、分位数和极端值。
- 使用 `CompareArtifact` 比较成对标准示例，确认异常模式是否稳定且可复现。
- 使用 `ExecutePython` 做只读统计诊断时，保留正常范围、异常频率和分组差异作为证据。
- 不把罕见值自动等同于错误；需要约束冲突、重复证据或标准示例支持。

## 推荐工具

- 先用 `InspectDataFile` 获取 schema、空值、枚举、分位数和样例。
- 用 `CompareArtifact` 对齐成对示例，定位可重复的单表差异。
- 需要分组统计或约束验证时，用 `ExecutePython` 运行 workspace 中的只读诊断脚本；不要在诊断脚本中直接改输入。

## 执行步骤

1. 确定表的记录粒度、主键候选和字段语义。
2. 分别检查 schema、枚举、范围、缺失、重复和表内时序约束。
3. 在成对示例中验证异常模式与标准形式的对应关系。
4. 优先形成可解释、可重复执行且只影响目标记录的修复规则。
5. 修改后重新检查行数、schema、主键、分布和未受影响记录。

## 修复与停止条件

- 只有唯一正确值能由同表证据或成对示例推出时才恢复具体值。
- 无法推出原值但能证明取值无效时，按任务契约选择置空、标记无效或保留并报告。
- 完全重复且没有独立业务含义的记录可以删除；近似重复必须先确认粒度。
- 证据不足、多个修复值同样合理或修改会扩大影响范围时停止自动修复并保持原值。

## 禁止事项

- 不使用隐藏答案、评分结果或当前数据集的预设错误清单。
- 不根据单个极端样本臆造全局阈值。
- 不为了让分布更平滑而修改合法的罕见记录。
- 不改变目录结构、文件名、schema、序列化格式或无关业务文件。
