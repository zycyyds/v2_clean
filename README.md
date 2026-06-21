# Gold-Guided Data Agent

当前系统基于 AgentScope `ReActAgent + Toolkit`，包含三个相互隔离的部分：

1. `DataExplorer`：读取训练 raw/gold 示例，递归发现目标叶子字段并分析来源。
2. `FeatureEngineer`：根据字段规则和细粒度任务，调用或派生 Skill 处理验证原始数据。
3. `Evaluator`：独占验证 gold，评分并产生不含 gold/pred 具体值的公开反馈。

原始 `skills/` 和 `lib/` 在 Agent 运行时只读。Engineer 可以在当前实验中创建 adapter/fork，也可以在没有合适 Skill 时编写单任务 Python，但不能修改全局 Skill。

## Skill 分层

- `explore`：Gold 分析、文件发现、schema 和样本检查，只用于探索阶段。
- `process`：结构化抽取、join、字段派生、文本抽取、OCR 和可逆规范化。
- `label`：仅在任务明确要求标签或预测目标时加载。
- `util`：工作簿导出、结果验证和生命周期辅助。

每个 Skill 必须声明 `name`、`layer`、`description`、`when_to_use`、`when_to_skip`、`capability_types`、`inputs`、`outputs` 等元数据。Agent 的选择顺序固定为：

```text
直接调用 -> 调整参数 -> 当前实验 adapter -> 当前实验 fork -> standalone Python
```

## 标准产物

Explorer 输出：

```text
field_extraction_rules.json
extraction_task_plan.json
data_analysis_report.json
explorer_run_record.json
```

Engineer 输出：

```text
final_dataset.xlsx
result_manifest.json
target_field_mapping.json
extraction_task_execution.json
skill_usage_report.json
```

`final_dataset.xlsx` 的业务 Sheet 根据 Gold 顶层类别动态生成，不绑定 MIMIC 或固定医疗字段；固定控制 Sheet 为 `_cases`、`_provenance`、`_unsupported`。

## 训练验证闭环

```bash
cd /Users/mkbk/PycharmProjects/v2_clean
conda activate py310

python main.py \
  --workflow train-validate \
  --train-examples /absolute/path/to/train_examples \
  --validation-raw /absolute/path/to/validation_raw \
  --validation-gold /absolute/path/to/validation_gold \
  --experiment-dir /absolute/path/to/experiment \
  --max-iters 60 \
  "描述目标字段、记录粒度、是否需要标签以及业务约束"
```

验证 gold 只传给 Evaluator，不得写入 prompt。需要限制本次执行轮数时使用 `--round-limit N`；再次使用同一个 `--experiment-dir` 可继续运行。

候选 bundle 只有在以下条件全部满足后才可能晋升：

- required extraction task 均有真实执行记录。
- 每个规则进入 mapping 或 `_unsupported`。
- `final_dataset.xlsx` 和控制 Sheet 可读取。
- 派生 Skill 有执行凭据且验证通过。
- `skill_usage_report.json` 状态为 `SUCCESS`。
- Evaluator 的 artifact validation 通过，综合分数至少提升 `0.001`。

连续两轮没有提升后冻结最佳 bundle。本期不自动运行测试集。

## 兼容入口

```bash
# 旧单轮双阶段入口
python main.py --two-phase --max-iters 60 "任务描述"

# 查看完整 Skill 元数据
python main.py --show-manifest
```

## 验证

```bash
conda run -n py310 python -m pytest -q tests
conda run -n py310 python -m compileall -q agent agent_tools skills workflow lib main.py
```
