# Cell Error Detection / Strict R-GCN 后续工作计划

> 字段级 `F_corr` 修复流程已经实现并单独记录在
> [cell_repair_candidate_workflow.md](cell_repair_candidate_workflow.md)。Strict R-GCN 是唯一
> 错误检测器；MiniMax M3 只根据 Train dirty-clean pairs 离线生成字段修复函数，冻结后在
> Validation/Test 阶段不访问模型 API。

更新时间：2026-08-13

本文用于窗口关闭后的工作交接。它记录已经完成的事实、当前实验结论、建议执行顺序、尚未解决的问题，以及需要项目负责人确认的分歧点。除“已冻结”部分外，其余内容不应被当作已经确定的最终论文方案。

## 1. 当前目标

在 MIMIC Typed-Value 图上进行 Cell 级错误检测：

```text
输入：图中已经存在的 (Row, Relation, Value) Cell
输出：dirty_score
标签：clean=0，dirty=1
```

任务不是链接预测。Clean 和 dirty Cell 都是图中已经存在的边；模型判断的是这条 Cell 所表达的值是否错误，而不是这条边是否存在。

## 2. 已冻结的数据与实验协议

### 2.1 数据规模

```text
图节点：2,280,547
Cell observations：13,784,782
有向边：27,569,564
有向 Relation 类型：680（约 340 个字段的正向/反向类型）

监督 Cell：30,000
dirty：20,000
clean：10,000
```

监督划分：

```text
fold 0/1/2：train，13,861
fold 3：validation，7,628
fold 4：internal test，8,511
```

在模型和超参数冻结前，不使用 fold 4，不根据 fold 4 修改模型。

外层 Agent 数据集的 `10 train / 20 validation / 970 test` 与上述 Cell 内部 folds 不是同一层划分。当前图模型开发只发生在 train 图内部；外层 validation/test 暂未进入当前模型选择。

### 2.2 Embedding v1

现有 Qwen3-Embedding-0.6B embedding 冻结，不在当前第一轮实验中重做：

```text
node embedding shape：[2,280,547, 1024]
relation embedding shape：[680, 1024]
storage dtype：float16
identity：197e4e3677ec665a690662790009ab676e926c4e38ca752484da60f2caaf2963
```

`embedding_text` 是 Qwen 的输入文本，不是最终向量。最终 1024 维向量位于 `node_embeddings.f16.npy` 或 `relation_embeddings.f16.npy`。

Strict R-GCN 的输入策略：

```text
Row：不使用 Row Qwen；使用可学习 RowType + Table embedding
Value：使用冻结 Value Qwen，再投影 1024 -> 128
Relation：使用冻结 Relation Qwen，再投影 1024 -> 128
Row node_id：仅用于图拓扑索引，不做 Qwen embedding
```

Row 节点不能直接删除。它负责表达“多个 Cell 属于同一行”的高阶结构。不同 Row 即使初始特征相同，也会因为邻居不同而在消息传播后得到不同表示。

### 2.3 当前 Strict R-GCN 默认结构

```text
hidden dimension：128
R-GCN layers：2
fanouts：16,8
bases：16
batch size：64
dropout：0.2
optimizer：AdamW
learning rate：3e-4
weight decay：1e-4
maximum epochs：40
early-stop patience：7
checkpoint selection：validation Average Precision
threshold selection：best checkpoint 上最大 validation Macro-F1
```

特殊处理：

1. 每个目标 Cell 独立构造局部子图。
2. 邻居采样前，同时精确删除目标 Cell 的正向 edge ID 和反向 edge ID。
3. 其他同 Relation 边和平行边不被误删。
4. 两跳采样从目标向外展开，消息按外层到目标的顺序传播。
5. Relation 使用 16-basis 分解，避免为 680 种 Relation 各训练完整 `128x128` 矩阵。
6. Validation 使用固定采样，训练采样由 seed、epoch 和 observation 决定并可复现。

## 3. 已完成开发实验（seed 666，fold 4 withheld）

| 模型 | Validation AP | AUROC | Macro-F1 | Dirty F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Triple MLP | 0.998467 | 0.997307 | 0.993617 | 0.995983 | 39 | 3 |
| FullRow R-GCN | 0.999224 | 0.998678 | 0.995902 | 0.997415 | 25 | 2 |
| Strict R-GCN | 0.999170 | 0.998617 | 0.996966 | 0.998084 | 19 | 1 |

目前可以说：

1. 两个 R-GCN 在 seed 666 上小幅优于 Triple MLP。
2. 移除完整 Row Qwen 后，Strict R-GCN 没有性能断崖。
3. FullRow 与 Strict 的 AP 差异约 `0.000054`，单 seed 下不能认为存在实质差异。
4. 不能仅凭这些结果宣称图消息有效，因为 `Relation + Value` 可能已经足以识别固定人工注入模式。
5. Validation 中所谓 `original_dirty` 也是最初人工注入日志中的错误，并非自然临床错误；它只有 9 条。`synthetic_dirty` 是扩展人工注入。

## 4. Git 与运行环境状态

真正用于开发和推送的工作树：

```text
/Users/mac/PycharmProjects/v2_clean-graph
branch：codex/graph-typed-value-builder
```

已推送提交：

```text
b14d1c60  feat: add Cell error graph training pipeline
93695a26  feat: add matched no-graph Cell baseline
```

Windows 当前状态：

```text
代码目录：D:\mimic_graph\v2_clean-graph
当前 HEAD：b14d1c6
git pull 失败原因：Recv failure: Connection was reset
旧版本测试：16 passed
```

`16 passed` 是正常的旧版本结果。只有 Windows 成功拉取 `93695a26` 后，才有新增的 Strict MLP 代码和测试。

恢复工作时第一步：

```bat
cd /d D:\mimic_graph\v2_clean-graph
git pull --ff-only origin codex/graph-typed-value-builder
git rev-parse --short HEAD
```

目标 HEAD：

```text
93695a2
```

如果 GitHub 网络仍被重置，先解决网络或稍后重试；不要在 `b14d1c6` 上运行 `--model-type strict_mlp`，该版本尚不支持它。

## 5. 下一步执行顺序

### P0：完成 Strict MLP 匹配无图对照

Strict MLP 与 Strict R-GCN 使用：

```text
相同 RowType + Table
相同 Relation Qwen
相同 Value Qwen
相同 hidden dimension
相同数量的节点 self-update
相同分类头、损失、fold 和选择规则
```

唯一核心区别：Strict MLP 不读取图边、不采样邻居、不聚合关系消息。

Windows 拉取 `93695a2` 后运行（只用 validation）：

```bat
python -m graph.cell_training_cli ^
  --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" ^
  --supervision-dir "D:\mimic_graph\private\graph_cell_supervision_v2" ^
  --embedding-dir "D:\mimic_graph\embeddings\qwen3_0.6b_full_train_v1" ^
  --output-dir "D:\mimic_graph\experiments\strict_mlp_dev_666" ^
  --model-type strict_mlp ^
  --seed 666 ^
  --device cuda
```

不要添加 `--evaluate-internal-test`。

判定：

```text
Strict R-GCN 明显优于 Strict MLP：图消息有证据支持
两者接近：主要可能由 Relation + Value/注入指纹解决
Strict MLP 更好：当前图消息可能无增益或引入噪声
```

主比较指标是 validation AP；Macro-F1、FP/FN 为阈值分类补充指标。

### P1：错误注入质量审计（必须在扩大数据前完成）

当前 20,000 dirty 是 4 个大类各 5,000，但 16 个 Cell-update subtype 严重不均衡。主要问题：

1. 固定脏值指纹：`999`、`-999`、`UNKNOWN_GENDER_CODE`、`990045`、`995158`、`ALIAS_`。
2. `unit_scale_error` 存在 `0 -> 0.0`，数值语义未改变，不应作为当前语义错误标签。
3. 存在 `6 -> -99900.0`，疑似规则叠加或同 Cell 多次注入，需要追踪 lineage。
4. `label_conflict_with_cohort` 只有 1 条；不能用总体 AP 掩盖 subtype 无法评价的问题。
5. 最终训练 mask 主要只保存 clean/dirty/source/fold，没有在训练报告中按 subtype 输出指标。

建议修改：

1. 为每个监督 Cell 持久化 `error_class`、`error_subtype`、注入参数和原值/脏值 lineage。
2. 移除或重新采样数值等价污染，如 `0 -> 0.0`。
3. 阻止同一 Cell 的非预期重复注入；若允许复合错误，必须显式标注为 compound subtype。
4. 增加按 subtype 的 AP、Recall、FPR、混淆统计和预测明细。
5. 增加 hard clean：极端但合法的数值、罕见类别、边界时间和合法缺失。
6. 降低固定哨兵值比例，增加必须依赖同行、跨表或时间上下文才能识别的 hard dirty。

### P2：邻域范围消融（仅在 Strict MLP 结果之后）

不要把 batch size、fanout 和层数混为一谈：

```text
layers：单个目标能看多远
fanout：每一跳看多宽
batch size：一次并行分类多少目标 Cell，不扩大单个 Cell 的视野
```

建议 validation-only、seed 666、一次只改变一个主要变量：

| 实验 | Layers | Fanout | 目的 |
|---|---:|---|---|
| E0 | 2 | `16,8` | 当前基准 |
| E1 | 2 | `32,16` | 扩大每跳宽度 |
| E2 | 2 | `64,32` | 更高覆盖 |
| E3 | 3 | `16,8,4` | 增加图距离 |
| E4 | 3 | `32,16,8` | 增加距离和宽度 |

每次记录：

```text
validation AP / AUROC / Macro-F1
FP/FN
训练时间
GPU 峰值显存
每个目标局部图的节点数/边数分布
```

只有 AP 提升稳定且成本合理时，才保留更大配置。更多层可能产生 over-smoothing；更大 fanout 可能传播噪声并造成显存组合爆炸。

### P3：数据规模决策

当前 30,000 个监督 Cell 足够第一轮管线开发和模型比较，但不足以证明真实临床错误泛化，也不足以可靠评估稀有 subtype。

不建议立即机械扩大到 300,000。先提升有效多样性：

```text
修复无效/复合注入
减少固定脏值
平衡 subtype 的有效坐标覆盖
增加 hard clean 与 hard dirty
按 subject/stay 隔离
保留 subtype 与注入参数
```

完成质量改造后，再决定扩展到约 `64k-100k` 或更大。高质量 100k 通常比重复固定规则的 300k 更有信息价值。

### P4：冻结正式实验

在模型结构、数据规则和超参数全部冻结后：

```text
正式 seeds：666、667、668
每个 seed 独立训练
validation 选 checkpoint 和阈值
然后一次性评估 fold 4 internal test
聚合均值、样本标准差和总混淆矩阵
```

不能查看 fold 4 后再修改模型，否则 internal test 退化为另一个 validation。

### P5：外层归纳评估

当前协议是：

```text
transductive_full_graph_preassigned_folds
```

目标标签彼此隔离，但 train/validation/internal-test Cell 位于同一张完整 train 图。Validation/test 节点的无标签结构可能参与传播。它不是自动等于标签泄露，但不应宣称严格的未见患者归纳泛化。

最终应增加独立图评估：

```text
train subjects 构建训练图
未见 validation subjects 构建独立验证图
未见 test subjects 构建独立测试图
```

外层 `20 validation / 970 test` 应在内部方案冻结后进入该阶段。

## 6. 暂不执行

在前置证据不足时，暂不做：

1. 不重新生成 Qwen Embedding v2。
2. 不直接把监督数据机械扩大到 300,000。
3. 不直接实现 27.6M 边的 full-batch 全图训练。
4. 不把 batch size 调到 30,000 并称为全图传播。
5. 不使用 fold 4 调参。
6. 不依据单 seed 宣称图消息稳定有效。
7. 不把 `dirty_score=0.8` 解释为 80% 临床错误概率。

## 7. 待确认的分歧与决策点

以下事项需要项目负责人明确选择；建议选项列在前面。

### D1：先跑 Strict MLP，还是先扩大 R-GCN？

建议：先跑 Strict MLP。没有匹配无图对照时，扩大 R-GCN 不能回答图消息是否必要。

备选：并行跑邻域消融，但仍不得打开 fold 4。

### D2：错误数据修复是现在做，还是保留 v1 后再做 v2？

建议：冻结现有数据为 `supervision_v1`，保留现有结果用于诊断；另建 `supervision_v2_quality` 修复无效、重复和固定指纹注入。不要覆盖现有数据，以保证可复现。

### D3：16 个 subtype 是否强制均衡？

建议：不做机械数量均衡；设定每类最低有效且多样的样本数，并使用 class-aware sampling/loss。稀有规则若没有足够真实坐标，应扩大患者覆盖或重设计规则，而不是重复同一条。

备选：每个 subtype 固定相同数量，但必须保证不同患者/坐标/数值，不允许简单复制。

### D4：RowType/Table 是否改成 Qwen 初始化？

建议：当前主线保留 learned RowType + learned Table。Qwen Table 作为后续消融，不阻塞主实验。

若测试 Qwen，应使用可靠自然语言 table 描述，而不是只输入 `icu/chartevents` 缩写。

### D5：是否追求“全图传播”？

建议：先做 fanout/layers 敏感性实验。若更大邻域持续带来实际增益，再设计 Cluster-GCN、GraphSAINT 或分块层级传播。

当前 sampler 中“fanout 不截断”只代表完整 L-hop 邻域，不等于传统 full-batch 全图传播；增大 batch size也不等于全图传播。

### D6：主要科学主张是什么？

需要在以下方向中明确主次：

```text
A. 检测当前人工注入 benchmark 的 Cell 错误
B. 证明图上下文相对 Relation+Value 语义的独立增益
C. 泛化到未见患者和真实临床错误
```

建议：近期以 A+B 为可验证目标，C 作为需要新数据、独立图和真实错误标注支持的后续目标。

## 8. 恢复会话后的最短检查清单

```text
[ ] Windows 网络恢复，git pull 到 93695a2
[ ] Windows Cell tests 全部通过
[ ] 运行 strict_mlp_dev_666，保持 internal test withheld
[ ] 读取 Strict MLP validation report
[ ] 比较 Strict R-GCN - Strict MLP
[ ] 决定是否执行 E1-E4 邻域消融
[ ] 决定建立 supervision_v2_quality，不覆盖 v1
[ ] 为监督记录和报告补 error subtype
[ ] 数据/模型冻结后再跑 666/667/668 与 fold 4
[ ] 最后进入外层独立图评估
```

## 9. 关键路径

Windows：

```text
代码：D:\mimic_graph\v2_clean-graph
图：D:\mimic_graph\data\graph_dirty_cell_supervised_v2
监督：D:\mimic_graph\private\graph_cell_supervision_v2
Embedding：D:\mimic_graph\embeddings\qwen3_0.6b_full_train_v1
实验：D:\mimic_graph\experiments
```

Mac 私有审计数据：

```text
/Users/mac/PycharmProjects/v2_clean/datasets/
  mimic_icu_mortality_v3_1_raw_error_fullraw_10train_20val_970test_seed20260804/
```

本文档所在位置：

```text
docs/cell_rgcn_next_steps.md
```
