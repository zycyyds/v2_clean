# MiniMax M3 字段级 F_corr 修复流程

## 1. 方法边界

本流程采用 GIDCL 的显式修复思路：针对每个 `table.column`，把 Train 中该字段的全部
dirty-clean demonstrations 和 sampled-clean examples 交给 MiniMax M3，让模型生成一个可解释、
确定性的 Python 修复函数：

```python
def Correction(input_string):
    ...
    return corrected_string
```

Strict R-GCN 是唯一错误检测器。`F_corr` 只处理它预测为 dirty 的 Cell，不承担错误检测，
也不读取 Row/Graph context。本实现不包含辅助 `F_det`、`F_gen`、微调修复模型 `M_corr`、
双候选排序、Top-K、score、margin 或独立 Agent workflow。

MiniMax 仅在 Train 规则合成阶段使用。规则冻结后，Validation、Internal Test 和未来外层 Test
只运行本地 Python 代码，不连接模型 API，也不修改规则。

## 2. 冻结数据协议

| Split | dirty-clean pairs | sampled clean | 用途 |
| --- | ---: | ---: | --- |
| folds 0/1/2 Train | 9,560 | 4,301 | 生成、反例迭代和 clean-preservation 审计 |
| fold 3 Validation | 5,210 | 2,418 | 只执行冻结规则，不反馈错例 |
| fold 4 Internal Test | 5,230 | 3,281 | development audit，不再作为 untouched final test |

未来外层 Test 才是论文的一次性最终测试集。`merged_injection_log.csv` 是宿主私有配对和评估
输入；error subtype、injection seed、fold、label、R-GCN score、定位 ID 和私有路径均不得进入
字段证据、MiniMax prompt 或规则源码。

定位列 `subject_id/hadm_id/stay_id/canonical_stay_id/patient_id` 不导出给 MiniMax。这些字段即使
被检测为 dirty，首版也会因为没有冻结规则而标记 `unresolved`。

## 3. 模块与产物

| 命令 | 主要产物 |
| --- | --- |
| `recover-raw` | 从图 Cell observations 恢复的 29 张 dirty CSV、`recovered_raw_manifest.json` |
| `build-pairs` | `field_pairs_manifest.json`、`fields/<field_id>.json` |
| `synthesize-fcorr` | 每字段 evidence、对话、原始响应、验收报告、`correction.py` |
| `freeze-registry` | `rule_registry.json`，通常已由 synthesis 自动生成 |
| `build-targets` | `repair_targets.csv`、`target_manifest.json` |
| `run-rules` | `repair_values.jsonl`、`repair_plan.jsonl` |
| `apply` | 新 raw 副本、`repair_apply_manifest.json` |
| `evaluate` | private `repair_evaluation_report.json` |

所有输出目录必须不存在或为空，避免旧会话、旧规则与本轮产物混合。

## 4. Windows 执行顺序

以下命令在 `D:\mimic_graph\v2_clean-graph` 的 `graph-embedding` 环境运行。`--raw-dir`
当前 30,000 个监督 Cell 的证据构建直接使用图中保存的原始字符串。完整 raw 只在最终 Apply
阶段需要；若要启用额外 raw/graph 交叉检查，可传入原 dirty raw 或由 4.1 恢复的逻辑副本。

### 4.1 原 dirty raw 缺失时恢复逻辑副本

本节可推迟到最终 Apply 之前。若构图时的 `raw_dirty_cell_supervised_v2` 已丢失，先检查 D 盘
可用空间：

```bat
fsutil volume diskfree D:
```

确认空间足够后，从 `cell_observations.jsonl` 恢复：

```bat
python -m graph.cell_repair_cli recover-raw ^
  --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" ^
  --output-dir "D:\mimic_graph\data\raw_dirty_cell_supervised_v2"
```

构图阶段为每个 CSV Cell（包括空值）顺序保存了 `table/row_number/column/raw_value`，因此宿主
可以流式恢复表头、行和字段值。恢复过程不读取 clean 数据或 injection log，也不尝试从当前只
有 10 张表的 `D:\database\mimic` 拼接缺失表。恢复副本在逻辑内容上等同于构图时的 dirty
raw，但不承诺与原文件字节级相同，例如 CSV quoting 和换行可能不同。

宿主逐表检查列顺序、行号连续性、manifest 表集合与行数，并记录每张表 SHA256。任一检查失败
都会删除不完整输出并失败关闭。成功后应确认 `recovered_raw_manifest.json` 中
`status=SUCCESS`、`table_count=29`。

### 4.2 构建字段配对证据

```bat
python -m graph.cell_repair_cli build-pairs ^
  --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" ^
  --supervision-dir "D:\mimic_graph\private\graph_cell_supervision_v2" ^
  --paired-log "D:\mimic_graph\private\graph_cell_supervision_v2\merged_injection_log.csv" ^
  --output-dir "D:\mimic_graph\experiments\fcorr_v1\field_pairs"
```

宿主会先审计全部 30,000 个监督 Cell。20,000 个 dirty 必须逐个唯一映射到 injection record，
`cell_observations.raw_value` 必须等于 `dirty_value`；split 数量必须严格等于上表，任何不一致
都会失败关闭。若完整 raw 可用，可额外传入 `--raw-dir` 启用 raw/graph 当前值交叉检查。

每个 LLM 可见证据文件只有：

```json
{
  "table": "icu/chartevents",
  "column": "valuenum",
  "dirty_clean_pairs": [{"dirty": "123.0", "clean": "1.23"}],
  "clean_examples": ["1.23", "7.38"]
}
```

列表不抽样、不去重，并按监督 observation 的确定性顺序保存。

### 4.3 每字段合成并冻结 F_corr

```bat
set MODEL_NAME=MiniMax-M3
python -m graph.cell_repair_cli synthesize-fcorr ^
  --evidence-dir "D:\mimic_graph\experiments\fcorr_v1\field_pairs" ^
  --output-dir "D:\mimic_graph\experiments\fcorr_v1\rules" ^
  --agent-key react_planner ^
  --max-attempts 12
```

不同字段使用完全隔离的 Chat 历史。同一字段第一轮读取全部 Train 证据；宿主执行 AST 安全
检查和全部 Train pairs 的 exact-match 回归。只有 `Accuracy > 0.85` 才接受。clean preservation
单独报告，但不作为首版 GIDCL 接受阈值。

若未通过，下一轮消息包含上一版完整函数和全部 Train 错修三元组
`dirty/expected_clean/actual_output`。最多 12 次总尝试，达到阈值立即停止；仍失败则字段为
`FCORR_REJECTED`。API 明确报告上下文过长时字段为 `CONTEXT_TOO_LARGE`，首版不会自行抽样。

实际请求强制 `temperature=0.0`、`seed=666`。解析模型名不是 `MiniMax-M3` 时失败关闭。
允许 `re.match/search/sub/fullmatch/compile`、简单字符串操作、条件分支、一个嵌套 `is_dirty`
helper，以及有 Train 证据支持的固定映射；禁止 import、文件/网络、随机、动态执行、定位 ID、
private/Test reference 和字段间访问。

GIDCL 论文说明 correction 规则采用与 detection 规则类似的错误反例再生成机制，但官方仓库未
公开 correction 第二轮的逐字 prompt。本实现据此发送完整 Train 错修反例，并在审计产物中保存
每轮完整对话、原始响应、请求/响应 SHA256、模型名和非敏感生成配置。

可用 `--field icu/chartevents.valuenum` 限制本次只生成某个字段；该参数可重复。

### 4.4 构建 Strict R-GCN repair targets

```bat
python -m graph.cell_repair_cli build-targets ^
  --predictions "D:\mimic_graph\experiments\strict_rgcn_formal_666\predictions.csv" ^
  --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" ^
  --raw-dir "D:\mimic_graph\data\raw_dirty_cell_supervised_v2" ^
  --split validation ^
  --output-dir "D:\mimic_graph\experiments\fcorr_v1\validation_targets"
```

目标构建只选择指定 split 中 `prediction=1` 的 Cell，不读取 `label/source/fold` 来决定目标。

### 4.5 无 API 离线执行规则

```bat
set OPENAI_API_KEY=
set OPENAI_API_BASE=
python -m graph.cell_repair_cli run-rules ^
  --targets "D:\mimic_graph\experiments\fcorr_v1\validation_targets\repair_targets.csv" ^
  --rule-dir "D:\mimic_graph\experiments\fcorr_v1\rules" ^
  --output-dir "D:\mimic_graph\experiments\fcorr_v1\validation_repairs"
```

每个 target 只调用所属字段的一个冻结 `Correction`。返回字符串不同于原值时 action 为
`replace`；无规则、字段被拒绝、运行异常、非字符串或返回原值时均为 `unresolved`。不存在
在线 LLM 回退，也不存在第二候选。

加载时会重新校验 synthesis manifest、字段 manifest、证据副本和源码哈希。相同 targets 与
相同冻结规则必须生成字节级相同的两个 JSONL。

### 4.6 应用到新的 raw 副本

```bat
python -m graph.cell_repair_cli apply ^
  --raw-dir "D:\mimic_graph\data\raw_dirty_cell_supervised_v2" ^
  --repair-plan "D:\mimic_graph\experiments\fcorr_v1\validation_repairs\repair_plan.jsonl" ^
  --output-dir "D:\mimic_graph\data\raw_repaired_fcorr_validation_v1"
```

输入目录永远不修改。宿主复制完整目录、检查每个 replacement 的当前值前置条件，仅修改 plan
声明的 Cell，并保持 CSV/CSV.GZ schema、行数、无关字段和无关文件。

### 4.7 Private 隔离评估

```bat
python -m graph.cell_repair_cli evaluate ^
  --repair-values "D:\mimic_graph\experiments\fcorr_v1\validation_repairs\repair_values.jsonl" ^
  --repair-plan "D:\mimic_graph\experiments\fcorr_v1\validation_repairs\repair_plan.jsonl" ^
  --injection-log "D:\mimic_graph\private\graph_cell_supervision_v2\merged_injection_log.csv" ^
  --predictions "D:\mimic_graph\experiments\strict_rgcn_formal_666\predictions.csv" ^
  --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" ^
  --output-dir "D:\mimic_graph\experiments\fcorr_v1\validation_private_evaluation"
```

报告包括 detector TP/FP/FN/TN、rule-only exact accuracy、候选覆盖、clean preservation、精确/
错误/未解决修复数，以及检测修复联合 Precision/Recall/F1。联合 Recall 的分母是该 split 的
全部真实 dirty Cell，包括检测器漏掉、从未进入 repair targets 的 Cell。

Validation 报告不得将单个错例反馈给当前 MiniMax 会话。Internal Test 使用相同命令但将
`--split internal_test`，其结果只能记为 `development_audit`，也不得据此修改当前规则。

## 5. 验证

```bat
python -m pytest -q tests\test_cell_repair.py
python -m pytest -q tests\test_cell_training.py tests\test_graph_builder.py
python -m compileall -q graph lib
```

`test_cell_repair.py` 使用伪 MiniMax completion，不访问真实 API。
