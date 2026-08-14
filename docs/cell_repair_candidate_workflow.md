# MiniMax M3 多候选 `F_corr` 流程

## 1. 当前方法边界

Strict R-GCN 是唯一错误检测器。MiniMax M3 只在 Train folds 0/1/2 上读取字段级
dirty-clean demonstrations、sampled-clean examples 和目标字段屏蔽后的行上下文，生成一个
可解释、确定性的 Python 候选函数：

```python
def GenerateCandidates(input_string, row_context):
    ...
    return [
        {"value": "...", "rule_id": "...", "evidence": "..."},
    ]
```

每个目标最多返回 5 个按可信顺序排列且 value 去重的候选。当前模块只生成候选，不执行候选
选择，不生成 replacement，也不 Apply。R-GCN clean confidence、ComplEx/MLP 排序和最终修复
属于后续模块。

MiniMax 仅用于 Train 规则合成与 Train 反例迭代。规则冻结后，Validation 和 Internal Test
使用完全相同的 `frozen_audit` 流程，不连接 MiniMax、不调整配置、不把任何错例反馈给规则
生成器。两者只是 split 名称、样本和输出目录不同。

## 2. 数据与隔离协议

| Split | dirty-clean | sampled clean | 用途 |
| --- | ---: | ---: | --- |
| folds 0/1/2 Train | 9,560 | 4,301 | 生成规则、Recall@K 验收和最多 12 轮反馈 |
| fold 3 Validation | 5,210 | 2,418 | 运行冻结候选规则并做 private frozen audit |
| fold 4 Internal Test | 5,230 | 3,281 | 与 Validation 完全相同的 frozen audit |

定位字段 `subject_id/hadm_id/stay_id/canonical_stay_id/patient_id` 不导出给 MiniMax。每条 Train
证据的 `row_context` 也删除目标字段和上述定位字段。Validation/Internal Test reference、error
subtype、injection seed、fold、label、R-GCN score、row/observation ID 和 private 路径不能进入
Train 证据、MiniMax prompt、规则源码或冻结 registry。

字段证据结构为：

```json
{
  "table": "icu/chartevents",
  "column": "valuenum",
  "dirty_clean_pairs": [
    {
      "dirty": "123.0",
      "clean": "1.23",
      "row_context": {"valueuom": "mg/dL"}
    }
  ],
  "clean_examples": [
    {
      "value": "7.38",
      "row_context": {"valueuom": "pH"}
    }
  ]
}
```

证据直接从 `cell_observations.jsonl` 恢复当前值和行上下文，不要求 29 张原始 raw 表。可选
`--raw-dir` 只用于额外 raw/graph 交叉检查；完整 raw 仍只在未来 Apply 阶段需要。

## 3. Train 验收

宿主对每个 Train dirty-clean pair 执行两次候选函数，强制检查：

- AST 安全、唯一顶层 `GenerateCandidates` 和固定两参数签名；
- 输出必须是 0 至 5 个结构化候选，value 不重复；
- 相同输入和上下文的输出必须完全一致；
- 禁止 import、文件/网络、随机、动态执行、定位 ID 和 private/Test reference；
- 计算 Candidate Recall@1/3/5、MRR、候选数量和 clean preservation。

字段接受条件为严格 `Train Candidate Recall@5 > 0.85`。这是对 GIDCL 单值 `F_corr` 的多候选
扩展；GIDCL 原文的 85% 阈值保留，但 Recall@5 是本实现新增指标，不应表述为论文原指标。

Gold 未进入 Top-5 时，下一轮只反馈 Train 反例：

```json
{
  "dirty": "X",
  "row_context": {"valueuom": "mg/dL"},
  "expected_clean": "225664",
  "generated_candidates": ["220045", "223761"],
  "failure": "expected_clean_missing_from_top_5"
}
```

每个字段最多 12 次总生成尝试。通过即冻结；未通过为 `FCORR_REJECTED`；API 明确报告上下文
超限时为 `CONTEXT_TOO_LARGE`。所有会话、原始响应、源码、配置和哈希均保存。

## 4. 产物

| 命令 | 主要产物 |
| --- | --- |
| `build-pairs` | `field_pairs_manifest.json`、`fields/*.json` |
| `synthesize-fcorr` | 每字段会话、响应、验收报告、`correction.py`、`rule_registry.json` |
| `build-targets` | `repair_targets.csv`，含 target-masked `row_context_json` |
| `run-rules` | `candidate_values.jsonl`、`candidate_execution_manifest.json` |
| `evaluate-candidates` | private `candidate_evaluation_report.json` |

`candidate_values.jsonl` 中每个 detector-positive Cell 有一个记录；无冻结规则、规则无候选或
运行异常时 candidates 为空并记录 reason。该文件不包含 Gold、label、subtype 或 injection log。

## 5. Windows 单行命令

以下命令在 `D:\mimic_graph\v2_clean-graph` 的 `graph-embedding` 环境中执行。

### 5.1 构建 Train 证据

```bat
python -m graph.cell_repair_cli build-pairs --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" --supervision-dir "D:\mimic_graph\private\graph_cell_supervision_v2" --paired-log "D:\mimic_graph\private\graph_cell_supervision_v2\merged_injection_log.csv" --output-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\field_pairs"
```

### 5.2 MiniMax M3 合成多候选函数

先用一个小字段 smoke test：

```bat
set MODEL_NAME=MiniMax-M3 && python -m graph.cell_repair_cli synthesize-fcorr --evidence-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\field_pairs" --output-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\rules_smoke" --agent-key react_planner --max-attempts 12 --field hosp/patients.gender
```

全字段运行时使用新的空输出目录：

```bat
set MODEL_NAME=MiniMax-M3 && python -m graph.cell_repair_cli synthesize-fcorr --evidence-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\field_pairs" --output-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\rules" --agent-key react_planner --max-attempts 12
```

### 5.3 Validation 冻结候选审计

```bat
python -m graph.cell_repair_cli build-targets --predictions "D:\mimic_graph\experiments\strict_rgcn_formal_666\predictions.csv" --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" --split validation --output-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\validation_targets"
```

```bat
python -m graph.cell_repair_cli run-rules --targets "D:\mimic_graph\experiments\fcorr_candidates_v1\validation_targets\repair_targets.csv" --rule-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\rules" --output-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\validation_candidates"
```

```bat
python -m graph.cell_repair_cli evaluate-candidates --candidate-values "D:\mimic_graph\experiments\fcorr_candidates_v1\validation_candidates\candidate_values.jsonl" --injection-log "D:\mimic_graph\private\graph_cell_supervision_v2\merged_injection_log.csv" --predictions "D:\mimic_graph\experiments\strict_rgcn_formal_666\predictions.csv" --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" --output-dir "D:\mimic_graph\experiments\fcorr_candidates_v1\validation_private_audit"
```

### 5.4 Internal Test 冻结候选审计

使用完全相同的三条命令，仅将 `--split validation` 改为 `--split internal_test`，并将输出目录
中的 `validation` 改为 `internal_test`。两种报告的 `evaluation_role` 均为 `frozen_audit`。

## 6. 指标解释

`rule_candidate_recall_at_k` 的分母是 R-GCN 已正确检出的 dirty Cell；它衡量候选规则本身。
`joint_candidate_recall_at_k` 的分母是 split 中全部真实 dirty Cell，检测漏检按无候选处理；它
衡量检测加候选生成链路的可修复上限。

例如共有 100 个真实 dirty Cell，R-GCN 检出 90 个，其中 81 个 Gold 位于 Top-5：

```text
rule_candidate_recall_at_5 = 81 / 90 = 0.90
joint_candidate_recall_at_5 = 81 / 100 = 0.81
```

当前没有候选排序器，因此报告明确记录 `selection_performed=false`，不能把 Candidate Recall@K
写成最终修复 Accuracy，也不能从该文件直接生成 repair plan。

## 7. 验证

```bat
python -m pytest -q tests\test_cell_repair.py
```

```bat
python -m pytest -q tests\test_cell_training.py tests\test_graph_builder.py
```

```bat
python -m compileall -q graph lib
```

测试中的 MiniMax completion 为伪响应，不访问真实 API。
