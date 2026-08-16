# Pi Agent Train-only Raw Repair Audit

该实验只评估 Pi Agent 从外层 10-stay Train 学习 raw 错误检测、候选生成和修复的能力。它不生成 17 文件 result package，不计算 downstream composite score，不使用 R-GCN，也不使用 Validation；结果必须标记为 `internal_development_test_audit`，不能作为图方法结果或 untouched final test。

## 数据边界

公开 Train view 由宿主从以下输入生成：

- `train/raw_dirty_cell_supervised_v2`：29 张 dirty raw 表；
- `train/clean_raw_reconstructed_v1`：结构对应的 29 张 clean raw 表；
- `host_private/graph_cell_supervision_v2`：20,000 dirty Cell 和 10,000 sampled clean Cell；

公开 view 会稳定伪名化 patient、admission、stay、provider、order 和事件定位值。它不导出 fold、label、error subtype、source error ID、原始行定位或私有路径。MiniMax 只能读取公开 view；Train/Test Gold 只能由宿主评分器读取。

## 1. 构建公开 Train View

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  python -m agent.pi_repair_audit_cli build-view \
  --dirty-raw <dataset>/train/raw_dirty_cell_supervised_v2 \
  --clean-raw <dataset>/train/clean_raw_reconstructed_v1 \
  --graph-dir <dataset>/train/graph_dirty_cell_supervised_v2 \
  --supervision-dir <dataset>/host_private/graph_cell_supervision_v2 \
  --paired-log <dataset>/host_private/graph_cell_supervision_v2/merged_injection_log.csv \
  --output-dir <experiment-root>/public_train
```

命令只有在 29 表、20,000 dirty 和 10,000 clean 全部匹配时才成功。输出为：

```text
public_train/
  public_train_manifest.json
  dirty_raw/
  clean_raw/
  evidence/evidence_manifest.json
  evidence/fields/*.json
```

## 2. Train-only Agent 与冻结

Agent 从空工作区构建确定性的检测和修复 pipeline，不继承格式化 snapshot。四个 correction skills 默认自动加载。独立 Train replay 失败时，宿主最多反馈两次结构或运行问题；所有反馈均来自公开 Train，不读取 Validation。

```bash
set -a
# 按项目现有方式提供 MiniMax M3 配置；不要把密钥写进命令或日志。
set +a

conda run -n py3102 env PYTHONPATH=. MODEL_NAME=MiniMax-M3 \
  python -m agent.pi_repair_audit_cli train \
  --experiment-dir <experiment-root>/train_agent \
  --public-train-root <experiment-root>/public_train \
  --train-gold-log <dataset>/host_private/graph_cell_supervision_v2/merged_injection_log.csv \
  --train-row-gold-log <dataset>/host_private/train_raw_modification_log.csv \
  --max-iters 10000 \
  --max-repair-turns 2
```

冻结 pipeline 的入口固定为：

```text
python pipeline/run.py --raw-root <raw> --output-dir <output> \
  --split-mode train|test
```

每次运行只生成 `corrected_raw/` 和 `repair_candidates.jsonl`。候选报告必须声明所有实际修改，使用零基 CSV data-row index，每个目标返回 1-5 个唯一候选并确定一个 `selected_value`；删除错误插入行使用 `__DELETE_ROW__`。

冻结快照只保留 `pipeline/` 下的文本源码/配置和 `repair_submission.json`。Train replay 输出、候选报告、`.agent_runs`、`__pycache__`、CSV 和模型文件不会进入 Test 运行快照；冻结后会再做一次无模型 Train replay，输出必须与冻结前逐文件一致。

## 3. 一次性 Internal Test

Test Harness 不创建 Agent，不读取模型配置，不继承 API 凭据，也不允许网络。冻结 pipeline 只执行一次，随后宿主读取 474 条 private raw Gold 评分。

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  python -m agent.pi_repair_audit_cli test \
  --train-experiment <experiment-root>/train_agent \
  --test-experiment <experiment-root>/internal_test \
  --test-raw <dataset>/test/raw \
  --test-gold-log <dataset>/host_private/test_raw_modification_log.csv \
```

`score_report.json` 报告：Detection Precision/Recall/F1、Candidate Recall@1/3/5、Exact Repair Precision/Recall/F1、clean preservation 和四类错误分项。同一个 Cell 上的连续注入按值链折叠为一个最终修复目标，同时分别记录原始 Gold 事件数、唯一修复坐标数和重叠事件数。`sanitized_failure_cases.json` 删除行定位、再次伪名化 ID 字段值，并将错误插入行的完整内容替换为不可逆短哈希，用于人工检查。

`full` 子命令可在 Train 成功冻结后立即执行同一套一次性 Test；Test 结果永远不会反馈给 Agent。若查看 Test 后继续修改规则，该结果只能作为开发审计，最终论文需要新的外层 Test。
