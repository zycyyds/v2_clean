# MIMIC ICU Mortality Data Cleaning Agent

这个项目只保留一条 MIMIC-IV ICU mortality 数据包复现链路：从公开的 train reference 学习转换规则，在 hidden validation 上迭代并沉淀脚本，停止后自动用全新上下文的同类 Agent 处理 test 并私有评分。

## 环境

```bash
conda env create -f environment.yml
conda activate py3102
```

Validation Agent 和自动 Test Agent 需要在 `model_config.local.yaml` 配置模型访问凭据；该文件只在本机保存，不能提交。数据划分、评分器和手工兼容测试入口本身不调用模型。

## 主链路

### 1. 生成物理划分

```bash
python main.py \
  --workflow prepare-reference-splits \
  --raw-root /absolute/path/to/mimiciv/3.1 \
  --reference-root /absolute/path/to/reference-package \
  --split-output datasets/mimic_icu_mortality_summary_labels_split_10train_4000val_5000test_materialized \
  --split-counts 10,4000,5000 \
  --raw-mode copy \
  --decompress-gzip
```

输出目录必须包含 `train/`、`validation/` 和 `test/`。Agent 只能读取 `train/raw`、`train/reference`、`validation/raw` 和 validation keys；validation/test 的 reference 都由宿主评分器私有读取。

### 2. 训练验证与自动 Test

使用 [reference_guided_loop_template.md](prompts/reference_guided_loop_template.md) 生成任务提示词后运行：

```bash
python main.py \
  --workflow reference-guided-train-validate \
  --dataset-split /absolute/path/to/10train_4000val_5000test \
  --experiment-dir experiments/mimic_icu_mortality_agentscope2_v1 \
  --round-limit 2 \
  --patience 2 \
  --max-attempts 30 \
  --max-iters 10000 \
  "<reference-guided task prompt>"
```

`--round-limit` 只统计分数严格提升的正式 loop。`--max-iters` 是整个持续 validation Agent 生命周期的 ReAct 总上限，并不要求必须执行满。下降、持平和门禁失败只记录为 attempt；默认连续两个有效但未提升的 attempt 才停止，无效 attempt 不占 patience。每个正式 loop 会归档累计 `script_bundle`、validation 结果、评分和 provenance。

Validation 因 `round-limit`、patience、`max-attempts`、`target-score` 自然停止后，会自动从最新正式 best 创建 checkpoint，并在同一 Python 进程中用全新 memory/toolkit/workspace 的 `Data Cleaning Agent` 处理 test。Validation 阶段第一次按 `Ctrl+C` 也会取消当前未提交 attempt 并进入该流程；Test 阶段再次按 `Ctrl+C` 才终止整个程序。

自动输出位于：

```text
test_checkpoints/checkpoint_XXXX_best_round_XXXX/
  frozen_script_bundle/
  agent_runs/data_cleaning_agent/
  test_run/result_package/
  test_evaluation/
  checkpoint_report.json
```

新评分器对 validation 和 test 使用同一套 schema-v2 规则：`20% core + 60% feature + 20% summary`。每个文件内部为 `10% Schema F1 + 20% Key/行结构 F1 + 40% 对齐后逐行单元格 F1 + 30% 完整行 F1`；文件和列使用固定等权，不受 chart 行数影响。

### 3. 手工兼容测试入口

正常主链路不需要 `--adapter-script`。仅在手工复评一个已有独立 adapter 时使用：

```bash
python main.py \
  --workflow reference-test-evaluate \
  --dataset-split datasets/mimic_icu_mortality_summary_labels_split_10train_4000val_5000test_materialized \
  --experiment-dir experiments/mimic_icu_mortality_5000test_from_best_0_9603 \
  --adapter-script experiments/mimic_icu_mortality_5000test_from_best_0_9603/build_result_package.py
```

测试结果写入 `test_run/result_package`，评分报告写入 `test_evaluation/` 和 `reference_test_stage_report.json`。旧报告保留原评分含义；新实验只使用 schema-v2 评分器，不重写历史分数。

## 保留的 Skills

Data Cleaning Agent 只注册以下 9 个 MIMIC Pipeline Skills：

- `pipeline_build_cohort`
- `pipeline_filter_disease_cohort`
- `pipeline_extract_diag_features`
- `pipeline_extract_proc_features`
- `pipeline_extract_lab_features`
- `pipeline_extract_med_features`
- `pipeline_extract_icu_event_features`
- `pipeline_clean_feature_table`
- `pipeline_assemble_reference_package`

它们只提供按需读取的业务说明，不能直接执行。实验规则只沉淀到当前实验的 `rule_ledger.json` 和累计 Pipeline。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n py3102 python -m pytest -p no:cacheprovider -q tests
```
