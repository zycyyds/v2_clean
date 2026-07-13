# MIMIC ICU Mortality Reference Agent

这个项目只保留一条 MIMIC-IV ICU mortality 数据包复现链路：从公开的 train reference 学习转换规则，在 hidden validation 上迭代 adapter，最后冻结最佳 adapter 并评估测试集。

## 环境

```bash
conda env create -f environment.yml
conda activate py310
```

训练验证阶段需要在 `model_config.local.yaml` 配置模型访问凭据；该文件只在本机保存，不能提交。数据划分、既有结果复评和测试阶段运行已冻结 adapter 时不读取该配置。

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

### 2. 三轮训练验证

使用 [reference_guided_loop_template.md](/Users/mac/PycharmProjects/v2_clean/prompts/reference_guided_loop_template.md) 生成任务提示词后运行：

```bash
python main.py \
  --workflow reference-guided-train-validate \
  --dataset-split datasets/mimic_icu_mortality_summary_labels_split_10train_4000val_5000test_materialized \
  --experiment-dir experiments/mimic_icu_mortality_summary_labels_10train_4000val_template_v4 \
  --round-limit 3 \
  --max-iters 400 \
  "<reference-guided task prompt>"
```

每轮只会在 `experiment-dir` 内创建或修改 adapter/fork；只有 validation composite score 提升时才会晋升 `active_bundle`。连续两轮无提升会冻结最佳 bundle。

### 3. 冻结 adapter 的测试集评估

```bash
python main.py \
  --workflow reference-test-evaluate \
  --dataset-split datasets/mimic_icu_mortality_summary_labels_split_10train_4000val_5000test_materialized \
  --experiment-dir experiments/mimic_icu_mortality_5000test_from_best_0_9603 \
  --adapter-script experiments/mimic_icu_mortality_5000test_from_best_0_9603/build_result_package.py
```

测试结果写入 `test_run/result_package`，评分报告写入 `test_evaluation/` 和 `reference_test_stage_report.json`。当前冻结 adapter 在 5000 例测试集上的 composite score 为 `0.9435`；训练验证最佳分数 `0.9603` 保留在历史实验目录中。

## 保留的 Skills

ReferenceCodeAgent 只注册以下 9 个 MIMIC Pipeline Skills：

- `pipeline_build_cohort`
- `pipeline_filter_disease_cohort`
- `pipeline_extract_diag_features`
- `pipeline_extract_proc_features`
- `pipeline_extract_lab_features`
- `pipeline_extract_med_features`
- `pipeline_extract_icu_event_features`
- `pipeline_clean_feature_table`
- `pipeline_assemble_reference_package`

它们是 Agent 创建 experiment-local adapter 时的可检查基线。最佳 0.9603 adapter 保持为独立脚本，只依赖 Python、pandas 和 numpy。

## 验证

```bash
conda run -n py310 python -m pytest -q
conda run -n py310 python -m compileall -q main.py agent agent_tools workflow skills lib
```
