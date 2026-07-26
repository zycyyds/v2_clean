---
name: pipeline_clean_feature_table
description: Use when Data Cleaning Agent needs to merge cohort and feature CSVs into a cleaned MIMIC feature table before writing custom table assembly code.
---

# pipeline_clean_feature_table

## 用途
包装 teacher pipeline 的特征表合并/基础清洗逻辑，把 cohort 和多个 feature CSV 合并成 `features_wide.csv`。

## 适用条件
- 已经有 `cohort.csv` 和 diagnosis/lab/med/ICU 等 feature CSV。
- reference 是宽表或需要最终 `features_wide.csv`/`final_dataset.csv`。

## 跳过条件
- reference 是多文件事件明细包，不需要强行合并宽表。
- feature CSV 没有可共享 key，需要先修复 feature 产物。

## 需要观察的证据
以下字段是应从 cohort、feature 文件和公开 reference 中确认的概念，不是可直接调用的 Skill 参数：

```json
{
  "cohort_path": "/path/to/cohort.csv",
  "feature_paths": ["/path/to/diagnosis_features.csv"],
  "record_grain": "hadm_id"
}
```

## 输出契约
- `features_wide.csv`
- `row_counts`
- `columns`
- merged feature path list

## source_pipeline_files
- `utils/outlier_removal.py`
- `utils/uom_conversion.py`

## expected_raw_files
无；本 skill 处理已生成产物。

## output_contract
CSV 以 `record_grain` 为 key，保留 cohort columns 并合并 feature columns。

## 失败模式
- feature file 缺少 `record_grain`。
- 重名列被去重策略丢弃。
- reference 要求保留多文件明细，不能合并。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 检查 join key、行粒度、列顺序、缺失 sentinel、数值精度和排序。
- 将清洗与装配规则写入对应 Pipeline 模块和 `config.yaml`。
- 用 `RunPipeline` 生成完整包，禁止运行后手改 CSV；最后调用 `ValidateDraft`。
- 不确定的归一化规则记录到 rule ledger，不得为了匹配样本硬编码患者值。
- 禁止读取 private reference、test raw 和历史实验。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
