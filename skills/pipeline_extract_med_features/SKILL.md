---
name: pipeline_extract_med_features
description: Use when Data Cleaning Agent needs MIMIC medication features from prescriptions or inputevents before writing custom medication scripts.
---

# pipeline_extract_med_features

## 用途
包装 teacher pipeline 的用药特征逻辑，从 prescriptions 生成 medication order/count/top drug flag。

## 适用条件
- reference/features 中有 drug、medication、prescriptions、dose、药物类别或具体药名特征。
- 已经有 `cohort.csv`。

## 跳过条件
- reference 不包含用药特征。
- raw 中 prescriptions/inputevents 均不可用。
- reference 只需要非结构化文本里的药物实体。

## 需要观察的证据
以下字段是应从 prescriptions/inputevents、字典和公开 reference 中确认的概念，不是可直接调用的 Skill 参数：

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "top_n": 30,
  "chunksize": 250000
}
```

## 输出契约
- `medication_features.csv`
- `row_counts`
- `columns`
- `source_files`

## source_pipeline_files
- `preprocessing/hosp_module_preproc/feature_selection_icu.py`
- `utils/hosp_preprocess_util.py`

## expected_raw_files
- `hosp/prescriptions.csv`
- `icu/inputevents.csv`

## output_contract
CSV 以 `hadm_id` 为 key，包含 medication feature columns。

## 失败模式
- prescriptions 缺少 `drug`。
- reference 要求 inputevents 药物逻辑时需要 adapter。
- 药名标准化与 reference 命名不一致。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 检查药物来源、itemid/药名、dose 单位、开始结束时间和 stay 映射。
- 将规则写入 `workspace/pipeline/med.py`，再运行 train 诊断和 validation Pipeline。
- 候选必须通过 `ValidateDraft`；药物映射不唯一时写入 rule ledger。
- 不得猜测 ATC/class、填充患者常量或创建另一套业务入口。
- 禁止读取 private reference、test raw 和历史实验。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
