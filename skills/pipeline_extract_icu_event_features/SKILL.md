---
name: pipeline_extract_icu_event_features
description: Use when Data Cleaning Agent needs MIMIC ICU event, chart, vital, or stay features from icustays, chartevents, d_items, outputevents, procedureevents, or inputevents.
---

# pipeline_extract_icu_event_features

## 用途
包装 teacher pipeline 的 ICU event/chart 特征逻辑，输出 ICU stay count、LOS 和常见 chart item 聚合。

## 适用条件
- reference/features 中有 ICU、vital signs、chartevents、d_items、outputevents、procedureevents、inputevents 相关特征。
- care setting 是 ICU，或 reference 需要 ICU 事件特征。

## 跳过条件
- Non-ICU reference 且没有 ICU event 字段。
- `icu/icustays.csv` 缺失。
- reference 需要图片、notes、ECG，不属于本 skill。

## 需要观察的证据
以下字段是应从 ICU 事件表、d_items 和公开 reference 中确认的概念，不是可直接调用的 Skill 参数：

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "top_n": 20,
  "chunksize": 250000
}
```

## 输出契约
- `icu_features.csv`
- `row_counts`
- `columns`
- `source_files`

## source_pipeline_files
- `preprocessing/hosp_module_preproc/feature_selection_icu.py`
- `utils/icu_preprocess_util.py`
- `utils/outlier_removal.py`
- `utils/uom_conversion.py`

## expected_raw_files
- `icu/icustays.csv`
- `icu/chartevents.csv`
- `icu/d_items.csv`
- `icu/outputevents.csv`
- `icu/procedureevents.csv`
- `icu/inputevents.csv`

## output_contract
CSV 以 `hadm_id` 为 key，包含 ICU event feature columns。

## 失败模式
- cohort 缺少 `hadm_id`。
- chartevents 太大，需要合理 chunksize。
- reference 需要固定 item label 或时间窗时，默认 top-N 可能不够。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 检查 stay_id、itemid、事件时间、单位、值列、时间窗和重复粒度。
- 将 chart/out/proc 规则分别落实到累计 Pipeline 模块，不创建 adapter。
- 对大表先采样诊断，再运行完整 validation Pipeline 和 `ValidateDraft`。
- 无法证明的 itemid/label 映射保留为未解决项，不得从隐藏反馈补值。
- 禁止读取 private reference、test raw 和历史实验。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
