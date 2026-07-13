---
name: pipeline_extract_med_features
description: Use when ReferenceCodeAgent needs MIMIC medication features from prescriptions or inputevents before writing custom medication scripts.
---

# pipeline_extract_med_features

## 用途
包装 teacher pipeline 的用药特征逻辑，从 prescriptions 生成 medication order/count/top drug flag。

## 什么时候使用
- reference/features 中有 drug、medication、prescriptions、dose、药物类别或具体药名特征。
- 已经有 `cohort.csv`。

## 什么时候不要直接使用
- reference 不包含用药特征。
- raw 中 prescriptions/inputevents 均不可用。
- reference 只需要非结构化文本里的药物实体。

## 输入
`spec_json` JSON object:

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "top_n": 30,
  "chunksize": 250000
}
```

## 输出
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

## common_failure_modes
- prescriptions 缺少 `drug`。
- reference 要求 inputevents 药物逻辑时需要 adapter。
- 药名标准化与 reference 命名不一致。

## adapter_policy
先直接调用；需要 ATC/class 映射、dose 聚合或 inputevents-only 逻辑时创建 adapter。
