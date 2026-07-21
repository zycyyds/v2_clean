---
name: pipeline_extract_icu_event_features
description: Use when Data Cleaning Agent needs MIMIC ICU event, chart, vital, or stay features from icustays, chartevents, d_items, outputevents, procedureevents, or inputevents.
---

# pipeline_extract_icu_event_features

## 用途
包装 teacher pipeline 的 ICU event/chart 特征逻辑，输出 ICU stay count、LOS 和常见 chart item 聚合。

## 什么时候使用
- reference/features 中有 ICU、vital signs、chartevents、d_items、outputevents、procedureevents、inputevents 相关特征。
- care setting 是 ICU，或 reference 需要 ICU 事件特征。

## 什么时候不要直接使用
- Non-ICU reference 且没有 ICU event 字段。
- `icu/icustays.csv` 缺失。
- reference 需要图片、notes、ECG，不属于本 skill。

## 输入
`spec_json` JSON object:

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "top_n": 20,
  "chunksize": 250000
}
```

## 输出
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

## common_failure_modes
- cohort 缺少 `hadm_id`。
- chartevents 太大，需要合理 chunksize。
- reference 需要固定 item label 或时间窗时，默认 top-N 可能不够。

## adapter_policy
先直接调用；需要固定 itemid/label、事件时间窗、first/last 聚合时创建 adapter。
