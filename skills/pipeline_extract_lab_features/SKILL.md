---
name: pipeline_extract_lab_features
description: Use when ReferenceCodeAgent needs MIMIC lab features from labevents and d_labitems with chunked reading before writing custom lab scripts.
---

# pipeline_extract_lab_features

## 用途
包装 teacher pipeline 的 lab 处理逻辑，分块读取 `labevents`，连接 `d_labitems`，输出常见 lab 项目的 count/mean/min/max。

## 什么时候使用
- reference/features 中有 lab、labevents、检验指标、`d_labitems.label` 或 lab 聚合字段。
- 需要处理较大的 lab 表，避免一次性全量读入。

## 什么时候不要直接使用
- reference 不包含 lab 特征。
- `labevents.csv` 缺失。
- reference 需要固定 itemid 精确字段但没有可传参数时，创建 adapter。

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
- `lab_features.csv`
- `row_counts`
- `columns`
- `source_files`

## source_pipeline_files
- `preprocessing/hosp_module_preproc/feature_selection_icu.py`
- `utils/labs_preprocess_util.py`
- `utils/outlier_removal.py`
- `utils/uom_conversion.py`

## expected_raw_files
- `hosp/labevents.csv`
- `hosp/d_labitems.csv`

## output_contract
CSV 以 `hadm_id` 为 key，包含 lab feature columns。

## common_failure_modes
- cohort 缺少 `hadm_id`。
- labevents 很大，chunksize 过大导致慢或内存高。
- reference 的 lab 字段要求与默认 top-N 聚合不同。

## adapter_policy
先直接调用；需要固定 label/itemid、时间窗、单位转换或异常值策略时创建 adapter。
