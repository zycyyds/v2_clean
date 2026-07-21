---
name: pipeline_extract_proc_features
description: Use when Data Cleaning Agent needs MIMIC procedure or operation features from procedures_icd, d_icd_procedures, or ICU procedure events.
---

# pipeline_extract_proc_features

## 用途
包装 teacher pipeline 的手术/操作特征逻辑，输出 procedure count、unique count、codes、titles 和 top code flags。

## 什么时候使用
- reference/features 中有 procedures、procedure_count、procedure_titles、operation/procedure code 字段。
- 已经有 `cohort.csv`。

## 什么时候不要直接使用
- reference 不包含手术/操作特征。
- raw 中没有 `procedures_icd.csv` 且没有可替代 procedure event 来源。

## 输入
`spec_json` JSON object:

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "top_n": 20
}
```

## 输出
- `procedure_features.csv`
- `row_counts`
- `columns`
- `source_files`

## source_pipeline_files
- `preprocessing/hosp_module_preproc/feature_selection_icu.py`
- `utils/icu_preprocess_util.py`
- `utils/hosp_preprocess_util.py`

## expected_raw_files
- `hosp/procedures_icd.csv`
- `hosp/d_icd_procedures.csv`
- `icu/procedureevents.csv`

## output_contract
CSV 以 `hadm_id` 为 key，包含 procedure feature columns。

## common_failure_modes
- `procedures_icd.csv` 缺失时只会生成空特征框架。
- reference 需要 ICU procedureevents 细粒度聚合时可能需要 adapter。

## adapter_policy
先直接调用；需要 ICU event 时间窗或更细粒度 operation 聚合时创建 adapter/fork。
