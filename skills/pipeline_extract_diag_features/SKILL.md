---
name: pipeline_extract_diag_features
description: Use when ReferenceCodeAgent needs MIMIC diagnosis ICD features from diagnoses_icd and d_icd_diagnoses before writing custom diagnosis extraction code.
---

# pipeline_extract_diag_features

## 用途
包装 teacher pipeline 的诊断 ICD 特征逻辑，输出诊断数量、唯一诊断数、全部 ICD、主诊断和 top prefix flag。

## 什么时候使用
- reference/features 中有 diagnosis、ICD、primary diagnosis、long_title、diagnosis_count 等字段。
- 已经有 `cohort.csv`，需要基于 `hadm_id` 过滤 diagnoses。

## 什么时候不要直接使用
- reference 不包含诊断特征。
- `diagnoses_icd.csv` 缺失。
- 需要特定非 top-N ICD 展开且参数不够时，创建 adapter。

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
- `diagnosis_features.csv`
- `row_counts`
- `columns`
- `source_files`

## source_pipeline_files
- `preprocessing/hosp_module_preproc/feature_selection_icu.py`
- `utils/hosp_preprocess_util.py`

## expected_raw_files
- `hosp/diagnoses_icd.csv`
- `hosp/d_icd_diagnoses.csv`

## output_contract
CSV 以 `hadm_id` 为 key，包含 diagnosis feature columns。

## common_failure_modes
- cohort 缺少 `hadm_id`。
- dictionary join keys 缺失。
- reference 要求的诊断列命名与 wrapper 默认命名不同。

## adapter_policy
先直接调用；列名、top-N、ICD root 规则不同则创建 adapter；固定逻辑不合适时 fork。
