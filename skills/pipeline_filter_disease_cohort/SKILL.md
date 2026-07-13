---
name: pipeline_filter_disease_cohort
description: Use when ReferenceCodeAgent needs to filter a MIMIC cohort by disease, ICD prefixes, or admitted-due-to rules before writing custom disease filtering code.
---

# pipeline_filter_disease_cohort

## 用途
包装 teacher pipeline 的疾病 cohort 筛选逻辑。输入已有 cohort 和 ICD/疾病规则，输出筛选后的 cohort。

## 什么时候使用
- reference 或任务中出现 disease cohort、admitted due to、ICD prefix、疾病名称筛选。
- 已有 `cohort.csv`，需要按 `diagnoses_icd` 和 `d_icd_diagnoses` 缩小 cohort。

## 什么时候不要直接使用
- reference 没有疾病筛选。
- 疾病名称无法映射到明确 ICD 候选时，不要硬猜；标记 ambiguous 或创建 adapter。

## 输入
`spec_json` JSON object:

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "disease_rule": {"icd_prefixes": ["K70", "K74"]},
  "primary_only": false
}
```

也可以传 `disease_text`，由 wrapper 查询 ICD 字典候选。

## 输出
- `disease_filtered_cohort.csv`
- `matched_hadm_count`
- `icd_prefixes`

## source_pipeline_files
- `preprocessing/day_intervals_preproc/disease_cohort.py`

## expected_raw_files
- `hosp/diagnoses_icd.csv`
- `hosp/d_icd_diagnoses.csv`

## output_contract
输出保留输入 cohort 的原始字段和记录粒度，只减少行数。

## common_failure_modes
- 没有 ICD prefix。
- 疾病文本对应多个候选，不能自动收敛。
- cohort 缺少 `hadm_id`。

## adapter_policy
先直接调用；需要复杂 ICD 规则、标题匹配或 primary-only 变体时创建 adapter；不得修改原始 skill。
