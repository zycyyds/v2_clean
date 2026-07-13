---
name: pipeline_build_cohort
description: Use when ReferenceCodeAgent needs to build a MIMIC-IV structured cohort from train/raw or validation/raw before writing custom cohort scripts.
---

# pipeline_build_cohort

## 用途
包装 teacher pipeline 的 visit/cohort/outcome 构建逻辑，生成 ICU 或 Non-ICU cohort。它只写当前 step 的输出目录，不写 teacher pipeline 项目。

## 什么时候使用
- 需要从 `hosp/patients.csv`、`hosp/admissions.csv`、`icu/icustays.csv` 构建 cohort。
- reference 包里有 cohort/label/visit 粒度文件，需要在 validation raw 上复现。
- 写自定义 cohort 大脚本前，必须先尝试这个 skill。

## 什么时候不要直接使用
- 输入不是 MIMIC-IV 3.1 结构化 raw。
- reference 不是 visit/cohort 粒度。
- 需要复杂疾病筛选时，先用本 skill 构建基础 cohort，再用 `pipeline_filter_disease_cohort`。

## 输入
`spec_json` JSON object:

```json
{
  "mimic_root": "/path/to/raw",
  "task_spec": {
    "care_setting": "ICU",
    "outcome_type": "Mortality",
    "record_grain": "stay_id"
  },
  "case_keys_path": "/path/to/keys.csv",
  "include_label": true
}
```

## 输出
- `cohort.csv`
- optional `visit_base.csv`
- `row_counts`
- `columns`
- `source_files`

## source_pipeline_files
- `preprocessing/day_intervals_preproc/day_intervals_cohort_v3.py`
- `mimic4_preprocess_util.py`

## expected_raw_files
- `hosp/patients.csv`
- `hosp/admissions.csv`
- `icu/icustays.csv`

## output_contract
`cohort.csv` 必须包含 `stay_id`、`hadm_id` 或 `subject_id` 中至少一个可用于 split/evaluation 的 key。

## common_failure_modes
- raw 目录缺少 admissions/patients/icustays。
- `case_keys_path` 与 cohort 没有共享 key。
- reference 的粒度不是 `hadm_id` 或 `stay_id`。

## adapter_policy
先直接调用；参数不能覆盖 reference 逻辑时创建 experiment 内 adapter；只有内部逻辑无法包装时 fork。
