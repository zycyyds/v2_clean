---
name: pipeline_clean_feature_table
description: Use when ReferenceCodeAgent needs to merge cohort and feature CSVs into a cleaned MIMIC feature table before writing custom table assembly code.
---

# pipeline_clean_feature_table

## 用途
包装 teacher pipeline 的特征表合并/基础清洗逻辑，把 cohort 和多个 feature CSV 合并成 `features_wide.csv`。

## 什么时候使用
- 已经有 `cohort.csv` 和 diagnosis/lab/med/ICU 等 feature CSV。
- reference 是宽表或需要最终 `features_wide.csv`/`final_dataset.csv`。

## 什么时候不要直接使用
- reference 是多文件事件明细包，不需要强行合并宽表。
- feature CSV 没有可共享 key，需要先修复 feature 产物。

## 输入
`spec_json` JSON object:

```json
{
  "cohort_path": "/path/to/cohort.csv",
  "feature_paths": ["/path/to/diagnosis_features.csv"],
  "record_grain": "hadm_id"
}
```

## 输出
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

## common_failure_modes
- feature file 缺少 `record_grain`。
- 重名列被去重策略丢弃。
- reference 要求保留多文件明细，不能合并。

## adapter_policy
先直接调用；需要特殊缺失值、异常值、单位转换或列命名策略时创建 adapter。
