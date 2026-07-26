---
name: pipeline_build_cohort
description: Use when Data Cleaning Agent needs to build a MIMIC-IV structured cohort from train/raw or validation/raw before writing custom cohort scripts.
---

# pipeline_build_cohort

## 用途
包装 teacher pipeline 的 visit/cohort/outcome 构建逻辑，生成 ICU 或 Non-ICU cohort。它只写当前 step 的输出目录，不写 teacher pipeline 项目。

## 适用条件
- 需要从 `hosp/patients.csv`、`hosp/admissions.csv`、`icu/icustays.csv` 构建 cohort。
- reference 包里有 cohort/label/visit 粒度文件，需要在 validation raw 上复现。
- 写自定义 cohort 大脚本前，必须先尝试这个 skill。

## 跳过条件
- 输入不是 MIMIC-IV 3.1 结构化 raw。
- reference 不是 visit/cohort 粒度。
- 需要复杂疾病筛选时，先用本 skill 构建基础 cohort，再用 `pipeline_filter_disease_cohort`。

## 需要观察的证据
以下字段是应从任务契约、raw 和公开 reference 中确认的概念，不是可直接调用的 Skill 参数：

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

## 输出契约
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

## 失败模式
- raw 目录缺少 admissions/patients/icustays。
- `case_keys_path` 与 cohort 没有共享 key。
- reference 的粒度不是 `hadm_id` 或 `stay_id`。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 用 `InspectDataFile` 确认 patients、admissions、icustays 的 key、时间和粒度。
- 用 `Write/Edit` 将规则落实到 `workspace/pipeline/cohort.py`，再调用 `RunPipeline`。
- train 结果仅作公开诊断；validation 候选必须由完整 Pipeline 生成并通过 `ValidateDraft`。
- 缺少关键来源或粒度无法证明时停止推导并写入 `rule_ledger.json`，不得猜测患者级值。
- 禁止读取 private reference、test raw、历史实验或把患者行复制进 Pipeline。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
