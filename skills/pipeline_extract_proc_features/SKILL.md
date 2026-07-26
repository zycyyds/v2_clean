---
name: pipeline_extract_proc_features
description: Use when Data Cleaning Agent needs MIMIC procedure or operation features from procedures_icd, d_icd_procedures, or ICU procedure events.
---

# pipeline_extract_proc_features

## 用途
包装 teacher pipeline 的手术/操作特征逻辑，输出 procedure count、unique count、codes、titles 和 top code flags。

## 适用条件
- reference/features 中有 procedures、procedure_count、procedure_titles、operation/procedure code 字段。
- 已经有 `cohort.csv`。

## 跳过条件
- reference 不包含手术/操作特征。
- raw 中没有 `procedures_icd.csv` 且没有可替代 procedure event 来源。

## 需要观察的证据
以下字段是应从 procedure 来源表、字典和公开 reference 中确认的概念，不是可直接调用的 Skill 参数：

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "top_n": 20
}
```

## 输出契约
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

## 失败模式
- `procedures_icd.csv` 缺失时只会生成空特征框架。
- reference 需要 ICU procedureevents 细粒度聚合时可能需要 adapter。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 检查 procedures_icd/procedureevents 的粒度、时间、ICD 版本和业务 key。
- 将确定性规则写入 `workspace/pipeline/proc.py`，再用 `RunPipeline` 回放。
- 公开 train 只作证据；validation 包通过 `ValidateDraft` 后才能提交。
- 来源粒度不支持目标字段时记录限制，不得以患者常量补齐。
- 禁止读取 private reference、test raw 和历史实验。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
