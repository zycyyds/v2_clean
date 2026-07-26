---
name: pipeline_extract_diag_features
description: Use when Data Cleaning Agent needs MIMIC diagnosis ICD features from diagnoses_icd and d_icd_diagnoses before writing custom diagnosis extraction code.
---

# pipeline_extract_diag_features

## 用途
包装 teacher pipeline 的诊断 ICD 特征逻辑，输出诊断数量、唯一诊断数、全部 ICD、主诊断和 top prefix flag。

## 适用条件
- reference/features 中有 diagnosis、ICD、primary diagnosis、long_title、diagnosis_count 等字段。
- 已经有 `cohort.csv`，需要基于 `hadm_id` 过滤 diagnoses。

## 跳过条件
- reference 不包含诊断特征。
- `diagnoses_icd.csv` 缺失。
- 需要特定非 top-N ICD 展开且参数不够时，创建 adapter。

## 需要观察的证据
以下字段是应从 diagnoses_icd、字典表和公开 reference 中确认的概念，不是可直接调用的 Skill 参数：

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "top_n": 20
}
```

## 输出契约
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

## 失败模式
- cohort 缺少 `hadm_id`。
- dictionary join keys 缺失。
- reference 要求的诊断列命名与 wrapper 默认命名不同。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 检查 ICD 版本、标准化、seq_num、去重和目标业务 key。
- 将映射和聚合写入 `workspace/pipeline/diag.py`，不要创建 Skill variant。
- 用 train 公开样本诊断后运行 validation Pipeline 和 `ValidateDraft`。
- 证据不足的映射写入 `rule_ledger.json` 未解决项，不得编造 code/title。
- 禁止读取 private reference、test raw、历史实验或嵌入患者级行。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
