---
name: pipeline_filter_disease_cohort
description: Use when Data Cleaning Agent needs to filter a MIMIC cohort by disease, ICD prefixes, or admitted-due-to rules before writing custom disease filtering code.
---

# pipeline_filter_disease_cohort

## 用途
包装 teacher pipeline 的疾病 cohort 筛选逻辑。输入已有 cohort 和 ICD/疾病规则，输出筛选后的 cohort。

## 适用条件
- reference 或任务中出现 disease cohort、admitted due to、ICD prefix、疾病名称筛选。
- 已有 `cohort.csv`，需要按 `diagnoses_icd` 和 `d_icd_diagnoses` 缩小 cohort。

## 跳过条件
- reference 没有疾病筛选。
- 疾病名称无法映射到明确 ICD 候选时，不要硬猜；标记 ambiguous 或创建 adapter。

## 需要观察的证据
以下字段是应从任务契约、诊断表和公开 reference 中确认的概念，不是可直接调用的 Skill 参数：

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "disease_rule": {"icd_prefixes": ["K70", "K74"]},
  "primary_only": false
}
```

也可以传 `disease_text`，由 wrapper 查询 ICD 字典候选。

## 输出契约
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

## 失败模式
- 没有 ICD prefix。
- 疾病文本对应多个候选，不能自动收敛。
- cohort 缺少 `hadm_id`。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 用 `InspectDataFile` 检查 ICD 版本、code/title、seq_num 和 cohort key。
- 将经证据支持的筛选规则写入 `workspace/pipeline/cohort.py` 或 `diag.py`。
- 用 `RunPipeline` 回放，记录已验证前缀和反例到 `rule_ledger.json`，再运行 `ValidateDraft`。
- 疾病名称无法唯一映射时停止并记录歧义，不得硬猜或沉淀患者名单。
- 禁止读取 private reference、test raw 和历史实验。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
