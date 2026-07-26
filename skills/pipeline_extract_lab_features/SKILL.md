---
name: pipeline_extract_lab_features
description: Use when Data Cleaning Agent needs MIMIC lab features from labevents and d_labitems with chunked reading before writing custom lab scripts.
---

# pipeline_extract_lab_features

## 用途
包装 teacher pipeline 的 lab 处理逻辑，分块读取 `labevents`，连接 `d_labitems`，输出常见 lab 项目的 count/mean/min/max。

## 适用条件
- reference/features 中有 lab、labevents、检验指标、`d_labitems.label` 或 lab 聚合字段。
- 需要处理较大的 lab 表，避免一次性全量读入。

## 跳过条件
- reference 不包含 lab 特征。
- `labevents.csv` 缺失。
- reference 需要固定 itemid 精确字段但没有可传参数时，创建 adapter。

## 需要观察的证据
以下字段是应从 labevents、d_labitems 和公开 reference 中确认的概念，不是可直接调用的 Skill 参数：

```json
{
  "mimic_root": "/path/to/raw",
  "cohort_path": "/path/to/cohort.csv",
  "top_n": 30,
  "chunksize": 250000
}
```

## 输出契约
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

## 失败模式
- cohort 缺少 `hadm_id`。
- labevents 很大，chunksize 过大导致慢或内存高。
- reference 的 lab 字段要求与默认 top-N 聚合不同。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 检查 itemid、label、单位、时间窗、异常值和缺失值序列化。
- 将规则写入累计 Pipeline 对应模块，不创建 adapter 或 Skill variant。
- 大表使用分块诊断；运行 `RunPipeline` 后通过 `ValidateDraft` 检查完整包。
- 单位转换或异常值规则缺少证据时保留原值并记录未解决项。
- 禁止读取 private reference、test raw 和历史实验。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
