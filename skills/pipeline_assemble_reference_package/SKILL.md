---
name: pipeline_assemble_reference_package
description: Use when Data Cleaning Agent needs to assemble generated cohort/features/detail files into a result_package matching train/reference shape without requiring reference.csv or package_manifest.json.
---

# pipeline_assemble_reference_package

## 用途
把已经生成的 cohort/features/detail CSV 复制到 `result_package/` 的指定相对路径，使其与 `train/reference` 同构。它不要求生成 `reference.csv`，`package_manifest.json` 也只是可选索引。

## 适用条件
- Agent 已经根据 train/reference 推断出 validation result_package 应该有哪些相对文件。
- 已经生成核心 CSV，需要由累计 Pipeline 统一输出并交给 `ValidateDraft`。

## 跳过条件
- 目标 CSV 尚未生成。
- train 回归仍未通过，需要先修复抽取逻辑。
- 需要生成新特征，而不是复制已有产物。

## 需要观察的证据
以下字段是应从公开 reference 目录和 Pipeline manifest 中确认的概念，不是可直接调用的 Skill 参数：

```json
{
  "package_name": "result_package",
  "write_manifest": false,
  "files": [
    {
      "source": "/path/to/cohort.csv",
      "target": "cohort/cohort.csv",
      "role": "cohort"
    }
  ]
}
```

## 输出契约
- `result_package/`
- copied file list
- optional `package_manifest.json`
- `row_counts`
- `columns`

## source_pipeline_files
- local reference package adapter wrapping MIMIC-IV-Data-Pipeline outputs

## expected_raw_files
无；本 skill 处理已生成产物。

## output_contract
`result_package` 包含 `files[].target` 声明的相对路径；`reference.csv` 和 `package_manifest.json` 都不是成功必要条件。

## 失败模式
- `files` 为空。
- source artifact 不存在。
- target 路径越权。
- 把多文件 reference 错误合并成单个 `reference.csv`。

## 可用工具
- 观察：`Read`、`Glob`、`Grep`、`InspectDataFile`、`CompareArtifact`。
- 实现：`Write`、`Edit`，且只能修改当前 `workspace/pipeline` 与 `rule_ledger.json`。
- 验证：`RunAnalysisPython` 只做受限诊断；`RunPipeline` 执行完整入口；`ValidateDraft` 做提交前验收。

## 执行步骤
- 用 `Glob/InspectDataFile` 确认预期相对路径、schema、列顺序和文件数量。
- 在 `pipeline/run.py` 统一调用各模块并写完整 `result_package`；不创建打包 adapter。
- `pipeline_manifest.json` 声明全部业务文件，运行 `RunPipeline(validation)` 后调用 `ValidateDraft`。
- 缺文件、额外业务文件、placeholder 或运行后手工修改均不得提交。
- 禁止读取 private reference、test raw、历史实验或复制 current best 结果文件。

## 停止条件
- 缺少必要 raw 字段、业务 key 或公开证据时，停止推导并把问题写入 `rule_ledger.json`，不得猜测。
- 当前模块已由完整 Pipeline 重放，且 `ValidateDraft` 通过、Pipeline hash 未变化时，本说明对应工作完成。

## 禁止路径
- validation/test private reference、test 评分报告、凭证、网络和其他实验目录。
- 不得把患者级数据、历史结果包或 validation 输出硬编码、复制或打包进 Pipeline。
