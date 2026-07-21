---
name: pipeline_assemble_reference_package
description: Use when Data Cleaning Agent needs to assemble generated cohort/features/detail files into a result_package matching train/reference shape without requiring reference.csv or package_manifest.json.
---

# pipeline_assemble_reference_package

## 用途
把已经生成的 cohort/features/detail CSV 复制到 `result_package/` 的指定相对路径，使其与 `train/reference` 同构。它不要求生成 `reference.csv`，`package_manifest.json` 也只是可选索引。

## 什么时候使用
- Agent 已经根据 train/reference 推断出 validation result_package 应该有哪些相对文件。
- 已经生成核心 CSV，需要打包交给 `ValidateResultPackage` 和 Evaluator。

## 什么时候不要直接使用
- 目标 CSV 尚未生成。
- train 回归仍未通过，需要先修复抽取逻辑。
- 需要生成新特征，而不是复制已有产物。

## 输入
`spec_json` JSON object:

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

## 输出
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

## common_failure_modes
- `files` 为空。
- source artifact 不存在。
- target 路径越权。
- 把多文件 reference 错误合并成单个 `reference.csv`。

## adapter_policy
先直接调用；如果 reference package 有复杂目录规则，创建 experiment 内 adapter；不要修改全局 skill。
