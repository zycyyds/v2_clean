# MIMIC-IV ICU Mortality 完整数据复现规格

日期：2026-07-13

## 1. 目标

基于 `/Users/mac/PycharmProjects/MIMIC-IV-Data-Pipeline-main` 的原始实现和本机 MIMIC-IV 3.1 原始数据，完整复现 ICU mortality 数据生成链路，执行到原 notebook 的 Block 7 为止。

本次包含 cohort、五类事件特征、诊断编码归一化、summary、labels、逐 stay 数据文件和全局字典；不运行任何机器学习、模型训练、特征重要性或预测评估步骤。

## 2. 固定配置

| 配置项 | 固定值 |
|---|---|
| MIMIC 版本 | 3 / 3.1 |
| 数据场景 | ICU |
| 任务 | Mortality |
| Disease Filter | No Disease Filter |
| 特征类型 | diagnosis、output、chart、procedures、medications 全部启用 |
| Diagnosis 处理 | ICD-9 转 ICD-10，并按 ICD-10 root 分组 |
| 二次特征筛选 | No |
| Chart 异常值处理 | No |
| 观察窗口 | First 72 hours |
| Prediction window | 2 hours |
| 时间 bucket | 1 hour |
| Imputation | No Imputation |

这组配置与当前 Agent 学习和生成的 ICU mortality 目标文件最一致。配置在本次执行器中显式声明，不读取 notebook 的缓存控件状态。

## 3. 输入与隔离目录

### 3.1 只读输入

- 原始数据：`/Users/mac/PycharmProjects/MIMIC-IV-Data-Pipeline-main/mimiciv/3.1`
- 原项目源码：`/Users/mac/PycharmProjects/MIMIC-IV-Data-Pipeline-main`
- ICD 映射：原项目 `utils/mappings/ICD9_to_ICD10_mapping.txt`

### 3.2 本次输出

默认运行根目录：

`/Users/mac/PycharmProjects/v2_clean/reproductions/mimic_icu_mortality_v3_1_full_v1`

执行器在运行根目录中创建 `mimiciv/3.1` 和映射文件的只读链接，并把所有新产物写入该目录。不得修改或覆盖原项目的 `data`，也不得复用其中日期混杂的中间产物。

## 4. 数据生成链路

### 阶段 1：Cohort

调用 Version 3 cohort 逻辑生成 ICU mortality cohort：

- 仅保留成人病例，`Age >= 18`。
- mortality label 按原版规则计算：患者 `dod` 位于 ICU `intime` 与 `outtime` 之间时为 1，否则为 0。
- 不使用 disease filter。
- 输出 cohort 主表及原版 summary。

原版 Version 3 存在输出路径不一致：cohort 被写到运行根目录，而后续阶段从 `data/cohort` 读取。执行器必须在阶段完成后将这两个原版产物移动到本次运行根目录的 `data/cohort`，不得修改其业务内容。

### 阶段 2：五类特征抽取

调用原版 ICU feature pipeline，从 MIMIC-IV 3.1 抽取：

- `preproc_chart_icu`
- `preproc_diag_icu`
- `preproc_med_icu`
- `preproc_out_icu`
- `preproc_proc_icu`

所有五类特征均启用，不进行第二次 feature selection。

### 阶段 3：Diagnosis 归一化

按原版 preprocessing 执行：

1. ICD-9 映射为 ICD-10。
2. 删除无法得到目标 ICD-10 root 的记录。
3. 以 root 作为最终 `new_icd_code`。
4. 将六列诊断抽取中间态转成最终四列诊断文件。

最终 diagnosis 文件必须与 Agent 目标契约一致，不把六列中间态作为最终金标准候选。

### 阶段 4：Summary

对 cohort 和五类 feature 文件生成原版 summary。预期共 10 个 summary CSV：每个业务文件各自对应的统计和列/特征摘要，具体文件名以原版函数实际输出为准并写入 manifest。

### 阶段 5：可选清洗关闭

- 不运行二次特征筛选。
- 不运行 chart 异常值检测或修正。

阶段日志必须明确记录这两项为 `disabled`，避免把“未执行”误判为遗漏。

### 阶段 6：逐 stay 数据生成

调用原版 `data_generation_icu.Generator`，固定参数：

- `include_time = 72`
- `bucket = 1`
- `predW = 2`
- `impute = False`

沿用原版病例纳入语义：只有可满足 72 小时观察窗口和 2 小时预测窗口的 ICU stay 才进入 labels 和逐 stay 产物。

每个纳入 stay 必须完整生成：

- `demo.csv`
- `static.csv`
- `dynamic.csv`

同时生成原版 `data/dict/*` 字典文件和 `data/csv/labels.csv`。

## 5. 产物分类

### 5.1 金标准候选

用于后续 Agent 评估的候选仅包括 17 个 CSV：

- 1 个 cohort 文件。
- 1 个 `csv/labels.csv`。
- 5 个最终 feature 文件。
- 10 个 summary 文件。

最终清单以 manifest 中的相对路径明确列出，不能混入六列 diagnosis 中间态。

### 5.2 完整复现但不作为金标准

- 所有逐 stay 的 `demo.csv`、`static.csv`、`dynamic.csv`。
- `data/dict/*`。
- 阶段日志、运行状态、耗时和校验报告。

这些文件用于证明原项目数据生成链路已完整执行，但不参与当前 Agent 输出评分。

## 6. 执行器设计

执行器采用单命令、分阶段、可恢复模式：

- 每个阶段开始前校验依赖输入。
- 每个阶段成功后写入 checkpoint，包括阶段名、配置、开始/结束时间、产物路径、文件大小、行数和 schema。
- 已完成阶段只有在配置和输入指纹一致时才允许跳过。
- 阶段失败时保留日志和已完成产物；重新运行从最近的有效 checkpoint 继续。
- `--force-stage` 仅重跑指定阶段及其下游，不能静默混用旧产物。
- 最终写出统一 `reproduction_manifest.json` 和 `validation_report.json`。

## 7. 验收门禁

### 7.1 输入门禁

- MIMIC-IV 3.1 所需原始表存在且非空。
- ICD-9 到 ICD-10 映射文件存在且非空。
- 原版目标模块能够从 `py310` 环境导入。

### 7.2 Cohort 门禁

- cohort 非空且 schema 与原版 Version 3 输出契约一致。
- `stay_id`、`hadm_id`、`subject_id` 的缺失和重复情况被报告。
- `Age >= 18`。
- label 仅包含 0 和 1，并按原版 `dod` 时间规则抽样回算。

### 7.3 Feature 门禁

- 五个最终 feature 文件全部存在且非空。
- 最终 diagnosis 文件是四列目标形态，`new_icd_code` 为 root；六列中间态不得冒充最终文件。
- 每个 feature 的 stay 范围属于 cohort，额外 stay 数必须为 0。

### 7.4 Labels 与逐 stay 门禁

- labels 非空，label 值域合法，stay 必须属于 cohort。
- labels 的纳入条件按原版整小时语义验证，满足 72 小时观察加 2 小时预测窗口。
- 每个 labels stay 恰好存在一组 `demo.csv`、`static.csv`、`dynamic.csv`。
- 三类逐 stay 文件缺失数、空文件数和多余目录数均为 0。
- `data/dict` 中原版要求的字典文件全部存在且非空。

### 7.5 最终门禁

- 17 个金标准候选全部存在，路径和 schema 写入 manifest。
- 10 个 summary 文件数量和来源关系正确。
- 所有产物来自同一次运行和同一配置，不允许引用原项目旧 `data` 文件。
- 输出目录中不生成模型、训练 checkpoint、预测或 ML 评估产物。
- 任一门禁失败时，整体状态标记为 `failed_validation`，不得宣称复现完成。

## 8. 测试策略

实现前先增加小规模和纯逻辑测试：

1. 固定配置正确映射到原版函数参数。
2. 隔离目录不会写入原项目 `data`。
3. Version 3 cohort 路径适配只移动文件，不改变内容。
4. Diagnosis 六列到四列转换符合原版 root 规则。
5. checkpoint 仅在输入和配置指纹一致时生效。
6. labels 与逐 stay 一一对应门禁能识别缺失、空文件和多余目录。
7. manifest 只把约定的 17 个 CSV 列为金标准候选。
8. 全链路结束后执行真实数据验收，并记录各阶段耗时和产物规模。

## 9. 非目标

- 不修改 MIMIC-IV-Data-Pipeline-main 的原始算法。
- 不修正原项目中可能存在的业务定义争议；只进行必要的路径适配和可复现封装。
- 不运行 notebook Block 7 之后的任何机器学习步骤。
- 不把逐 stay 文件或字典纳入当前 Agent 金标准评分。
- 不覆盖现有验证集、测试集、实验目录或旧的复现结果。

## 10. 完成定义

只有同时满足以下条件才算完成：

1. 固定配置的全部数据阶段真实执行成功。
2. `validation_report.json` 的所有强制门禁通过。
3. 17 个金标准候选、全部逐 stay 文件和字典均可定位。
4. manifest 能证明每个文件的来源、schema、行数、大小和同次运行关系。
5. 原项目现有 `data` 和当前 Agent 数据集未被修改。
