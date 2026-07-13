# Reference-Guided Codex-Style Loop Prompt Template

这个模板用于 `--workflow reference-guided-train-validate` 的 Codex-style Agent 运行。

适用场景：

- `train/raw` 是原始数据。
- `train/reference` 是处理后的示例结果包。
- `validation/raw` 是需要 Agent 处理的验证原始数据。
- `validation/reference` 只允许 Evaluator 读取。
- 目标是让 Agent 学习 `raw -> reference package` 的转换关系，并通过 validation loop 迭代 adapter/rule。

需要替换的占位符：

- `<DATASET_SPLIT>`：包含 `train/validation/test` 的 split 目录。
- `<EXPERIMENT_DIR>`：本次实验输出目录。
- `<VALIDATION_KEY_COUNT>`：验证集 key 数量，例如 `4000`。
- `<KEY_COLUMN>`：主键列，例如 `stay_id`。
- `<REFERENCE_FILES>`：`train/reference` 中真实业务文件列表。

## Prompt

```text
请执行 reference-guided Codex-style 数据整理任务。

【目标】
这个 dataset-split 里：
- train/raw 是原始数据。
- train/reference 是处理后的参考结果包。
- validation/raw 是需要处理的验证原始数据。
- validation/reference 不能由 Agent 读取，只能由 Evaluator 读取。

请分析 train/raw 到 train/reference 的数据关系，学习它的文件结构、字段、key、join、筛选、时间窗、派生逻辑和输出形态，然后用同一套规则处理 validation/raw，生成和 train/reference 同构的 result_package。

【严格限制】
1. 只能读取当前 dataset-split 目录内部文件和实验目录内部文件。
2. 不要读取 validation/reference 或任何 private reference。
3. 不要修改原始 skills/、lib/ 或 teacher pipeline 项目。
4. 如果现有 skill 接口和 train/reference 输出形态不匹配，允许在当前 experiment 内创建 adapter/fork。
5. standalone Python 只能作为实验内 adapter 草稿或 glue code，不能污染全局代码。

【Reference 形态要求】
1. 不要把 package_manifest.json、reference.csv 当作必须业务结果；如果存在，只作为索引或辅助信息。
2. 业务结果以 train/reference 下真实存在的文件为准。
3. validation result_package 必须尽量保持和 train/reference 同构。
4. 本次需要复现的业务文件是：
<REFERENCE_FILES>
5. 如果某个文件在 train/reference 中不存在，不要凭空生成。
6. 如果 reference 是事件明细表，不要强行合并成 features_wide。

【Agent 执行策略】
1. 先读取 reference_contract.json、train/reference 文件清单、train/reference 每个 CSV 的 schema 和样例。
2. 再读取 train/raw 和 validation/raw 的文件清单、schema、关键列和样例。
3. 优先使用已注册的 pipeline_* skills；如果 skill 输出接口不满足 reference 形态，创建 experiment-local adapter。
4. adapter 必须先在 train/raw 上生成 train result package，并和 train/reference 做回归比较。
5. train 回归达标后，必须立即用同一 adapter/rule 处理 validation/raw。
6. validation 阶段不得读取 validation/reference。
7. 最终必须发布 canonical result_package，并运行 ValidateResultPackage。
8. 本轮不强制包装 Skill。loop 阶段优先允许脚本、adapter 或 fork 持续迭代；只有当你已经自然整理好稳定能力时，才可选写 `skill_packaging_plan.json` 和 `workspace/capabilities/<skill_name>/`。缺失或格式不完整不得阻塞 result_package 进入评估。

【Train 回归门禁规则】
1. train/reference 只用于学习转换关系和做回归检查，不是最终优化目标。
2. 不要求 train/reference 100% exact match。
3. 当 train 结果满足以下任一条件时，必须停止继续诊断 train 差异，并立即处理 validation/raw：
   - 每个 reference 文件的行数误差 <= 1%，且关键列存在；
   - 或 cell-level recall >= 0.98；
   - 或只剩少量可解释的边界差异，并已写入 train_regression_report.json。
4. 不允许为了少量边界差异无限写诊断脚本。
5. 如果已经做过 3 轮 train 差异诊断，且差异只剩时间窗、格式化或少量边界差异，必须停止 train 诊断并进入 validation。
6. train 达标后必须把当前最新结果包发布为 canonical result_package。
7. 发布后必须立即用同一套规则处理 validation/raw。
8. 真正决定是否保留本轮修改的是 validation evaluator 的 composite_score，而不是 train 分数。

【Validation Loop 规则】
1. 从第 2 轮开始，必须先读取上一轮 public_feedback.json。
2. 本轮修改必须对应 public_feedback 里的具体缺口。
3. 不能重新从 train/reference 学一套全新的逻辑。
4. train 回归只用于确认本轮修改没有破坏基础结构。
5. 只要 validation composite_score 高于历史 best，就应晋升当前 candidate。
6. 如果 validation 分数没有提升，保留上一轮 best，不要用本轮结果覆盖 active bundle。
7. 连续两轮没有提升或没有可执行修复目标时，应冻结 best bundle。

【Validation 输出要求】
1. 必须处理 validation keys 中全部 <VALIDATION_KEY_COUNT> 个 <KEY_COLUMN>。
2. result_package 中主结果文件必须覆盖 <VALIDATION_KEY_COUNT> 个唯一 <KEY_COLUMN>。
3. result_package 必须生成和 train/reference 同名、同目录层级的业务文件。
4. 结果必须运行 ValidateResultPackage，expected_key_count=<VALIDATION_KEY_COUNT>，key_column=<KEY_COLUMN>。
5. 成功前必须真实读取或验证最终 result_package。

【报告要求】
1. train_regression_report.json 需要说明每个文件的 train 行数、reference 行数、误差比例、是否通过、边界差异原因。
2. skill_usage_report.json 需要说明调用、跳过、适配或派生 skill 的理由。
3. result_package_validation_report.json 需要记录 result_package 结构、文件、行数、key 覆盖率。
4. 如果创建或修改 adapter/fork，必须记录到当前 experiment bundle，不得写入全局 skills/。
5. 如果本轮可选生成 skill_packaging_plan.json，它需要说明每个 Skill 的名称、负责的 reference artifact、来源脚本、base skills 和拆分理由；未生成不算失败。

任何零产出、缺少 result_package、未处理 validation/raw、未覆盖 <VALIDATION_KEY_COUNT> 个 <KEY_COLUMN>、未发布结果包、未验证真实产物，都不得返回 SUCCESS。
```

## Command Skeleton

```bash
cd /Users/mkbk/PycharmProjects/v2_clean
conda activate py310

PROMPT="$(python - <<'PY'
from pathlib import Path

template = Path('prompts/reference_guided_loop_template.md').read_text()
prompt = template.split('```text', 1)[1].split('```', 1)[0].strip()
prompt = prompt.replace('<VALIDATION_KEY_COUNT>', '4000')
prompt = prompt.replace('<KEY_COLUMN>', 'stay_id')
prompt = prompt.replace(
    '<REFERENCE_FILES>',
    '- cohort/cohort_icu_mortality_0__.csv\n'
    '- features/preproc_chart_icu.csv\n'
    '- features/preproc_diag_icu.csv\n'
    '- features/preproc_med_icu.csv\n'
    '- features/preproc_out_icu.csv\n'
    '- features/preproc_proc_icu.csv'
)
print(prompt)
PY
)"

python main.py \
  --workflow reference-guided-train-validate \
  --dataset-split <DATASET_SPLIT> \
  --experiment-dir <EXPERIMENT_DIR> \
  --round-limit 4 \
  --max-iters 400 \
  "$PROMPT"
```

## Notes From Successful Run

- 这类任务里 `train/reference` 是公开示例，允许 Agent 用它回归；`validation/reference` 必须隐藏给 Evaluator。
- `train` 内部自修是有价值的，但只能作为回归门禁；真正优化目标是 validation 分数。
- 如果 reference 是目录包，就不要强行合成 `reference.csv` 或 `features_wide.csv`。
- 对 MIMIC ICU mortality 这次实验，效果好的核心是：先修 schema 和过抽取，再通过 validation feedback 逐轮修字段边界。
