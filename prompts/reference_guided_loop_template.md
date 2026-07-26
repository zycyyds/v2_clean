# Data Cleaning Agent Validation Prompt Template

该模板用于 AgentScope 2.x 工作树中的：

```text
--workflow reference-guided-train-validate
```

宿主程序会提供真实路径、受限工具、隐藏评分、best 晋升和自动 test。用户 prompt 只描述本次数据任务和必须保持的业务约束。

需要替换：

- `<VALIDATION_KEY_COUNT>`：validation key 数，例如 `4000`。
- `<KEY_COLUMN>`：业务主键，例如 `stay_id`。
- `<REFERENCE_FILES>`：`train/reference` 下全部业务文件。

## Prompt

```text
请执行 MIMIC-IV ICU mortality reference-guided 数据清洗任务。

【任务目标】
1. train/raw 是公开原始示例，train/reference 是对应标准结果包。
2. 自主学习 raw 到 reference package 的文件结构、字段、业务 key、join、筛选、时间窗、派生、去重、排序、缺失值和序列化规则。
3. 使用同一套累计 Pipeline 处理 validation/raw，生成与 train/reference 同构的完整 result_package。
4. validation 停止后，宿主会冻结历史 best Pipeline，并在全新 Test Agent 上处理 test；不要提前读取或处理 test。

【数据隔离】
1. 只能读取宿主明确授权的当前实验 train/raw、train/reference、train/keys、validation/raw、validation/keys 和通用项目代码。
2. 禁止读取 validation/test private reference、隐藏评分报告、其他实验目录或历史脚本和结果包。
3. 不得把患者级数据、stay_id、label、validation 结果或 private 信息硬编码进脚本或静态资产。
4. 不修改全局 skills/、workflow/、lib/；只写当前 workspace。

【累计 Pipeline】
1. 唯一业务实现位于 workspace/pipeline，唯一入口是 pipeline/run.py。
2. 首个候选建立完整 Pipeline；后续候选必须继承 current best，只修改当前反馈涉及的模块。
3. 不创建 adapter、Skill variant、capability 包、build_test.py 或第二套业务入口。
4. Skill 只是按需阅读的业务说明；规则必须落实到累计 Pipeline 或 rule_ledger.json。
5. Pipeline 每次都必须生成全部业务文件，不得只生成本次修改的文件，不得在运行后手工修改 CSV。
6. 固定入口必须支持：
   python pipeline/run.py --raw-root <raw> --output-dir <output> --split-mode train|validation|test

【需要复现的17个业务文件】
<REFERENCE_FILES>

【Agent 工作方式】
1. 先用 Glob、InspectDataFile、Read 检查 train/reference 全部文件及 train/raw 来源，再建立3到5个短期 Task，只保留一个主要任务 in_progress。
2. 已验证规则、反例和未解决问题持续写入 workspace/rule_ledger.json。
3. 可以用 RunPipeline(train) 和 CompareArtifact 做公开诊断，但 train 不要求逐值100%复现，也不能因少量 train 差异无限停留。
4. 用 RunPipeline(validation) 生成 workspace/result_package。
5. 候选提交前必须调用 ValidateDraft；失败时根据机器报告继续修复，失败不会产生 attempt。
6. 只有 ValidateDraft 通过且 Pipeline hash 未变化，才调用 SubmitCandidate。
7. SubmitCandidate 后由宿主执行正式门禁、隐藏评分和严格提升晋升，并把结果返回当前同一个 Agent 上下文。
8. 未晋升时从恢复后的 current best 继续；失败候选仅作为负面经验，不继续使用失败代码。

【输出契约】
1. validation cohort 必须覆盖全部 <VALIDATION_KEY_COUNT> 个 <KEY_COLUMN>，主键唯一且无未知 key。
2. 其他业务文件不得包含未知 key；事件/特征/summary 的业务粒度、重复规则和 schema 以 train/reference 证据为准。
3. 输出文件名、目录层级、列顺序、值格式、gzip 形式和确定性排序必须与 train/reference 契约一致。
4. 每次候选包含完整 result_package、pipeline_manifest.json、run.py、固定模块和可重放的父 Pipeline lineage。
5. 不要求生成 reference.csv；package_manifest.json 仅可作为索引，不能代替17个业务文件。

不要返回文字上的 SUCCESS 来代替工具执行。持续观察、修改、运行和验收，直到调用 SubmitCandidate 或宿主停止流程。
```

## 真实目录 Smoke Command

```bash
cd /Users/mac/PycharmProjects/v2_clean-agentscope2
conda activate py3102

DATASET="/Users/mac/PycharmProjects/v2_clean/datasets/mimic_icu_mortality_v3_1_reproduced_random_10train_4000val_5000test_seed_1759733077"
EXPERIMENT="/Users/mac/PycharmProjects/v2_clean-agentscope2/experiments/mimic_icu_mortality_agentscope2_smoke_v1"

PROMPT="$(python - <<'PY'
from pathlib import Path

template = Path("prompts/reference_guided_loop_template.md").read_text(encoding="utf-8")
prompt = template.split("```text", 2)[2].split("```", 1)[0].strip()
prompt = prompt.replace("<VALIDATION_KEY_COUNT>", "4000")
prompt = prompt.replace("<KEY_COLUMN>", "stay_id")
prompt = prompt.replace(
    "<REFERENCE_FILES>",
    "\n".join(
        f"- {path}"
        for path in (
            "cohort/cohort_icu_mortality_0__.csv",
            "csv/labels.csv",
            "features/preproc_chart_icu.csv",
            "features/preproc_diag_icu.csv",
            "features/preproc_med_icu.csv",
            "features/preproc_out_icu.csv",
            "features/preproc_proc_icu.csv",
            "summary/chart_features.csv",
            "summary/chart_summary.csv",
            "summary/diag_features.csv",
            "summary/diag_summary.csv",
            "summary/med_features.csv",
            "summary/med_summary.csv",
            "summary/out_features.csv",
            "summary/out_summary.csv",
            "summary/proc_features.csv",
            "summary/proc_summary.csv",
        )
    ),
)
print(prompt)
PY
)"

python main.py \
  --workflow reference-guided-train-validate \
  --dataset-split "$DATASET" \
  --experiment-dir "$EXPERIMENT" \
  --round-limit 2 \
  --patience 2 \
  --max-attempts 30 \
  --max-iters 10000 \
  "$PROMPT"
```

`--max-iters` 是整个 validation Agent 生命周期的模型 ReAct 上限，不是要求必须执行10000次。达到2个正式晋升、patience、attempt 上限或第一次 `Ctrl+C` 后，宿主都会从正式 best 自动进入 test。
