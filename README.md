# Data Cleaning Agent

基于 AgentScope `2.0.4.post1` 的结构化医疗数据清洗与纠错实验系统。当前版本保留稳定的双层执行结构：宿主程序管理 attempt、评分、晋升和 Test；每个 attempt 内创建一个全新的 AgentScope ReAct Agent，自主观察数据、编写代码、运行工具并提交结果。

项目提供两条主要实验链路：

- `reference-guided-train-validate`：从 `train/raw -> train/reference` 示例学习累计 Pipeline，在 hidden validation 上迭代，冻结最佳版本后自动进入独立 Test。
- `reference-guided-correct`：从少量成对标准示例学习数据错误模式，修复 `correction/raw` 完整数据包，并由宿主使用 private reference 评估。

## 运行架构

```text
宿主 Attempt Loop
  -> 创建全新 Agent / AgentState / Toolkit / LocalWorkspace
  -> AgentScope ReAct: 思考 -> 工具调用 -> 观察 -> 继续推理
  -> Agent发布候选结果
  -> 宿主执行门禁、隐藏评分、晋升或回滚
  -> 下一 attempt 使用新的 Agent，并接收公开反馈
```

- `attempt`：一次候选提交，不一定成为正式 Loop。
- 正式 Loop：候选通过门禁且隐藏分数严格高于历史 best。
- `--max-iters`：每个 validation attempt 的 ReAct 上限；纠错模式下是该次独立 Agent 会话的上限。
- Test 使用全新的上下文、workspace 和精简 Toolkit，不继承 validation 对话。
- MiniMax M3 使用约 `89.99%` 上下文占用触发 AgentScope 语义压缩，压缩本身不会结束 attempt。

## 环境

```bash
conda env create -f environment.yml
conda activate py3102
```

核心环境：

```text
Python 3.11
AgentScope 2.0.4.post1
pandas 2.3.3
```

真实模型配置写入本机 `model_config.local.yaml`。该文件已被 Git 忽略，禁止提交API密钥：

```yaml
react_planner:
  api_keys:
    - "<PRIMARY_API_KEY>"
    - "<SECONDARY_API_KEY>"
  base_url: "<OPENAI_COMPATIBLE_BASE_URL>"
  model: "MiniMax-M3"
  temperature: 0.0
  seed: 666
```

配置多个 `api_keys` 时，模型客户端在当前密钥触发限流后依次尝试下一密钥。也可以使用 `OPENAI_API_KEY`、`OPENAI_API_BASE` 和 `MODEL_NAME` 环境变量覆盖本地配置。

## 数据目录契约

### Train / Validation / Test

```text
<dataset-split>/
  train/
    raw/
    reference/
    keys.csv
  validation/
    raw/
    reference_private/
    keys.csv
  test/
    raw/
    reference_private/
    keys.csv
```

Agent可以观察 `train/raw`、`train/reference` 和当前处理 split 的 raw 数据；validation/test private reference 仅允许宿主评分器读取。

### Correction

```text
<correction-dataset>/
  train/raw/
  train/reference/
  train/keys.csv
  correction/raw/
  correction/reference_private/
  correction/keys.csv
  host_private/correction_modification_log.csv
  split_manifest.json
```

Correction Agent不能读取 `correction/reference_private`、`host_private`、隐藏评分或历史实验。

## 工作流

### 1. 生成 10:4000:5000 物理划分

```bash
python main.py \
  --workflow prepare-reference-splits \
  --raw-root /absolute/path/to/mimiciv/3.1 \
  --reference-root /absolute/path/to/reference-package \
  --split-output /absolute/path/to/dataset-split \
  --split-counts 10,4000,5000 \
  --split-seed 666 \
  --raw-mode copy \
  --decompress-gzip
```

### 2. Validation Loop 与自动 Test

任务Prompt模板位于 [`prompts/reference_guided_loop_template.md`](prompts/reference_guided_loop_template.md)。每次实验必须使用新的空目录：

```bash
python main.py \
  --workflow reference-guided-train-validate \
  --dataset-split /absolute/path/to/dataset-split \
  --experiment-dir /absolute/path/to/new-experiment \
  --round-limit 5 \
  --patience 2 \
  --max-attempts 20 \
  --max-iters 10000 \
  "<REFERENCE_GUIDED_PROMPT>"
```

常用实验开关：

```text
--disable-pipeline-skills  Validation消融：不暴露9个Pipeline Skills
--enable-codegraph         向Validation/Correction Agent提供受限CodeGraphExplore
--target-score 0.99        达到目标分数后冻结best并进入Test
```

Validation自然停止或第一次收到 `Ctrl+C` 后，宿主从正式 best 创建 checkpoint，并启动独立 Test Agent。Test不能修改冻结Pipeline；业务结果由宿主运行冻结入口后评分。

### 3. 构造纠错数据划分

```bash
python main.py \
  --workflow prepare-correction-split \
  --source-archive /absolute/path/to/error-package.zip \
  --split-output /absolute/path/to/correction-dataset \
  --train-count 10
```

其余 stay 自动进入 correction，形成例如 `10:990` 的标准示例/待修复划分。

### 4. 成对示例纠错

```bash
python main.py \
  --workflow reference-guided-correct \
  --dataset-split /absolute/path/to/correction-dataset \
  --experiment-dir /absolute/path/to/new-correction-experiment \
  --max-iters 10000 \
  "请根据train/raw与train/reference成对示例，自主归纳错误规律并修复correction/raw完整数据包。保持目录、文件名、schema和gzip格式；证据不足时保持原值。"
```

纠错消融与恢复：

```text
--disable-error-view-skills  不暴露四类错误先验Skill
--enable-codegraph           启用受限源码查询
--resume                     仅恢复同一个失败的纠错实验目录
```

Skill开关、Prompt、数据集哈希和CodeGraph身份写入 `run_manifest.json`；配置变化后不能错误恢复到同一实验目录。

### 5. 手工复评已有 Adapter

正常主链路不需要该入口。仅在复评一个已有独立Adapter时使用：

```bash
python main.py \
  --workflow reference-test-evaluate \
  --dataset-split /absolute/path/to/dataset-split \
  --experiment-dir /absolute/path/to/test-experiment \
  --adapter-script /absolute/path/to/build_result_package.py
```

## Skills 与工具

### Pipeline Skills

Validation默认暴露9个MIMIC Pipeline Skills：

- `pipeline_build_cohort`
- `pipeline_filter_disease_cohort`
- `pipeline_extract_diag_features`
- `pipeline_extract_proc_features`
- `pipeline_extract_lab_features`
- `pipeline_extract_med_features`
- `pipeline_extract_icu_event_features`
- `pipeline_clean_feature_table`
- `pipeline_assemble_reference_package`

这些Skill包含 `SKILL.md`说明和受限可执行实现。关闭Pipeline Skills后，其目录同时从Agent授权读根中移除。

### Error View Skills

Correction默认暴露4个纯知识型 `SKILL.md`：

- `correction_intra_table_errors`
- `correction_entity_alignment_errors`
- `correction_cross_table_errors`
- `correction_task_oriented_errors`

它们提供证据要求、检测视角、保守修复原则和停止条件，不包含当前数据集的具体错误值、Gold答案或行号，也不能通过 `RunSkill` 执行。

### 数据质量工具

Correction Toolkit包含：

- `ProfileDataQuality`：单文件类型、缺失、唯一值和分布画像。
- `ProfileByGroup`：按实体或业务字段分组诊断。
- `TestDataConstraint`：验证范围、枚举、唯一性和关系约束。
- `AuditRepairDelta`：审计修复前后的行与单元格变化，不判断修改是否等于Gold。

工具是否注册和Agent是否实际调用是两件事；实际使用情况记录在 `skill_usage_report.json` 和 `tool_audit.jsonl`。

## CodeGraph

CodeGraph默认关闭，只用于理解授权项目源码，不索引或读取数据集、历史实验和private reference。

```bash
cd /absolute/path/to/v2_clean-agentscope2-parity
codegraph init

python main.py ... --enable-codegraph "<PROMPT>"
```

- `.codegraph`数据库和日志仅保存在本机，不进入Git。
- 模型侧只看到 `CodeGraphExplore(query, max_files)`，不能覆盖项目路径。
- Test Agent不连接CodeGraph。
- MCP启动失败发生在attempt创建前；单次查询失败时Agent可以回退到 `Read/Grep`。

## 门禁与结果

Validation候选需要通过结果包结构、累计Pipeline、隔离重放、train公开回归、validation完整重放和候选/重放哈希一致性检查。门禁失败会生成失败反馈，允许后续attempt继续修改；无效attempt不计入patience。

主要Validation输出：

```text
<experiment>/
  run_manifest.json
  experiment_state.json
  candidates/candidate_XXXX/attempt_outcome.json
  rounds/round_XXXX/
  active_bundle/
  test_checkpoints/checkpoint_XXXX_best_round_XXXX/
```

Correction输出：

```text
<experiment>/
  run_manifest.json
  correction_run_report.json
  correction_result_gate.json
  skill_usage_report.json
  result_package/
  evaluation/correction_evaluation_report.json
```

Correction门禁验证完整文件、schema、gzip可解析性、key覆盖以及没有额外/缺失业务文件。隐藏评估进一步区分正确修复、漏修、错误修复、额外误改和干净数据保留率。

## Git 与本地数据边界

以下内容不上传GitHub：

```text
datasets/
experiments/
output/
outputs/
reports/
figures/
.codegraph数据库与日志
model_config.local.yaml
host_private/
reference_private/
```

代码、测试、环境文件、Prompt模板和通用Skills可以提交。任何真实API密钥、患者级数据、隐藏Gold或实验结果都不得进入Git。

## 验证

```bash
conda run -n py3102 python -m pytest -q
conda run -n py3102 env PYTHONPYCACHEPREFIX=/private/tmp/v2_clean_pycache \
  python -m compileall -q main.py agent agent_tools workflow skills lib
```

当前基线测试：`160 passed`。
