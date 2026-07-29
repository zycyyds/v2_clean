# Data Cleaning Agent

基于 AgentScope `2.0.4.post1` 的结构化医疗数据清洗与纠错实验系统。本开发线同时保留原有双层实验工作流，并新增一个独立的 Pi 式通用持续 Agent 运行时。通用运行时不接入旧版 attempt、候选发布、隐藏评分和结果门禁。

项目提供两条主要实验链路：

- `reference-guided-train-validate`：从 `train/raw -> train/reference` 示例学习累计 Pipeline，在 hidden validation 上迭代，冻结最佳版本后自动进入独立 Test。
- `reference-guided-correct`：从少量成对标准示例学习数据错误模式，修复 `correction/raw` 完整数据包，并由宿主使用 private reference 评估。

## Pi 式通用 Agent

入口为 `agent.pi_cli`。一次进程只创建一个 Agent、AgentState、模型、Toolkit 和 LocalWorkspace；模型调用工具后会继续推理，直到不再生成工具调用或达到本轮迭代上限。

```bash
conda run -n py3102 env PYTHONPATH=. \
  python -m agent.pi_cli \
  --workdir /absolute/path/to/task-workdir \
  --max-iters 10000 \
  "<TASK_PROMPT>"
```

可选按 AgentScope 原生 Skill 机制加载一个或多个目录：

```bash
conda run -n py3102 env PYTHONPATH=. \
  python -m agent.pi_cli \
  --workdir /absolute/path/to/task-workdir \
  --skills-dir /absolute/path/to/skills \
  "<TASK_PROMPT>"
```

- 不传 `--skills-dir` 时，Toolkit 中不存在 Skill Viewer，也不会自动加载项目 Skills。
- 固定工具为 AgentScope 原生 `Read / Write / Edit / Glob / Grep / Bash`。
- `Read / Glob / Grep` 可并发；`Write / Edit / Bash` 串行执行。
- 工具失败或参数 JSON 不完整时，错误作为 Tool Result 返回同一 Agent继续修正。
- 被模型输出长度截断的工具调用不会执行；Agent会收到纠错观察并补发一次。
- 一次 Runtime 内再次调用 `run_turn(feedback)` 会保留上下文，并为新一轮重置 ReAct 计数。当前CLI首版只调用一次。

MiniMax-M3配置为 `1,000,000` token上下文。项目按Pi策略在本地估算达到 `983,616` token时触发压缩，保留系统信息、摘要、工具schema及近期消息合计约 `20,000` token。AgentScope 2.0.4使用近似token估算而非MiniMax官方 tokenizer，因此该触发值是工程估算，不应解释为服务端精确计数。若服务端先报告context overflow，运行时会强制压缩并重试一次。

运行记录写入：

```text
<workdir>/.agent_runs/<run_id>/
  run_manifest.json
  transcript.log
  tool_audit.jsonl
  run_report.json
  workspace/
```

这些文件只用于追踪模型、工具、压缩、token估算、耗时和结束原因，不参与评分或门禁。当前使用 `PermissionMode.BYPASS`；允许/禁止路径仅由任务Prompt约束，原生本地工具不是强安全沙盒。处理private reference或敏感数据时，必须由调用方提供额外隔离。

## Pi 式 Validation Harness

`agent.pi_harness_cli`在通用持续Agent外增加宿主隐藏评分。多轮共用同一个AgentState和上下文；未提升时宿主恢复正式best文件并清除AgentScope文件缓存，但保留失败经验。评分器不会注册为Agent工具，Gold和宿主报告由macOS `sandbox-exec`隔离。

```bash
conda run -n py3102 env PYTHONPATH=. \
  python -m agent.pi_harness_cli \
  --experiment-dir /absolute/path/to/new-empty-experiment \
  --train-raw /absolute/path/to/train/raw \
  --train-reference /absolute/path/to/train/reference \
  --validation-raw /absolute/path/to/validation/raw \
  --validation-gold /absolute/path/to/validation/reference_private \
  --evaluation-manifest evaluation_manifests/mimic_icu_mortality_v3_1.json \
  --prompt-file /absolute/path/to/prompt.txt \
  --max-rounds 20 \
  --patience 3 \
  --target-score 1.0 \
  --max-iters 10000
```

Agent不需要采用固定脚本布局，只需在工作区写出下面的描述文件。`result_root`必须是工作区内的专用子目录；`replay.argv`必须是参数数组，不能使用`sh -c`等Shell命令字符串：

```json
{
  "schema_version": 1,
  "result_root": "result_package",
  "replay": {
    "argv": [
      "python", "scripts/build_pipeline.py",
      "--raw", "{raw_root}",
      "--train-reference", "{train_reference}",
      "--out", "{output_dir}"
    ]
  }
}
```

评分按Gold动态发现文件并等权平均。单文件分数为`10% schema F1 + 20% key structure F1 + 40% row-aligned cell F1 + 30% exact-row F1`。结束时，宿主在一次性空目录中以参数数组独立重跑best，禁用网络并再次评分；重放临时目录在本次评分返回后自动删除。终端只输出轮次开始、隐藏评分、晋升/回滚和独立重放等低频阶段状态，不输出高频心跳。不存在旧版17文件门禁、固定Pipeline模块、train百分百复现或候选哈希一致门禁。

## 两条运行架构

Pi式Validation Harness使用一个持续Agent：

```text
宿主 Validation Round Loop
  -> 创建一个沙盒Worker、Agent和AgentState
  -> AgentScope ReAct自然结束当前turn
  -> 宿主隐藏评分并保存或恢复best
  -> 脱敏反馈返回同一个Agent上下文
  -> 停止后在空目录独立重放best
```

- `round`是一次Agent自然结束后进行的隐藏评分；不同round共享上下文和压缩历史。
- `--max-iters`在每个round重新获得完整预算，不限制宿主round数量。
- 分数严格提升至少`1e-6`才晋升；未提升时恢复best文件，但保留失败经验。
- 该Pi式Harness第一阶段只实现Validation，尚未接入独立Test Agent。

原有`reference-guided-*`命令仍保留旧parity架构：宿主管理attempt，每个attempt创建全新的Agent和AgentState，冻结best后进入原有Test流程。两条链路互不共享上下文或实验目录。

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
ripgrep
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
