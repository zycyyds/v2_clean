# Data Cleaning Agent: AgentScope 2.x 重构计划

## 实施状态（2026-07-22）

- 独立工作树：`/Users/mac/PycharmProjects/v2_clean-agentscope2`。
- 分支：`codex/agentscope2-refactor`；旧稳定工作树保持不变。
- 运行环境：Python `3.11.11`、AgentScope `2.0.4.post1`、Conda 环境 `py3102`。
- validation 已迁移为单个持续 Agent，并通过外部执行事件完成候选评分回传。
- validation/test Toolkit、Skill loader、LocalWorkspace、上下文压缩、原子状态、累计 Pipeline、checkpoint 和 Test Agent 均已接入。
- 旧 Skill 可执行注册器、Skill variant、`RunSkill`、`PublishDirectoryArtifact` 和每-attempt Agent 路径已删除。
- 验收只使用合成数据和 mock Agent；未调用真实模型 API，也未读取历史实验作为规则来源。

## 1. 背景

当前系统已经能够完成 reference-guided 数据清洗主链路，但现有实现同时存在两层循环：

- 外层 workflow 负责 attempt、隐藏验证、best 晋升和停止。
- 内层 Agent 使用 ReAct 完成观察、写代码、运行工具和修复。

这会造成 Agent 的思考上下文在不同 attempt 之间割裂，任务计划、失败经验、已验证规则和当前 best 容易错位。此次重构基于 AgentScope 2.x，把整个 validation 生命周期收敛为一个持续的 Data Cleaning Agent ReAct 循环，同时保留宿主程序对隐藏数据、评分、晋升和安全边界的最终控制权。

本次重构必须在独立工作树和独立环境中完成，不修改当前已经可以运行的稳定版本。

## 2. 重构目标

1. validation 阶段只创建一个持续存在的 Data Cleaning Agent，由它在同一上下文中反复观察、计划、修改和提交候选。
2. 使用 AgentScope 2.x 原生 state、task、workspace、toolkit 和上下文压缩能力，减少重复实现。
3. 将“Agent 如何思考”和“宿主如何验收”分开：Agent负责发现规则和生成 Pipeline，宿主负责隔离执行、隐藏评分和 best 晋升。
4. 保留累计、可执行、可验证的唯一 Pipeline；失败候选不能污染 current best。
5. validation 冻结后，使用全新上下文的 Test Agent 执行冻结 Pipeline，严格隔离 test private reference。
6. 保持原 validation 行为可回归、可恢复；新架构稳定前不替换旧版本。

## 3. 数据角色

### 3.1 Train

- `train/raw` 和 `train/reference` 是公开示例，供 Agent 观察输入输出结构、推导清洗规则和做局部实验。
- Agent 可以主动在 train 上运行脚本、检查反例和验证假设。
- 宿主不要求 train 逐值 100% 复现，也不把 train 匹配率作为进入 validation 隐藏评分的硬门禁。
- train 检查结果是诊断证据，帮助 Agent 判断规则是否可靠，但不能代替 validation 的隐藏评估。

### 3.2 Validation

- `validation/raw` 和 `validation/keys` 对 Agent 可见。
- `validation/reference`、隐藏评分器和详细 gold 差异对 Agent 不可见。
- Agent 使用 validation raw 反复运行累计 Pipeline，并通过工具提交候选。
- 宿主检查候选完整性、隔离性和可重放性后进行隐藏评分。
- 只有分数严格高于 current best 的候选才晋升为正式 round。

### 3.3 Test

- test 只评估 validation 阶段冻结的 best Pipeline，不用于继续学习业务规则。
- Test Agent 只能处理入口、路径、环境和参数衔接，不能重写清洗规则。
- Test Agent 会话彻底结束后，宿主才读取 test private reference 并评分。

## 4. 总体架构

```text
Host Workflow
├── RunManifest / Checkpoint / Promotion Journal
├── Validation Data Cleaning Agent（单实例、持续 ReAct）
│   ├── AgentState + tasks_context
│   ├── LocalWorkspace + context/tool-result offload
│   ├── Rule Ledger
│   └── Restricted Toolkit
│       ├── Read / Search / Inspect
│       ├── Restricted Python / Pipeline execution
│       ├── TaskCreate / TaskGet / TaskList / TaskUpdate
│       ├── ValidateDraft
│       └── SubmitCandidate
├── Validation Gate + Hidden Evaluator + Best Promotion
└── Fresh Test Agent + Test Gate + Private Evaluator
```

核心原则是：只有 Agent 自己的观察、计划、代码修改和公开诊断位于 ReAct 内；隐藏 reference、评分、晋升、回滚和冻结始终由宿主掌握。

## 5. Validation 的单 Agent 大 ReAct

validation 启动后只实例化一次 Data Cleaning Agent。完整生命周期如下：

1. 宿主创建实验 manifest、AgentState、LocalWorkspace、受限 toolkit 和初始 Pipeline。
2. Agent 观察 train 示例、validation raw 的结构和当前 Pipeline。
3. Agent 使用 Task 工具建立 3 至 5 个短期任务，并将一个主要任务设为 `in_progress`。
4. Agent读取 `rule_ledger.json`，确认已验证规则、反例、未解决问题和当前 best 的反馈。
5. Agent修改 `workspace/pipeline`，并在受限环境中运行局部诊断或完整 Pipeline。
6. Agent调用 `ValidateDraft` 做预提交检查。
7. 如果预检查失败，工具结果直接返回同一 Agent；Agent继续分析和修复，不产生 attempt。
8. 最近一次 `ValidateDraft` 通过且 Pipeline 哈希没有变化后，Agent调用 `SubmitCandidate`。
9. `SubmitCandidate` 暂停 Agent，宿主接管候选快照、正式门禁和隐藏验证评分。
10. 宿主把结果封装为 `ExternalExecutionResultEvent` 返回同一个 Agent。
11. Agent根据结果更新 Task 和 rule ledger，并开始下一次思考。
12. 候选晋升时，宿主更新 current best；无效、持平或下降时恢复 current best Pipeline，但保留失败经验。
13. 达到停止条件或第一次 `Ctrl+C` 后冻结正式 best，结束 validation Agent，再进入 test。

定义保持明确：

- `attempt`：一次通过预检查后调用 `SubmitCandidate` 的候选提交。
- 正式 `round/loop`：候选隐藏分数严格超过 best 并完成原子晋升。
- `--max-iters`：整个 validation Agent 生命周期的 ReAct 总迭代上限，不是每个 attempt 单独重置。

## 6. Agent 状态

### 6.1 AgentState

使用 AgentScope 2.x `AgentState` 保存模型可恢复状态，包括：

- 对话消息和压缩摘要引用。
- `tasks_context` 中的轻量任务。
- 当前 session、workspace 和工具调用状态。
- 当前 best 标识和最近一次宿主反馈的脱敏摘要。

宿主定期将其持久化为 `agent_state.json`。恢复时必须校验 run manifest、数据 key 哈希、prompt 哈希和 scorer 版本，避免把其他实验状态导入当前实验。

### 6.2 Rule Ledger

`rule_ledger.json` 是长期业务认知，不是待办列表。它至少记录：

- 已验证规则及证据来源。
- 适用文件、字段和边界条件。
- 被否定的假设和反例。
- 当前 Pipeline 中承载该规则的模块。
- 尚未解决的问题。

Task 表示“下一步做什么”，rule ledger 表示“已经学到了什么”。两者不得互相替代。

### 6.3 Pipeline

`workspace/pipeline` 始终是当前可编辑的累计 Pipeline。正式候选未晋升时，宿主将其恢复到 current best 的文件内容；对应失败原因以事件和 ledger 记录保留，避免代码基线与经验基线错位。

## 7. 原生轻量任务计划

第一版直接将 AgentScope 2.x 原生工具加入 Data Cleaning Agent toolkit：

- `TaskCreate`
- `TaskGet`
- `TaskList`
- `TaskUpdate`

不创建独立 Planner Agent。任务状态保存在 `AgentState.tasks_context`，随 AgentState 一起 checkpoint。

约束：

- 活跃任务保持 3 至 5 个，避免任务列表本身消耗大量上下文。
- 同时只允许一个主要任务处于 `in_progress`。
- 收到宿主反馈后，先更新当前任务状态，再创建后续任务。
- 任务描述必须指向可验证动作，例如“修正 diag 映射并通过 ValidateDraft”，不能只写“继续优化”。
- 已完成任务可以压缩归档，结论同步进入 rule ledger。

## 8. ValidateDraft 与 SubmitCandidate

### 8.1 ValidateDraft

`ValidateDraft` 是自定义宿主工具，用于候选提交前的快速、无隐藏信息检查：

- 不创建 attempt。
- 不读取 validation/test private reference。
- 不计算隐藏分数。
- 不触发 best 晋升或 patience 计数。
- 结果作为普通工具观察返回同一个 Agent，Agent可以继续思考。

检查内容包括：

- Pipeline manifest、唯一入口和声明模块完整。
- 预期17个业务文件存在、可解析、schema 合法。
- validation key 覆盖、未知 key、主键和重复业务键诊断。
- Pipeline 无患者/`stay_id` 硬编码，无 private reference 和其他实验路径。
- 能在干净目录受限执行，输出确定且可重放。
- 当前 Pipeline 与最近一次通过检查的哈希一致。

### 8.2 SubmitCandidate

`SubmitCandidate` 是 Agent 与宿主验证循环之间的明确边界。调用后 Agent 暂停，宿主负责：

1. 生成不可变候选快照和 attempt 编号。
2. 再执行完整正式门禁。
3. 对有效候选调用隐藏 validation 评分器。
4. 根据严格提升规则决定是否晋升。
5. 原子更新 active best、round 归档和 promotion journal。
6. 恢复可编辑 Pipeline 到晋升后的 current best。
7. 返回结构化 `ExternalExecutionResultEvent`。

只有“最近一次 ValidateDraft 通过”且“通过后 Pipeline 哈希未变化”时才能调用 `SubmitCandidate`，从机制上避免预检查后继续改代码再提交。

## 9. 中间件设计

第一版只采用与主链路直接相关的中间件：

- **运行身份与审计中间件**：为每次模型调用、工具调用和宿主事件附加 experiment、session、attempt 和 trace id。
- **权限中间件**：在工具执行前校验读取根、写入根、命令类型和网络策略。
- **预算中间件**：限制 ReAct 迭代、工具超时、输出体积、生成文件数和资源使用。
- **状态持久化中间件**：在关键工具调用、候选提交和压缩后原子保存 AgentState 与 ledger。
- **敏感信息过滤中间件**：确保 private reference 路径、隐藏评分细节和凭证不会进入模型消息。

第一版明确不采用 AgentScope App 的 `ToolOffloadMiddleware`。该组件用于将运行过慢的工具转为后台任务，依赖 `BackgroundTaskManager`、`MessageBus` 和 App service。当前 validation 要求工具结果按顺序返回同一 ReAct，并且 Pipeline 存在唯一写入者；引入后台并发会增加提交竞态和恢复复杂度。

## 10. LocalWorkspace 与结果 Offload

Data Cleaning Agent 通过以下方式构造：

```python
agent = Agent(
    ...,
    offloader=LocalWorkspace(...),
)
```

这里的 offload 指“上下文和大型工具结果落盘”，不是“把慢工具放到后台运行”。

AgentScope 2.x LocalWorkspace 负责：

- 将被上下文压缩替换掉的完整历史写入 `workspace/sessions/<session_id>/context.jsonl`。
- 将超出消息承载范围的大型工具结果写入 `workspace/sessions/<session_id>/tool_result-<id>.txt`。
- 模型上下文只保留摘要、必要片段、文件路径和内容元数据。
- Agent需要细节时使用受限 `Read`/搜索工具定向读取文件，而不是把整份大结果重新灌回上下文。

应设置单条工具结果进入消息的阈值，并对 CSV、日志、diff 和运行报告优先返回结构化摘要及样本。offload 文件仍受 workspace 权限和实验生命周期管理，不能成为绕过 private reference 隔离的通道。

## 11. 受限执行

Agent不获得 unrestricted Bash。执行工具只允许运行 workspace 内的 Python 和 Pipeline，并采用显式策略：

### 只读范围

- 当前实验 `train/raw`、`train/reference`、`train/keys`。
- 当前实验 `validation/raw`、`validation/keys`。
- 当前项目的 `skills/` 和获准通用代码。
- current workspace 和由 LocalWorkspace 创建的 session 文件。

### 可写范围

- 当前 Agent workspace。
- 工具为本次运行分配的临时输出目录。
- 宿主指定的 draft/replay 目录。

### 禁止范围和操作

- validation/test private reference。
- test raw（validation 阶段）。
- `experiments/` 总目录和其他实验目录。
- 网络访问、凭证读取和任意 package 安装。
- 危险 subprocess、shell 拼接、后台进程和写入授权根之外的路径。

每次执行还要限制超时、stdout/stderr 大小、生成文件数量、磁盘占用和可用 CPU/内存。工具返回退出码、结构化统计、日志文件路径和少量末尾日志，完整日志通过 LocalWorkspace offload。

## 12. 上下文压缩

上下文管理同时保留“可恢复的完整历史”和“模型当前需要的信息”：

1. 使用模型对应 token counter，以上下文窗口占用比例触发语义压缩。
2. 保留最近若干轮原始消息、当前 Task、current best 反馈和最近一次工具结果。
3. 压缩摘要固定包含：任务目标、当前状态、已验证规则、反例、候选历史、关键路径、约束和下一步。
4. 完整历史由 LocalWorkspace 落盘，摘要中保留可定向读取的路径。
5. 压缩后校验 current best、Pipeline hash、进行中 Task 和关键约束没有丢失。
6. 压缩本身不能触发停止；停止只由正式 round、有效未提升、attempt 上限、target score、迭代预算或用户中断决定。

## 13. 候选反馈与晋升

`ExternalExecutionResultEvent` 至少包含：

- attempt id、candidate hash、base best hash。
- draft/正式门禁状态和机器可读失败原因。
- 是否完成隐藏评分、当前分数、best 分数和是否晋升。
- 脱敏的文件级公开反馈。
- current best round、Pipeline hash 和恢复结果。
- patience、attempt、round 和迭代预算余量。

无效候选不评分、不计 patience；有效但持平或下降的候选计入连续未提升；严格提升才形成正式 round。下一次思考始终从 current best Pipeline 和其对应反馈出发，上一个失败候选只作为失败经验附加。

## 14. Checkpoint、恢复与 Ctrl+C

- AgentState、Task、rule ledger、run manifest 和宿主 experiment state 使用临时文件、`fsync` 和 `os.replace` 原子写入。
- best 晋升使用 promotion journal，恢复时先完成或回滚未完成的目录切换。
- 恢复同一实验前必须验证不可变 manifest；不读取其他历史实验作为脚本或经验来源。
- 第一次 `Ctrl+C` 取消当前未提交工作，恢复最近一次正式 best，保存 checkpoint 并结束 validation Agent。
- 如果正在进行原子 promotion，先完成一致性恢复再 checkpoint。
- 第二次 `Ctrl+C` 立即终止整个流程，不运行 test private evaluation。
- 没有正式 best 时记录 `checkpointed_without_best`，不进入 test。

## 15. Test Agent

validation 冻结后，在同一 Python 进程内创建一个全新的 Data Cleaning Agent test 实例：

- 新 AgentState、新 model context、新 LocalWorkspace 和新 toolkit。
- 只读取 frozen Pipeline、脱敏 test contract、`test/raw` 和 `test/keys`。
- 只允许写 `runner_spec.json` 和 test workspace 运行记录。
- 不允许创建新的业务清洗脚本、修改 frozen Pipeline 或读取 validation 会话历史。
- 如果 Pipeline 不能在 test 上执行，报告 `pipeline_rule_failure`，不能就地重写规则。

Test Agent结束并清理模型上下文后，宿主验证 frozen Pipeline 哈希、运行统一业务门禁，再读取 test private reference 评分。test 失败不能反向修改 validation best。

## 16. 第一版不采用 Agent Team

第一版不引入 Agent Team，也不预留无实际用途的 Team CLI 开关。

原因：

- Team 属于 AgentScope App 的多 session、storage 和 message bus 协作体系，会显著扩大迁移范围。
- 多 Agent 同时修改 Pipeline 会破坏唯一写入者、父 hash 和候选 lineage。
- validation 隐藏评分已经提供了稳定的外部评审信号，当前瓶颈不是缺少多个意见角色。
- 第一版优先证明单 Agent 大 ReAct、状态恢复、受限执行和候选闭环稳定。

稳定后可以单独评估只读分析 worker：它只能分析公开数据并向 leader 返回建议，leader 仍是唯一 Pipeline 写入者和 `SubmitCandidate` 调用者。

## 17. 分阶段实施

### 阶段 0：基线固化

- 在新工作树记录旧版本关键行为和测试结果。
- 使用 mock model、mock Agent 和合成数据固化 attempt、晋升、patience、checkpoint 与 test 边界。
- 建立旧 CLI 到新内部组件的兼容表。

### 阶段 1：AgentScope 2.x 最小 Agent

- 接入模型、AgentState、Toolkit 和 LocalWorkspace。
- 实现单 session 的基础 ReAct 与状态保存/恢复。
- 暂不接隐藏验证和 test。

### 阶段 2：任务、ledger 与压缩

- 加入原生 Task 工具和任务数量约束。
- 实现 rule ledger 读写工具与原子持久化。
- 接入 token 语义压缩、context offload 和 tool-result offload。

### 阶段 3：受限工具和 ValidateDraft

- 实现精确 read/write roots、受限执行器和资源预算。
- 实现 `ValidateDraft`、Pipeline hash 绑定和合成门禁测试。
- 验证 private reference、其他实验和网络不可达。

### 阶段 4：SubmitCandidate 大循环

- 实现暂停、宿主接管和 `ExternalExecutionResultEvent` 回传。
- 接入正式 validation 门禁、隐藏评分、best 晋升、回滚和反馈。
- 确认所有 attempt 都由同一个 Agent 连续处理。

### 阶段 5：原子状态与中断恢复

- 实现原子 JSON、promotion journal 和 session checkpoint。
- 覆盖模型异常、工具超时、进程终止和第一次 `Ctrl+C`。

### 阶段 6：Test Agent

- 新建全新上下文 Test Agent 和精简 toolkit。
- 实现 runner spec、冻结 hash 验证、统一门禁和私有评分隔离。
- 接入自然停止和第一次 `Ctrl+C` 后自动 test。

### 阶段 7：兼容与真实 smoke test

- 保持现有主 CLI 参数，新增行为由内部适配层承接。
- 先使用全新合成实验完成端到端测试。
- 再使用新数据 split 做低 round/attempt smoke test。
- 新版本达到稳定性标准前，旧工作树保持可运行且不被修改。

## 18. 测试计划

1. 单一 validation Agent 在多个 attempt 之间保持同一 session 和任务状态。
2. Task 工具最多维护 3 至 5 个活跃任务，且只有一个主要任务 `in_progress`。
3. rule ledger 经压缩和恢复后仍保留规则、反例和 Pipeline 模块映射。
4. `ValidateDraft` 失败不产生 attempt、不评分、不计 patience。
5. `ValidateDraft` 通过后修改 Pipeline，`SubmitCandidate` 必须拒绝 hash 不一致。
6. `SubmitCandidate` 后 Agent暂停，并能根据宿主事件继续 ReAct。
7. 无效、持平和下降候选不会替换 best；失败后代码恢复 best，失败经验仍保留。
8. train 回归不完全时仍可提交 validation，但诊断会返回 Agent。
9. 受限执行无法读取 private reference、test、其他实验、凭证或网络。
10. 超大日志和工具结果落盘，模型消息只包含摘要和可读路径。
11. context offload 可定向读取，且压缩不会丢失 current best、Task 和 hash。
12. promotion 临界区中断后可通过 journal 恢复一致状态。
13. 第一次 `Ctrl+C` 从最近正式 best 进入 test；第二次终止整个流程。
14. Test Agent 是全新状态，只能写 runner spec，不能改 frozen Pipeline。
15. test 门禁或评分失败不会修改 validation best。
16. 不启用 Agent Team 和 App `ToolOffloadMiddleware` 时，主链路可完整运行。
17. 旧 CLI 的关键参数、实验隔离和评分口径保持兼容。

## 19. 验收标准

- 同一次 validation 运行中只有一个持续的 Data Cleaning Agent session。
- 每个候选都经过 `ValidateDraft -> SubmitCandidate -> Host Evaluation Event`。
- Agent可根据每次工具和宿主返回继续思考，任务、规则和代码基线不再错位。
- current best 的 Pipeline、反馈、hash 和正式 round 在任意恢复点一致。
- 大型上下文和工具结果可审计地落盘，不挤占模型主要上下文。
- Agent不能通过工具、offload 文件或恢复机制接触任何 private reference。
- validation 和 test 端到端通过合成测试，旧稳定版本仍可独立运行。
