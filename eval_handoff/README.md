# Raha 评测交付手册

> **给接手的 AI / 工程师**：本文档描述如何独立完成一次「data-cleaning agent 在
> 错误注入数据集上的评测」。目标是不需要原开发者介入，就能跑出可与历史结果横向
> 比较的数字。
>
> 本文档基于 2026-10-24 真实跑通的 icu5000 1.6pct 与 4.7pct 两轮整理。
> 文中所有命令都验证过。

---

## 0. TL;DR — 五分钟看懂

我们要评的东西：**一个 agent 拿到「注入了错误的 MIMIC 原始数据」，能不能清洗干净、
按指定格式产出结果包**。

评测分三段（**这是整个流程最关键的设计**）：

```
①  val 阶段   agent 拿着 train(小样本) + validation(小样本) 反复迭代
               每回合 host 打分并把分数反馈给 agent，agent 据此改规则
               ↑ 这一段有 LLM，是整个流程唯一的「训练」环节
               ↓
②  test 阶段  把 val 阶段冻结下来的 pipeline 原封不动重放到 test 上
               ↑ 这一段没有 LLM，纯确定性重放。test 只被跑一次
               ↓
③  打分       Raha composite（总分）+ 错误级指标（细分明细）
```

**为什么这样切**：agent 从头到尾**没有见过 test**，test 是一次性的、无 LLM 参与的
确定性评测。这样 test 分数不掺任何「看着答案改」的水分。

**一次完整评测 ≈ 5-5.5 小时**（详见 §7 耗时）。

---

## 1. 术语与目录约定

### 1.1 关键角色

| 名字 | 是什么 | 谁看得到 |
|---|---|---|
| **raw** | 注入了错误的原始数据（hosp/ icu/ ecg/ note/ 四棵子目录，28 张表） | agent |
| **reference** | 干净的金标准（13 个结果 csv） | agent 看 train/validation 的；**validation 的会被 harness 隐藏** |
| **modification_log.csv** | 注入日志：每个被注入 cell 的 clean_value(真值) / dirty_value / error_subtype | **只有 host 能看。这是答案。** |
| **keys.csv** | 该 split 的 stay_id 列表（切分键） | agent |
| **结果包** | agent 产出的 13 个 csv + submission.json | agent 产出，host 打分 |

### 1.2 数据集目录约定

```
datasets/注入错误数据集/
  icu_5000_v2_rawerror_test1.6pct_614M/     ← 完整数据集（含 test/reference + 全部 modification_log）
    ├── v2_rawerror_icu5000_final_report.json   ← 注入统计报告（host 侧用）
    ├── train/{raw,reference}/  + keys.csv
    ├── validation/{raw,reference}/ + keys.csv
    └── test/{raw,reference}/ + keys.csv + modification_log.csv   ← ⚠️ 含答案

  icu5000_more1/                              ← 扣留视图（codex 侧用，test 无 reference）
  icu5000_less1/                              ← 扣留视图

datasets/val_views/                           ← ⭐ agent 侧唯一允许碰的目录
  icu5000_less1/{train,validation}/           ← 密封视图：无 test/、无 modification_log
  icu5000_more1/{train,validation}/
```

---

## 2. ⚠️ 五个必须知道的坑

> 这些都是踩过的。每个都能让一整轮跑批白费几小时。

### 坑 1：绝不能让 agent 看到 modification_log.csv

`datasets/注入错误数据集/<完整数据集>/test/modification_log.csv` 里是**每个被注入
cell 的 clean_value 真值** —— 等于直接给答案。

**为什么危险**：val prompt 明确允许 agent「枚举 train/raw 与 validation/raw 的
公共父目录」。如果 `--train-raw` 指向完整数据集目录，agent 第一轮就会去读这个日志。

**实测**：2026-10-24 第一次跑 icu5000 就中招了，agent 第一轮就在打印
`SUBTYPE: label_leakage_feature ... clean_value ... dirty_value`，被紧急 kill。

**正确做法**：给 agent 一个**独立父目录下的密封视图**：

```
datasets/val_views/<name>/          ← 这个父目录下只放密封视图
  ├── train/{keys.csv,raw/,reference/}
  └── validation/{keys.csv,raw/,reference/}
```

- 不含 `test/`
- 不含任何 `modification_log.csv`
- 父目录 `datasets/val_views/` 里**只有**密封视图，没有完整数据集

**验证**：`grep -r modification_log datasets/val_views/` 必须无输出。

### 坑 2：`--train-raw` 的父目录会被整体放行

harness 的 sandbox profile 里：

```
(allow file-read* (literal "<train_raw 的父目录>") (subpath "<train_raw 的父目录>"))
```

即 **train split 整个目录递归可读**（含 train/reference，这是对的，agent 要拿它自检）。
但如果 `train/` 里有 modification_log.csv，agent 就能读到 —— 这就是坑 1 的机制。

### 坑 3：validation 的 gold 会被隐藏，但 train 的不会

- `train/reference` → agent **可以读**（这是它推导规则的「少量标准示例」）
- `validation/reference` → agent **读不到**（sandbox 只放行 `validation/raw`）

harness 会把 validation gold 的路径 redact 成 `<hidden_gold>`。**不要试图去找它**，
那属于违规且会被 harness 的隔离检查拒绝。

### 坑 4：replay 契约只注入 4 个占位符

harness 重放时**只**注入这些：

| 占位符 | val 阶段 | test 阶段 |
|---|---|---|
| `{raw_root}` | validation/raw | **test/raw** |
| `{train_reference}` | train/reference | train/reference |
| `{output_dir}` | 结果包目录 | 结果包目录 |
| `{workdir}` | 工作目录 | 工作目录 |
| `{train_raw}` | train/raw | **不存在！** |

**用 `{train_raw}` 会在 test 重放时直接失败**：
`placeholder unavailable in this replay context: train_raw`。

另外 `{output_dir}` 必须**直接就是结果包目录**。如果 agent 的 pipeline 写成
`{output_dir}/result/xxx.csv`，重放后 scorer 找不到文件 —— 2026-10-24 的 v1 就是
这么栽的（重放得分 0.497189，而正确写法是 0.892868）。

**交付前必须验**：

```bash
python3 -c "
import json
d = json.load(open('<snapshot>/submission.json'))
print('result_root:', d['result_root'])
print('argv:', d['replay']['argv'])
"
# result_root 相对 workdir 解析后，那 13 个 csv 必须都在里面
```

### 坑 5：`_prepare()` 要求实验目录必须是全新的空目录

harness **不支持 resume**。实验目录已存在会直接报
`experiment directory must be new and empty`。

- 每次重跑要换一个新的实验目录名
- 想续跑只能改 harness 源码（`agent/pi_harness.py`，约 40 行），
  且有不可逾越的限制：worker 进程内的对话上下文无法恢复

---

## 3. 三条 pi 入口的语义（极易混淆）

| 入口 | 作用 | 有 LLM？ | 读 test gold？ |
|---|---|---|---|
| `agent.pi_cli` | one-shot 跑一次 | ✅ | 可能（**不要用，口径不可比**） |
| `agent.pi_harness_cli` | **val 阶段**：多回合 + host 反馈 | ✅ | ❌ |
| `agent.pi_test_harness_cli` | **test 阶段**：重放冻结 pipeline + 打分 | ❌ | ✅（host 侧） |
| `agent.pi_test_scorer` | 单独给一个结果包打分 | ❌ | ✅（host 侧） |

**只有后两个能用于本文档描述的评测流程。**

---

## 4. 执行流程（照着做）

### Step 0 — 环境

```bash
# conda 环境名
conda run -n py311agent2 --no-capture-output <cmd>
# 项目根
export PROJ=/Users/mac/PycharmProjects/v2_clean
# 必须 cd /tmp —— worktree 里有个叫 agent 的包会遮蔽真正的 agent 包
cd /tmp
```

模型配置：`model_config.yaml` / `model_config.local.yaml`（含 API key）。
**不要把 key 写进任何交付文件。**

### Step 1 — 建密封 val 视图

用交付包里的 `scripts/build_sealed_view.py`：

```bash
python3 scripts/build_sealed_view.py \
  --source "$PROJ/datasets/注入错误数据集/icu5000_more1" \
  --dest   "$PROJ/datasets/val_views/icu5000_more1"
```

它会：拷 train+validation 的 raw/reference/keys.csv → 逐字节校验 → 复核无
modification_log、无 test/。

**注意**：父目录 `datasets/val_views/` 下只放密封视图。

### Step 2 — 写 val prompt

模板见 `prompts/`。**关键是要写对三个数字**，写错 agent 就会算错工作量：

1. train / validation / test 各自的错误条数与 cell 错误率
2. 三个阶段的规模倍数
3. **错误密度在各阶段的变化方向**（见 §6 的「密度反转」）

数字从数据集的 `v2_rawerror_icu5000_final_report.json` 里读。

### Step 3 — 跑 val harness

```bash
export PROJ=/Users/mac/PycharmProjects/v2_clean
export RUN=$PROJ/<你的工作目录>/k29_agent_run
export DS=$PROJ/datasets/val_views/icu5000_more1
export FULL=$PROJ/datasets/注入错误数据集/icu_5000_v2_rawerror_test4.7pct_614M
export EXP=$RUN/experiments/icu5000_4p7_val_seed666_v1

cd /tmp
conda run -n py311agent2 --no-capture-output env PYTHONPATH="$PROJ" \
  python -m agent.pi_harness_cli \
    --experiment-dir "$EXP" \
    --train-raw "$DS/train/raw" \
    --train-reference "$DS/train/reference" \
    --validation-raw "$DS/validation/raw" \
    --validation-gold "$DS/validation/reference" \
    --evaluation-manifest "$PROJ/evaluation_manifests/icu_5000_v1.json" \
    --dataset-manifest "$FULL/v2_rawerror_icu5000_final_report.json" \
    --prompt-file "$RUN/prompts/icu5000_4.7pct_val_prompt_v1.md" \
    --max-rounds 20 \
    --patience 3 \
    --max-iters 10000
```

**产物**（都在 `$EXP/host/` 下）：

| 文件 | 含义 |
|---|---|
| `rounds/round_*.json` | 每回合的分数 |
| `best_snapshot/` | 最好那版的 agent workdir |
| `reproducible_snapshot/` | 独立重放验证过的那版 ⭐ **test 阶段用这个** |
| `run_report.json` | 总报告（status / best_score / stop_reason） |
| `public_feedback/round_*.json` | 反馈给 agent 的内容 |

**正常结束的 status**：`SUCCESS_REPRODUCIBLE`。
`stop_reason` 可以是 `patience`（正常）、`target_score`（达标）、
`worker_crash_finalized`（worker 崩了但成果保住了）。

### Step 3.5 —（可选但推荐）容错驱动

harness 对 worker 的模型调用**零容错**，一次 ReadTimeout / 429 / 畸形 JSON 就整跑完蛋。
交付包里的 `scripts/run_val_harness_ft.py` 在 harness 外面加两层保护：

1. `run_turn` 失败自动重试（4 次，间隔 15 秒）
2. 兜底收尾：harness 抛异常时走它自己的收尾流程（独立重放 best_snapshot →
   可接受就冻结 reproducible_snapshot + 写 run_report.json），
   这样「崩在第 1 轮」也能留下前面的成果

```bash
python3 scripts/run_val_harness_ft.py more1 v1     # 参数：<profile> <实验后缀>
```

**实测**：不加固时连续两批全废（v1 死在 `'key_commands' is a required property`，
v2 死在 `ReadTimeout`）；加固后一次跑通 13 轮。

### Step 4 — 跑 test harness

```bash
export EXP=$RUN/experiments/icu5000_4p7_val_seed666_v1
export TEST_EXP=$RUN/experiments/icu5000_4p7_test_seed666_v1
export FULL=$PROJ/datasets/注入错误数据集/icu_5000_v2_rawerror_test4.7pct_614M

cd /tmp
conda run -n py311agent2 --no-capture-output env PYTHONPATH="$PROJ" \
  python -m agent.pi_test_harness_cli \
    --validation-experiment "$EXP" \
    --test-experiment "$TEST_EXP" \
    --test-raw "$FULL/test/raw" \
    --test-gold "$FULL/test/reference" \
    --evaluation-manifest "$PROJ/evaluation_manifests/icu_5000_v1.json" \
    --replay-timeout 7200 \
    --scoring-timeout 10800
```

它做三件事：
1. **strict preflight** —— 在 validation raw 上再跑一遍冻结的 pipeline，核对可复现性
   并生成 attestation（保证 test 只被跑一次）
2. 用四个占位符在 test raw 上**重放**
3. 用 `pi_test_scorer` 打分

**产物**：`$TEST_EXP/host/score_report.json`（含 composite_score）、
`$TEST_EXP/host/test_result_package/`（重放出的结果包）、
`$TEST_EXP/host/frozen_snapshot/`。

**超时**要按数据量调：4.7pct 的 test raw 是 5.30 GB，1.6pct 是 4.88 GB。
replay 7200s / scoring 10800s 是 4.7pct 实测够用的值。

### Step 5 — 错误级指标（可选，但强烈建议）

Raha composite 是**一个数**，看不出「哪类错没修」。错误级指标给的是明细。

```bash
bash scripts/score_error_metrics.sh \
  <完整数据集目录> <结果包目录> <ds_key> <输出json>
```

它内部两步：
1. `reproduction/build_eval_cache.py` —— 把结果包与 gold 建成可逐 cell 比对的缓存
2. `reproduction/compute_error_metrics.py` —— 逐 cell 分类记账

**耗时**：4.7pct 约 1.5 小时（扫 3,200 万条注入日志 + 逐 cell 比对）。纯 CPU，
跟 agent 无关。**可以放后台跑，不阻塞总分结论。**

---

## 5. 指标定义

### 5.1 Raha composite（总分）

13 个文件**等权**（每文件 1/13），文件内四项加权：

```
file_score = 0.1×schema_f1 + 0.2×key_structure_f1 + 0.4×row_aligned_cell_f1 + 0.3×exact_row_f1
composite  = mean(13 个 file_score)
```

- `schema_f1` —— 列名/列数匹配
- `key_structure_f1` —— 主键结构匹配
- `row_aligned_cell_f1` —— **按主键对齐后逐 cell 的 F1**（权重最大）
- `exact_row_f1` —— 完全一致的行占比

`key_columns` 由 `evaluation_manifests/<task>_v1.json` 定义。

### 5.2 错误级指标

**检测（detection）**：注入的 cell 里，agent 改对了多少？
- `P` = 改对的 / (改对的 + 误改干净格的)
- `R` = 改对的 / 全部注入 cell
- `F1`

**修复（repair）**：改对了的 cell 里，有多少改得**完全正确**？
（值对但精度/格式不同也算错）

**cell 分桶**：

| 桶 | 含义 |
|---|---|
| `b_clean` | 干净 cell，agent 改对了 |
| `b_dirty` | 脏 cell，agent 没动（漏检） |
| `b_wrong` | 脏 cell，agent 改了但改错了 |
| `b_missing` | 整行丢了（key 对不上） |
| `fp_det` | 干净 cell 被误改（假阳性） |
| `dup_removed` / `dup_kept` | 重复行：清掉了 / 留下了 |
| `leak_removed` / `leak_kept` | 标签泄露行：清掉了 / 留下了 |
| `crow_recovered` | key 被破坏但用干净 key 救回的行 |
| `natural_noise` | raw↔gold 的天然落差（**已从 P/R 分母剔除**） |

### 5.3 ⚠️ 两种口径 —— 绝对不能混用

| 口径 | 开关 | 行为 |
|---|---|---|
| **新口径** | 默认（`recover_clean_key=true`） | key 列被注入破坏时，先把该行 key 还原成干净值再定位，能救回这一行所有非 key cell |
| 旧口径 | `--no-recover-clean-key` | 不还原，key 被破坏的行**整行记 b_missing** |

**实测差距（icu5000 1.6pct，我们的 agent）**：

| | b_missing | Det F1 | Rep F1 |
|---|---:|---:|---:|
| 新口径 | 8,972 | 0.996719 | 0.851930 |
| 旧口径 | **2,826,200** | 0.829509 | 0.733281 |

**旧口径下的掉分不是 agent 变差，是记账 artifact**（把「key 被注入破坏」误记成
「agent 弄丢了一整行」）。

**规矩**：
- **对外只报新口径**（`recover_clean_key: true`）
- 旧口径只用于跟更早的历史 run 横比
- **比较双方时必须确认口径一致** —— 每个 json 里有 `recover_clean_key` 字段，
  报之前先看这个字段

---

## 6. ⚠️ 「密度反转」—— 4.7pct 上我们输给 codex 的根因

这是本项目最重要的一条经验，务必读。

### 现象

| 档位 | val composite | test composite | 落差 |
|---|---:|---:|---:|
| 1.6pct | 0.903785 | 0.897831 | −0.006 |
| 4.7pct | **0.915858**（更高！） | 0.867853 | **−0.048** |

**val 分数更高，test 分数更低** —— 说明 agent 在 val 上过拟合了。

### 根因

| 档位 | train 错误率 | validation 错误率 | test 错误率 |
|---|---:|---:|---:|
| 1.6pct | 1.30% | 2.24% | 1.58% |
| 4.7pct | **2.34%** | **0.89%** | **4.72%** |

4.7pct 的 **validation 错误率只有 0.89%，test 是 4.72%，差 5.3 倍**。
1.6pct 的 val(2.24%) 和 test(1.58%) 基本对齐。

**val 循环的反馈信号密度远低于 test**，agent 在 val 上校准出的规则天然偏向
「少修、不动」，到 test 上就大面积失准。

### 具体表现

4.7pct 上我们（agent）vs codex 逐文件：

| 文件 | codex | agent | 差 |
|---|---:|---:|---:|
| admissions | 0.6640 | 0.8024 | **+0.1384** ✅ |
| chartevents（3,300 万行） | 0.9988 | 0.9979 | −0.0009 |
| **inputevents** | 0.8932 | 0.7971 | **−0.0961** ❌ |
| **labs** | 0.9438 | 0.8743 | **−0.0695** ❌ |
| **medications** | 0.9573 | 0.8951 | **−0.0622** ❌ |
| **procedures** | 0.9622 | 0.9029 | **−0.0593** ❌ |
| **diagnoses** | 0.9738 | 0.9310 | **−0.0427** ❌ |
| cohort | 0.9819 | 0.9463 | −0.0356 ❌ |

**规律**：输的都是**靠聚合/筛选规则从 raw 切出来的大表**，赢的是「保住行数」的表。
**chartevents（纯逐 cell 修复，3,300 万行）几乎打平** —— 说明 agent 的
逐 cell 修复规则学得很好，**输的是行级筛选规则**。

### 结论

- 这是**评测设计的缺口**，不是 agent 能力不足
- 修法：让 val 阶段的错误密度与 test 对齐（在 sealed view 里重新注入一个
  密度匹配的 validation 切片）
- **但绝不能靠调 val 参数去过拟合 test** —— 那只是把 overfitting 从模型挪到评测里

### prompt 里的对应写法

在 val prompt 里显式提醒 agent 密度差异（4.7pct prompt 里的原文）：

```
- **注意 train 和 validation 的错误密度是反的**：train 的 cell 错误率（2.34%）
  比 validation（0.89%）**高**，但 validation 的行数更多、覆盖的 stay 更多。
  这说明错误密度和样本规模是两件独立的事，**不要用 validation 上"每行错误数少"
  来推断测试集会更容易** —— 测试集 cell 错误率是 4.72%，比 train 还高近一倍。
```

---

## 7. 耗时（4.7pct 实测）

| 阶段 | 耗时 | 备注 |
|---|---:|---|
| 准备（密封视图 + prompt + 脚本） | 0.5 h | 一次性，第二个数据集可复用模板 |
| **val harness** | **3.2 h** | 最不可预测：13 轮。不加固容错可能翻倍 |
| test 重放 + Raha 打分 | 0.4 h | 纯重放，最稳的一段 |
| 错误级指标 | 1.5 h | 纯 CPU，可放后台 |
| **合计** | **5-5.5 h** | |

加上 codex 对照（已有产出只补打分）：**+1.5 h**。

**压缩时间的三个点**：
1. 错误级指标可延后 —— 不影响总分结论
2. test 重放不必等 val 跑满 patience
3. 多数据集可并行，但内存是瓶颈（16 GB，test 重放峰值 ~5 GB），最多同时 2 个

---

## 8. 交付包内容

```
eval_handoff/
  README.md                          ← 本文档
  scripts/
    build_sealed_view.py             ← 建密封 val 视图（含逐字节校验 + 泄漏复核）
    run_val_harness_ft.py            ← 容错 val harness 驱动（重试 + 兜底收尾）
    run_val_harness.sh               ← val 阶段启动脚本
    run_test_harness.sh              ← test 阶段启动脚本
    score_error_metrics.sh           ← 错误级指标
    final_table.py                   ← 出最终对照表
    selftest.sh                      ← 交付包自检（语法/文件齐全/无硬编码路径/无密钥）
  prompts/
    icu5000_4.7pct_val_prompt_v1.md  ← 4.7pct val prompt 范例
    icu5000_1.6pct_val_prompt_v1.md  ← 1.6pct val prompt 范例
  reference/
    compute_error_metrics.py         ← 错误级指标核心（必须用这个，别重写）
    build_eval_cache.py              ← eval cache 构建
    icu_5000_v1.json                 ← evaluation manifest 范例
    依赖清单.md                       ← ⚠️ 换机器必读：git 里缺的 13 个核心文件
    已知结果.md                       ← 历史基线数字，用来校验新结果合理性
```

**自检**：

```bash
bash eval_handoff/scripts/selftest.sh
```

**⚠️ 重要**：`agent/pi_harness*.py`、`workflow/pi_harness_evaluation.py`、
`evaluation_manifests/` 这些文件**不在 git 版本控制里**（仓库只跟踪了 125 个文件）。
所以本交付包是**实体文件交付**，不是「clone 仓库就能跑」。
换机器时必须按 `reference/依赖清单.md` 一并拷贝那 13 个文件。

**所有脚本的路径都用环境变量可覆盖**，默认指向本机：

```bash
export PROJ=/path/to/v2_clean    # 项目根
export RUN=$PROJ/k29_agent_run    # agent run 目录
export DS=$PROJ/datasets/val_views/icu5000_more1
export FULL=$PROJ/datasets/注入错误数据集/icu_5000_v2_rawerror_test4.7pct_614M
export EXP=$RUN/experiments/icu5000_4p7_val_seed666_v1
```

不设就用脚本里的默认值（跟原环境一致）。

---

## 9. 历史基线（用来校验新结果是否合理）

### icu5000 1.6pct (less)

| | codex | 我们的 agent |
|---|---:|---:|
| Raha composite | 0.896692 | **0.897831** |
| Det F1 | 0.980339 | **0.996719** |
| Rep F1 | 0.800252 | **0.851930** |

val composite 0.903785（6 轮，worker 崩了兜底）

### icu5000 4.7pct (more)

| | codex | 我们的 agent |
|---|---:|---:|
| Raha composite | **0.885118** | 0.867853 |
| Det F1 | **0.996362** | 0.995987 |
| Rep F1 | **0.829340** | 0.824043 |

val composite 0.915858（13 轮，正常 patience 停）

**新结果落在这些区间附近才算合理**。差太远先查：
1. replay 契约写对了吗（`{output_dir}` 有没有多套一层 `result/`）
2. 口径对吗（`recover_clean_key` 字段）
3. val 视图密封吗（有没有泄漏）
4. prompt 里的量级数字对吗

---

## 10. 禁止事项（写给接手的人）

| ❌ 不要 | 原因 |
|---|---|
| 硬编码任何 subject_id / hadm_id / stay_id / note_id / 行号 | 换数据集就废，且是作弊 |
| 让 agent 读 `pi_test_scorer.py` / `evaluation_manifest.json` | 评分元数据 |
| 让 agent 读 test gold / validation gold | harness 有隔离检查，违规 |
| 读 `modification_log.csv` 给 agent | 那是答案 |
| 改 `agent/` 下的评测代码 | 会让新旧结果口径不可比 |
| 把 API key 写进任何交付文件 | 安全 |
| 混用新旧口径报数字 | 见 §5.3 |

---

## 11. 快速排障

| 症状 | 原因 | 处理 |
|---|---|---|
| agent 第一轮就在读 modification_log | val 视图没密封 | 重建密封视图，父目录只放视图 |
| `placeholder unavailable in this replay context: train_raw` | replay.argv 用了 `{train_raw}` | 只用 4 个占位符 |
| test 重放得分 0.497 而 val 0.89 | pipeline 写进了 `{output_dir}/result/` | 13 个 csv 必须直接落在 `{output_dir}` |
| `experiment directory must be new and empty` | 实验目录已存在 | 换新目录名 |
| `ambiguous_file` 报错 | 结果包里有重名文件 | 检查 result_root 下的文件树 |
| val harness 跑几轮就死 | 模型 API 抖动/限流 | 用 `run_val_harness_ft.py` |
| 内存告警 / OOM | 16 GB 机器，test raw 5.3 GB | 关掉其他吃内存的进程；test 阶段峰值 ~5 GB |
| 错误级指标跑很久 | 正常，4.7pct 要 1.5 h | 放后台 |
| 分数明显偏离历史基线 | 见 §9 的四个排查点 | 逐个查 |

---

## 12. 附：给另一个 AI 的直接指令模板

> 你要在一个 MIMIC-IV 派生的错误注入数据集上评测一个 data-cleaning agent。
> 请严格按 `eval_handoff/README.md` 的流程执行，重点注意：
> 1. §2 的五个坑（尤其是 modification_log 泄漏和 replay 契约）
> 2. §5.3 的口径规矩
> 3. §6 的密度反转问题（不要在 val 上过拟合 test）
> 4. 遇到 §11 表里的症状先查表再动手
> 5. 只汇报进度，不要擅自 kill 跑批或修改评分代码
