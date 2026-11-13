# score_only —— 独立评分包

让**任何一个 AI**（codex / 豆包 / workbuddy / 另一个 Claude）给一个 data-cleaning
agent 的产出打分，而不需要它理解本项目的代码。

**核心就一件事：拿结果包跟 gold 逐文件比对。** 本包把这件事包成一条命令。

---

## 快速开始（Raha composite，总分）

需要三样东西：**结果包目录**、**gold 目录**、**manifest**。

```bash
bash run_raha.sh \
  <结果包目录> \
  <gold 目录> \
  manifests/<manifest>.json \
  <输出json>
```

例（N18 1.6pct）：

```bash
bash run_raha.sh \
  /Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments/n18_1p6_test_seed666_v1/host/test_result_package \
  "/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_test1.6pct_490M/test/reference" \
  manifests/n18_ckd_ED_notes_v1.json \
  /tmp/score.json
```

### ⚠️ manifest 必须挑对

| 数据集 | manifest | 输出文件数 | 切分键 |
|---|---|---:|---|
| icu5000 | `manifests/icu_5000_v1.json` | 13 | stay_id |
| **N18** | `manifests/n18_ckd_ED_notes_v1.json` | **18** | subject_id |

主键定义完全不同（N18 的 labs key 含 `valuenum`/`valueuom`、outputevents 含 `value`，
还有 6 个 ED/文本输出）。**用错 manifest 会全盘对不上。**

---

## 加分项：错误级指标（修了多少 / 改错多少 / 丢行 / 误改）

额外需要**完整数据集目录**（里面有 `test/modification_log.csv` = 注入日志）。
本机跑才现实（N18 4.7pct 的 log 有 4.59 GB）。

```bash
bash run_error_metrics.sh \
  <完整数据集目录> \
  <结果包目录> \
  <ds_key> \
  <输出json> \
  <日志路径>
```

⚠️ **口径**（v29）：`recover_clean_key=true` + `exclude_rows_not_in_gold=true`。
详见 `评测规范.md` §4 —— 口径不一致时分数不可比。

---

## 环境要求

* Python 3.9+ 和 pandas（就这两个）
* 不需要 conda、不需要项目代码、不需要网络
* macOS / Linux 都可以

先跑一遍自检确认环境没问题：

```bash
bash selftest.sh
```

---

## 包里有什么

| 文件 | 作用 |
|---|---|
| **`评测规范.md`** | **主体** —— 评什么、怎么评、口径规矩、诊断表、基线。给另一个 AI 读这份 |
| `多AI交叉评估规程.md` | 多个评分方并行时的角色分工、盲评要求、分歧归因 |
| `run_raha.sh` | 一条命令 → composite + 逐文件分数 |
| `run_error_metrics.sh` | 一条命令 → 错误级指标明细 |
| `selftest.sh` | 环境自检（6 项） |
| `pi_test_scorer.py` | Raha 评分器（也可以直接调） |
| `error_metrics/` | 错误级指标的建缓存 + 逐 cell 记账 |
| `workflow/` | 评分器依赖的模块（自带，不引用项目代码） |
| `manifests/` | 各数据集的主键定义 |
| `总表.md` | 历史结果总表（自检参照） |

---

## 交给另一个 AI 的话（可直接贴）

> 你要评测一个 data-cleaning agent 的产出。请严格按 `score_only/评测规范.md` 执行：
>
> 1. 跑 §2.2 拿 Raha composite（一���命令 `bash run_raha.sh <结果包> <gold> manifests/<manifest>.json out.json`），
>    跑 §2.3 拿错误级指标（4.7pct 约 1.5h，放后台）
> 2. 按 §6 的 JSON 格式输出，并附 3-5 句人话摘要
> 3. 按 §7 对照基线自检，异常就按 §5 的诊断表查
> 4. **不要修改评分代码**（§9）
> 5. 遇到 §8 的故障先查表再动手
>
> 关键：**manifest 要挑对**（icu5000 → `icu_5000_v1.json`；N18 → `n18_ckd_ED_notes_v1.json`）。
> 如果报出的 composite 明显偏低，先看是不是 manifest 用错了。

---

## 已知边界

* Raha composite 与口径无关（由 gold 算出，不看注入日志）→ **任何 agent 跑出来都应该完全一致**，
  这是交叉验证的第一道关。
* 错误级指标依赖注入日志和口径 → 不同 agent 跑可能有差异，差异先按 §4 查口径，再按 §4.2 归因。
* 某些文件谁都修不回来（outputevents 的 charttime、demographics 的 anchor_age）
  —— 是数据的信息上限，不是 agent 水平。
