# ceshi1 三维度横向对比 —— **新增的 8 个结果目录**（2026-11-25 补齐）

> 本文件是 `00_三维度横向对比.md` / `00_紧凑横向对比.md` / `00_评估进度表.md` 的**增量**：
> 那三张表是既有手写文件（未改动），这里给出补齐 n18 / I48 后**新增的行**。
> 数字全部从 JSON 读出、无手抄；composite 用 harness 自己的打分器重打，
> 错误级用 `score_only/error_metrics/`，原生层用 `reproduction/eval_*_native_cells.py`。

## 维度一：整体分数（composite）

| 结果目录 | 跑者 | 档位 | 数据集 | composite | schema | key | cell | exact |
|---|---|---|---|---:|---:|---:|---:|---:|
| `n18/1.6pct_我们的agent_pi+val_mimicraw` | 我们 | 1.6pct | mimicraw | **0.918531** | 1.0000 | 0.9367 | 0.9267 | 0.8684 |
| `n18/1.6pct_cc基线_ClaudeCode` | cc | 1.6pct | raw(无原生层) | **0.904403** | 1.0000 | 0.9212 | 0.9111 | 0.8524 |
| `n18/4.7pct_我们的agent_pi+val_mimicraw` | 我们 | 4.7pct | mimicraw | **0.922014** | 1.0000 | 0.9561 | 0.9427 | 0.8457 |
| `n18/4.7pct_cc基线_ClaudeCode` | cc | 4.7pct | raw(无原生层) | **0.885828** | 1.0000 | 0.9006 | 0.8905 | 0.8317 |
| `n18/deepprep-less` | deepprep | 1.6pct | mimicraw | **0.203910** | 0.9426 | 0.1650 | 0.1396 | 0.0694 |
| `n18/deepprep-more` | deepprep | 4.7pct | mimicraw | **0.199036** | 0.9506 | 0.1513 | 0.1291 | 0.0736 |
| `I48/deepprep-less` | deepprep | 1.6pct | mimicraw | **0.401026** | 0.9231 | 0.4532 | 0.3994 | 0.1944 |
| `I48/deepprep-more` | deepprep | 4.7pct | mimicraw | **0.464301** | 0.9846 | 0.5189 | 0.4506 | 0.2728 |

## 维度二：错误级指标（str / oldsw / old / goldv29）

> 口径定义见 `五档七口径横向对比.md` §1.1：**oldsw = 数值比对**、**old = 纯字符串**（两者只差「①值比对」这一个开关）；`old` 一并给出便于对照。

| 结果目录 | 跑者 | 口径 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `n18/1.6pct_我们的agent_pi+val_mimicraw` | 我们 | **str** | **0.997021** | **0.770292** | 2,518,246 | 926,404 | 13,315 | 6,348 |
| `n18/1.6pct_我们的agent_pi+val_mimicraw` | 我们 | **oldsw** | **0.589662** | **0.582126** | 2,503,270 | 40,525 | 255 | 4,407,917 |
| `n18/1.6pct_我们的agent_pi+val_mimicraw` | 我们 | **old** | **0.588310** | **0.464639** | 1,867,550 | 664,543 | 11,957 | 4,407,917 |
| `n18/1.6pct_我们的agent_pi+val_mimicraw` | 我们 | **goldv29** | **0.998211** | **0.955918** | 3,284,567 | 173,114 | 284 | 6,348 |
| `n18/4.7pct_我们的agent_pi+val_mimicraw` | 我们 | **str** | **0.993245** | **0.813892** | 7,990,218 | 2,273,779 | 122,103 | 48,167 |
| `n18/4.7pct_我们的agent_pi+val_mimicraw` | 我们 | **oldsw** | **0.527422** | **0.524070** | 5,939,693 | 52,881 | 14,846 | 14,894,606 |
| `n18/4.7pct_我们的agent_pi+val_mimicraw` | 我们 | **old** | **0.524704** | **0.445464** | 4,686,547 | 1,247,773 | 73,100 | 14,894,606 |
| `n18/4.7pct_我们的agent_pi+val_mimicraw` | 我们 | **goldv29** | **0.997321** | **0.987942** | 10,248,038 | 119,396 | 18,666 | 48,167 |
| `n18/4.7pct_cc基线_ClaudeCode` | cc | **str** | **0.798627** | **0.457561** | 2,599,182 | 3,672,775 | 1,478,177 | 2,684,133 |
| `n18/4.7pct_cc基线_ClaudeCode` | cc | **oldsw** | **0.438065** | **0.357886** | 3,034,792 | 1,201,461 | 940,714 | 15,725,059 |
| `n18/4.7pct_cc基线_ClaudeCode` | cc | **old** | **0.435032** | **0.274950** | 1,783,935 | 2,394,094 | 998,938 | 15,725,059 |
| `n18/4.7pct_cc基线_ClaudeCode` | cc | **goldv29** | **0.803126** | **0.621452** | 4,389,472 | 1,963,799 | 1,396,863 | 2,684,133 |
| `n18/1.6pct_cc基线_ClaudeCode` | cc | **str** | **0.854190** | **0.552883** | 1,756,713 | 1,300,267 | 80,982 | 326,351 |
| `n18/1.6pct_cc基线_ClaudeCode` | cc | **oldsw** | **0.510105** | **0.450586** | 1,922,304 | 337,024 | 63,806 | 4,628,833 |
| `n18/1.6pct_cc基线_ClaudeCode` | cc | **old** | **0.508829** | **0.338894** | 1,287,142 | 960,949 | 75,043 | 4,628,833 |
| `n18/1.6pct_cc基线_ClaudeCode` | cc | **goldv29** | **0.855332** | **0.721691** | 2,491,173 | 577,800 | 68,989 | 326,351 |
| `n18/deepprep-less` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |
| `n18/deepprep-more` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |
| `I48/deepprep-less` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |
| `I48/deepprep-more` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |

> ⚠️ 四个 deepprep 目录的**错误级算不出来**，但坏法各不相同（都是 deepprep 产物自身的缺陷，
> 不是工具问题 —— 工具要按 key 列给每张表建 sqlite 索引，坏表直接建不出来）：
> | 目录 | 坏在哪 |
> |---|---|
> | `n18/deepprep-less` | `admissions.csv` 0 行；`icu/outputevents.csv` **缺 subject_id** 且 0 行 |
> | `n18/deepprep-more` | `admissions.csv` **缺 hadm_id**；`icu/{icustays,inputevents}.csv` 0 行；`icu/{chartevents,outputevents}.csv` **缺 subject_id** 且 0 行 |
> | `I48/deepprep-less` | `admissions.csv` 是 **1 字节空文件**（连表头都没有）⇒ 建表 SQL 语法错 `near ")"` |
> | `I48/deepprep-more` | 只坏 `icu/outputevents.csv`（**缺 charttime**），其余 6 张表都齐 |
> ⇒ 对照：K29 / ICU5000 的 deepprep `admissions.csv` 有 8,093 / 6,476 行且 key 齐 ⇒ 工具跑得通，
> 所以那两档在原表里有错误级数字。

## 维度三：MIMIC 原生错误格（string 口径）

| 结果目录 | 跑者 | 口径 | 总格数 | 修对 | 没动 | 缺行 | **修复率** |
|---|---|---|---:|---:|---:|---:|---:|
| `n18/1.6pct_我们的agent_pi+val_mimicraw` | 我们 | string | 715 | 711 | 0 | 3 | **0.994406** |
| `n18/4.7pct_我们的agent_pi+val_mimicraw` | 我们 | string | 586 | 578 | 0 | 7 | **0.986348** |
| `n18/4.7pct_cc基线_ClaudeCode` | cc | string | 586 | 0 | 444 | 142 | **0.000000** |
| `n18/1.6pct_cc基线_ClaudeCode` | cc | string | 715 | 23 | 626 | 26 | **0.032168** |
| `n18/deepprep-less` | deepprep | string | 715 | 0 | 0 | 715 | **0.000000** |
| `n18/deepprep-more` | deepprep | string | 586 | 0 | 0 | 586 | **0.000000** |
| `I48/deepprep-less` | deepprep | string | 1264 | 0 | 200 | 59 | **0.000000** |
| `I48/deepprep-more` | deepprep | string | 1238 | 0 | 785 | 217 | **0.000000** |

## 读表提醒（补齐部分）

1. **n18 的 1.6pct cc 基线是本轮新跑的**（跑在 mimicraw 1.6pct 上，
   数据集切片 `datasets/注入错误数据集/n18_codex_16/` + prompt
   `~/Downloads/codexfuxian/n18_codex_16_prompt.md`）。
   ⚠️ 早先 `agent_work/n18_less_cc_v1` 那次 cc 跑批**没有入库**：它跑的是
   `n18_less` 切片（已从盘上删除），cohort **7,445 行** ≠ 当前 mimicraw 的 **7,048 行**
   ⇒ 另一个数据集构建，不可比。`n18_more_v1`（4.7pct，cohort 7,048）才是可比的 ⇒ 已入库。

2. **codex 基线在 train 上就没做到「与 reference 一致」**（prompt 的硬要求）：
   独立核它的 `work/train_out` vs 切片 `train/reference` ⇒ **31 个错 cell**，
   且集中在四列：`demographics.anchor_age` 错 9/10 行、`multimodal/ed_triage.pain`
   错 **18/18** 行、`admissions.insurance` 错 3 行、`icu/icustays.los` 错 1 行；
   另 `icu/chartevents.csv` 少 162 行、`icu/outputevents.csv` 少 2 行。

3. **cc 基线修不动 MIMIC 原生错误格**：n18 4.7pct 修复率 **0**（444 没动 / 142 缺行）、
   n18 1.6pct 修复率仅 **0.032168**（23 修对 / 626 没动 / 26 缺行）；
   对照我们的 agent：1.6pct **0.994406**、4.7pct **0.986348** ⇒ 差 30 倍以上。
   原因：规则推导法碰不到哨兵码与生理不可能值（原生错误类）。

4. **cc 基线的错误级 Det F1 也全面低于我们的 agent**（1.6pct：str 0.854 vs 0.997、
   goldv29 0.855 vs 0.998；Rep F1 0.553 vs 0.770、0.722 vs 0.956）⇒
   差距不只在原生层，整张表上 codex 都留下了更多脏数据
   （1.6pct str 口径 b_wrong 130 万 vs 我们 92.6 万）。

4. **n18 的 deepprep composite 只有 0.20**（I48 的 deepprep 是 0.40~0.46）——
   原因是 **key 几乎全错**（0.165 / 0.151 vs K29 的 0.83）⇒ deepprep **没做 cohort 筛选**，
   而 n18/i48 的输出表是**筛选之后**的表 ⇒ 结构上就对不上。

