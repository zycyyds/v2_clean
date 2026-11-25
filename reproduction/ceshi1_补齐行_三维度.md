# ceshi1 三维度横向对比 —— **新增的 7 个结果目录**（2026-11-25 补齐）

> 本文件是 `00_三维度横向对比.md` / `00_紧凑横向对比.md` / `00_评估进度表.md` 的**增量**：
> 那三张表是既有手写文件（未改动），这里给出补齐 n18 / I48 后**新增的行**。
> 数字全部从 JSON 读出、无手抄；composite 用 harness 自己的打分器重打，
> 错误级用 `score_only/error_metrics/`，原生层用 `reproduction/eval_*_native_cells.py`。

## 维度一：整体分数（composite）

| 结果目录 | 跑者 | 档位 | 数据集 | composite | schema | key | cell | exact |
|---|---|---|---|---:|---:|---:|---:|---:|
| `n18/1.6pct_我们的agent_pi+val_mimicraw` | 我们 | 1.6pct | mimicraw | **0.918531** | 1.0000 | 0.9367 | 0.9267 | 0.8684 |
| `n18/4.7pct_我们的agent_pi+val_mimicraw` | 我们 | 4.7pct | mimicraw | **0.922014** | 1.0000 | 0.9561 | 0.9427 | 0.8457 |
| `n18/4.7pct_cc基线_ClaudeCode` | cc | 4.7pct | raw(无原生层) | **0.885828** | 1.0000 | 0.9006 | 0.8905 | 0.8317 |
| `n18/deepprep-less` | deepprep | 1.6pct | mimicraw | **0.203910** | 0.9426 | 0.1650 | 0.1396 | 0.0694 |
| `n18/deepprep-more` | deepprep | 4.7pct | mimicraw | **0.199036** | 0.9506 | 0.1513 | 0.1291 | 0.0736 |
| `I48/deepprep-less` | deepprep | 1.6pct | mimicraw | **0.401026** | 0.9231 | 0.4532 | 0.3994 | 0.1944 |
| `I48/deepprep-more` | deepprep | 4.7pct | mimicraw | **0.464301** | 0.9846 | 0.5189 | 0.4506 | 0.2728 |

## 维度二：错误级指标（str / oldsw / goldv29）

| 结果目录 | 跑者 | 口径 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `n18/deepprep-less` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |
| `n18/deepprep-more` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |
| `I48/deepprep-less` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |
| `I48/deepprep-more` | deepprep | str/oldsw/goldv29 | — | — | — | — | — | — |

> ⚠️ 四个 deepprep 目录的**错误级算不出来**：I48 / n18 的 deepprep 输出里有**空表**
> （`admissions.csv` 只有表头；n18 的 `icu/outputevents.csv` 同样为空）⇒
> `compute_error_metrics.py` 建 sqlite 索引时直接炸（`KeyError: 'subject_id'` /
> `sqlite3.OperationalError`）。这是**数据本身的缺陷**（deepprep 的结构性输出问题），
> 不是工具问题 —— K29 / ICU5000 的 deepprep 输出 `admissions.csv` 有 8,093 / 6,476 行，
> 工具跑得通，所以那两档在原表里有错误级数字。

## 维度三：MIMIC 原生错误格（string 口径）

| 结果目录 | 跑者 | 口径 | 总格数 | 修对 | 没动 | 缺行 | **修复率** |
|---|---|---|---:|---:|---:|---:|---:|
| `n18/1.6pct_我们的agent_pi+val_mimicraw` | 我们 | 1.6pct | string | 715 | 711 | 0 | 3 | **0.994406** |
| `n18/4.7pct_我们的agent_pi+val_mimicraw` | 我们 | 4.7pct | string | 586 | 578 | 0 | 7 | **0.986348** |
| `n18/4.7pct_cc基线_ClaudeCode` | cc | 4.7pct | string | 586 | 0 | 444 | 142 | **0.000000** |
| `n18/deepprep-less` | deepprep | string | 715 | 0 | 0 | 715 | **0.000000** |
| `n18/deepprep-more` | deepprep | string | 586 | 0 | 0 | 586 | **0.000000** |
| `I48/deepprep-less` | deepprep | string | 715 | 0 | 200 | 59 | **0.000000** |
| `I48/deepprep-more` | deepprep | string | 586 | 0 | 785 | 217 | **0.000000** |

## 读表提醒（补齐部分）

1. **n18 的 1.6pct 没有 cc/codex 基线** —— 不是漏拷，是**从来没跑过**：
   cc/codex 基线当时跑的是 `datasets/注入错误数据集/n18_less` / `n18_more` 两个切片
   （已从盘上删除），其 cohort 分别是 7,445 / 7,048 行；当前 mimicraw n18 的 cohort 是 **7,048** 行
   ⇒ `n18_more`（4.7pct）**可比、已入库**，`n18_less`（1.6pct，cohort 7,445）**不可比、没入库**。
   要补 1.6pct 的 cc 基线，只能在 mimicraw 1.6pct 上**重跑一次 cc/codex**。

2. **n18 4.7pct 的 cc 基线原生修复率 = 0**（444 没动 / 142 缺行）——
   与包里其它 cc/codex 基线一致：规则推导法碰不到哨兵码与生理不可能值（原生错误类）。

3. **n18 的 deepprep composite 只有 0.20**（I48 的 deepprep 是 0.40~0.46）——
   原因是 **key 几乎全错**（0.165 / 0.151 vs K29 的 0.83）⇒ deepprep **没做 cohort 筛选**，
   而 n18/i48 的输出表是**筛选之后**的表 ⇒ 结构上就对不上。

