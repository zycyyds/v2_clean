# 磁盘占用说明（2026-11-20 清理）

> 起因：磁盘 96% 满（856/926 GB，只剩 42 GB）。清理后 **180 GB 可用（80%）**。

## 一、评分器的 sqlite 索引缓存（最大头，220.9 GB）

**位置**：`/Users/mac/.claude/jobs/910902cd/tmp/v31_idx_cache`
（硬编码在 `score_only/error_metrics/compute_error_metrics.py:34` 与
`score_only/error_metrics/build_eval_cache.py:17` 的 `CACHE_DIR`）

**它是什么**：`build_eval_cache.py` 为「agent 输出」和「gold」各建一份 sqlite 索引，
加速错误级指标的逐格比对。**纯中间产物** —— 删掉不影响任何结果，只是重跑评分要重建。

⚠️ **必须先建缓存再打分**：`compute_error_metrics.py` **不会自动重建**。
缓存不存在时它会把**每个文件都 SKIP** ⇒ 指标全 0（看起来像「agent 一格没修」）。
所以重跑评分前先跑：

```bash
python3 score_only/error_metrics/build_eval_cache.py \
  --result-dir <agent 输出目录> --ref-dir <gold 目录> --ds-key <ds_key>
```

**每个 ds-key 约 10~20 分钟**（大表 `icu/chartevents.csv` 的索引单个就 4 GB、建 5 分钟）。

### 本次删了什么

删了 **43 个历史 ds-key / 602 个 db / 134.5 GB**，保留当前 `ceshi1` 评估用到的
**16 个 ds-key / 86.4 GB**。

**保留清单**（当前评估流水线实际引用的）：

| ds-key | 用于 |
|---|---|
| `k29_mimicraw_16_agent` / `k29_mimicraw_47_agent` | build_final_table：K29 我们 1.6/4.7pct |
| `k29_16_mimic` / `k29_47_mimic` | eval_tasks：K29 我们（重算口径） |
| `icu5000_mimicraw_16_agent` / `icu5000_mimicraw_47_agent` | ICU5000 我们 1.6/4.7pct |
| `k29dp16` / `k29dp47` / `icudp16` / `icudp47` | deepprep 基线四档 |
| `k29cx16new` / `k29cx47new` / `icucx16` / `icucx47` / `i48cc16` | codex / cc 基线 |
| `i48_mimicraw2_16_agent` / `i48_mimicraw2_47_agent` | I48 mimicraw2 两档 |

**删除的**（历史实验的死缓存）：`icu5000_{less,more}_{agent,codex,cc_v1}`、
`i48_{less,more}_{agent,cc_v1}`、`i48_mimicraw_v{1,3}_*`、`i48_*_numericcheck`、
`i48_old_*`、`i48_val_*`、`k29_1800_*`、`k29_less_model`、`k29_47_v4_model`、
`k29_test4p7`、`n18_*`、`score_any_*` 等。

> 判据：**「当前 ceshi1 评估流水线（eval_tasks.py + build_final_table.py）引用到的
> ds-key 及其别名」之外的，一律当死缓存**。

## 二、worktree 里与 `ceshi1` 重复的结果包（8.9 GB → 已删）

`ceshi1/<档位>/结果包/` 里已有**逐字节相同**的副本（md5 全验证过），
worktree 里的 `*/host/test_result_package/` 可删：

| 删掉的 | 大小 |
|---|---:|
| `k29_mimicraw_1p6_test_seed666_v1/host/test_result_package` | 767 MB |
| `k29_mimicraw_4p7_test_seed666_v1/host/test_result_package` | 755 MB |
| `icu5000_mimicraw_1p6_test_seed666_v1/host/test_result_package` | 2,087 MB |
| `icu5000_mimicraw_4p7_test_seed666_v1/host/test_result_package` | 3,964 MB |
| `i48_mimicraw2_1p6_test_seed666_v1/host/test_result_package` | 881 MB |

⚠️ **`i48_mimicraw2_4p7_test_seed666_v1` 保留** —— 那一档已按要求移出 `ceshi1`，
worktree 里这份是**唯一副本**。

删的只是 `test_result_package`（输出结果）；**`frozen_snapshot` / `reproducible_snapshot`
/ `run_report.json` / `score_report.json` 全部保留**（重放验证的依据）。

## 三、K29 log 的改前原件（4.1 GB → gzip 后 248 MB）

`/Users/mac/.claude/jobs/448aad57/tmp/log_backup/*.bak` 是
`fix_k29_log_rawform.py` 改 K29 log **之前**的原件 ——
K29 1.6pct 的 log 被按比例改过（`--ratio 0.2629`），**这些是还原原始 log 的唯一副本**。
⇒ **不删，只 gzip**（CSV 文本压缩率 ~16 倍）。解包：`gunzip -k xxx.bak.gz`。

## 四、下次清理时看哪里

| 位置 | 说明 |
|---|---|
| `/Users/mac/.claude/jobs/910902cd/tmp/v31_idx_cache` | sqlite 索引缓存，220 GB 的大头，可重建 |
| `/Users/mac/.claude/jobs/*/tmp/` | 各会话的临时产物（`log_backup` / `dbcopy` / `ziptest` 等） |
| `<worktree>/*_agent_run/experiments/*/host/test_result_package` | 与 `ceshi1` 重复的结果包 |
| `~/Downloads/ceshi1` | **交付物，不要动** |

⚠️ 删缓存前先确认**没有评分进程在跑**（`pgrep -fl compute_error_metrics`）——
运行中的进程正读着这些 db。
