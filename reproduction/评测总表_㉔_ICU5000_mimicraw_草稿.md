# ㉔ 条目草稿 —— ICU5000「MIMIC 原生错误」增强版（mimicrawerror）

> **这是草稿**：本次会话的 worktree 隔离在 `k29-mimicraw`，
> 不允许直接写 `eval-movies-cleaning/评测总表.md`（那要动另一个 worktree 的 git）。
> ⇒ 下面这段可**直接粘贴**进 `评测总表.md`（接在 §㉓ 之后）。
> 数据来源：`reproduction/icu5000_mimicraw_report.md`（完整版）与 `reproduction/台账.md`。
> ⚠️ **4.7pct 那档还没跑完**（见末尾「待补」）。

---

### ㉔ ICU5000「MIMIC 原生错误」增强版（mimicrawerror）—— **1.6pct（已跑完）+ 4.7pct（待补）**

> 这是 **㉓（K29 mimic-raw-error）的 ICU5000 版**：同一套**原生错误检测 + 修复规则**
> （T1 哨兵码 / T2 生理不可能区间 / R2 用该患者该项的中位数修复），
> 换到 **ICU5000**（按 ICU 住院 stay 切片）这个数据集上。
> ⇒ 与 ㉓ 最大的不同不是规则，而是 **① 索引空间是混合的　② 多一层 key 列注入
>   ③ `value` 文本列只有约三成可用　④ train 分片大 46 倍 ⇒ agent 收敛快得多**。

#### ㉔① ICU5000 mimicraw 1.6pct（v1）—— 我们的 agent（pi + val harness）

| 项 | 值 |
|---|---|
| **完整数据集** | `$PROJ/datasets/注入错误数据集/icu5000/icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror` |
| **源数据集** | `icu_5000_v2_rawerror_test1.6pct_614M`（`raw/` 只在「叠脏格」上被改回 MIMIC 原生错值，其余一字节没动） |
| **prompt** | `$WT/eval_handoff/prompts/icu5000_1.6pct_mimicraw_val_prompt_v1.md`（sha256 `c2185b4c16d7b957a536ed9329d19b26d0b025f3b5570a1622f621835cb537b9`）<br>长期副本：`…_mimicrawerror/prompt/icu5000_1.6pct_mimicraw_val_prompt_v1.md` |
| **跑法** | pi + val harness（`eval_handoff/scripts/run_val_harness_ft.py`，profile `icu5000_mimicraw_16`，max_rounds=20 / patience=3 / seed=666）+ test 阶段**纯重放**（`agent.pi_test_harness_cli`，frozen snapshot，无 LLM 调用） |
| **结果包** | test：`$WT/icu5000_agent_run/experiments/icu5000_mimicraw_1p6_test_seed666_v1/host/test_result_package`（**平铺**，13 个 csv）<br>val：`…_1p6_val_seed666_v1/host/best_snapshot/result` |
| **val composite** | **0.914914**（**7 轮** / patience 收尾 / `SUCCESS_REPRODUCIBLE` / 独立重放一致）<br>逐轮：0 → 0.903684 → 0.911372 → **0.914914**(最优) → 0.914914 ×3 |
| **test composite** | **0.899143**（`status=SUCCESS`，`test_execution_count=1`，`equal_file_row_aligned_v1`） |
| **原生错误格（2,450）** | b_clean **2,431（99.2%）** / b_wrong **0** / b_dirty 4 / b_missing 15 |
| **注入规模（test）** | cell_update 8,270,908 + duplicate 1,336,035 + insert 404,755 = 10,011,698 条；评估 8,271,002 格 |

**七口径错误级指标（test）**

| 口径 | 值比对 | key 还原 | 排除 gold 外行 | Det F1 | Rep F1 | b_clean | b_wrong | b_missing |
|---|---|---|---|---:|---:|---:|---:|---:|
| **v29** | 数值 | ✅ | ✅ | **0.999105** | **0.953289** | 7,804,229 | 451,758 | 14,877 |
| str | 纯字符串 | ✅ | ✅ | 0.996726 | 0.851130 | 6,774,771 | 1,433,444 | 14,877 |
| oldsw | 数值 | ❌ | ❌ | 0.832522 | 0.807671 | 5,228,512 | 209,841 | 2,832,514 |
| **old（完全老口径）** | 纯字符串 | ❌ | ❌ | **0.829542** | **0.733017** | 4,580,158 | 814,434 | 2,832,514 |
| goldv29 | 数值（gold 判据） | ✅ | ✅ | 0.999105 | 0.949830 | 7,770,072 | 485,915 | 14,877 |
| goldstr | 纯字符串（gold 判据） | ✅ | ✅ | 0.999153 | 0.947049 | 7,742,175 | 513,818 | 14,877 |
| goldold | 纯字符串 + gold 判据 | ❌ | ❌ | 0.832569 | 0.803488 | 5,192,728 | 245,631 | 2,832,514 |

#### ㉔ 与 ㉓（K29）的横向对比

**① 原生错误层修复率（这一项跨数据集可比 —— 都是「MIMIC 自己录错的值」）**

| 档位 | 原生格 | 修对 | 修对率 | 改错 |
|---|---:|---:|---:|---:|
| K29 4.7pct | 873 | 773 | 88.5% | 0 |
| K29 1.6pct | 873 | 836 | 95.8% | 1 |
| **ICU5000 1.6pct** | **2,450** | **2,431** | **99.2%** | **0** |

⇒ ICU5000 哨兵码只丢 3 格 ⇒ **K29 那个「44 格哨兵码一格没修」（prompt 里两条规则打架）没有重演**。
⇒ 归因：ICU5000 的 train 分片大 46 倍（97,950 行 vs 2,000 行）⇒ agent 有 16 个 HARD itemid 的
对照例子可学（K29 只有 5 个）；且 prompt 里「先修 key 列 → 再修 valuenum → 最后去重」提前堵住了
K29 踩过的「去重跑在修 key 之前」的坑。

**② Rep F1 的差距不是「没检测到」**（Det 0.9991 ⇒ 几乎全检测到），是「**知道错了但恢复不出原值**」：

| 表 / 列 | 注入类型 | 抽样归因 |
|---|---|---|
| outputevents / **charttime** | implausible_negative_event_time（407,694） | **46%「agent 找不到该行」** ⇒ agent 改对了值但**改了 key 列** ⇒ 按旧 key 对不上（㉓ §4.2 同一个坑，K29 58 格 / 这里 43 万格） |
| inputevents / endtime | med_end_before_start（837,909） | **74% 猜错** ⇒ 真值不可推断 |
| inputevents / amount | sentinel_or_negative_med_amount（837,909） | **90% 猜错** ⇒ 真值不可推断 |
| chartevents（原生层所在文件） | valuenum 各类 | **b_wrong 仅 0.26%** ✅ |

⇒ K29 的 Rep 更高（0.979 / 0.990）**不是它更会修**，而是它的注入层里这类「真值不可推断」错误的占比小得多。

**③ log 的「写法形态」：ICU5000 与 K29 4.7pct 同源，K29 1.6pct 是唯一异类**

| 档位 | log 里 `clean_value` 整数形式占比 | 写法税（v29 Rep − str Rep） |
|---|---:|---:|
| **ICU5000 1.6pct** | **0.11%** | **0.102** |
| ICU5000 4.7pct | 0.01% | — |
| K29 4.7pct | 0.02% | 0.108 |
| **K29 1.6pct** | **70.49%** | **0.233** |

⇒ **ICU5000 不需要改 log**（K29 1.6pct 当初要按比例改 log 是因为 legacy 引擎写整数）⇒
纯字符串口径可直接跨「ICU5000 1.6pct / ICU5000 4.7pct / K29 4.7pct」三档比较。

#### ㉔④ 待补

- [ ] 4.7pct 档的 val / test / 七口径 / 原生格（native 层 2,174 格）
- [ ] **两档交集 2,059 格的跨档对比**（ICU5000 特有：两档原生层格数不同 ——
      2,450 vs 2,174 —— 因为「itemid 也被注入器改过」的格数不同：141 vs 417，
      这些格形态同构不可判定 ⇒ 划归注入层。跨档比修复率要用交集 2,059 格做分母）
