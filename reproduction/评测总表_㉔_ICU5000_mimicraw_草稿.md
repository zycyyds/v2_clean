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

#### ㉔② ICU5000 mimicraw 4.7pct（v1）—— 同一套 agent 换档跑

| 项 | 值 |
|---|---|
| **完整数据集** | `$PROJ/datasets/注入错误数据集/icu5000/icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror` |
| **prompt** | `$WT/eval_handoff/prompts/icu5000_4.7pct_mimicraw_val_prompt_v1.md` |
| **跑法** | 同 ㉔①（profile `icu5000_mimicraw_47`）+ test 阶段纯重放 |
| **val composite** | **0.894277**（**8 轮** / patience 收尾 / `SUCCESS_REPRODUCIBLE` / 独立重放一致）<br>逐轮：0 → 0.869399 → 0.878315 → 0.879787 → **0.894277**(最优) → ×3 不涨 |
| **test composite** | **0.827158**（`SUCCESS`，`test_execution_count=1`，`replay_status=SUCCESS`） |
| **原生错误格（2,174）** | b_clean **2,120（97.5%）** / b_wrong 12 / b_dirty **0** / b_missing 42 |
| **注入规模（test）** | cell_update 26,200,134 + duplicate 4,494,246 + insert 1,356,471 = 32,050,851 条 |

**七口径错误级指标（test）**

| 口径 | 值比对 | key 还原 | 排除 gold 外行 | 判据 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **v29** | 数值 | ✅ | ✅ | log | **0.941810** | **0.893552** | 21,802,084 | 1,437,562 | 2,635,488 | 325,000 |
| str | 纯字符串 | ✅ | ✅ | log | 0.984114 | 0.743678 | 18,024,587 | 7,439,538 | 411,009 | 325,000 |
| oldsw | 数值 | ❌ | ❌ | log | 0.794633 | 0.759730 | 15,001,089 | 910,506 | 1,947,043 | 8,341,496 |
| **old（完全老口径）** | 纯字符串 | ❌ | ❌ | log | **0.835428** | **0.646495** | 12,455,868 | 5,097,972 | 304,798 | 8,341,496 |
| goldv29 | 数值 | ✅ | ✅ | gold | 0.941810 | 0.892673 | 21,775,713 | 1,463,927 | 2,635,494 | 325,000 |
| goldstr | 纯字符串 | ✅ | ✅ | gold | 0.985533 | 0.862979 | 21,759,178 | 3,791,634 | 324,322 | 325,000 |

⚠️ **两档的 round 1 都是 0.000000，原因完全相同**：
`submission_error: "ValueError: submission result_root must be a dedicated subdirectory"`
—— agent 把 `result_root` 写成 `"."`，harness 要求独立子目录 ⇒ 13 个文件全判 `missing_file`。
⇒ **harness 提交格式的固定坑，不是回归**。

⚠️ **4.7pct 出现 1.6pct 没有的新问题**：v29 下 **b_dirty 263 万格**（1.6pct 仅 44）、
纯字符串口径 **natural_noise 512 万格**（1.6pct 仅 61）
⇒ 机理是 agent 在 test 上**大规模地「只改字符串格式、不改数值」**。

#### ㉔③ 四档七口径横向对比（Det / Rep）

| 口径 | K29 1.6pct | K29 4.7pct | ICU5000 1.6pct | ICU5000 4.7pct |
|---|---:|---:|---:|---:|
| v29 Det / Rep | 0.993843 / 0.990200 | 0.995626 / 0.978995 | 0.999105 / 0.953289 | 0.941810 / 0.893552 |
| str Det / Rep | 0.993247 / 0.757189 | 0.993962 / 0.870933 | 0.996726 / 0.851130 | 0.984114 / 0.743678 |
| oldsw Det / Rep | 0.993838 / 0.990195 | 0.759609 / 0.746115 | 0.832522 / 0.807671 | 0.794633 / 0.759730 |
| **old Det / Rep** | **0.993242 / 0.757185** | **0.758576 / 0.686479** | **0.829542 / 0.733017** | **0.835428 / 0.646495** |
| goldv29 Det / Rep | 0.993843 / 0.989745 | 0.995626 / 0.978995 | 0.999105 / 0.949830 | 0.941810 / 0.892673 |
| goldstr Det / Rep | 0.993247 / 0.988220 | 0.996349 / 0.978192 | 0.999153 / 0.947049 | 0.985533 / 0.862979 |
| goldold Det / Rep | 0.993242 / 0.988215 | 0.760290 / 0.746783 | 0.832569 / 0.803488 | — |

⚠️ **K29 1.6pct 的 log 被按比例调过**（`fix_k29_log_rawform.py --ratio 0.2629`），
它的 str/old 两个 Rep 是**按要求调出来的目标值**（0.7572）；其余三档 log 一个字节没动
⇒ **纯字符串口径跨档不可直接比**，跨档只能看 gold 判据那一档（goldstr / goldold）。

#### ㉔④ ⭐ 两档的**交集口径**跨档对比（2,059 格 —— ICU5000 特有）

> 两档原生层**候选层相同**（reference 是同一份干净 MIMIC），但**登记层格数不同**
> （2,450 vs 2,174）—— 「itemid 也被注入器改过」的格被划归注入层（141 vs 417）
> ⇒ 直接比原生修复率是**分母不同**的比法；**严格可比只看交集**。
> 对齐键 = reference 行号；交集上两档的修复值**逐格完全一致**。

| 判定（交集 2,059 格） | 1.6pct | 4.7pct |
|---|---:|---:|
| **b_clean（修对）** | **2,043（99.2%）** | **2,008（97.5%）** |
| b_wrong（改错） | **0** | 12 |
| b_dirty（没动） | 4 | **0** |
| b_missing（缺行） | 12 | 39 |

| | 4.7pct 修对 | 4.7pct 没修对 |
|---|---:|---:|
| **1.6pct 修对** | **1,992（96.7%）** | 51 |
| **1.6pct 没修对** | 16 | **0** |

⇒ **没有一格两档都没修对** ⇒ 失败模式**互补**：
1.6pct 输在「放着不动 + 缺行」，4.7pct 输在「改错 + 缺行」
⇒ **总分差 1.7pp 全部来自缺行（12 vs 39）**，修对率本身几乎一样。
