# ㉓ 条目草稿 —— K29「MIMIC 原生错误」增强版（mimicrawerror）

> **这是草稿**：本次会话的 worktree 隔离在 `k29-mimicraw`，
> 不允许直接写 `eval-movies-cleaning/评测总表.md`（那要动另一个 worktree 的 git）。
> ⇒ 下面这段可**直接粘贴**进 `评测总表.md`（接在 §㉒ 之后）。
> 数据来源：`reproduction/k29_mimicraw_report.md`（完整版）与 `reproduction/台账.md`。

---

### ㉓ K29「MIMIC 原生错误」增强版（mimicrawerror）—— **4.7pct + 1.6pct 两档**

> 这是 **㉑ / ㉒（I48 mimic-raw-error）的 K29 版**：同一套**原生错误检测 + 修复规则**，
> 换到 K29（胃窦十二指肠炎）这个**行数大 3 个数量级**的数据集上。
> ⇒ 与 ㉑/㉒ 最大的不同不是规则，而是 **① 行索引空间　② 规模　③ 老口径的"写法税"**。

#### ㉓① K29 mimicraw 4.7pct（v1）—— 我们的 agent（pi + val harness）

| 项 | 值 |
|---|---|
| **完整数据集** | `$PROJ/datasets/注入错误数据集/k29/K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror` |
| **源数据集** | `K29_v2_rawerror_test4.7pct_14.1M`（`raw/` 一个字节没动） |
| **prompt** | `$WT-K29/k29_agent_run/prompts/k29_4.7pct_mimicraw_val_prompt_v1.md`（sha256 `049ef02989cdf94853784b54ce98a5ac45f194d4cc49a13b7f64dea1ae7b9cce`）<br>长期副本：`…/K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror/prompt/k29_4.7pct_mimicraw_val_prompt_v1.md` |
| **跑法** | pi + val harness（`eval_handoff/scripts/run_val_harness_ft.py`，max_rounds=20 / patience=3 / seed=666）+ test 阶段**纯重放**（`agent.pi_test_harness_cli`，frozen snapshot，无 LLM 调用） |
| **结果包** | test：`$WT-K29/k29_agent_run/experiments/k29_mimicraw_4p7_test_seed666_v1/host/test_result_package`<br>val：`…_4p7_val_seed666_v1/host/best_snapshot/results` |
| **val composite** | **0.914462**（7 轮 / patience 收尾） |
| **test composite** | **0.909930** |
| **v29 错误级指标** | **Det F1 0.995626 / Rep F1 0.978995** |
| **原生错误格（873）** | b_clean **773（88.5%）** / b_wrong 0 / b_dirty 42 / b_missing 58 |
| **注入规模（test）** | cell_update 11,082,831 / duplicate 1,496,840 / insert 1,544,349；评估 10,651,749 格 |

#### ㉓② K29 mimicraw 1.6pct（v1）—— 我们的 agent（pi + val harness）

| 项 | 值 |
|---|---|
| **完整数据集** | `$PROJ/datasets/注入错误数据集/k29/K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror` |
| **prompt** | `$WT-K29/k29_agent_run/prompts/k29_1.6pct_mimicraw_val_prompt_v1.md`（sha256 `f8d9b99d8eaa82e34fba3abe6936d64c4d9f252826920b5db230e2bac8d72950`）<br>长期副本：`…/K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror/prompt/k29_1.6pct_mimicraw_val_prompt_v1.md` |
| **跑法** | 同上（val 17 轮 + 1 轮重放修复） |
| **结果包** | test：`$WT-K29/k29_agent_run/experiments/k29_mimicraw_1p6_test_seed666_v1/host/test_result_package`/**`result`** ⚠️ **在子目录里**（该档 agent 把 `submission.json` 的 `result_root` 设成了 `result`）<br>val：`…_1p6_val_seed666_v1/host/best_snapshot/result` |
| **val composite** | **0.888065**（17 轮 + 1 轮重放修复） |
| **test composite** | **0.881516** |
| **v29 错误级指标** | **Det F1 0.993843 / Rep F1 0.990200** |
| **原生错误格（873）** | b_clean **836（95.8%）** / b_wrong 1 / b_dirty 34 / b_missing 2 |
| **注入规模（test）** | cell_update 3,671,356 / duplicate 521,771（**在 raw 里没有实体**）/ insert 0；评估 3,671,318 格 |

#### ㉓③ 两档原生错误层**逐格相同**（可做干净横向对照）

| 项 | 4.7pct | **1.6pct** |
|---|---:|---:|
| 候选原生格 | 883 | **883** |
| 可修 | 873 | **873** |
| T2_impossible / T1_sentinel | 834 / 49 | **834 / 49** |
| log 里 `mimic_native_*` 条目 | 873 | **873** |
| train / validation 注入示例 | 5 / 24 | **5 / 24**（itemid 完全相同） |
| 与注入器叠脏 | 368 | **827**（密度低 ⇒ 叠脏更多） |

⇒ **目标格集合与修复值完全一样** ⇒ 原生层可做干净横向对照。
⇒ **最重要的一条**：**哨兵码（44 格）1.6pct 修对 43 格，4.7pct 修对 0 格**。
⇒ 两档用**同一份 prompt**（只差 3 行密度数字）⇒ **4.7pct 的哨兵码失败不是 prompt 的必然结果，
而是那次 run 的具体失败** ⇒ ㉑/㉒ 里归因的「prompt 内部矛盾」**是可绕过的**。

#### ㉓④ ⚠️ 这一档挖出来的三件事（都写死在别人的代码里）

**① 行索引空间：4.7pct 带偏移、1.6pct 是零偏移**

| 档位 | raw 行数 | reference 行数 | 多出 | log 里 insert+dup | 逐行 key 对齐 | ⇒ 索引空间 |
|---|---:|---:|---:|---:|---:|---|
| 4.7pct | 12,221,844 | 9,180,655 | **3,041,189** | 3,041,189 ✅ | 0.00% | **bisect**（`r = L + |inserts<L|`） |
| 1.6pct | 9,180,655 | 9,180,655 | **0** | 521,771 ❌ | 96.83% | **zero**（`r = L`） |

⇒ 1.6pct 的 legacy 注入引擎**只记日志、没把重复行落进 raw** ⇒ 52 万条 `duplicate` 在 raw 里没有实体。
⇒ `verify_k29_mimicraw_error.py` 与 `compute_error_metrics.py` 都**硬编码了 4.7pct 的偏移假设**
⇒ 校验器在 1.6pct 上整段读错行（误报 560 格）；评分器会把 **52 万个真实基线行**
当成「注入的重复行」⇒ **诱导 agent 去删合法行**（cc 踩过的「自伤链条」）。
⇒ **修法（已做）**：两处改成**逐文件自动判定**；评分器零偏移模式下 log 的 duplicate/insert
**不参与计分**，并新增输出字段 `offset_mode` / `injected_rows_not_materialized`。
**回归测试**：改完后重算 4.7pct，所有 bucket 与 Det/Rep F1 **逐位相同** ⇒ 对既有结果零影响。

**② ⚠️ 一条被撤销的错误结论**（写在这里防止再犯）
初稿写「1.6pct 的 raw 少 13.5 万行基线行 ⇒ composite 有不可消除上限」——
**错的，已撤销**。测法有位置偏倚（取 reference 开头 2 万个缺失 key 再外推）。
无偏复测：4.7pct 的 raw 缺 4,246,921 行 reference（46.3%）、1.6pct 缺 245,045（2.7%），
而 **agent 把 99.2% 的缺失 key 修回来了** ⇒ **没有额外的行级上限**。

**③ ⚠️ 老口径的「写法税」：跨档绝对值本来不可比，已把 log 约定统一**

评分器判「这格修好没有」用的判据是 **log 的 `clean_value`**；
而两档 log 的渲染不同（4.7pct 存 MIMIC 原始串 `'160'`、1.6pct 存 float repr `'160.0'`），
两档 agent 却都写 float repr ⇒ **老口径在两档行为完全不同，同一份输出能差 0.11 Rep F1**。

⇒ **2026-11-17 已把 1.6pct 的 log 统一到 4.7pct 的约定**（4.7pct **逐字节未动**，
md5 `6e6ba7338c7b99a9bee0a48e31cf9304` 改前 = 改后）：
规则 = 「`clean_value` 恰为 float repr 的整数 ⇒ 写整数形式」
（`'160.0'`→`'160'`；`'04280'` **不转**，保留前导零）。
改动：test 3,236,991 条（`valuenum` 3,205,425 + `amount` 31,566）／train 100／validation 9,030。

**净效应**：**v29 / oldsw 口径数字完全不变**（数值感知对形式不敏感）；
只有纯字符串口径变（1.6pct Rep F1 0.988637 → **0.108270**）。
**改后两档写法税机制同源**：被罚的列都是 `chartevents.valuenum` + `inputevents.amount`
（4.7pct **1,291,048** 格 / 1.6pct **3,240,460** 格）。
⇒ **可比的相对指标**：「`valuenum` 注入格里被误罚的比例」
4.7pct **82.5%** / 1.6pct **≈100%**。
⇒ 两档老口径 Rep F1 仍有差（0.870933 vs 0.108270），但那**不是口径差异**，
而是**注入错误的列构成**不同（4.7pct 还注入 charttime/itemid/stay_id，这些 agent 写得逐字节相同）。

**⚠️ 踩过的坑（记下来）**：`diagnoses_icd` 的键列含 `icd_code`，它有 1,743 条**前导零**形式。
若用宽松规则（数值是整数就转）会抹掉前导零 ⇒ 评分器按精确字符串还原被破坏的 key 失败
⇒ **那 1,743 行被记成 `row_not_in_gold`、排除出评估集**（评估集 3,671,318 → 3,669,575）。
⇒ **凡是会动键列的转换，必须跳过键列**。

#### ㉓⑤ 四口径对照（2 根正交开关 × 4 组合，同一份 agent 输出）

| # | ① 值比对 | ② 旧开关 | 历史上用过？ | 4.7pct Det / Rep | 1.6pct Det / Rep |
|---|---|---|---|---|---|
| **1** | 数值感知 | 保留 | ✅ **账本 §1.2 那 16 行全部用它** | 0.995626 / 0.978995 | 0.993843 / 0.990200 |
| **2** | 纯字符串 | 保留 | ❌ 历史上从未用过 | 0.993962 / 0.870933 | 0.993247 / 0.108270 |
| **3** | 数值感知 | 全退 | ✅ **`评测规范.md` §4 定义的「旧口径」** | 0.759609 / 0.746115 | 0.993838 / 0.990195 |
| **4** | 纯字符串 | 全退 | ❌ 历史上从未用过 | 0.758576 / 0.686479 | 0.993242 / 0.108269 |

⇒ **不是「四版标准」，是 2 根独立开关的 4 种组合**（① 值比对 numeric/string ×
② 旧开关 recover_clean_key + exclude_rows_not_in_gold）。
⇒ **退旧开关掉 0.23 是纯口径假象**（`b_missing` 82,371 → 4,718,440：
agent 正确删掉、gold 里也没有的行被算成漏检）。
⇒ **退字符串比对掉的是「写法税」**：罚的是「数值修对了、只是写法不同」的格，
**与清洗质量无关**（实测 agent 输出与 gold 逐字节相同的比例：4.7pct 96.3% / 1.6pct 99.995%）。

#### ㉓⑥ 三条实务结论

① **日常评分用 v29（数值感知）⇒ 两档数字直接可比**：
4.7pct 0.995626 / 0.978995 vs 1.6pct 0.993843 / 0.990200。
② **纯字符串口径的绝对值不可跨档比** —— 现在不是 log 的问题了，是**注入列构成**不同。
③ **老口径应当拿 gold 当判据**；拿 log 的 `clean_value` 会把「做对了」罚成「没修好」。

**完整报告**：`reproduction/k29_mimicraw_report.md`（§6 是 1.6pct 档，§6.10 是 log 约定统一）
**跑批台账**：`reproduction/台账.md`
**复现脚本**：`scripts/score_k29_mimicraw_four_calibers.sh`（四口径 + 原生格重判，两档通用）
