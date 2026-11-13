# ㉓ K29 4.7pct「MIMIC 原生错误」增强版（mimicrawerror）

> 跑批时间：2026-11-13 21:07 → 2026-11-14 02:28（val 约 4h20m，test 重放约 25m）
> 结论一句话：**873 格 MIMIC 原生错误，agent 修对 773 格（88.5%）、改错 0 格**；
> 放宽"键对不上但值算对"的口径是 **830/873 = 95.1%**。

---

## 1. 这一档是什么

| 项 | 值 |
|---|---|
| **完整数据集** | `$PROJ/datasets/注入错误数据集/k29/K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror` |
| **扣留视图（给 agent）** | `$PROJ/datasets/注入错误数据集/k29_mimicraw_47`（只有 train/validation，无 test、无 `modification_log*`、无 `mimic_native_error_log*`） |
| **与源数据集的唯一区别** | `reference/` 里 MIMIC 原生错误已修好；`modification_log.csv` 登记了原生错误；368 格叠脏格的 raw 值恢复成 MIMIC 原生错值 |
| **prompt** | 数据集内 `prompt/k29_4.7pct_mimicraw_val_prompt_v1.md`（25,681 bytes，与 `k29_agent_run/prompts/` 里跑的那份 md5 相同） |
| **跑法** | pi + val harness（`run_val_harness_ft.py k29_mimicraw v1`）+ test 阶段纯重放（`agent.pi_test_harness_cli`） |
| **harness profile** | `k29_mimicraw`（`eval_handoff/scripts/run_val_harness_ft.py`，新增） |
| **数据集自检** | `reproduction/verify_k29_mimicraw_error.py`（**17 项全部通过**） |
| **原生格专用判定** | `reproduction/eval_native_cells.py`（只出聚合，不打印患者标识） |
| **结果** | `runs/k29_mimicraw_47_agent_error_metrics_v29.json`、`runs/k29_mimicraw_47_agent_native_cells.json` |

---

## 2. 数据集构造（与 I48 ㉑/㉒ 的差异）

| 维度 | I48 mimicrawerror | **k29 mimicrawerror（本次）** |
|---|---|---|
| 原生错误类型 | 只有「生理不可能区间」 | **「生理不可能区间」+ 「MIMIC 哨兵错误码」**（后者对所有 itemid 生效） |
| 判定依据 | 硬区间 + `!= 0` | 硬区间 + 哨兵码；**另加两条**：与该患者自身常态偏离 ≥ 20%、其余读数都在区间内 |
| train 里的示例 | **天然存在**（I48 的 cohort 患者多） | **注入的**（k29 train 只有 10 个 hadm / 2k 行，`(subject,itemid)` 够条件的只有 6 组）⇒ train 5 格 + validation 24 格 |
| 与注入重叠的格 | 199 格"同时被污染" | **368 格叠脏 ⇒ raw 值恢复成 MIMIC 原生错值、同格的注入条目从 log 删除**（每格只剩一个错误，目标唯一可判定） |
| prompt 额外内容 | — | 哨兵码类、**枚举型多值不是重复行**（230033/229864 共 147 组合法多值）、去重顺序、**`value` 文本列判据** |

### 2.1 关键构造决策：为什么删掉重叠的注入条目

883 格原生错误格里，**368 格原本被注入器叠了一层脏值**（1.6pct 档是 836 格）。
处理方式：**把这 368 格的 raw 值恢复成 MIMIC 原生错值，并删掉同格的注入 log 条目**。

- 理由一：**单格 ×100 与单格原生错误在形态上完全同构**（都只是"一个比正常值大两个数量级的孤立尖峰"）⇒ 两层错误叠在一起时**不可判定**。
- 理由二：用户决定"少几百个注入错误无所谓"（1414 万条里的 0.003%）。
- 理由三：这样这 883 格**每格只有一个错误**，目标唯一，`b_clean`/`b_dirty`/`b_wrong` 三态干净可分。

### 2.2 关键发现：`value` 文本列是原生错误的判据（数据自带）

**这是本次最有价值的发现**，而且是 agent 在啃 train 对照时自己撞出来、我顺着它的疑问查出来的。

| 组 | raw 的 `value` 文本列 | raw 的 `valuenum` | 数量 |
|---|---|---|---:|
| **纯原生格**（只有 MIMIC 原生错误） | **= 那个坏数**（与 valuenum 一致） | 坏数 | **510 / 510 = 100%** |
| 叠脏格 | != valuenum | 注入值 | 372 / 373 |
| `unit_scale_error` 注入格 | = **干净数** | 脏值 | 495,623 / 497,408 |
| `sentinel_chart_value` 注入格 | = **干净数** | 脏值 | 634,397 / 1,935,724 |

⇒ **判据**：`value == valuenum` ⇒ MIMIC 原生错误（原始数据里 value 记的也是坏数，**没有任何线索能还原**）⇒ 只能用中位数；
`value != valuenum` ⇒ 注入错误 ⇒ **直接采用 `value` 文本列**（比 ÷100 可靠）。

⇒ 副作用（意外收获）：这条规则让**注入层**也能直接还原 ⇒ Rep F1 从 I48 的 0.9382 提到 **0.9790**。

⚠️ **被否掉的两条错误判据**（都留了实测记录）：
- ❌「÷100 后落不落回区间 ⇒ 量纲放大」—— **在 CVP 上 245/245 全错**（原生错值 277~361 ÷100 = 2.77~3.61，落在 CVP 的 [-20,80] 里，看着完全合理）。
- ❌「只看绝对区间」—— 边界形态（舒张压 10~19、HR 15~19）生理上可能真实 ⇒ 改用「**与该患者自身常态的偏离 ≥ 20%**」（实测 883 格里**偏离 < 20% 的 = 0**，与 I48 同口径）。

### 2.3 构造过程中被 agent 抓出来的 bug（已修）

注入示例时**只改了 `valuenum`、没改 `value` 文本列** ⇒ 示例形态与测试集真实分布**相反**。
校验器已加自动检查（原生层每格必须 `value == valuenum`），这个 bug 以后会被自动抓出来。

---

## 3. 跑分结果

### 3.1 val（7 轮，patience 收尾）

| 轮次 | composite |
|---|---:|
| r1 | 0.900066 |
| r2 | 0.906229 |
| r3 | 0.911156 |
| **r4** | **0.914462** |
| r5 / r6 / r7 | 0.914462（同分，patience=3 收尾） |

`status=SUCCESS_REPRODUCIBLE`、`stop_reason=patience`、**`best_score == reproducible_score == 0.914462`**
（独立重放逐位相同 ⇒ pipeline 确定性没问题）。

> r3 花了约 2 小时，且有约 2 小时是**纯推理零文件产出** —— agent 陷进了「逆向推导评分公式」的兔子洞
> （它在猜「每个文件比较几列」，而 harness 不给公式、这题无解）。
> 后来它自己走出来了，并逐文件定位到该修什么（chartevents 的 valuenum、inputevents 的 amount 哨兵码）。
> ⚠️ 记一笔：**这类兔子洞在别的任务族上也出现过**，值得在 prompt 里加一句「不要反推评分公式，把反馈当方向信号」。

### 3.2 test（纯重放，无 LLM）

| 项 | 值 |
|---|---:|
| status | SUCCESS，`test_execution_count=1` |
| **composite** | **0.909930** |
| schema_f1 / file_coverage | 1.0 / 1.0（12/12 文件匹配） |
| key_structure_f1 | 0.949911 |
| row_aligned_cell_f1 | 0.931399 |
| exact_row_f1 | 0.824626 |

**横向对照（不同任务族，同一套方法）**：

| | I48 1.6pct mimicraw | I48 4.7pct mimicraw | **k29 4.7pct mimicraw** |
|---|---:|---:|---:|
| test composite | 0.916227 | 0.912463 | **0.909930** |
| Det F1 | 0.999641 | 0.987321 | **0.995626** |
| Rep F1 | 0.944613 | 0.938215 | **0.978995** |

⇒ composite 三族可比（k29 略低但同量级）；**错误级指标 k29 明显更好**（`value` 文本列判据的功劳）。

### 3.3 v29 错误指标（数值感知口径）

```
recover_clean_key=true   exclude_rows_not_in_gold=true   numeric_compare=true
injected = 14,124,020（cell_update 11,082,831 / duplicate 1,496,840 / insert 1,544,349）
evaluated = 10,651,749 cells

Det F1 = 0.995626   (P 0.998290 / R 0.992977)
Rep F1 = 0.978995   (P 0.981615 / R 0.976390)

b_clean 10,363,470   b_dirty 3,382   b_wrong 202,526   b_missing 82,371
fp_det       20,767   natural_noise 45
dup_removed 1,187,744 / dup_kept 0        leak_removed 370,652 / leak_kept 0
crow_recovered 5,205,203
```

### 3.4 🎯 核心指标：873 个原生错误格上的判定

| 判定 | 格数 | 占比 |
|---|---:|---:|
| **b_clean（改对）** | **773** | **88.5%** |
| **b_wrong（改错）** | **0** | **0.0%** ⭐ |
| b_dirty（没动） | 42 | 4.8% |
| b_missing（键对不上） | 58 | 6.6% |

**按 itemid**：

| itemid | 测量项 | 合计 | 修对 | 改错 | 未动 | 缺行 |
|---|---|---:|---:|---:|---:|---:|
| 220074 | 中心静脉压 | 398 | **380** | 0 | 0 | 18 |
| 220052 | 动脉血压均值 | 234 | **208** | 0 | 2 | 24 |
| 220180 | 无创舒张压 | 60 | **57** | 0 | 0 | 3 |
| 220050 | 动脉收缩压 | 46 | **43** | 0 | 0 | 3 |
| 220051 | 动脉舒张压 | 38 | **36** | 0 | 0 | 2 |
| 220179 | 无创收缩压 | 14 | **13** | 0 | 0 | 1 |
| 220181 | 无创平均压 | 11 | **11** | 0 | 0 | 0 |
| 223762 | 体温 ℃ | 8 | **7** | 0 | 0 | 1 |
| 220045 | 心率 | 5 | 3 | 0 | 0 | 2 |
| 224689 | 呼吸频率-自主 | 5 | **5** | 0 | 0 | 0 |
| **227456 / 227465 / 227467 / 227457 / 220228 / 220545 / 220546 / 225624 / …** | **哨兵码类** | **44** | **0** | 0 | **44** | 0 |

---

## 4. 两处失败的逐格归因（实测，不是猜）

### 4.1 44 格哨兵码一格没修 —— **我 prompt 的内部矛盾造成的**

agent 的 `pipeline.py` 里**规则全都实现了**（`NATIVE_SENTINELS` 我 prompt 里那 10 个、`IMPOSSIBLE_RANGES` 20 行表、`_clean_via_value_text` 这条 `value` 文本列判据 —— 它甚至**独立**发现了这条判据）。但：

```python
def _clean_via_value_text(val, vn):
    ...
    if vn in SENTINEL_VALUES or vn == -1000:
        return val, False          # ← 哨兵码在这里就 return 了
    ...
    return vn, abs(vn - val) < 1e-12   # ← value==valuenum ⇒ 原生错误候选
```

⇒ **哨兵码在到达"value==valuenum ⇒ 原生错误"那条判断之前就 return 了** ⇒ 44 格哨兵码原生错误永远进不了第 7 步的「中位数修复」。

**根因是我的 prompt 内部两条规则打架**：
- 「类 2：哨兵错误码」说 ⇒ 这些码根本不是测量值 ⇒ 用中位数修
- 「怎么区分」那节说 ⇒ `value != valuenum` ⇒ 是注入 ⇒ 修成 `value` 文本列的值（举的例子只有 ×100）

⇒ **哨兵码同时满足两条**，agent 选了"注入"那一支。

⇒ **修法（下一版 prompt）**：把区分规则写成**先判哨兵码**、或明确写
「`value == valuenum` ⇒ 原生错误（**哨兵码也要先看这一条**）⇒ 中位数；`value != valuenum` ⇒ 注入 ⇒ 用 `value`」。
⇒ 修好后这 44 格应全部变成 `b_clean` ⇒ 修复率 **88.5% → 93.5%**。

### 4.2 58 格键对不上 —— **值是对的，键错了**（与 I48 ㉒③ 同一个坑）

| 形态 | 格数 |
|---|---:|
| 放宽 key（去掉 charttime）找到**多行**（重复行） | **58 / 58** |
| charttime 与 gold 不同（agent 给的是"第三种时间"） | **58 / 58** |
| **值 == 修复值（值算对了，只是键不同）** | **57 / 58** |

⇒ agent 的去重（第 6 步 `drop_duplicates`）**位置是对的**（在 charttime/valuenum 修复之后 ✓），
但**去重键含 charttime** ⇒ charttime 被修成另一个值 ⇒ 没去成重 ⇒ 同一 `(subject, hadm, stay, itemid)` 留下多行，
按完整 5 键查 gold 就对不上。

⇒ **修法（下一版 prompt）**：去重键应当**不含 charttime**（`(subject_id, hadm_id, stay_id, itemid)`）。
⇒ 按 I48 的宽松口径（"值算对就算"）计，**实际修复率 = (773 + 57) / 873 = 95.1%**。

---

## 5. 三条结论

1. **原生错误规则跨任务族迁移成功**：k29 是 12 个输出文件 + `note/` 多模态 + hadm_id 切片，
   I48 是 13 个文件 + ECG + subject_id 切片 ⇒ **换了一个完全不同的任务族，873 格里修对 773、误改 0**。
2. **哨兵码这一类（I48 没做过）暴露了 prompt 的漏洞**，不是 agent 的能力问题 ⇒ 下一版 prompt 修一处即可补回 44 格。
3. **本数据集的价值不是抬分，是让这个类别从"不可测"变成"可测"**：
   883 格原生错误 vs 1414 万格注入错误 ⇒ 只占 0.006%。
   但**在源数据集上，agent 修对了原生错误反而会被记成 `fp_det`（误改）**（源 reference 存的就是坏值）
   ⇒ 换到本数据集，同一个行为从"误改"变成"修对"。
   ⇒ 没有它，就不会有任何 agent 有动力去学这一类。

## 6. 待办

- [ ] 下一版 prompt：① 哨兵码先判 `value == valuenum`；② 去重键不含 charttime
- [ ] 补跑 1.6pct 档（原生层与 4.7pct 完全相同：883 格；只是注入密度低 ⇒ 注入层更难）
- [ ] 往 `评测总表.md` 加 ㉓ 条目（该台账在 `eval-movies-cleaning` worktree 里，本次未改）
