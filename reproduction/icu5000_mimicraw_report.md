# ㉔ ICU5000「MIMIC 原生错误」增强版（mimicrawerror）

> 构造时间：2026-11-17
> 一句话：**icu5000 的两档（1.6pct / 4.7pct）都加上了「MIMIC 自己录错的值」这一层评测**，
> 判定规则、修复规则（R2 中位数）、prompt 附加节全部沿用 K29 ㉓ 的那套，
> 但**有三处 icu5000 特有的地方必须单独处理**，否则构造出来的东西是错的。

---

## 1. 这一档是什么

| 项 | 值 |
|---|---|
| **完整数据集（1.6pct）** | `$PROJ/datasets/注入错误数据集/icu5000/icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror` |
| **完整数据集（4.7pct）** | `$PROJ/datasets/注入错误数据集/icu5000/icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror` |
| **扣留视图（给 agent）** | `$PROJ/datasets/注入错误数据集/icu5000/icu5000_mimicraw_{16,47}`（只有 train/validation，无 test、无 `modification_log*`、无 `mimic_native_error_log*`） |
| **与源数据集的唯一区别** | `reference/` 里 MIMIC 原生错误已修好；`modification_log.csv` 末尾登记了原生错误；叠脏格的 raw 值恢复成 MIMIC 原生错值 |
| **溯源表** | 各 split 下的 `mimic_native_error_log.csv`（19 列，逐格记录原生错值 / 修复值 / 修复规则 / 可用读数个数） |
| **prompt** | `eval_handoff/prompts/icu5000_{1.6pct,4.7pct}_mimicraw_val_prompt_v1.md` |
| **构造脚本** | `reproduction/build_icu5000_mimic_raw_error.py` |
| **校验脚本** | `reproduction/verify_icu5000_mimic_raw_error.py` |
| **统计脚本** | `reproduction/icu5000_native_stats.py`（prompt 里引用的每个数字都来自它） |
| **prompt 生成** | `reproduction/make_icu5000_mimicraw_prompt.py` |
| **一键准备** | `scripts/prepare_icu5000_mimicraw_run.sh {16\|47}`（manifest → agent 视图 → 统计 → prompt → prompt 同步进数据集） |

---

## 1.1 manifest 的两处说明

- **两档的 `cell_level_error_rate` 单位统一成「百分数」**（1.58 / 4.72），
  并另存 `cell_level_error_rate_fraction`（0.015799 / 0.047219）——
  4.7pct 源数据集自带的 manifest 用的是**分数**，1.6pct 那份用**百分数**，直接对比会差 100 倍。
- ⚠️ **4.7pct 档的 `validation_failures` 有一条**：`actual_count 32,049,531 too far from target 28,998,744`
  （实际注入数比目标高 10.5%）。这是**源数据集本身的状态**（注入器超额注入），
  **不是 mimicraw 层引入的** —— manifest 只是把这个事实记下来，harness 不看这个字段。

---

## 2. 与 K29 ㉓ 的差异（**三处必须单独处理**）

| 维度 | K29 ㉓ | **ICU5000 ㉔（本次）** |
|---|---|---|
| **① raw 的行索引空间** | 4.7pct 带偏移、1.6pct 零偏移（**两档不同**） | **混合**：chartevents 是 bisect 偏移（raw 比 reference 多 1,740,790 行）；inputevents / outputevents 是零偏移 ⇒ **必须逐文件自动判定**，写死任何一种都会读错行 |
| **② itemid 列上的别名注入** | K29 没有这一层 | **1,632,972 条 `chart_itemid_alias_split`**（chartevents）+ 574,828 条 `medication_itemid_alias_split` ⇒ **raw 的 itemid 被改成了 `992{2\|3}XXXX`**（实测 99.94% 符合「`992` + (2 或 3) + 原 itemid 末 4 位」）。这是 **key 列**上的错误，不修就大面积键对不上 |
| **③ 两层错误叠加（itemid 那一层）** | 构建器只处理了 valuenum 那一层 | **141 格**（4.7pct 417 格）的 itemid 也被注入器改过、同时又是原生错误格 ⇒ 两层错误形态同构、不可判定 ⇒ **不进 native 层**，留给注入层评测（K29 报告 §2.1 原则的延伸） |
| **④ `value` 文本列的可靠性** | `unit_scale_error` 里 **99.6%** 的格 `value` == clean_value ⇒ prompt 敢说「修注入错误直接抄 `value` 比 ÷100 可靠」 | **只有约 29%** —— icu5000 的 chartevents 里有大量**文本型 chart 行**，`value` 列记的是描述性文本（`Normal` / `Medium` / `3 rails up`）⇒ **照抄 K29 那句会误导 agent**，prompt 已改（见 §2.2） |

### 2.1 ②的实测：别名规则

```
column=itemid 的注入共 2,207,800 条（chartevents 1,632,972 + inputevents 574,828）
  全部 2,207,800 条（100.0%）符合：dirty = "992" + (2 或 3) + clean_value 末 4 位
  前缀分布：9922 = 2,206,749 条、9923 = 1,051 条
  涉及文件：icu/chartevents.csv（1,632,972）、icu/inputevents.csv（574,828）
```

⇒ agent 只要把 `992{2,3}XXXX` 还原成 `220XXX` 就能修好这 163 万条 key 列错误。
这条规则**写进了 prompt**（K29 的 prompt 里没有这一层）。

### 2.2 ⭐ `value` 文本列在 icu5000 上**只有约三成可用**（K29 是 99.6%）

K29 的 prompt 里有这么一段（直接抄过来会出事）：

> ⇒ **反过来也成立，对你修注入错误同样有用**：注入类的错误……**在 raw 里的 `value` 文本列记的就是那个正确值**
> —— 实测 `unit_scale_error` 的 497,408 格里 495,623 格（99.6%）、`sentinel_chart_value` 的 1,935,724 格里
> 634,397 格的 `value` 直接等于 log 里登记的 clean_value。
> ⇒ **修这类注入错误时，直接采用 `value` 文本列比"÷100"或"猜一个中位数"都可靠**。

**icu5000 实测（抽样 20,000 格/类）**：

| 类别 | 档位 | log 里共 | `value` == clean_value | `value` **不是数值** | 别的数值 |
|---|---|---:|---:|---:|---:|
| `unit_scale_error` | 1.6pct | 1,360,895 | 5,849（**29.2%**） | **14,131（70.7%）** | 20 |
| | 4.7pct | 4,750,154 | 6,075（**30.4%**） | **13,897（69.5%）** | 28 |
| `sentinel_chart_value` | 1.6pct | 559,135 | 5,995（**30.0%**） | **13,992（70.0%）** | 13 |
| | 4.7pct | 8,033,664 | 5,966（**29.8%**） | **13,996（70.0%）** | 38 |

⇒ **两档结论一致（约 30% 可用 / 70% 是文本）**，不是抽样噪声。

⇒ 那七成的 `value` 记的是**护理描述性文本**，不是数值 —— 例如
`Face tent`、`Medium`、`Normal`、`Refer to restraint policy`、`3 rails up`、`2 Person Assist`。
**根因**：icu5000 的 `icu/chartevents.csv` 里有大量**文本型 chart 行**（氧疗方式、约束等级、
活动能力、疼痛评估…），它们的 `value` 列本来就不该是数值，而注入器给它们的 `valuenum`
注入了数值类错误（×100 / 哨兵码）。

⚠️ **但判据本身仍然成立**：验证器确认 test 的 2,450 个原生格 **`value` 文本列 100% == `valuenum`** ——
原生错误格是数值型 itemid，`value` 记的就是那个坏数本身。
⇒ **「`value == valuenum` ⇒ 原生错误」这条判据照常有效**；
   失效的只是「反过来拿 `value` 去修注入错误」这条**修复建议**。

⇒ prompt 已改：改成「**只有当 `value` 能解析成数值、且与 `valuenum` 明显不在一个量级时，
才优先采用 `value`；`value` 不是数值的那些格（文本型 chart 行）不要套这条**」。
（工具：`reproduction/icu5000_value_reliability.py`，输出按 6 类归并的计数，不含病人级信息。）

---

## 3. 数据集构造

### 3.1 判定规则（沿用 K29 ㉓，未改）

| 规则 | 内容 |
|---|---|
| **T1 哨兵错误码** | `valuenum ∈ {-999,-998,-997,-990,9999,99999,999999,9999999,220000,220045}` —— 对**所有** itemid 生效 |
| **T2 生理不可能区间** | itemid 命中 20 项 `HARD_RANGE` 且越界且 `!= 0` |
| **R2 修复** | 该 (subject_id, itemid) 在**最终干净表**里「可用读数」的中位数；类 1 的可用 = 区间内且 ≠ 0，类 2 的可用 = 非哨兵且 ≠ 0；一个可用读数都没有 ⇒ **保持原样不动** |

**值 0 一律不算错** —— MIMIC 里 0 = 未测到 / 呼吸机暂停，真值不可判定。

### 3.2 构造实测（1.6pct 档）

| 项 | 数量 |
|---|---:|
| reference 里的原生错误候选 | **2,658**（T2 越界 2,496 + T1 哨兵 162） |
| 可修（该患者该项有可用读数） | **2,591** |
| 不可修（无可用读数 ⇒ 保持原样） | **67** |
| ① 同时被注入器叠了脏值 ⇒ raw 恢复成原生错值、丢弃同格的注入条目 | **164** |
| ② itemid 也被注入器改过 ⇒ **两层错误叠加、不可判定 ⇒ 不进 native 层** | **141** |
| **最终登记进 native 层（log 追加条目数）** | **2,450** |

log 侧：原 log 10,009,412 条；chartevents 插入行 1,740,790 行；被注入改过 valuenum 的格 1,920,194。

### 3.2b ⚠️ 两档的**候选层**相同，但**最终登记层不同**（K29 不是这样，icu5000 是）

4.7pct 档的构建实测：候选 **2,658** 格（T2 2,496 + T1 162）、可修 2,591、不可修 67 —— **与 1.6pct 档逐项相同**。

原因：**两档的 `reference/` 是同一份干净的原始 MIMIC**（同一个 baseline `icu_5000_clean_eval_10_20_rest`），
注入器只改 `raw`、不改 `reference` ⇒ **MIMIC 原生错误是原始数据自带的，两档的候选层完全一样**。

**但最终登记进 native 层的格数两档不同**：

| 项 | 1.6pct | 4.7pct |
|---|---:|---:|
| 候选层（reference 里的原生错误） | 2,658 | **2,658（相同）** |
| ① 叠脏（⇒ 恢复 raw 值 + 丢注入条目，仍进 native 层） | 164 | **854** |
| ② itemid 也被注入器改过（⇒ **不进** native 层） | 141 | **417** |
| **最终登记进 native 层** | **2,450** | **2,174** |

差异来源：**注入强度不同** ⇒ ① 被注入器叠脏的格更多；② itemid 别名注入率更高
（K29 实测 4.7pct 46% vs 1.6pct 2.7%）⇒ 更多原生格撞上「两层错误叠加」而被划归注入层。

⇒ ⚠️ **跨档比原生错误修复率时，分母不同（2,450 vs 2,174），不是同一批格**。
要严格跨档可比，得只看**两档都登记了的交集格**；否则要把「4.7pct 少评 276 格」这件事记在结论里。
⇒ 这与 K29 不同（K29 两档的原生层是**逐格相同**的 873 格）—— 差别就在 §2 的第②③条上。

### 3.3 ⚠️ 构造过程中抓到的两个 bug（都已修）

1. **`fixes` 没剔除「itemid 也被改过」的 141 格** ⇒ reference 改了、log 却没登记 ⇒ agent 修对了反而被判错。构建器的自检（`改了 N 格但登记 M 条`）把它抓了出来。已修：排除的格同时从 `fixes` 里剔掉，`rewrite_reference` / `drop_valuenum_rows` 都用剔过的集合。
2. **`NATIVE_REPR` 一开始按 raw 的 itemid 分组** ⇒ 把注入器改出来的 `9923xxxx` 当成了测量项（清单里出现 57 项，其中一半是假 itemid）。已修：改按 **reference 的 itemid** 分组 ⇒ 57 项全部是真 itemid。

### 3.4 `NATIVE_REPR` 必须按 icu5000 自己统计

注入 train/validation 示例时，脏值的**形态**必须和 test 的真实分布一致，否则 agent 学成「只认 K29 的形态」、在 test 上大面积漏检。

| | itemid 数 | 来源 |
|---|---:|---|
| K29 ㉓ | 20 | 统计自 K29 test 的 510 个纯原生格 |
| **ICU5000 ㉔** | **57** | 统计自 icu5000 test 的 2,473 个纯原生格（`--stat-repr` 生成） |

统计口径：候选格 = reference 里「区间越界且 ≠ 0」或「哨兵码」；**纯原生** = 该格在 raw 里的 `value` 文本列 == `valuenum` 且 raw 的 itemid 没被注入器改过；⇒ per-itemid 中位数。
（哨兵码形态的 itemid 一律是 `999999.0` / `9999.0`。）

---

## 4. 校验

`reproduction/verify_icu5000_mimic_raw_error.py` 独立重算、不信构建器的自检。

### 4.1 1.6pct 档：**✅ 全部通过**（exit 0）

| # | 检查项 | 实测 |
|---|---|---|
| ① | test 的 raw 与源数据集的差异**只**出现在「叠脏格」上 | 差异 **164 行** = 叠脏 164 格 ✅ |
| ② | reference 里残留的原生错误格 = ①不可判定 + ②itemid 叠脏 | 残留 **208** = 67 + 141 ✅ **精确吻合** |
| ③④ | log 里 `mimic_native_*` 条目数 == 溯源表行数；每格**恰好 1 条** valuenum 条目 | 2,450 == 2,450；重复 0、缺 0 ✅ |
| ①b | test 每个原生格的 raw `value` 文本列 == `valuenum` | **2,450 / 2,450** ✅ |
| ⑤ | log 的 `raw_row_index` 与 reference 行号对齐 | 抽样 500 条，不一致 **0** ✅ |
| ⑤b | 用**源数据集** reference 独立重算 R2 中位数 | **2,450 / 2,450** 一致，跳过 0 ✅ |
| ⑥ | train / validation 的注入示例 | train 20 例、validation 50 例，全部通过 ✅ |
| ⑦ | 溯源表列齐全 | 19 列 ✅ |

⚠️ **② 是对新排除逻辑的一次独立验证**：verify 从产物重算「残留的原生错误格」，
得到 **208**，正好等于构造时判定的「67（无可用读数）+ 141（itemid 叠脏）」——
两处独立算出来的数分毫不差。

类别分布：`T2_impossible` 2,307 + `T1_sentinel` 143 = 2,450；
其中同时被注入器叠脏 **164**（与构造值一致）。

### 4.2 4.7pct 档

（构建完成后跑，结论见下）

---

## 5. prompt 的改动

icu5000 的基础 prompt（`eval_handoff/prompts/icu5000_{1.6pct,4.7pct}_val_prompt_v1.md`）已经定稿，
`make_icu5000_mimicraw_prompt.py` 是**插入式**的：

1. 把「错误注入量级」那三行的数字换成 mimicraw 档的实测值（多了一类原生错误）；
2. 插入 `## 新增错误类：MIMIC 自身的原生错误（不是注入器造成的）` 一节 —— **K29 ㉓ 的同款结构**
   （T1/T2 两类、R2 中位数修复、`value` 文本列判据、去重必须放在所有修复之后、枚举型多值不是重复行），
   但里面引用的每个数字都换成 **icu5000 自己的实测值**（偏离度分布、CVP 坏值范围与「÷100」判据的失败率、
   `value` 文本列的可靠性比例、train/validation 的示例覆盖）；
3. 插入 `### ⚠️ 另外：itemid 列上还有一批 CareVue 别名注入` 一节 —— **icu5000 特有**，
   告诉 agent `992{2|3}XXXX` 的还原规则，并强调「先修 key 列，再修 valuenum，最后去重」。

---

## 6. 待办

- [ ] 4.7pct 档的构建与校验
- [ ] 枚举型 itemid 全量扫描（确认 icu5000 没有 K29 之外的高频多值 itemid）
- [ ] harness 跑分（本次 scope 只到「可直接跑」的产物，跑分留待后续）
