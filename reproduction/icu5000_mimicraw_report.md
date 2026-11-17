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

## 2. 与 K29 ㉓ 的差异（**三处必须单独处理**）

| 维度 | K29 ㉓ | **ICU5000 ㉔（本次）** |
|---|---|---|
| **① raw 的行索引空间** | 4.7pct 带偏移、1.6pct 零偏移（**两档不同**） | **混合**：chartevents 是 bisect 偏移（raw 比 reference 多 1,740,790 行）；inputevents / outputevents 是零偏移 ⇒ **必须逐文件自动判定**，写死任何一种都会读错行 |
| **② itemid 列上的别名注入** | K29 没有这一层 | **1,632,972 条 `chart_itemid_alias_split`**（chartevents）+ 574,828 条 `medication_itemid_alias_split` ⇒ **raw 的 itemid 被改成了 `992{2\|3}XXXX`**（实测 99.94% 符合「`992` + (2 或 3) + 原 itemid 末 4 位」）。这是 **key 列**上的错误，不修就大面积键对不上 |
| **③ 两层错误叠加（itemid 那一层）** | 构建器只处理了 valuenum 那一层 | **141 格**的 itemid 也被注入器改过、同时又是原生错误格 ⇒ 两层错误形态同构、不可判定 ⇒ **不进 native 层**，留给注入层评测（K29 报告 §2.1 原则的延伸） |

### 2.1 ②的实测：别名规则

```
chart_itemid_alias_split 共 1,632,972 条
  其中 1,631,929 条（99.94%）符合：dirty = "992" + (2 或 3) + clean_value 末 4 位
  反例形态：230045 → 99230045（前缀是 9923 而不是 9922）
  涉及文件：全部在 icu/chartevents.csv
```

⇒ agent 只要把 `992{2,3}XXXX` 还原成 `220XXX` 就能修好这 163 万条 key 列错误。
这条规则**写进了 prompt**（K29 的 prompt 里没有这一层）。

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

`reproduction/verify_icu5000_mimic_raw_error.py` 独立重算、不信构建器的自检：

| # | 检查项 |
|---|---|
| ① | test 的 raw 与源数据集的差异**只**出现在「叠脏格」上（差异后的值 == MIMIC 原生错值） |
| ② | reference 里残留的原生错误格数 = ①不可判定 + ②itemid 叠脏（两类之和） |
| ③④ | log 里 `mimic_native_*` 条目数 == 溯源表行数；每个原生格在 log 里**恰好 1 条** valuenum 条目 |
| ①b | test 的每个原生格，raw 的 `value` 文本列都 == `valuenum`（与真实 MIMIC 原生错误同构） |
| ⑤ | log 的 `raw_row_index` 与 reference 行号对齐（抽样 500 条核对 `clean_value`） |
| ⑤b | 用**源数据集**的 reference 独立重算 R2 中位数，与登记的 `repaired_value` 逐格一致 |
| ⑥ | train/validation 的注入示例：raw 是脏值、reference 是中位数、log 有条目、dirty 值确实越界、`value` 文本列 == 脏值 |
| ⑦ | 溯源表列齐全 |

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
