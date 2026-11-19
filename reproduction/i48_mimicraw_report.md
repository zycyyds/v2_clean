# ㉕ I48「MIMIC 原生错误」增强版（mimicraw2）

> **本档是什么**：在 I48（房颤 ECG 任务族）上，把 **MIMIC 自己录错的原生错误**
> （T1 哨兵占位码 + T2 生理不可能区间）按「该患者该 itemid 的可用读数中位数」修好，
> 并登记进 `modification_log.csv` ⇒ agent 在 train 对照里第一次能看见这类错误模式。
>
> **产物**：`datasets/注入错误数据集/i48/I48_atrial_fibrillation_v2_rawerror_mimicraw2_test{1.6,4.7}pct_411M/`
> （**新建目录**，旧的 `mimicrawerror_*` 及其历史跑分记录原封不动保留）
>
> **本会话的边界**：只到「可直接跑」的产物（数据集 + 校验 + 视图 + prompt + manifest +
> harness profile + 跑批脚本），**不跑 agent**（用户明确要求）。

---

## 1. 与旧版（`mimicrawerror_*`）的差别

> 旧版是 2026-07-07 ~ 07-09 建的，在另一条分支（`k29-agent-rawerror-run`）上。
> 审计发现它相对 icu5000 的标准缺 4 处，本版全部补齐。

| # | 旧版 | **本版（mimicraw2）** |
|---|---|---|
| 1 | 原生错误规则**只有 T2**（label 硬区间越界），**没有哨兵码类** | **T1 哨兵占位码 + T2 越界**（与 icu5000 同构） |
| 2 | 「itemid 也被注入器改过」的格**留在 native 层**（两层错误叠加、形态同构、不可判定） | **排除**，交给注入层评测（1.6pct test 排除 7 格） |
| 3 | 叠脏格在 raw 里躺的是**注入层的脏值**（9999 / -888），形态与真实原生错误相反 | **把 raw 恢复成 MIMIC 原生错值**（`value` 文本列同步）⇒ 一格只剩一个错误（1.6pct test 恢复 203 格） |
| 4 | 独立校验只有 1 类检查、只查 train/validation、不写 JSON | **8 大项**、三 split 全查、**写 JSON** |

另外两处工程修正：
- **评分器的索引空间判定**改成「数据探测」（见 §4）—— 旧判据对 i48 会**误判**。
- **prompt** 加了 3 节（见 §5）—— 旧版没有哨兵码类、没有 itemid 别名节、
  也没有「哨兵码清单不是封闭的」。

---

## 2. 构造规则

### 2.1 检测（T1 + T2）

| 类 | 判据 | 覆盖 |
|---|---|---|
| **T1 哨兵占位码** | `valuenum ∈ {-999,-998,-997,-990,9999,99999,999999,9999999,220000,220045}`<br>**对所有 itemid 生效** | test 34 格（1.6pct 档） |
| **T2 生理不可能区间** | itemid 命中「硬不可能区间」且越界（**且 ≠ 0**）<br>区间按 **PhysioNet d_items 的权威 label 文本**匹配 ⇒ **i48 自己的区间**，不是照抄 K29 的 20 个 itemid 表 | test 1,273 格 |

- **值 0 一律不算错**（MIMIC 里 0 = 未测到 / 呼吸机暂停，真值不可判定）。
- 区间的 11 条 label 子串见 `reproduction/build_i48_mimic_raw_error.py` 的
  `HARD_BY_LABEL`；含 loss/unintentional/note/comment/finding/change/alarm 的 label 排除。

### 2.2 修复（R2：患者-itemid 中位数）

修复值 = **该 (subject_id, itemid) 在 reference 里「可用读数」的中位数**
（偶数个取中间两个的平均，与 numpy/pandas/statistics.median 一致）。

- 类 1（越界）的可用读数 = **落在该 itemid 区间内**且 ≠ 0；
- 类 2（哨兵码）的可用读数 = **不是哨兵码**且 ≠ 0。

**中位数必须在 pipeline 的最后一步算**（cohort 筛好、重复行删掉、注入错误修完之后），
否则池子里会混进别的住院 / 重复行 / 被注入弄坏的读数。

### 2.3 三条自检断言（构建器内部）

1. 改写 reference 的格数 == 登记进 log 的条数
2. 改写格数 + 被排除的 itemid 叠脏格数 == 可修格数
3. 恢复 raw 的格数 == 同时被注入器改过的格数

任何一条不过就 `SystemExit`（构建不落盘完整状态）。

### 2.4 ⚠️ i48 特有的「孪生行」：同一个 key 在 raw 里有两行

注入器在**部分原生格上**造了重复行（`duplicate_row` + 把那一行的 valuenum 改成
`-999` / ×100）⇒ **同一个 (subject, hadm, stay, charttime, itemid) 在 raw 里有两行**：

```
native_value=307.0 ⇒ raw 里这个 key 的所有行：
    valuenum='307.0'    value='307.0'   ← 原生格本身（value == valuenum ✅）
    valuenum='-999'     value='307'     ← 孪生行（重复注入）
```

⇒ **任何「按 key 找那一行」的检查都会踩坑**：
- 校验器 ①b（形态检查）按 key 取「最后一行」⇒ 撞上孪生行 ⇒ 误报 11 格「形态错」
  ⇒ 已改成按 **(key, valuenum == 原生错值)** 定位原生格那一行；
- 构建器按 key 取「第一行」匹配 ⇒ 有可能匹配到孪生行 ⇒ 把注入形态误当成原生形态
  ⇒ 已改成收集全部候选行、**优先选 valuenum == 原生错值**的那行。

⚠️ 但**构建器的产物逐字节不变**：按 log 行号定位（`build_raw2log`）本来就是对的，
重建对比 12/12 文件 md5 完全一致 ⇒ **两档数据集都不需要重建**。

---

## 3. 构造实测（两档）

> 两档的原生错误**候选层相同**（1,307 格 = T1 34 + T2 1,273）—— reference 都源自
> 同一份 MIMIC 清洗结果，原生错误只取决于 MIMIC 自己的数据，与注入密度无关。
> 但**最终登记层不同**（1,264 vs 1,238），差别来自「itemid 也被注入器改过」的格
> 被划归注入层（**7 vs 33**）—— 密度越高，itemid 别名注入碰到的原生格越多。

| | 1.6pct | 4.7pct |
|---|---:|---:|
| 注入层（test） | 6,709,417 | 21,544,193 |
| 原生候选（test） | **1,307**（T1 34 + T2 1,273） | **1,307**（同上） |
| 可修 | 1,271 | 1,271 |
| 不可修（该患者该项无可用读数 ⇒ 真值不可判定） | 36 | 36 |
| **itemid 叠脏格排除** | **7** | **33** |
| **登记进 native 层（test）** | **1,264**（T1 23 + T2 1,241） | **1,238**（T1 22 + T2 1,216） |
| raw 恢复成原生错值 | 203 | 957 |
| 丢弃与原生格重叠的注入条目 | 203 | 957 |
| train / validation 登记 | 9 / 11 | 11 / 11 |

### 3.1 ⭐ 两档的**交集口径**（跨档比原生修复率的唯一严格分母）

| | 格数 |
|---|---:|
| 1.6pct 登记 | 1,264 |
| 4.7pct 登记 | 1,238 |
| **交集（两档都登记）** | **1,231** |
| 只在 1.6pct（被 4.7pct 的 itemid 别名注入碰掉） | 33 |
| 只在 4.7pct（被 1.6pct 的碰掉） | 7 |

⇒ 交集上两档的**修复值逐格完全一致**（`intersect_repaired_mismatch = 0`、
`intersect_native_value_mismatch = 0`）⇒ 跨档比原生修复率时用**这 1,231 格**做分母。
复现：`python3 reproduction/i48_native_overlap.py --a <1.6pct> --b <4.7pct> --split test`

⚠️ 旧版登记的是 1,249 格（test）—— 差 15 格：
旧版**只做 T2**、**没排除 itemid 叠脏格**、且对同 key 的重复行会**重复登记**。

---

## 4. ⚠️ 评分器的索引空间判定（必须改，否则 i48 的分数是垃圾）

`score_only/error_metrics/compute_error_metrics.py` 要逐文件判定 log 的行号空间：

- **zero**：raw 行号 == log 行号
- **bisect**：raw 行号 = log 行号 + 它前面插入的行数

**旧判据**（看行数猜）：`raw 行数 − reference 行数 == log 里 insert/dup 条数` ⇒ bisect，否则 zero。

⚠️ 这只在「reference = 注入前的干净表」时成立：

| 数据集 | raw − reference | log insert+dup | 旧判据 | **真值** |
|---|---:|---:|---|---|
| icu5000 4.7pct | 5,850,717 | 5,850,717 | bisect | bisect ✅（碰巧相等） |
| **i48 1.6pct** | **4,470,588** | **1,222,255** | **zero ❌** | **bisect** |

**根因**：i48 的 `reference` 是**处理后的输出表**（cohort 筛选后 1,277 万行），
不是注入前的干净表（1,602 万行）⇒ 差值里还混着「被 cohort 筛掉的行」。

⇒ **改成数据探测**（`probe_offset_mode()`）：抽 log 条目（带 `dirty_value`），
分别按 zero / bisect 映射到 raw 行号，看**哪个映射能让 raw 那一行的值真的等于 dirty_value**。

**实测（三个数据集，chartevents）**：

| 数据集 | zero 命中 | bisect 命中 | 判定 |
|---|---:|---:|---|
| icu5000 1.6pct | 3 / 210 | **210 / 210** | bisect（旧判据一致） |
| icu5000 4.7pct | 12 / 253 | **253 / 253** | bisect（旧判据一致） |
| **i48 1.6pct** | 7 / 210 | **210 / 210** | **bisect**（旧判据会误判成 zero） |

探测结果写进输出 JSON 的 `offset_probe` 字段（可审计）；
探测无法判定时（zero/bisect 映射相同）才退回行数判据。

### 4.2 ⭐ gold 取值空间（`--clean-from-gold` 那三个口径）—— **又一个 i48 会踩的坑**

`goldv29 / goldstr / goldold` 三个口径把「正确答案」改从 **reference** 取，
而**取 reference 那一行**历史上是按 **rowid**（= log 的 `raw_row_index` + 1）——
这只在「reference 与 log 同源同序」时成立。

| 数据集 | reference 是什么 | log 的行空间 | rowid 能对上吗 |
|---|---|---|---|
| icu5000 / K29 | 注入前的干净表 | 同一份 ⇒ 同一行号 | ✅ |
| **i48** | **处理后的输出表**（cohort 筛选后 12,768,419 行） | 源 raw（16,016,752 行） | ❌ |

**实测（i48 1.6pct，1,264 个原生格）**：按 key 在 reference 里定位到 1,264 / 1,264，
但 **reference 行号 == log 行号的只有 8 格**（碰巧），行号差最大 **324 万**
⇒ 照旧取 rowid 就会把**随机行**的值当「正确答案」⇒ **三个 gold 口径的分数是垃圾**。

**改法**（与 §4 同一套思路：数据探测，不写死）：
`probe_gold_mode()` 抽 log 条目，看 **reference 第 L 行的 key** 是不是等于
**raw 里那一行（log 行号 L 对应的那一行）的干净 key**（被注入改过的 key 列
用 log 的 `clean_value` 还原）⇒ 命中率高 = **rowid 模式**（历史行为，不变），
命中率极低 = **key 模式**（按干净 key 查；同 key 多行时取「该列值 == clean_value」的那一行）。

⚠️ **两个坑**：
① **取样必须跨全日志均匀铺开**：只取最前面的 200 条会有 **23% 假命中**
（i48 的 reference 保留了源 raw 开头的行）⇒ 判定不了「未定」⇒ 退回 rowid ⇒ 依然错。
② **判据必须比 key，不能只比「列值 == clean_value」**：注入的列常是**低基数**的
（insurance / language / expire_flag）⇒ 拿随机行的值也会撞上 ⇒ 命中率虚高。
实测 i48 的 `hosp/admissions.csv`：rowid 取到的行 **key 全不同（400/400）**，
但列值命中 **59.5%** ⇒ 判成「未定」⇒ 退回 rowid ⇒ 依然取错行。

**回归实测**：

| 数据集 | 判定 | key 命中率 | 行为 |
|---|---|---:|---|
| icu5000 1.6pct | **rowid** | 200/200 | 与改动前**逐字节一致** ✅ |
| icu5000 4.7pct | **rowid** | 200/200 | 同上 ✅ |
| **i48 mimicraw2 1.6pct** | **key** | 1/200 | 改走干净 key 查询 ✅ |

判定结果写进输出 JSON 的 `gold_probe` 字段（可审计）。

### 4.3 ⭐ 原生格专项判定器也要 i48 专用版

`reproduction/eval_native_cells.py`（icu5000 版）靠「log 行号 == reference 行号」
把原生格和 gold 行对上 ⇒ **对 i48 不成立**（同上，行空间不同）。
⇒ 新增 `reproduction/eval_i48_native_cells.py`：**按溯源表里的干净 key** 在 agent
输出里查（溯源表 `mimic_native_error_log.csv` 带 subject/hadm/stay/charttime/itemid 五列）。

自测（拿数据集自己的 reference 当 agent 输出）：**1264 / 1264 全 b_clean** ✅
（reference 那一格的值就等于修复值）⇒ 定位与判定链路通。

---

### 4.4 ⚠️ **old 口径在 i48 上有结构性偏差**（`--keep-rows-not-in-gold` + `--no-recover-clean-key`）

i48 的任务本身含 **cohort 筛选**（prompt 里明写「cohort 选择：入选依据需要你从
train 对照里推导出来」）⇒ raw 的行不是都该出现在结果里：

| | raw 行 | reference 行 | agent **该删**的行 |
|---|---:|---:|---:|
| i48 1.6pct（test chartevents） | 17,239,007 | 12,768,419 | **3,837,707（22.3%）** |
| i48 4.7pct（test chartevents） | 20,132,205 | 12,768,419 | **5,533,731（27.5%）** |

old 口径有**两个**开关会把与 agent 能力无关的格记成 **b_missing（假漏检）**：

| 来源 | 开关 | 1.6pct | 4.7pct |
|---|---|---:|---:|
| ① **cohort 筛掉的行**上的注入格 | `--keep-rows-not-in-gold` | 650,544（20.3%） | 3,070,497（20.3%） |
| ② **key 列被注入改过**的行上的注入格<br>（agent 修好了 key ⇒ 按脏 key 查不到它的行） | `--no-recover-clean-key` | 782,704（24.4%） | 4,052,166（26.7%） |
| **①+② 合计** | | **1,433,248（44.6%）** | **7,122,663（47.0%）** |
| ③ 真正该评的（gold 里有该行） | | 1,777,268（55.4%） | 8,028,651（53.0%） |

⇒ **在 i48 上，超过 44% 的注入格在 old 口径下都会变成假漏检**。
`b_missing` 直接进 `fn_det`（漏报）⇒ **old 口径的 Det F1 被结构性压到 ~0.71 上限**
（1.6pct；4.7pct 更低）。

**ICU5000 对照（证明这是 i48 独有的）**：

| | icu5000 4.7pct | i48 1.6pct | i48 4.7pct |
|---|---:|---:|---:|
| 注入格总数 | 22,295,033 | 3,210,516 | 15,151,314 |
| old 口径记成假漏检 | **0（0.0%）** | 1,433,248（44.6%） | 7,122,663（47.0%） |

⇒ ICU5000 的注入错误**天然只落在 gold 存在的行上**（它的 1.6pct 日志是事后
diff raw↔gold 得到的）⇒ 它的 old 口径没有这一项。
⇒ **跨档比 old 口径时，i48 天然吃亏** ⇒ 必须同时给出 v29 / goldv29 才能比。

---

## 5. prompt（v5）
基线 = 旧版的 v3（1.6pct）/ v4（4.7pct），加 4 处：

1. **「类 2：哨兵占位码」**（T1，对所有 itemid 生效）+ 两类原生错误的修复池子差异；
2. **「哨兵码清单不是封闭的」** + 要求 agent 自己统计 `valuenum` 取值分布
   —— **ICU5000 实测**：prompt 把哨兵码清单写成封闭集合时，agent 会漏掉注入层用的
   `-1` / `-888`，4.7pct 档因此有 **281 万格**哨兵码一格没修（`b_dirty` 263 万）。
   i48 的 `sentinel_chart_value` 注入量级是 **107.9 万（1.6pct）/ 771.1 万（4.7pct）**，
   同样的坑必须提前堵；
3. **itemid 别名注入一节** —— i48 的 test 有 **12.3 万（1.6pct）/ 39.7 万（4.7pct）**条
   `chart_itemid_alias_split`，旧版 prompt 完全没提；
   还原规则「原 itemid = `"2" + alias[3] + alias[-4:]`」**在 i48 上实测 100% 成立**
   （1.6pct test：122,942 / 122,942）；
4. **量级三行按实测重算** + 追加**原生错误层的规模**（train 9 / validation 11 / test 1,264）。

---

## 6. 独立校验（**从产物重算，不信构建器的自检**）

`reproduction/verify_i48_mimic_raw_error.py`，8 大项 × 三 split × 两档
⇒ **两档各 38 项检查，38 项全部通过、0 失败**
（报告落在各数据集的 `verify_native_report.json`）。

| # | 检查项 | 1.6pct | 4.7pct |
|---|---|---|---|
| ① | raw 与源 raw 的差异**只**出现在叠脏格上 | 203 行 == 203 格 ✅ | 957 == 957 ✅ |
| ② | reference 里**残留**的原生错误格（真值不可判定那批） | 0 ✅ | 0 ✅ |
| ③ | log 里 `mimic_native_*` 条目数 == 溯源表行数 | 1,264 == 1,264 ✅ | 1,238 == 1,238 ✅ |
| ④ | 每个原生格在 log 里**恰好 1 条** valuenum 条目 | ✅ | ✅ |
| ①b | 每个原生格的 raw `value` 文本列 **== `valuenum`**（原生错误形态） | **1,264 / 1,264** ✅ | **1,238 / 1,238** ✅ |
| ⑤ | reference[该格] == 修复值 | **1,264 / 1,264** ✅ | **1,238 / 1,238** ✅ |
| ⑤b | 用**源数据集** reference **独立重算** R2 中位数 | **1,264 / 1,264**，不一致 0 ✅ | **1,238 / 1,238**，不一致 0 ✅ |
| ⑥ | 溯源表 19 列齐全 + T1/T2 分类 | T1 23 + T2 1,241 ✅ | T1 22 + T2 1,216 ✅ |
| ⑦ | manifest / prompt / 各 split 产物齐备 | ✅ | ✅ |

⚠️ **这三轮调试查出的都是校验器自身的 bug，数据本身一直是对的**：
①b 按 key 定位撞上「孪生行」（注入器在同一 key 上造重复行）；
①b 按 clean key 找不到行（有一批格的 key 列本身被注入器改过）；
改 ①b 时误删了 ⑤ 用的 `want_keys`（NameError）。
⇒ **校验器与构建器/评分器统一成同一套定位：raw 行号 = log 行号 L + 插入偏移。**

⚠️ 另外做过一次**重建逐字节对比**（重建到临时目录）：
12/12 个文件 md5 完全一致 ⇒ 构建器那处修改对产物**零影响** ⇒ 两档都不需要重建。

---

## 7. 产物清单

| | 1.6pct | 4.7pct |
|---|---|---|
| 完整数据集 | `datasets/注入错误数据集/i48/I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M` | `…_mimicraw2_test4.7pct_411M` |
| agent 视图 | `datasets/val_views/i48_mimicraw2_16` | `datasets/val_views/i48_mimicraw2_47` |
| prompt | `eval_handoff/prompts/i48_1.6pct_mimicraw_val_prompt_v5.md`<br>sha256 `beeed442c825f18d…` | `…_4.7pct_…_v5.md`<br>sha256 `36a90bb561faf311…` |
| manifest | `<数据集>/v2_rawerror_i48_final_report.json` | 同左 |
| harness profile | `i48_mimicraw2_16` | `i48_mimicraw2_47` |

**脚本**：

| 脚本 | 作用 |
|---|---|
| `reproduction/build_i48_mimic_raw_error.py` | 构建器（T1+T2 / 叠脏排除 / raw 恢复 / 三断言 / `--stat-repr`） |
| `reproduction/verify_i48_mimic_raw_error.py` | 独立校验器（8 大项，写 JSON） |
| `reproduction/make_i48_manifest.py` | manifest（注入层取自源数据集，原生层从产物实测） |
| `reproduction/make_i48_mimicraw_prompt.py` | prompt v5 生成器 |
| `reproduction/i48_alias_verify.py` | itemid 别名规则实测 |
| `scripts/prepare_i48_mimicraw_run.sh` | manifest → 视图 → prompt → 同步 → 独立校验 |
| `scripts/run_val_harness_i48_mimicraw.sh` | val harness（跑前自检 + 泄漏检查） |
| `scripts/run_test_harness_i48_mimicraw.sh` | test 纯重放 |
| `scripts/score_i48_mimicraw_four_calibers.sh` | 七口径 + 原生格重判 |

---

## 8. 跑分结果（1.6pct）

### 8.1 主结果

| 项 | 值 |
|---|---|
| **val composite** | **0.895719**（最佳轮 = 第 3 轮；第 1/2/3 轮 = 0.894668 / 0.895131 / 0.895719） |
| **val 独立重放** | **0.895719**（完全复现 ⇒ `SUCCESS_REPRODUCIBLE`） |
| **test composite** | **0.896069**（`status=SUCCESS`、`replay_status=SUCCESS`、`test_execution_count=1`） |
| test 结果包 | `i48_agent_run/experiments/i48_mimicraw2_1p6_test_seed666_v1/host/test_result_package`（chartevents 12,768,205 行，比 reference 少 **214 行 / 0.0017%**） |
| **原生错误格（1,264）** | b_clean **1,240（98.1%）** / b_wrong **0** / b_dirty 22 / b_missing 2 |

> ⚠️ **val 跑到第 6 轮时撞上 provider 侧的流式分块错误**（MiniMax 报
> `ValueError: Separator is found, but chunk is longer than limit`，连接零字节 10.5 小时），
> 后续轮次无法完成 ⇒ 用**驱动自带的兜底收尾** `finalize()`（= harness 自己的
> `except BaseException` 分支）跑完独立重放并冻结快照。
> ⇒ **val 的 status 仍是 `SUCCESS_REPRODUCIBLE`**（独立重放真跑了一遍、分数逐位复现），
> 但只覆盖了前 5 轮记录的最好结果，不是 20 轮跑满。

### 8.2 七口径错误级指标（test）

| 口径 | 值比对 | key 还原 | 排除 gold 外行 | 判据 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **v29** | 数值 | ✅ | ✅ | log | **0.998638** | **0.907893** | 3,855,349 | 491,244 | 5,266 | 2,458 |
| str | 纯字符串 | ✅ | ✅ | log | 0.996127 | 0.763085 | 3,055,882 | 1,257,052 | 38,925 | 2,458 |
| oldsw | 数值 | ❌ | ❌ | log | 0.733084 | 0.716098 | 2,645,012 | 87,809 | 5,083 | **2,750,319** |
| **old（完全老口径）** | 纯字符串 | ❌ | ❌ | log | **0.729628** | **0.594346** | 2,003,505 | 697,365 | 37,034 | **2,750,319** |
| goldv29 | 数值 | ✅ | ✅ | gold | 0.998638 | 0.907893 | 3,855,349 | 491,244 | 5,266 | 2,458 |
| goldstr | 纯字符串 | ✅ | ✅ | gold | 0.999251 | **0.908271** | 3,854,401 | 492,213 | 5,245 | 2,458 |
| goldold | 纯字符串 | ❌ | ❌ | gold | 0.733555 | 0.716555 | 2,645,012 | 87,830 | 5,062 | **2,750,319** |

三个可验证的一致性：

1. **v29 与 goldv29 逐位相同**（Det/Rep/b_clean 全一致）⇒ gold 判据没有改变数值口径的语义；
2. **goldstr 的 Rep（0.908271）比 str（0.763085）高 0.145** ⇒ 差额全是「写法税」
   （`natural_noise` **3,742,039** 格：log 写 `'116'`、gold 写 `'116.0'`）；
3. **old / oldsw 的 b_missing 正好 2,750,319** ⇒ 与 §4.4 独立量化的结构性惩罚
   （cohort 筛掉的行 + key 被注入的行）**完全对上** ⇒ old 口径在 i48 上的 Det/Rep
   被系统性压低是**口径问题，不是 agent 问题**。

### 8.3 ⭐ 原生错误层：**越界类几乎全修好，哨兵码类几乎全漏**

| 类 | 格数 | 修对 | 率 |
|---|---:|---:|---:|
| **T2_impossible（生理不可能区间）** | 1,241 | **1,239** | **99.8%** |
| **T1_sentinel（哨兵占位码）** | 23 | **1** | **4.3%** |

⇒ 22 格没修的，**错值清一色 `999999.0`**（`prothrombin time` 7、`inr` 7、
`creatinine`/`glucose`/`bun`/`ph`/`base excess`/`o2 pressure`/`co2 pressure`/`specific gravity` 各 1）。

**归因（两条独立证据链）**：

| split | 原生层构成 | 注入层哨兵码取值 |
|---|---|---|
| **train** | T2 **9** 格、T1 **0** 格 | `-1` / `9999` / `-888` / `-999`（**不含 999999**） |
| **validation** | T2 **11** 格、T1 **0** 格 | 同上 |
| **test** | T2 1,241 + T1 23（**全是 999999**） | — |

① **train/validation 里一个原生哨兵码样本都没有**（i48 的 train 是 15.7 万行的小分片，
test 是 1,702 万行 ⇒ 原生层密度相同但样本量差 100 倍）⇒ agent 没有任何
「MIMIC 自己录错的哨兵码」可学；
② **注入层在 train 里用的哨兵码只有 -1/-888/-999/9999** ⇒ agent 学到的是
「这几个具体值脏」，不是「哨兵码这一类」⇒ 到 test 见到 999999 就不认了。

⇒ 而 T2 越界类**不需要 train 样本**（物理常识 + label 文本即可推出区间）⇒ 修得极好。

⚠️ **这指向数据集设计的两个可改点**（不是 agent 的锅）：
- **train/validation 分片应放大**（对齐 icu5000：train 分片大 46 倍），
  让原生错误类在 train 里就有足够样本；
- 或者在 prompt 里把哨兵码**按类**讲清（含 999999），并强调「不是一个固定清单」。

---

## 9. 待办

- [x] **跑 agent（1.6pct）** ⇒ val 0.895719 / test 0.896069 / 七口径 / 原生格 1,240（98.1%）✅
- [ ] **4.7pct 档跑 agent**（第 1 次尝试撞上 provider 流式分块错误、第 1 轮未完成 ⇒ 无快照可用；
      已归档为 `…_attempt1_provider_hang`，第 2 次尝试重跑中）
- [ ] 两档原生层的**交集口径**跨档对比（1,231 格；脚本 `reproduction/i48_native_cross_tier.py` 已就绪，
      4.7pct 跑完后即可出）
- [ ] 数据集设计改进（见 §8.3）：train/validation 分片放大 + prompt 的哨兵码按类讲清
- [ ] icu5000 4.7pct 的 gold 探针在**严格版**（全量 cell 字典）下的判定复核
      —— 先前测到的「未定」是我测试脚本按条目采样导致某些行缺 key 列还原信息，
      正式评分器用全量字典不受影响；直连诊断也确认 icu5000 的 reference 行数
      （32,315,301）== log 行空间上限+1 ⇒ rowid 路径本来就是对的。

---

## 10. 本轮改动清单（评分器 / 工具）

| 文件 | 改了什么 |
|---|---|
| `score_only/error_metrics/compute_error_metrics.py` | ① `probe_gold_mode()`：新增「gold 取值空间」数据探测（rowid / key 两种模式，判定结果写进输出 JSON 的 `gold_probe`）<br>② `clean_key_of()` + `DB.rows_with_key()`：key 模式下的 gold 行定位<br>③ 探针判据改为**比 key**（低基数列上「列值相等」会假命中） |
| `reproduction/eval_i48_native_cells.py`（新增） | i48 专用原生格判定器（按干净 key 查；带 any 口径 + 歧义统计） |
| `reproduction/i48_native_cross_tier.py`（重写） | 跨档交集口径对比（1,231 格） |
| `scripts/score_i48_mimicraw_four_calibers.sh` | 原生格重判改调 i48 专用版（icu5000 版对 i48 不成立） |

**回归验证**：

- icu5000 1.6pct / 4.7pct ⇒ gold 探针判 **rowid** ⇒ 与改动前**逐字节一致** ✅
- i48 mimicraw2 1.6pct ⇒ 七张表判 **key**（patients 除外 ⇒ rowid，正确）✅
- i48 冒烟（agent 输出 = reference）：v29 与 goldv29 **完全一致**（Det 1.000000 / Rep 0.998710，
  b_clean 4,347,329 / b_dirty 0 / b_missing 0）⇒ 修复前的 rowid 路径会多出 10 个假 b_dirty ✅
