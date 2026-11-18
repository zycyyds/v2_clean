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

---

## 3. 构造实测（两档）

> 两档的原生错误层**逐格同源**（reference 是同一份 MIMIC 清洗结果，
> 原生错误只取决于 MIMIC 自己的数据，与注入密度无关）
> ⇒ **跨档比原生修复率时，分子分母都天然同源**，不像 icu5000 还要先取交集。

| | 1.6pct | 4.7pct |
|---|---:|---:|
| 注入层（test） | 6,709,417 | 21,544,193 |
| 原生候选（test） | 1,307（T1 34 + T2 1,273） | 1,307（同上） |
| 可修 | 1,271 | 1,271 |
| 不可修（该患者该项无可用读数 ⇒ 真值不可判定） | 36 | 36 |
| **itemid 叠脏格排除** | **7** | **7** |
| **登记进 native 层（test）** | **1,264**（T1 23 + T2 1,241） | **1,264** |
| raw 恢复成原生错值 | 203 | 203 |
| 丢弃与原生格重叠的注入条目 | 203 | 203 |
| train / validation 登记 | 9 / 11 | 9 / 11 |

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

## 6. 产物清单

| | 1.6pct | 4.7pct |
|---|---|---|
| 完整数据集 | `datasets/注入错误数据集/i48/I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M` | `…_mimicraw2_test4.7pct_411M` |
| agent 视图 | `datasets/val_views/i48_mimicraw2_16` | `datasets/val_views/i48_mimicraw2_47` |
| prompt | `eval_handoff/prompts/i48_1.6pct_mimicraw_val_prompt_v5.md` | `…_4.7pct_…_v5.md` |
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

## 7. 待办

- [ ] **跑 agent**（用户明确要求「先不跑」）⇒ val / test / 七口径 / 原生格
- [ ] 两档原生层的**交集口径**跨档对比（两档 native 层已逐格同源 ⇒ 直接比即可，
      但跑完才能有分子）
