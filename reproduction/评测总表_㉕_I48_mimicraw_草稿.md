# ㉕ 条目草稿 —— I48「MIMIC 原生错误」增强版（mimicraw2）

> **这是草稿**：本次会话的 worktree 隔离在 `k29-mimicraw`，
> 下面这段可**直接粘贴**进 `评测总表.md`（接在 §㉔ 之后）。
> 数据来源：`reproduction/i48_mimicraw_report.md`（完整版）与 `reproduction/台账.md`。
> ✅ **两档（1.6pct / 4.7pct）都已跑完**；4.7pct 见 §㉕⑤~⑧。

---

### ㉕ I48「MIMIC 原生错误」增强版（mimicraw2）—— **1.6pct（已跑完）**

> 这是 **㉔（ICU5000 mimicrawerror）的 I48 版**：同一套**原生错误检测 + 修复规则**
> （T1 哨兵码 / T2 生理不可能区间 / R2 用该患者该项的中位数修复），
> 换到 **I48**（房颤 ECG 任务族）上，**并按 icu5000 的标准补齐了 4 处**
> （T1 类、itemid 叠脏排除、raw 恢复成原生错值、8 大项独立校验）。
>
> ⭐ **I48 与前两档最大的不同：它是一个「任务型」数据集** ——
> agent 不只要清洗，还要**自己推导 cohort 入选规则**、按入选规则筛掉非队列行。
> ⇒ 这直接导致 **i48 的 raw 行数 ≠ reference 行数**（1,723.9 万 → 1,276.8 万），
>   而 ICU5000/K29 的 reference 是「注入前的干净表」（行数与 raw 的原始行数一致）。
> ⇒ **评分器必须逐数据集判定「行空间」**，否则 i48 的分数是垃圾（详见 §㉕③）。

#### ㉕① I48 mimicraw2 1.6pct —— 我们的 agent（pi + val harness）

| 项 | 值 |
|---|---|
| **完整数据集** | `$PROJ/datasets/注入错误数据集/i48/I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M` |
| **agent 视图** | `$PROJ/datasets/val_views/i48_mimicraw2_16`（train 15.7 万行 / validation 13.8 万行 / test 1,723.9 万行） |
| **prompt** | `$WT/eval_handoff/prompts/i48_1.6pct_mimicraw_val_prompt_v5.md`（sha256 `beeed442c825f18d…`） |
| **跑法** | pi + val harness（profile `i48_mimicraw2_16`，max_rounds=20 / patience=3 / seed=666）+ test 阶段**纯重放**（frozen snapshot，无 LLM 调用） |
| **val composite** | **0.895719**（最佳轮 = **第 3 轮**；第 1/2/3 轮 = 0.894668 / 0.895131 / 0.895719）<br>**独立重放 = 0.895719（逐位复现）⇒ `SUCCESS_REPRODUCIBLE`** |
| **test composite** | **0.896069**（`status=SUCCESS`、`replay_status=SUCCESS`、`test_execution_count=1`） |
| **原生错误格（1,264）** | b_clean **1,240（98.1%）** / b_wrong **0** / b_dirty 22 / b_missing 2 |
| 注入规模（test） | cell_update 3,210,516 格 + duplicate/insert 1,222,255 行 |

⚠️ **val 只跑到第 6 轮**：撞上 **provider 侧流式分块错误**
（MiniMax `ValueError: Separator is found, but chunk is longer than limit`；
连接零字节 10.5 小时）⇒ 用**驱动自带的兜底收尾 `finalize()`**（= harness 自己的
`except BaseException` 分支）跑完独立重放并冻结快照 ⇒ status 仍是 `SUCCESS_REPRODUCIBLE`
（重放真跑了一遍、分数逐位复现），但**不是 20 轮跑满**。

**七口径错误级指标（test）**

| 口径 | 值比对 | key 还原 | 排除 gold 外行 | 判据 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **v29** | 数值 | ✅ | ✅ | log | **0.998638** | **0.907893** | 3,855,349 | 491,244 | 5,266 | 2,458 |
| str | 纯字符串 | ✅ | ✅ | log | 0.996127 | 0.763085 | 3,055,882 | 1,257,052 | 38,925 | 2,458 |
| oldsw | 数值 | ❌ | ❌ | log | 0.733084 | 0.716098 | 2,645,012 | 87,809 | 5,083 | **2,750,319** |
| **old（完全老口径）** | 纯字符串 | ❌ | ❌ | log | **0.729628** | **0.594346** | 2,003,505 | 697,365 | 37,034 | **2,750,319** |
| goldv29 | 数值 | ✅ | ✅ | gold | 0.998638 | 0.907893 | 3,855,349 | 491,244 | 5,266 | 2,458 |
| goldstr | 纯字符串 | ✅ | ✅ | gold | 0.999251 | **0.908271** | 3,854,401 | 492,213 | 5,245 | 2,458 |
| goldold | 纯字符串 | ❌ | ❌ | gold | 0.733555 | 0.716555 | 2,645,012 | 87,830 | 5,062 | **2,750,319** |

**三个自洽性检查（可复现）**：
① **v29 与 goldv29 逐位相同**（Det/Rep/b_clean 全一致）⇒ gold 判据没改数值口径语义；
② **goldstr 的 Rep 比 str 高 0.145** ⇒ 差额全是「写法税」（`natural_noise` **3,742,039** 格：
log 写 `'116'`、gold 写 `'116.0'`）；
③ **old/oldsw 的 b_missing 正好 2,750,319** ⇒ 与 §㉕③ 独立量化的结构性惩罚**完全对上**。

#### ㉕② ⭐ 原生层：**越界类几乎全修好，哨兵码类几乎全漏**（这一条最有信息量）

| 类 | 格数 | 修对 | 率 |
|---|---:|---:|---:|
| **T2_impossible（生理不可能区间）** | 1,241 | **1,239** | **99.8%** |
| **T1_sentinel（哨兵占位码）** | 23 | **1** | **4.3%** |

⇒ 22 格没修的**错值清一色 `999999.0`**（`prothrombin time` 7、`inr` 7，
其余 creatinine/glucose/bun/ph/base excess/o2 pressure/co2 pressure/specific gravity 各 1）。

**归因（两条独立证据链，都不是 agent 的锅）**：

| split | 原生层构成 | 注入层哨兵码取值 |
|---|---|---|
| **train** | T2 **9** 格、T1 **0** 格 | `-1`/`9999`/`-888`/`-999`（**不含 999999**） |
| **validation** | T2 **11** 格、T1 **0** 格 | 同上 |
| **test** | T2 1,241 + T1 23（**全是 999999**） | — |

① **train/validation 里一个原生哨兵码样本都没有**（i48 的 train 是 15.7 万行的小分片，
test 是 1,702 万行 ⇒ 原生层密度相同但样本量差 100 倍）；
② **注入层在 train 里用的哨兵码只有 -1/-888/-999/9999** ⇒ agent 学到的是
「这几个具体值脏」，不是「哨兵码这一类」。
⇒ 而 T2 越界类**不需要 train 样本**（物理常识 + label 文本即可推出区间）⇒ 修得极好。

⇒ **数据集设计的两点可改**：train/validation 分片放大（对齐 ICU5000 的 46 倍）；
prompt 把哨兵码**按类**讲清（含 999999）。

#### ㉕③ ⚠️ **i48 独有的口径偏差：`old` 口径被结构性惩罚**（跨档比必须知道）

i48 的任务含 **cohort 筛选**（agent 要自己推导入选规则并筛掉非队列行）
⇒ raw 1,723.9 万行 → reference 1,276.8 万行 ⇒ **agent 该删 383.8 万行（22.3%）**。
而 `old` 口径的 `--keep-rows-not-in-gold` **不排除「gold 里也没有的行」**
⇒ 这些行上的注入格被记成 **b_missing（假漏检）**。

| 来源 | 开关 | 1.6pct | 4.7pct |
|---|---|---:|---:|
| ① **cohort 筛掉的行**上的注入格 | `--keep-rows-not-in-gold` | 650,544（20.3%） | 3,070,497（20.3%） |
| ② **key 列被注入改过**的行上的注入格 | `--no-recover-clean-key` | 782,704（24.4%） | 4,052,166（26.7%） |
| **①+② 合计（old 口径的假漏检）** | | **1,433,248（44.6%）** | **7,122,663（47.0%）** |
| ③ 真正该评的 | | 1,777,268 | 8,028,651 |

**ICU5000 对照（0 格 ⇒ i48 独有）**：icu5000 4.7pct 的 2,229.5 万个注入格
**全部**落在 gold 存在的行上（它的 1.6pct 日志是事后 diff raw↔gold 得到的）
⇒ 它的 old 口径没有这一项。

⇒ **跨档比 old 口径时 i48 天然吃亏**（Det 0.73 vs icu5000 0.83）
⇒ **必须同时给 v29 / goldv29**（i48 的 v29 Det 0.998638 与 icu5000 的 0.999105 同档）。

#### ㉕④ ⭐ 评分器为此改了两处（**不改就会把 i48 的分数算成垃圾**）

| # | 坑 | 现象 | 改法 | 回归 |
|---|---|---|---|---|
| 1 | **raw 的索引空间** | 旧判据（`raw 行数 − reference 行数 == insert/dup 条数`）只在「reference = 注入前的干净表」时成立 ⇒ i48 被误判成 zero ⇒ 索引平移修正被关掉 | `probe_offset_mode()` **数据探测**：抽 log 条目按 zero/bisect 两种映射投到 raw 行号，看哪个映射能让 raw 那一行的值真的等于 dirty_value | i48 1.6pct：zero 7/210 vs **bisect 210/210**；icu5000 两档判定不变 |
| 2 | **gold 的取值空间** | `--clean-from-gold` 按 reference 的 rowid 取「正确答案」⇒ i48 的 reference 是 cohort 筛选后的输出表（12,768,419 行）vs log 行空间（16,016,752 行）⇒ **1,264 格里只有 8 格碰巧对上、行号差最大 324 万** ⇒ 把随机行当正确答案 | `probe_gold_mode()` **数据探测**：比 **key**（不是比列值！低基数列会假命中：i48 的 admissions rowid 取到的行 key 全不同、列值却命中 59.5%）⇒ rowid / key 两模式 | icu5000 两档判 **rowid**（与改动前逐字节一致）；i48 七张表判 **key**；i48 冒烟 v29 与 goldv29 **完全一致** |

⚠️ 另外：原生格判定器也不能直接用 icu5000 版（它靠「log 行号 == reference 行号」对齐 gold）
⇒ 新增 `reproduction/eval_i48_native_cells.py`（按**干净 key** 查；自测 1264/1264 全 b_clean）。

#### ㉕⑤ I48 mimicraw2 4.7pct —— 我们的 agent（pi + val harness）✅ **已跑完**

| 项 | 值 |
|---|---|
| **完整数据集** | `$PROJ/datasets/注入错误数据集/i48/I48_atrial_fibrillation_v2_rawerror_mimicraw2_test4.7pct_411M` |
| **agent 视图** | `$PROJ/datasets/val_views/i48_mimicraw2_47` |
| **prompt** | `$WT/eval_handoff/prompts/i48_4.7pct_mimicraw_val_prompt_v5.md` |
| **跑法** | pi + val harness（profile `i48_mimicraw2_47`，max_rounds=20 / patience=3 / seed=666）+ test 阶段**纯重放** |
| **val composite** | **0.932402**（最佳轮 = **第 4 轮**；第 1~7 轮 = 0.910608 / 0.929932 / 0.932364 / **0.932402** / 0.932402 / 0.932402 / 0.932402）<br>**独立重放 = 0.932402（逐位复现）⇒ `SUCCESS_REPRODUCIBLE`** |
| **test composite** | **0.921177**（`status=SUCCESS`、`replay_status=SUCCESS`、`test_execution_count=1`） |
| **原生错误格（1,238）** | b_clean **1,197（96.7%）** / b_wrong 1 / b_dirty 21 / b_missing 19 |
| **停机原因** | `patience`（连续 3 轮分数持平）；`rounds=7`、`repair_rounds=1` |

**七口径错误级指标（test）**

| 口径 | 值比对 | key 还原 | 排除 gold 外行 | 判据 | Det F1 | Rep F1 | b_clean | b_wrong | b_dirty | b_missing |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **v29** | 数值 | ✅ | ✅ | log | **0.998319** | **0.982855** | 13,558,777 | 261,878 | 6,853 | 47,654 |
| str | 纯字符串 | ✅ | ✅ | log | 0.994823 | 0.771103 | 9,909,961 | 3,792,141 | 125,406 | 47,654 |
| oldsw | 数值 | ❌ | ❌ | log | 0.794925 | 0.786776 | 10,283,334 | 138,291 | 5,662 | **7,001,734** |
| **old（完全老口径）** | 纯字符串 | ❌ | ❌ | log | **0.791465** | **0.618291** | 7,373,107 | 2,950,529 | 103,651 | **7,001,734** |
| goldv29 | 数值 | ✅ | ✅ | gold | 0.998319 | 0.982855 | 13,558,777 | 261,878 | 6,853 | 47,654 |
| goldstr | 纯字符串 | ✅ | ✅ | gold | 0.998345 | 0.982247 | 13,548,694 | 272,672 | 6,142 | 47,654 |
| goldold | 纯字符串 | ❌ | ❌ | gold | 0.794951 | 0.786767 | 10,283,334 | 138,900 | 5,053 | **7,001,734** |

**三个自洽性检查（可复现）**：
① **v29 与 goldv29 逐位相同**（Det/Rep/四个桶全一致）⇒ 数值口径下换判据不改变任何一格；
② **goldstr 的 Rep 比 str 高 0.211** ⇒ 差额全是「写法税」（`natural_noise` **910,943** 格）；
③ **old/oldsw 的 b_missing 正好 7,001,734**（1.6pct 是 2,750,319）⇒
   同一个结构性惩罚，随注入量放大 —— **old 口径下 i48 两档都天然吃亏，跨档不可比**。

#### ㉕⑥ ⭐ 两档原生层的**交集口径**跨档对比（1,231 格）—— 结果是「完全一致」

脚本 `reproduction/i48_native_cross_tier.py`；输出 `runs/i48_mimicraw2_native_cross_tier_16_vs_47.json`。

| 项 | 1.6pct | 4.7pct |
|---|---:|---:|
| 登记格数 | 1,264 | 1,238 |
| 命中格数（agent 输出里能定位到） | 1,262 / 1,264 | 1,219 / 1,238 |
| 多行歧义格 | 11 | 154 |
| 全档修复率 | 98.1% | 96.7% |

**交集 1,231 格的一致性矩阵**（A = 1.6pct，B = 4.7pct）：

| A \ B | b_clean | b_dirty | b_missing | b_wrong |
|---|---:|---:|---:|---:|
| **b_clean** | **1,188** | 0 | 19 | 1 |
| **b_dirty** | 0 | **21** | 0 | 0 |
| **b_missing** | 2 | 0 | 0 | 0 |

⇒ **两档都修对 1,188 / 两档都没动 21 / A 修对 B 没修 20 / B 修对 A 没修 2**
⇒ **交集上两档「修复值不一致 0 格」、「原生错值不一致 0 格」**

⭐ **这是最硬的一条交叉验证**：两次**完全独立**的 val 训练循环（不同档位、不同 log、
不同训练/验证切片）在同一批 1,231 个原生错误格上给出了**逐格一致的判断**
⇒ 说明评分器的原生格判定和 agent 的行为都是稳定的，不是某一次跑的偶然。

#### ㉕⑦ ⚠️ 本轮踩到的第三个坑：**`submission.json` 的 `replay.argv` 必须是完整命令**

第 7 轮后的独立重放**秒挂**（`rc=64`，0.011 s）：

```
/usr/bin/sandbox-exec: illegal option -- -
Usage: sandbox-exec [options] command [args]
```

- **根因**：harness 把 `submission.json` 的 `replay.argv` 直接拼在 `sandbox-exec -f <profile>` 后面
  （`agent/pi_harness.py:885`）。macOS 的 `sandbox-exec` 在**第一个非选项参数处**停止选项解析 ⇒
  `argv` 若以 `--raw_root` 开头，`--raw_root` 会被 sandbox-exec 当成**它自己的选项** ⇒ 直接报错。
  **harness 只认 `replay.argv`，不认 `entrypoint` 字段** ⇒ argv 必须是
  `["python3", "replay.py", "--raw_root", "{raw_root}", …]` 这种**完整命令**。
- **harness 自带修复回路**（`agent/pi_harness.py:481`）：重放失败 ⇒ 把错误回灌给 agent
  （"Fix the replayable pipeline"）⇒ agent 修 `submission.json` + 让 `replay.py` 兼容所有调用形式
  ⇒ 重放 **agent_workdir** ⇒ 直到重放分数 ≥ 正式最好分。本轮 **repair_rounds=1** 即通过。
- **agent 自己定位了根因**：它复现了 `sandbox-exec --raw_root /tmp` 并得到**逐字节相同**的 stderr，
  还写了 `analysis/diff78.py` 把所有可能的调用形式逐个验证（干净目录重放字节级复现全部 13 个文件）。
- ⚠️ **其它档的 agent 也可能有同样的坑** ⇒ 值得回头查各档 `submission.json` 的 argv 是否都是完整命令。

#### ㉕⑧ 交付

| 位置 | 内容 |
|---|---|
| `i48_agent_run/experiments/i48_mimicraw2_4p7_test_seed666_v1/host/test_result_package` | 13 个 csv（agent 实际产出，与 1.6pct 目录结构一致）<br>⚠️ **按要求先不放进 `ceshi1` 做对比** ⇒ 三维度表里不含这一档 |
| `runs/i48_mimicraw2_47_agent_error_metrics_*_run20261119_4p7.json` | 七个口径 |
| `runs/i48_mimicraw2_47_agent_native_cells_run20261119_4p7_{numeric,string}.json` | 原生格 |
| `runs/i48_mimicraw2_native_cross_tier_16_vs_47.json` | 两档交集对比 |

> 要把 4.7pct 加回 `ceshi1` 对比时：`reproduction/build_final_table.py` 的 `SPEC` 里
> 取消注释 `I48/4.7pct_我们的agent_pi+val_mimicraw2` 那一段 + 把结果包拷进
> `ceshi1/I48/4.7pct_我们的agent_pi+val_mimicraw2/结果包/` 即可（数据都在，一分钟能恢复）。
