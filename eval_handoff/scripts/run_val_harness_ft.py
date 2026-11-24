"""带容错的 val harness 驱动（不改 harness 源码）。

为什么需要它：连续两次跑批都死在 worker 的模型调用上 ——
  v1: 'key_commands' is a required property（第 5 轮，4 轮成果保住了）
  v2: Pi worker ReadTimeout（第 1 轮，一轮都没交付，全丢）
harness 的 run() 对 worker.run_turn() 抛出的异常零容错，一次抖动整跑完蛋。

这个驱动在 harness 外面加两层保护：
  1. 给 run_turn 加重试：模型调用失败（ReadTimeout / 429 / 畸形响应）自动重试
  2. 兜底收尾：harness.run() 抛异常时，不让整跑白费 —— 走它自己的收尾流程
     （独立重放 best_snapshot → 可接受就冻结 reproducible_snapshot + 写 run_report.json）
     这样「崩在第 1 轮」也能留下第 1 轮之前的成果

用法: python run_val_harness_ft.py <profile> [实验目录后缀]
  profile: less1 | more1 | n18 | n18_more | i48
           （决定数据集视图、prompt、实验名前缀、dataset manifest、evaluation manifest）

环境变量可覆盖默认路径（换机器/换数据集时用）：
  PROJ       项目根，默认 /Users/mac/PycharmProjects/v2_clean
  RUN        agent run 目录，默认 $PROJ/k29_agent_run
  PROMPT_DIR prompt 目录，默认 $RUN/prompts
"""
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

PROJ = Path(os.environ.get("PROJ", "/Users/mac/PycharmProjects/v2_clean"))
sys.path.insert(0, str(PROJ))

from agent.pi_harness import (  # noqa: E402
    PiHarnessResult,
    PiValidationHarness,
    PiValidationHarnessConfig,
    build_worker_model_environment,
    load_evaluation_manifest,
    restore_agent_workdir,
    snapshot_agent_workdir,
)
from agent.pi_worker_client import JsonlWorkerClient  # noqa: E402

RUN = Path(os.environ.get("RUN", str(PROJ / "k29_agent_run")))
PROMPT_DIR = Path(os.environ.get("PROMPT_DIR", str(RUN / "prompts")))
# 本 driver 所在的**工作树**根：evaluation_manifests / prompts 这些「跟代码走」的东西
# 从这里取，别从 PROJ（共享 checkout，可能在别的分支上）取。
WT = Path(os.environ.get("WT", str(Path(__file__).resolve().parents[2])))

# profile → (视图目录, prompt 文件, 实验名前缀, dataset manifest 所在的完整数据集)
PROFILES = {
    "less1": {
        "ds": PROJ / "datasets/val_views/icu5000_less1",
        "prompt": PROMPT_DIR / "icu5000_1.6pct_val_prompt_v1.md",
        "exp_prefix": "icu5000_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "icu_5000_v2_rawerror_test1.6pct_614M"
                     / "v2_rawerror_icu5000_final_report.json"),
    },
    "more1": {
        "ds": PROJ / "datasets/val_views/icu5000_more1",
        "prompt": PROMPT_DIR / "icu5000_4.7pct_val_prompt_v1.md",
        "exp_prefix": "icu5000_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "icu_5000_v2_rawerror_test4.7pct_614M"
                     / "v2_rawerror_icu5000_final_report.json"),
    },
    # N18 是另一个任务（不是 icu5000）：18 个输出文件、切分键 subject_id、
    # 有 ed/ + note/ 多模态、manifest 也不同。
    "n18": {
        "ds": PROJ / "datasets/注入错误数据集/n18_less_1",
        "prompt": PROMPT_DIR / "n18_1.6pct_val_prompt_v1.md",
        "exp_prefix": "n18_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "N18_ckd_ED_v2_rawerror_test1.6pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # N18 4.7pct（more）：与 icu5000 4.7pct 同样的密度剖面
    #   train 2.33% / validation 0.89% / test 4.72%（照抄 K29 的分 split 率）
    "n18_more": {
        "ds": PROJ / "datasets/注入错误数据集/n18_more_1",
        "prompt": PROMPT_DIR / "n18_4.7pct_val_prompt_v1.md",
        "exp_prefix": "n18_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "N18_ckd_ED_v2_rawerror_test4.7pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # I48（房颤）：13 个输出文件（7 顶层 + 4 icu + 2 ecg）、切分键 subject_id、
    # 多模态是 ecg/ 而不是 ed/+note/，manifest 也不同。
    # 密度剖面跟 N18 1.6pct 一样：train 1.30% / validation 2.24% / test 1.58%。
    "i48": {
        "ds": PROJ / "datasets/注入错误数据集/i48_less_1",
        "prompt": PROMPT_DIR / "i48_1.6pct_val_prompt_v1.md",
        "exp_prefix": "i48_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # I48 1.6pct「MIMIC 原生错误」增强版（mimic-raw-error）：
    # 母数据集 I48_..._mimicrawerror_...，与原版的唯一区别是
    # **train/validation 的 reference 里，MIMIC 自身的原生错误（生理不可能的孤立尖峰）
    # 已经被修好**，并在 modification_log 里登记 —— 于是 agent 在 train 对照里
    # 第一次能看见这类错误模式，有 train 证据可学。
    # 视图只含 train/validation，**没有 test**（val 阶段不需要）。
    "i48_mimicraw": {
        "ds": PROJ / "datasets/注入错误数据集/i48_mimicraw_1",
        "prompt": PROMPT_DIR / "i48_1.6pct_mimicraw_val_prompt_v1.md",
        "exp_prefix": "i48_mimicraw_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "I48_atrial_fibrillation_v2_rawerror_mimicrawerror_test1.6pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # i48_mimicraw 的 v3 prompt：修掉 v1 的两个实测缺陷
    #   ① v1 只给了 train 里出现过的 7 个 itemid 的区间；train 分片里根本不存在
    #      另外 9 个（220045/220059/220061/220179/220181/220210/223762/224690/
    #      227466），test 上这些 itemid 有 74 格原生错误 -> v3 给完整 20 行对照表
    #   ② v1 没写死中位数池的口径；agent 在 cohort 筛选/去重/注入修复**之前**就算
    #      中位数，池子混进别的住院和重复行 -> val 11 格里只对 1 格。
    #      v3 明确要求「pipeline 最后一步对自己产出的干净表做后处理」
    "i48_mimicraw_v3": {
        "ds": PROJ / "datasets/注入错误数据集/i48_mimicraw_1",
        "prompt": PROMPT_DIR / "i48_1.6pct_mimicraw_val_prompt_v3.md",
        "exp_prefix": "i48_mimicraw_v3_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "I48_atrial_fibrillation_v2_rawerror_mimicrawerror_test1.6pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # I48 4.7pct「MIMIC 原生错误」增强版 —— 1.6pct 版的姊妹数据集，
    # 原生错误层规则一字未改，只有注入层预设不同（4.7pct）。
    # ⚠️ 4.7pct 的密度剖面是**反转**的：train 2.33% / validation 0.89% / test 4.72%
    #    ⇒ validation 比 test 稀疏 5.3 倍，val 上验证过的组级规则未必在 test 上成立
    #    （flights 上踩过同一个坑，账本 ⑳②）。prompt v4 把这段密度描述改成了 4.7pct 的。
    "i48_mimicraw_47_v4": {
        "ds": PROJ / "datasets/注入错误数据集/i48_mimicraw_47_1",
        "prompt": PROMPT_DIR / "i48_4.7pct_mimicraw_val_prompt_v4.md",
        "exp_prefix": "i48_mimicraw_v4_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "I48_atrial_fibrillation_v2_rawerror_mimicrawerror_test4.7pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # I48 4.7pct（more）：密度剖面是「反转」的 —— train 2.33% / validation 0.89% /
    # test 4.72%。validation 的错误密度比 train 低三倍、比 test 低五倍。
    "i48_more": {
        "ds": PROJ / "datasets/注入错误数据集/i48_more_1",
        "prompt": PROMPT_DIR / "i48_4.7pct_val_prompt_v1.md",
        "exp_prefix": "i48_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "I48_atrial_fibrillation_v2_rawerror_test4.7pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # K29（胃炎/十二指肠炎）低密度档：train 400 / validation 800 / test 1,800 条错误
    # （fixed-v1 引擎造的，0 自然噪声）。切分键 hadm_id、12 个输出文件、
    # 多模态是 note/。视图走 datasets/val_views/（密封，父目录里没有别的东西）。
    # 任务书用 v2 —— v1 里写的密度是 4.7pct 的数字，跟这一档对不上；
    # v1 的 sha256 已被台账记为旧跑分（k29_1800_val_seed666_v1）的证据，不能改。
    "k29_1800": {
        "ds": PROJ / "datasets/val_views/k29_1800",
        "prompt": PROMPT_DIR / "k29_1800_val_prompt_v2.md",
        "exp_prefix": "k29_1800_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集"
                     / "K29_clean_v2_rawerror_10train_400_800_1800"
                     / "v2_rawerror_k29_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/k29_gastritis_duodenitis_v3_1.json",
    },
    # K29（胃炎/十二指肠炎）4.7pct「MIMIC 原生错误」增强版（mimic-raw-error）：
    #   母数据集 K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror
    #   与原版的唯一区别：**reference 里 MIMIC 自身的原生错误（生理不可能的孤立尖峰
    #   + 哨兵错误码）已经修好**，并在 modification_log 里登记 ⇒ agent 在 train 对照里
    #   第一次能看见这类错误模式，有 train 证据可学。
    #   ⚠️ 与原生错误格重叠的**注入条目已从 log 里删除**（原生层 883 格独占），
    #      所以 agent 面对这一层只有「改成同患者同 itemid 的中位数」一个目标。
    #   视图只含 train/validation，**没有 test**（val 阶段不需要）。
    "k29_mimicraw": {
        "ds": PROJ / "datasets/注入错误数据集/k29_mimicraw_47",
        "prompt": PROMPT_DIR / "k29_4.7pct_mimicraw_val_prompt_v1.md",
        "exp_prefix": "k29_mimicraw_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集" / "k29"
                     / "K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror"
                     / "v2_rawerror_k29_rebuild_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/k29_gastritis_duodenitis_v3_1.json",
    },
    # K29 1.6pct「MIMIC 原生错误」增强版（低错误档，与 4.7pct 同规则、不同注入密度）
    "k29_mimicraw_16": {
        "ds": PROJ / "datasets/注入错误数据集/k29_mimicraw_16",
        "prompt": PROMPT_DIR / "k29_1.6pct_mimicraw_val_prompt_v1.md",
        "exp_prefix": "k29_mimicraw_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集" / "k29"
                     / "K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"
                     / "v2_rawerror_k29_rebuild_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/k29_gastritis_duodenitis_v3_1.json",
    },
    # ICU5000 1.6pct「MIMIC 原生错误」增强版（mimic-raw-error）：
    #   母数据集 icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror
    #   与原版（less1）的唯一区别：**train/validation 的 reference 里，MIMIC 自身的
    #   原生错误（生理不可能的孤立尖峰 + 哨兵错误码）已经修好**，并在 modification_log
    #   里登记 ⇒ agent 在 train 对照里第一次能看见这类错误模式，有 train 证据可学
    #   （train 注入 20 例 / validation 注入 50 例，覆盖 16 个 HARD itemid）。
    #   ⚠️ icu5000 特有的三处（K29 没有）：
    #     ① raw 的行索引空间是**混合**的（chartevents 带插入偏移、input/outputevents 零偏移）
    #     ② itemid 列上多一层 CareVue 别名注入（220 万条 key 列错误，规则 100% 符合
    #        「992 + (2|3) + 原 itemid 末 4 位」）⇒ prompt 里专门写了一节
    #     ③ 「itemid 也被注入器改过」的 141 格是两层错误叠加、形态同构不可判定
    #        ⇒ 不进 native 层、留给注入层评测（native 层 2,450 格）
    #   视图只含 train/validation，**没有 test**（val 阶段不需要）。
    "icu5000_mimicraw_16": {
        "ds": PROJ / "datasets/val_views/icu5000_mimicraw_16",
        "prompt": WT / "eval_handoff/prompts/icu5000_1.6pct_mimicraw_val_prompt_v1.md",
        "exp_prefix": "icu5000_mimicraw_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/icu5000"
                     / "icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror"
                     / "v2_rawerror_icu5000_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/icu_5000_v1.json",
    },
    # ICU5000 4.7pct「MIMIC 原生错误」增强版 —— 1.6pct 版的姊妹数据集，
    # 原生错误层规则一字未改，只有注入层预设不同（4.7pct，native 层 2,174 格）。
    # ⚠️ 密度剖面是**反转**的：train 2.34% / validation 0.89% / test 4.72%
    #    ⇒ validation 比 test 稀疏 5.3 倍，val 上验证过的规则未必在 test 上成立。
    "icu5000_mimicraw_47": {
        "ds": PROJ / "datasets/val_views/icu5000_mimicraw_47",
        "prompt": WT / "eval_handoff/prompts/icu5000_4.7pct_mimicraw_val_prompt_v1.md",
        "exp_prefix": "icu5000_mimicraw_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/icu5000"
                     / "icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror"
                     / "v2_rawerror_icu5000_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/icu_5000_v1.json",
    },
    # ---- I48「MIMIC 原生错误」增强版（mimicraw2，对齐 icu5000 标准）----
    # 与母数据集 I48_atrial_fibrillation_v2_rawerror_test{1.6,4.7}pct_411M 的区别：
    #   **各 split 的 reference 里，MIMIC 自身的原生错误已按「该患者该 itemid 的
    #   可用读数中位数」修好**，并在 modification_log 里登记（error_subtype=
    #   mimic_native_*）⇒ agent 在 train 对照里第一次能看见这类错误模式。
    #   规则 = T1 哨兵占位码（对所有 itemid 生效）+ T2 生理不可能区间
    #   （区间按 PhysioNet d_items 的权威 label 匹配 ⇒ i48 自己的区间，不照抄 K29）。
    # ⚠️ 与旧版 mimicrawerror（Nov 2026-07 建的）的四处差别：
    #   ① 旧版只有 T2（没有哨兵码类）；这版补上 T1
    #   ② 旧版把「itemid 也被注入器改过」的格留在 native 层（两层错误叠加、
    #      形态同构、不可判定）；这版排除 ⇒ 交给注入层评测
    #   ③ 旧版没把 raw 恢复成 MIMIC 原生错值；这版恢复 ⇒ 一格只剩一个错误
    #   ④ 旧版的 prompt 没有「哨兵码清单不是封闭的」和 itemid 别名一节；
    #      这版（v5）都补上了（ICU5000 实测：封闭清单 ⇒ 4.7pct 漏修 281 万格）
    # ⚠️ i48 特有的结构：**reference 是处理后的输出表**（cohort 筛选后），
    #    不是注入前的干净表 ⇒ log 的 raw_row_index 与 reference 行号**不同空间**；
    #    评分器的索引空间判定已改成「数据探测」（见 compute_error_metrics.py）。
    "i48_mimicraw2_16": {
        "ds": PROJ / "datasets/val_views/i48_mimicraw2_16",
        "prompt": WT / "eval_handoff/prompts/i48_1.6pct_mimicraw_val_prompt_v5.md",
        "exp_prefix": "i48_mimicraw2_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/i48"
                     / "I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # I48 4.7pct 版 —— 与 1.6pct 版规则一字未改，只有注入层预设不同。
    # ⚠️ 密度剖面是**反转**的：train 2.33% / validation 0.89% / test 4.72%
    #    ⇒ validation 比 test 稀疏 5.3 倍。
    "i48_mimicraw2_47": {
        "ds": PROJ / "datasets/val_views/i48_mimicraw2_47",
        "prompt": WT / "eval_handoff/prompts/i48_4.7pct_mimicraw_val_prompt_v5.md",
        "exp_prefix": "i48_mimicraw2_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/i48"
                     / "I48_atrial_fibrillation_v2_rawerror_mimicraw2_test4.7pct_411M"
                     / "v2_rawerror_i48_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json",
    },
    # ---- N18「MIMIC 原生错误」增强版（mimicraw）----
    # N18 = ICD-10 慢性肾病（chronic kidney disease），18 个输出文件
    # （7 顶层 + 4 icu + 6 multimodal/ed+notes + 1 edstays 主表）、切分键 subject_id。
    # 与母数据集 N18_ckd_ED_v2_rawerror_test1.6pct_490M 的区别：
    #   **各 split 的 reference 里，MIMIC 自身的原生错误已按「该患者该 itemid 的
    #   可用读数中位数」修好**（T1 哨兵占位码 + T2 生理不可能区间），并在
    #   modification_log 里登记（error_subtype=mimic_native_*）⇒ agent 在 train 对照里
    #   第一次能看见这类错误模式，有 train 证据可学（train 注入 15 例同形态示例、
    #   validation 28 例 + 3 格真实原生错误）。
    # ⚠️ 与原生错误格重叠的**注入条目已从 log 里删除**（原生层独占），
    #    「itemid 也被注入器改过」的格划归注入层（两层错误叠加、形态同构、不可判定）。
    # ⚠️ N18 是 **cohort 型**任务（reference 是筛选后的输出表）⇒ log 的 raw_row_index
    #    与 reference 行号**不同空间**；评分器的索引空间判定是「数据探测」。
    # 视图只含 train/validation，**没有 test**（val 阶段不需要）。
    "n18_mimicraw_16": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_16",
        "prompt": WT / "eval_handoff/prompts/n18_1.6pct_mimicraw_val_prompt_v5.md",
        "exp_prefix": "n18_mimicraw_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # N18 4.7pct 版 —— 与 1.6pct 版规则一字未改，只有注入层预设不同。
    # ⚠️ 密度剖面与 1.6pct 相同（train 2.33% / validation 0.89% / test 4.72%）
    #    ⇒ validation 比 test 稀疏 5.3 倍，val 上验证过的组级规则未必在 test 上成立。
    "n18_mimicraw_47": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_47",
        "prompt": WT / "eval_handoff/prompts/n18_4.7pct_mimicraw_val_prompt_v5.md",
        "exp_prefix": "n18_mimicraw_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # N18 的 v6 prompt 版（相对 v5 只多一节：**显式 6 步顺序**，写死
    # 「先 cohort 筛选、再修原生错误」—— v5 只说「最后一步算」，agent 自己排，
    # 排在 cohort 筛选之前 ⇒ 池子混进非 cohort stay 的读数 ⇒ 原生格修成错值。
    # 实测：4.7pct 的 validation 31 格里 14 格受影响 ⇒ 修复率 96.8% → 54.8%。)
    "n18_mimicraw_16_v6": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_16",
        "prompt": WT / "eval_handoff/prompts/n18_1.6pct_mimicraw_val_prompt_v6.md",
        "exp_prefix": "n18_mimicraw_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    "n18_mimicraw_47_v6": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_47",
        "prompt": WT / "eval_handoff/prompts/n18_4.7pct_mimicraw_val_prompt_v6.md",
        "exp_prefix": "n18_mimicraw_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # N18 的 v7 prompt 版（相对 v6 多一节：**上下文预算 + 续跑摘要格式**）。
    # 这一节针对的是「上下文压缩摘要」的 schema 校验失败 —— 实测 5 个压缩事件里
    # 2 个先失败一次，连挂 4 次 ⇒ worker 崩 ⇒ 整个 run 报废（N18 4.7pct 第 1 遍就是这样死的）。
    "n18_mimicraw_16_v7": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_16",
        "prompt": WT / "eval_handoff/prompts/n18_1.6pct_mimicraw_val_prompt_v7.md",
        "exp_prefix": "n18_mimicraw_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    "n18_mimicraw_47_v7": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_47",
        "prompt": WT / "eval_handoff/prompts/n18_4.7pct_mimicraw_val_prompt_v7.md",
        "exp_prefix": "n18_mimicraw_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # N18 的 v8 prompt 版（相对 v7 多两处：原生错误节「注意」清单 +1 条，
    # 末尾多一节「⏱ 时间锚点」）。
    # 针对的是**任务窗口泄漏把 chartevents.charttime 往后推 30~48 小时**
    # ⇒ 原生错误格的行在 raw 里不在 reference 那个时刻上
    # ⇒ 按精确 5 段 key 定位必然漏检（实测 test：1.6pct 44/715、4.7pct 185/586 对不上）。
    "n18_mimicraw_16_v8": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_16",
        "prompt": WT / "eval_handoff/prompts/n18_1.6pct_mimicraw_val_prompt_v8.md",
        "exp_prefix": "n18_mimicraw_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    "n18_mimicraw_47_v8": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_47",
        "prompt": WT / "eval_handoff/prompts/n18_4.7pct_mimicraw_val_prompt_v8.md",
        "exp_prefix": "n18_mimicraw_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # N18 的 v9 prompt 版（相对 v8 多一节：**submission.json 的 replay.argv 契约**）。
    # 针对的是「独立重放这一关判 0」—— 1.6pct v7 run 实测：
    #   [Harness] independent replay: status=EXECUTION_FAILED score=0.000000
    #   [Harness] replay repair 1: status=INVALID_SUBMISSION score=0.000000
    # 成因两条（源码 agent/pi_harness.py + pi_harness_sandbox.py）：
    #   ① sandbox 只放行 harness 自己那个 python ⇒ /usr/bin/env、/bin/sh 不可执行
    #      ⇒ shebang 启动脚本 / sh -c 一律 ENOENT；
    #   ② 校验器拒绝「不含占位符的绝对路径」⇒ 脚本路径要写成 {workdir}/xxx.py。
    "n18_mimicraw_16_v9": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_16",
        "prompt": WT / "eval_handoff/prompts/n18_1.6pct_mimicraw_val_prompt_v9.md",
        "exp_prefix": "n18_mimicraw_1p6_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    "n18_mimicraw_47_v9": {
        "ds": PROJ / "datasets/val_views/n18_mimicraw_47",
        "prompt": WT / "eval_handoff/prompts/n18_4.7pct_mimicraw_val_prompt_v9.md",
        "exp_prefix": "n18_mimicraw_4p7_val_seed666",
        "manifest": (PROJ / "datasets/注入错误数据集/n18"
                     / "N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"
                     / "v2_rawerror_n18_final_report.json"),
        "eval_manifest": PROJ / "evaluation_manifests/n18_ckd_ED_notes_v1.json",
    },
    # ---- 校正类（单表纠错）----
    # 和上面那几族完全不同：不是 MIMIC cohort 清洗，是「给 10 行脏↔干净对照，
    # 归纳破坏规则，修好整张单表」。切分是 Raha 的 curated10 切分。
    #   切分目录里 train/{raw,reference}、validation/{raw,gold}、correction/{raw,reference_private}
    #   ⇒ 注意 gold 目录叫 **gold** 不是 reference（所以下面要覆盖 validation_gold_subdir）
    #   ⇒ correction/reference_private 与 host_private 对 agent 不可读（sandbox 只放行
    #     train raw / train reference / validation raw 三个根）
    #
    # ⚠️ 盘上**现存的四个切分**（datasets/correction/）：
    #     beers / rayyan / tax / raha_hospital(dense)
    #   ⇒ flights 的两个切分（flights_curated10_val20_v1、flights_curated10_groupdense_v1）
    #     **已从盘上删除** ⇒ 下面两个 flights profile 现在指向不存在的目录，
    #     启动脚本会先检查并在缺失时直接报错（不会静默跑错）。
    "hospital_correction": {
        "ds": PROJ / "datasets/correction/raha_hospital_curated10_val20_dense_v1",
        "prompt": WT / "eval_handoff/prompts/hospital_correction_val_prompt_v1.md",
        "exp_prefix": "hospital_correction_val_seed666",
        "manifest": (PROJ / "datasets/correction/raha_hospital_curated10_val20_dense_v1"
                     / "split_manifest.json"),
        "eval_manifest": WT / "evaluation_manifests/raha_correction_hospital_v1.json",
        "validation_gold_subdir": "validation/gold",
    },
    # 下面三个是 10:20:rest 切分（train 10~30 行 + validation 20 行 + correction=rest）。
    # train 与旧的 10:rest split 逐字节一致；validation 的 20 行从旧 correction(=test) 抽。
    "beers_correction": {
        "ds": PROJ / "datasets/correction/beers_curated10_val20_v1",
        "prompt": WT / "eval_handoff/prompts/beers_correction_val_prompt_v1.md",
        "exp_prefix": "beers_correction_val_seed666",
        "manifest": (PROJ / "datasets/correction/beers_curated10_val20_v1"
                     / "split_manifest.json"),
        "eval_manifest": WT / "evaluation_manifests/raha_correction_beers_v1.json",
        "validation_gold_subdir": "validation/gold",
    },
    "rayyan_correction": {
        "ds": PROJ / "datasets/correction/rayyan_curated10_val20_v1",
        "prompt": WT / "eval_handoff/prompts/rayyan_correction_val_prompt_v1.md",
        "exp_prefix": "rayyan_correction_val_seed666",
        "manifest": (PROJ / "datasets/correction/rayyan_curated10_val20_v1"
                     / "split_manifest.json"),
        "eval_manifest": WT / "evaluation_manifests/raha_correction_rayyan_v1.json",
        "validation_gold_subdir": "validation/gold",
    },
    "tax_correction": {
        "ds": PROJ / "datasets/correction/tax_curated10_val20_v1",
        "prompt": WT / "eval_handoff/prompts/tax_correction_val_prompt_v1.md",
        "exp_prefix": "tax_correction_val_seed666",
        "manifest": (PROJ / "datasets/correction/tax_curated10_val20_v1"
                     / "split_manifest.json"),
        "eval_manifest": WT / "evaluation_manifests/raha_correction_tax_v1.json",
        "validation_gold_subdir": "validation/gold",
    },
    # ⚠️ flights 的两个切分**已从盘上删除**（保留 profile 只作历史参考，勿直接启动）
    "flights_correction": {
        "ds": PROJ / "datasets/correction/flights_curated10_val20_v1",
        "prompt": WT / "eval_handoff/prompts/flights_correction_val_prompt_v1.md",
        "exp_prefix": "flights_correction_val_seed666",
        "manifest": (PROJ / "datasets/correction/flights_curated10_val20_v1"
                     / "split_manifest.json"),
        "eval_manifest": WT / "evaluation_manifests/raha_correction_flights_v1.json",
        "validation_gold_subdir": "validation/gold",
    },
    # flights 的**组密集**切分版：train 里每个航班给多行，
    # 让「同航班共享常量」这条结构在 train 上可学、在 validation 上可验证
    # （旧的 curated 选法每个航班只留 1 行 ⇒ 这条结构在 train/val 上根本不存在）。
    "flights_correction_dense": {
        "ds": PROJ / "datasets/correction/flights_curated10_groupdense_v1",
        "prompt": WT / "eval_handoff/prompts/flights_correction_val_prompt_v1.md",
        "exp_prefix": "flights_correction_dense_val_seed666",
        "manifest": (PROJ / "datasets/correction/flights_curated10_groupdense_v1"
                     / "split_manifest.json"),
        "eval_manifest": WT / "evaluation_manifests/raha_correction_flights_v1.json",
        "validation_gold_subdir": "validation/gold",
    },
}

MAX_TURN_RETRIES = 4
RETRY_SLEEP = 15

_orig_run_turn = JsonlWorkerClient.run_turn


async def _retrying_run_turn(self, prompt: str):
    """模型调用失败自动重试。worker 进程死了就没救了，直接抛。"""
    last: Exception | None = None
    for attempt in range(MAX_TURN_RETRIES):
        try:
            return await _orig_run_turn(self, prompt)
        except Exception as exc:
            last = exc
            dead = getattr(self, "process", None) is not None and self.process.returncode is not None
            print(f"[retry] run_turn 失败 ({type(exc).__name__}: {exc}) "
                  f"{'worker 已死，放弃' if dead else f'重试 {attempt + 1}/{MAX_TURN_RETRIES}'}",
                  flush=True)
            if dead:
                raise
            await asyncio.sleep(RETRY_SLEEP)
    raise last  # type: ignore[misc]


JsonlWorkerClient.run_turn = _retrying_run_turn


def make_config(exp_name: str, prof: dict) -> PiValidationHarnessConfig:
    ds = prof["ds"]
    return PiValidationHarnessConfig(
        project_root=PROJ,
        experiment_dir=RUN / "experiments" / exp_name,
        train_raw=ds / "train/raw",
        train_reference=ds / "train/reference",
        validation_raw=ds / "validation/raw",
        validation_gold=ds / prof.get("validation_gold_subdir", "validation/reference"),
        evaluation_manifest=prof.get(
            "eval_manifest",
            PROJ / "evaluation_manifests/icu_5000_v1.json"),
        dataset_manifest=prof["manifest"],
        max_rounds=20,
        patience=3,
        max_iters=10000,
    )


async def finalize(harness: PiValidationHarness, best_score: float, why: str) -> int:
    """harness 自己最后那段收尾：独立重放 → 通过就冻结 + 写 report。"""
    if not harness.best_snapshot.is_dir():
        print(f"!! {why}：没有 best_snapshot，无可收尾", file=sys.stderr)
        return 2
    print(f"\n收尾（{why}）：独立重放 best_snapshot ...", flush=True)
    replay = await harness._run_replay(harness.best_snapshot)
    print(f"  replay status={replay.status} score={replay.score:.6f}", flush=True)
    if not harness._replay_is_acceptable(replay, best_score):
        print(f"!! 重放不可接受（{replay.score:.6f} < {best_score:.6f}），不收尾", file=sys.stderr)
        return 3
    final_snapshot = harness.host_dir / "reproducible_snapshot"
    snapshot_agent_workdir(harness.best_snapshot, final_snapshot)
    result = PiHarnessResult(
        status="SUCCESS_REPRODUCIBLE",
        stop_reason=why,
        rounds=len(list((harness.host_dir / "rounds").glob("round_*.json"))),
        repair_rounds=0,
        best_score=max(best_score, 0.0),
        reproducible_score=replay.score,
        best_snapshot=harness.best_snapshot,
        reproducible_snapshot=final_snapshot,
    )
    harness._atomic_json(harness.host_dir / "run_report.json",
                         harness._run_report_payload(result))
    print("收尾完成：reproducible_snapshot + run_report.json")
    print(json.dumps(harness._run_report_payload(result), ensure_ascii=False, indent=1))
    return 0


def best_score_of(harness: PiValidationHarness) -> float:
    best = 0.0
    for p in sorted((harness.host_dir / "rounds").glob("round_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        best = max(best, float((d.get("metrics") or {}).get("composite_score") or 0.0))
    return best


async def main() -> int:
    profile = sys.argv[1] if len(sys.argv) > 1 else "less1"
    if profile not in PROFILES:
        print(f"未知 profile: {profile}（可选: {', '.join(PROFILES)}）", file=sys.stderr)
        return 2
    prof = PROFILES[profile]
    suffix = sys.argv[2] if len(sys.argv) > 2 else "v1"
    exp_name = f"{prof['exp_prefix']}_{suffix}"
    config = make_config(exp_name, prof)
    harness = PiValidationHarness(config)
    prompt = prof["prompt"].read_text(encoding="utf-8")
    print(f"[profile={profile}] experiment={config.experiment_dir}", flush=True)
    print(f"[profile={profile}] prompt={prof['prompt']}", flush=True)
    print(f"[profile={profile}] dataset={prof['ds']}", flush=True)

    # 续跑：把上一轮实验的 best_snapshot 灌进新实验的 workdir，
    # 让 agent 从自己上一轮的状态继续（analysis/ pipeline/ train_analysis.json 都在），
    # 而不是从头重推一遍。上一次跑被 API 限流掐断时用得上。
    #
    # 实现：harness 的 _prepare 要求「实验目录全新且为空」，且最后才 mkdir workdir。
    # 所以不能在 run() 之前预置 workdir —— 改为包装 _prepare：
    # 先让它按原样建好空的 workdir 和 host 目录，再把种子灌进去。
    seed_from = os.environ.get("SEED_FROM", "").strip()
    if seed_from:
        seed = Path(seed_from)
        if not (seed / "result").is_dir():
            print(f"!! SEED_FROM 不是合法的快照（缺 result/）：{seed}", file=sys.stderr)
            return 5
        _orig_prepare = harness._prepare

        # 注意：harness._prepare 是**实例属性**，不是绑定方法，
        # 所以 self._prepare(prompt) 只会传 prompt 一个参数 —— 这里用闭包。
        def _prepare_then_seed(prompt: str = "") -> None:
            _orig_prepare(prompt)
            restore_agent_workdir(harness.agent_workdir, seed)
            print(f"[seed] 已从 {seed} 恢复 agent workdir（续跑）", flush=True)

        harness._prepare = _prepare_then_seed
        print(f"[seed] 续跑模式，种子 = {seed}", flush=True)

    try:
        result = await harness.run(prompt)
        print(json.dumps({
            "status": result.status, "stop_reason": result.stop_reason,
            "rounds": result.rounds, "repair_rounds": result.repair_rounds,
            "best_score": result.best_score,
            "reproducible_score": result.reproducible_score,
        }, ensure_ascii=False, indent=1))
        return 0 if result.status == "SUCCESS_REPRODUCIBLE" else 1
    except BaseException:
        traceback.print_exc()
        print("\n!! harness 抛异常 —— 转入兜底收尾，保留已有轮次", file=sys.stderr)
        harness._key_columns = load_evaluation_manifest(
            config.evaluation_manifest, config.validation_gold)
        harness._model_environment = build_worker_model_environment("react_planner", "MiniMax-M3")
        return await finalize(harness, best_score_of(harness), "worker_crash_finalized")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
