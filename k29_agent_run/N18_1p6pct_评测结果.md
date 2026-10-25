==========================================================================================
N18 1.6pct 评测结果（cell 错误率 1.58%）  ·  我们的 agent (pi + val harness)
==========================================================================================

> 本文件只记**我们 agent 这一条**结果。跨 agent / 跨档位的对比见
> `../../评测总表.md`（台账）或 `../../score_only/总表.md`。

## 一、总分 + 错误级指标（v29 口径）

| 谁 | composite | Det F1 | Rep F1 | b_dirty 没动 | b_wrong 修错 | b_missing 丢行 | 误改干净格 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我们的 agent (pi+val) | **0.941817** | 0.997427 | 0.792025 | 13,061 | 839,490 | 3,247 | 4,607 |

（旧口径 v28 对照：Det F1 0.699140 / Rep F1 0.555167 / b_missing 3,490,901 —— 把 agent 正确删掉的 3,487,654 个格子误记成漏检）

## 二、结果包路径

  结果包 : $PROJ/.claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments/n18_1p6_test_seed666_v1/host/test_result_package  (18 csv)
  gold   : $PROJ/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_test1.6pct_490M/test/reference
  分数   : $PROJ/.claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments/n18_1p6_test_seed666_v1/host/score_report.json
  错误分 : $PROJ/runs/n18_agent_error_metrics_v29.json

## 三、逐文件 composite

| 文件 | file_score |
|---|---:|
| icu/outputevents.csv | 0.5842 |
| demographics.csv | 0.6389 |
| icu/inputevents.csv | 0.8867 |
| admissions.csv | 0.9067 |
| multimodal/ed_diagnosis.csv | 0.9901 |
| multimodal/ed_pyxis.csv | 0.9902 |
| multimodal/ed_vitalsign.csv | 0.9914 |
| multimodal/ed_medrecon.csv | 0.9927 |
| cohort.csv | 0.9930 |
| multimodal/ed_triage.csv | 0.9930 |
| multimodal/edstays.csv | 0.9930 |
| diagnoses.csv | 0.9963 |
| icu/chartevents.csv | 0.9975 |
| labs.csv | 0.9994 |
| medications.csv | 0.9998 |
| procedures.csv | 0.9998 |
| icu/icustays.csv | 1.0000 |
| multimodal/notes.csv | 1.0000 |
| **composite（18 文件等权）** | **0.941817** |

## 四、val → test 落差

| val composite | test composite | 落差 |
|---:|---:|---:|
| 0.949407 | 0.941817 | **-0.007590** |
