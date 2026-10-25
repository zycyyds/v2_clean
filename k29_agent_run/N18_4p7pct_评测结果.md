==========================================================================================
N18 4.7pct 评测结果（cell 错误率 4.72%）  ·  我们的 agent (pi + val harness)
==========================================================================================

> 本文件只记**我们 agent 这一条**结果。跨 agent / 跨档位的对比见
> `../../评测总表.md`（台账）或 `../../score_only/总表.md`。

## 一、总分 + 错误级指标（v29 口径）

| 谁 | composite | Det F1 | Rep F1 | b_dirty 没动 | b_wrong 修错 | b_missing 丢行 | 误改干净格 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我们的 agent (pi+val) | **0.947216** | 0.991523 | 0.812107 | 105,048 | 2,270,464 | 108,885 | 356 |

## 二、结果包路径

  结果包 : $PROJ/.claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments/n18_4p7_test_seed666_v1/host/test_result_package  (18 csv)
  gold   : $PROJ/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_test4.7pct_490M/test/reference
  分数   : $PROJ/.claude/worktrees/eval-movies-cleaning/k29_agent_run/experiments/n18_4p7_test_seed666_v1/host/score_report.json
  错误分 : $PROJ/runs/n18_more_agent_error_metrics_v29.json

## 三、逐文件 composite

| 文件 | file_score |
|---|---:|
| icu/outputevents.csv | 0.6260 |
| demographics.csv | 0.6407 |
| icu/inputevents.csv | 0.8919 |
| admissions.csv | 0.9124 |
| icu/chartevents.csv | 0.9888 |
| labs.csv | 0.9915 |
| multimodal/ed_pyxis.csv | 0.9998 |
| cohort.csv | 0.9998 |
| multimodal/ed_triage.csv | 0.9998 |
| multimodal/ed_vitalsign.csv | 0.9998 |
| multimodal/ed_diagnosis.csv | 0.9999 |
| medications.csv | 0.9999 |
| multimodal/ed_medrecon.csv | 0.9999 |
| multimodal/edstays.csv | 0.9999 |
| diagnoses.csv | 0.9999 |
| procedures.csv | 1.0000 |
| icu/icustays.csv | 1.0000 |
| multimodal/notes.csv | 1.0000 |
| **composite（18 文件等权）** | **0.947216** |

## 四、val → test 落差

| val composite | test composite | 落差 |
|---:|---:|---:|
| 0.947905 | 0.947216 | **-0.000689** |
