#!/bin/bash
# 我们的 agent（4.7pct mimic-raw-error val harness 冻结的 pipeline）在
# **4.7pct mimic-raw-error 版 K29 test** 上的错误级指标。
#
# v29 口径：recover_clean_key=true + exclude_rows_not_in_gold=true
#   —— raw 比 gold 多一截时，agent 正确删掉的行里的注入错误不能记成漏检。
#
# 关键：数据集是 4.7pct mimic-raw-error 版 ⇒ test/modification_log.csv 里
#   多了 873 条 mimic_native_* 条目，agent 对这些格的修复会被单独记成
#   b_clean / b_wrong / b_dirty，而不是 natural_noise。
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCORE_ONLY="${SCORE_ONLY:-$WT/score_only}"
DS="${DS:-$PROJ/datasets/注入错误数据集/k29/K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror}"
RESULT="${RESULT:-$WT/k29_agent_run/experiments/k29_mimicraw_4p7_test_seed666_v1/host/test_result_package}"
OUT="${OUT:-$PROJ/runs/k29_mimicraw_47_agent_error_metrics_v29.json}"
LOG="${LOG:-/Users/mac/.claude/jobs/448aad57/tmp/err_k29_mimicraw_47_agent.log}"

if [ ! -d "$RESULT" ]; then
  echo "!! 找不到结果包：$RESULT" >&2
  exit 3
fi
echo "dataset = $DS"
echo "result  = $RESULT"
echo "out     = $OUT"
echo "log     = $LOG"

bash "$SCORE_ONLY/run_error_metrics.sh" "$DS" "$RESULT" k29_mimicraw_47_agent "$OUT" "$LOG"
