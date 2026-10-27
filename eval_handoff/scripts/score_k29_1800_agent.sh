#!/bin/bash
# K29 1800（低密度档，fixed-v1）test 结果的独立打分：
#   1) Raha composite（零项目依赖，只要 python3 + pandas）
#   2) 错误级指标 v29（recover_clean_key=true + exclude_rows_not_in_gold=true）
#
# ⚠️ 两个都要读 test gold。台账里的 K29 1800 一条就是这两个数。
#
# 用法: bash score_k29_1800_agent.sh
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$PROJ/.claude/worktrees/eval-movies-cleaning}"
DS="${DS:-$PROJ/datasets/注入错误数据集/K29_clean_v2_rawerror_10train_400_800_1800}"
PKG="${PKG:-$WT/k29_agent_run/experiments/k29_1800_test_seed666_fix1/host/test_result_package/result}"
RUNS="${RUNS:-$PROJ/runs}"

mkdir -p "$RUNS"

echo "=== 1/2 Raha composite"
bash "$WT/score_only/run_raha.sh" \
  "$PKG" \
  "$DS/test/reference" \
  "$WT/score_only/manifests/k29_gastritis_duodenitis_v3_1.json" \
  "$RUNS/k29_1800_agent_overall_score.json"
rc1=$?

echo
echo "=== 2/2 错误级指标 v29"
bash "$WT/score_only/run_error_metrics.sh" \
  "$DS" \
  "$PKG" \
  k29_1800_agent \
  "$RUNS/k29_1800_agent_error_metrics_v29.json" \
  "$WT/k29_agent_run/logs/k29_1800_test_err_metrics.log"
rc2=$?

echo
echo "=== composite → $RUNS/k29_1800_agent_overall_score.json   (rc=$rc1)"
echo "=== 错误级   → $RUNS/k29_1800_agent_error_metrics_v29.json (rc=$rc2)"
exit $(( rc1 != 0 ? rc1 : rc2 ))
