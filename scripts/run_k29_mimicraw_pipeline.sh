#!/bin/bash
# k29 mimicrawerror 的完整评测流水线（照抄 I48 的跑法）：
#   ① prepare（manifest + agent 视图 + prompt）
#   ② verify（独立校验数据集）
#   ③ val harness（LLM 多回合，带每回合反馈）
#   ④ test 纯重放
#   ⑤ v29 错误指标 + 原生错误格专用判定
#
# 用法: bash run_k29_mimicraw_pipeline.sh [47|16] [stage]
#   stage = all(默认) | prepare | verify | val | test | score
set -uo pipefail

WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
DSROOT="$PROJ/datasets/注入错误数据集"
STAGE="${2:-all}"

case "${1:-47}" in
  47) BASE="K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror"; SRC="K29_v2_rawerror_test4.7pct_14.1M"; TAG="4.7pct" ;;
  16) BASE="K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"; SRC="K29_v2_rawerror_test1.6pct_4.2M"; TAG="1.6pct" ;;
  *) echo "档位只能是 47 或 16" >&2; exit 2 ;;
esac
FULL="$DSROOT/k29/$BASE"
SRCD="$DSROOT/k29/$SRC"
RUN="$WT/k29_agent_run"
VIEW="$DSROOT/k29_mimicraw_$1"
PROMPT="$RUN/prompts/k29_${TAG}_mimicraw_val_prompt_v1.md"

run_stage() { case "$STAGE" in all|"$1") return 0;; *) return 1;; esac; }

if run_stage prepare; then
  echo "############ ① prepare"
  bash "$WT/scripts/prepare_k29_mimicraw_run.sh" "$1" || exit 5
fi

if run_stage verify; then
  echo; echo "############ ② verify（独立校验数据集）"
  python3 "$WT/reproduction/verify_k29_mimicraw_error.py" --src "$SRCD" --dst "$FULL"
fi

if run_stage val; then
  echo; echo "############ ③ val harness（LLM 多回合，可能要几小时）"
  bash "$WT/scripts/run_val_harness_k29_mimicraw.sh" v1
fi

if run_stage test; then
  echo; echo "############ ④ test 纯重放"
  bash "$WT/scripts/run_test_harness_k29_mimicraw.sh"
fi

if run_stage score; then
  echo; echo "############ ⑤ 打分"
  RESULT="$RUN/experiments/k29_mimicraw_${1}p7_test_seed666_v1/host/test_result_package"
  RESULT="$RUN/experiments/k29_mimicraw_4p7_test_seed666_v1/host/test_result_package"
  bash "$WT/scripts/score_k29_mimicraw_agent_err.sh" || true
  echo; echo "---- 原生错误格专用判定 ----"
  python3 "$WT/reproduction/eval_native_cells.py" \
    --dataset "$FULL" --result "$RESULT" --split test \
    --out "$PROJ/runs/k29_mimicraw_${1}_agent_native_cells.json" || true
fi
