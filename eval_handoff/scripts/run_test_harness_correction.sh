#!/bin/bash
# test 阶段（校正类 / 单表纠错）：把 val 阶段冻结下来的 pipeline 在 **test(=correction)**
# 上重放一次（全程无 LLM），然后用校正类口径出最终分数。
#
# 用法:
#   bash eval_handoff/scripts/run_test_harness_correction.sh <dataset> <exp_suffix> [test_suffix]
#     dataset     : flights | hospital
#     exp_suffix  : val 阶段的实验后缀（默认 v1）
#     test_suffix : test 阶段的实验后缀（默认 v1）
#
# 前置：val 阶段必须跑完（要有 host/reproducible_snapshot）。
set -uo pipefail

DATASET="${1:-flights}"
EXP_SUFFIX="${2:-v1}"
TEST_SUFFIX="${3:-v1}"

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning}"
RUN="${RUN:-$WT/k29_agent_run}"

case "$DATASET" in
  flights)
    SPLIT="$PROJ/datasets/correction/flights_curated10_val20_v1"
    EVAL_MANIFEST="$WT/evaluation_manifests/raha_correction_flights_v1.json"
    ;;
  hospital)
    SPLIT="$PROJ/datasets/correction/raha_hospital_curated10_val20_dense_v1"
    EVAL_MANIFEST="$WT/evaluation_manifests/raha_correction_hospital_v1.json"
    ;;
  *) echo "FATAL: 未知 dataset: $DATASET" >&2; exit 2;;
esac

EXP="$RUN/experiments/${DATASET}_correction_val_seed666_${EXP_SUFFIX}"
TEST_EXP="$RUN/experiments/${DATASET}_correction_test_seed666_${TEST_SUFFIX}"
LOG="$RUN/logs/${DATASET}_correction_test_${TEST_SUFFIX}.log"

for d in "$SPLIT/correction/raw" "$SPLIT/correction/reference_private"; do
  if [ ! -d "$d" ]; then echo "FATAL: 缺目录: $d" >&2; exit 5; fi
done
if [ ! -d "$EXP/host/reproducible_snapshot" ]; then
  echo "!! val 阶段没跑完（缺 $EXP/host/reproducible_snapshot）" >&2; exit 3
fi
if [ -e "$TEST_EXP" ]; then
  echo "FATAL: test 实验目录已存在（harness 要求全新且为空）: $TEST_EXP" >&2; exit 2
fi

mkdir -p "$(dirname "$LOG")"
echo "=== dataset        : $DATASET"
echo "=== validation exp : $EXP"
echo "=== test exp       : $TEST_EXP"
echo "=== test raw       : $SPLIT/correction/raw"
echo "=== test gold      : $SPLIT/correction/reference_private"
echo "=== log            : $LOG"
echo

cd /tmp
conda run -n py311agent2 --no-capture-output env PYTHONPATH="$PROJ" \
  python -m agent.pi_test_harness_cli \
    --validation-experiment "$EXP" \
    --test-experiment "$TEST_EXP" \
    --test-raw "$SPLIT/correction/raw" \
    --test-gold "$SPLIT/correction/reference_private" \
    --evaluation-manifest "$EVAL_MANIFEST" \
    --replay-timeout "${REPLAY_TIMEOUT:-3600}" \
    --scoring-timeout "${SCORING_TIMEOUT:-3600}" > "$LOG" 2>&1
rc=$?
echo
echo "test harness exit=$rc   log=$LOG"
tail -20 "$LOG"

if [ "$rc" = "0" ]; then
  echo
  echo "=== 校正类口径的最终分数 ==="
  RESULT="$TEST_EXP/output/flights.csv"
  [ "$DATASET" = "hospital" ] && RESULT="$TEST_EXP/output/hospital.csv"
  if [ -f "$RESULT" ]; then
    python3 "$WT/eval_handoff/scripts/score_correction_result.py" \
      --dataset "$DATASET" --result "$RESULT" \
      --label "our_agent_${DATASET}_correction" \
      --out "$PROJ/runs/correction_${DATASET}_agent_score.json"
  else
    echo "!! 找不到结果文件: $RESULT" >&2
  fi
fi
exit $rc
