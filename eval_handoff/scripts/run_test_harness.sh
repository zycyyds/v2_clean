#!/bin/bash
# test 阶段启动脚本（icu5000 4.7pct 范例）。
#
# 把 val 阶段冻结下来的 pipeline 在 TEST 上重放一次（全程无 LLM），
# 然后用 pi_test_scorer 打分。
#
# 换数据集时改：EXP / TEST_EXP / FULL
#
# 用法: bash run_test_harness.sh
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
RUN="${RUN:-$PROJ/k29_agent_run}"
EXP="${EXP:-$RUN/experiments/icu5000_4p7_val_seed666_v1}"
TEST_EXP="${TEST_EXP:-$RUN/experiments/icu5000_4p7_test_seed666_v1}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/icu_5000_v2_rawerror_test4.7pct_614M}"
LOG="${LOG:-$RUN/logs/icu5000_4p7_test_seed666_v1.log}"

if [ ! -d "$EXP/host/reproducible_snapshot" ]; then
  echo "!! val 阶段没跑完（缺 $EXP/host/reproducible_snapshot）" >&2
  exit 3
fi

mkdir -p "$(dirname "$LOG")"

echo "=== validation experiment : $EXP"
echo "=== test experiment       : $TEST_EXP"
echo "=== test raw              : $FULL/test/raw"
echo "=== log                   : $LOG"
echo

cd /tmp
conda run -n py311agent2 --no-capture-output env PYTHONPATH="$PROJ" \
  python -m agent.pi_test_harness_cli \
    --validation-experiment "$EXP" \
    --test-experiment "$TEST_EXP" \
    --test-raw "$FULL/test/raw" \
    --test-gold "$FULL/test/reference" \
    --evaluation-manifest "$PROJ/evaluation_manifests/icu_5000_v1.json" \
    --replay-timeout "${REPLAY_TIMEOUT:-7200}" \
    --scoring-timeout "${SCORING_TIMEOUT:-10800}" > "$LOG" 2>&1
rc=$?
echo
echo "test harness exit=$rc   log=$LOG"
tail -30 "$LOG"
exit $rc
