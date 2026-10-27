#!/bin/bash
# test 阶段启动脚本（K29 1800 低密度档）。
#
# 把 val 阶段冻结下来的 pipeline 在 TEST 上重放一次（全程无 LLM），
# 然后用 harness 自带的隐藏评分打分。
#
# ⚠️ 这一档 test raw 有 1807 万行 ⇒ replay 约 270 s、scoring 约 350 s，
#    timeout 给得比别的档宽松。
#
# 前置：val 阶段必须已经跑完（要有 host/reproducible_snapshot）。
# 用法: bash run_test_harness_k29_1800.sh
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
RUN="${RUN:-$PROJ/k29_agent_run}"
EXP="${EXP:-$RUN/experiments/k29_1800_val_seed666_fix1}"
TEST_EXP="${TEST_EXP:-$RUN/experiments/k29_1800_test_seed666_fix1}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/K29_clean_v2_rawerror_10train_400_800_1800}"
LOG="${LOG:-$RUN/logs/k29_1800_test_seed666_fix1.log}"

for d in "$FULL/test/raw" "$FULL/test/reference"; do
  if [ ! -d "$d" ]; then
    echo "FATAL: 缺目录: $d" >&2
    exit 5
  fi
done

if [ ! -d "$EXP/host/reproducible_snapshot" ]; then
  echo "!! val 阶段没跑完（缺 $EXP/host/reproducible_snapshot）" >&2
  exit 3
fi

mkdir -p "$(dirname "$LOG")"

echo "=== validation experiment : $EXP"
echo "=== test experiment       : $TEST_EXP"
echo "=== test raw              : $FULL/test/raw"
echo "=== test gold             : $FULL/test/reference"
echo "=== log                   : $LOG"
echo

cd /tmp
conda run -n py311agent2 --no-capture-output env PYTHONPATH="$PROJ" \
  python -m agent.pi_test_harness_cli \
    --validation-experiment "$EXP" \
    --test-experiment "$TEST_EXP" \
    --test-raw "$FULL/test/raw" \
    --test-gold "$FULL/test/reference" \
    --evaluation-manifest "$PROJ/evaluation_manifests/k29_gastritis_duodenitis_v3_1.json" \
    --replay-timeout "${REPLAY_TIMEOUT:-10800}" \
    --scoring-timeout "${SCORING_TIMEOUT:-14400}" > "$LOG" 2>&1
rc=$?
echo
echo "test harness exit=$rc   log=$LOG"
tail -30 "$LOG"
exit $rc
