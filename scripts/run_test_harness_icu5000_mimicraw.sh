#!/bin/bash
# 把 icu5000 mimic-raw-error 版 val harness 冻结下来的 pipeline 在 TEST 上重放一次
# （全程无 LLM，纯重放）。
#
# 与 K29 的 run_test_harness_k29_mimicraw.sh 同一套流程，只换数据集 / manifest：
#   * 完整数据集 = icu_5000_v2_rawerror_test{1.6,4.7}pct_614M_mimicrawerror
#     ⇒ test/reference 里 MIMIC 自身的原生错误已经修好，
#       test/modification_log.csv 里多登记了原生错误条目
#       （1.6pct 2,450 条 / 4.7pct 2,174 条）
#   * gold = 新数据集的 test/reference（不是密封视图）
#   * evaluation manifest = icu_5000_v1.json
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$WT/icu5000_agent_run}"

TIER="${TIER:-16}"
case "$TIER" in
  16) FULLBASE="icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror"; EXPPREFIX="icu5000_mimicraw_1p6" ;;
  47) FULLBASE="icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror"; EXPPREFIX="icu5000_mimicraw_4p7" ;;
  *) echo "TIER 只能是 16 或 47" >&2; exit 2 ;;
esac

EXP="${EXP:-$RUN/experiments/${EXPPREFIX}_val_seed666_v1}"
TEST_EXP="${TEST_EXP:-$RUN/experiments/${EXPPREFIX}_test_seed666_v1}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/icu5000/$FULLBASE}"
LOG="${LOG:-$RUN/logs/${EXPPREFIX}_test_seed666_v1.log}"

for d in "$FULL/test/raw" "$FULL/test/reference"; do
  if [ ! -d "$d" ]; then
    echo "FATAL: 缺目录: $d" >&2
    exit 5
  fi
done

if [ ! -d "$EXP/host/reproducible_snapshot" ]; then
  echo "!! val 实验还没跑完（缺 $EXP/host/reproducible_snapshot）" >&2
  exit 3
fi

mkdir -p "$RUN/logs"

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
    --evaluation-manifest "$PROJ/evaluation_manifests/icu_5000_v1.json" \
    --replay-timeout "${REPLAY_TIMEOUT:-21600}" \
    --scoring-timeout "${SCORING_TIMEOUT:-28800}" > "$LOG" 2>&1
rc=$?
echo
echo "test harness exit=$rc   log=$LOG"
tail -40 "$LOG"
exit $rc
