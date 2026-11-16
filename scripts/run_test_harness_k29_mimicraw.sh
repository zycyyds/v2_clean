#!/bin/bash
# 把 4.7pct mimic-raw-error 版 val harness 冻结下来的 pipeline 在 TEST 上重放一次
# （全程无 LLM，纯重放）。
#
# 与 I48 的 run_test_harness_i48_mimicraw_47.sh 同一套流程，只换数据集/profile：
#   * 完整数据集 = K29 4.7pct mimicrawerror 版 ⇒ test/reference 里 MIMIC 自身的
#     原生错误已经修好，test/modification_log.csv 里多登记了 873 条原生错误条目
#   * gold = 新数据集的 test/reference（不是密封视图）
#
# 跑完之后用 score_k29_mimicraw_agent_err.sh 打 v29 错误指标（数值感知口径）。
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$WT/k29_agent_run}"

# 档位：TIER=47（默认，4.7pct）/ TIER=16（1.6pct）
TIER="${TIER:-47}"
case "$TIER" in
  47) FULLBASE="K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror"; EXPPREFIX="k29_mimicraw_4p7" ;;
  16) FULLBASE="K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"; EXPPREFIX="k29_mimicraw_1p6" ;;
  *) echo "TIER 只能是 47 或 16" >&2; exit 2 ;;
esac

EXP="${EXP:-$RUN/experiments/${EXPPREFIX}_val_seed666_v1}"
TEST_EXP="${TEST_EXP:-$RUN/experiments/${EXPPREFIX}_test_seed666_v1}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/k29/$FULLBASE}"
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
    --evaluation-manifest "$PROJ/evaluation_manifests/k29_gastritis_duodenitis_v3_1.json" \
    --replay-timeout "${REPLAY_TIMEOUT:-21600}" \
    --scoring-timeout "${SCORING_TIMEOUT:-28800}" > "$LOG" 2>&1
rc=$?
echo
echo "test harness exit=$rc   log=$LOG"
tail -40 "$LOG"
exit $rc
