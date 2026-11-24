#!/bin/bash
# 把「校正类」切分的 val harness 冻结下来的 pipeline 在 **test 段**上重放一次
# （全程无 LLM，纯重放 + 打分）。
#
# 校正类的 test 段 = 切分里的 **correction/**（=「剩下的全部行」）：
#   test raw  = correction/raw
#   test gold = correction/reference_private（**对 agent 不可读**，host 专用）
#   ⇒ 与 MIMIC 那一族不同：不是 test/ 子目录，而是 correction/。
#
# ⚠️ evaluation manifest **val 与 test 两段共用同一份**：
#    loader 只看 `files.<name>.key_columns`（key_columns 必须存在于 gold 表里），
#    `expected_rows` 只是说明性字段、scorer 不校验。
#
# 用法:
#   TIER=hospital bash scripts/run_test_harness_correction.sh <val实验目录> [test后缀]
set -euo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$WT/correction_agent_run}"

TIER="${TIER:-hospital}"
case "$TIER" in
  hospital)
    TAG="hospital"; SPLITDIR="raha_hospital_curated10_val20_dense_v1"
    MANIFEST="$WT/evaluation_manifests/raha_correction_hospital_v1.json"
    ;;
  beers)
    TAG="beers"; SPLITDIR="beers_curated10_val20_v1"
    MANIFEST="$WT/evaluation_manifests/raha_correction_beers_v1.json"
    ;;
  rayyan)
    TAG="rayyan"; SPLITDIR="rayyan_curated10_val20_v1"
    MANIFEST="$WT/evaluation_manifests/raha_correction_rayyan_v1.json"
    ;;
  tax)
    TAG="tax"; SPLITDIR="tax_curated10_val20_v1"
    MANIFEST="$WT/evaluation_manifests/raha_correction_tax_v1.json"
    ;;
  *) echo "TIER 只能是 hospital / beers / rayyan / tax" >&2; exit 2 ;;
esac

DS="${DS:-$PROJ/datasets/correction/$SPLITDIR}"
VAL_EXP="${1:?必须给 val 实验目录}"
TSUFFIX="${2:-v1}"
TEST_EXP="${TEST_EXP:-$RUN/experiments/${TAG}_correction_test_seed666_$TSUFFIX}"
LOG="${LOG:-$RUN/logs/${TAG}_correction_test_seed666_$TSUFFIX.log}"

for f in "$MANIFEST"; do
  [ -f "$f" ] || { echo "FATAL: 缺文件: $f" >&2; exit 5; }
done
for d in "$DS/correction/raw" "$DS/correction/reference_private"; do
  [ -d "$d" ] || { echo "FATAL: 缺目录: $d" >&2; exit 5; }
done

if [ ! -d "$VAL_EXP/host/reproducible_snapshot" ]; then
  echo "!! val 实验还没跑完（缺 $VAL_EXP/host/reproducible_snapshot）" >&2
  exit 3
fi

mkdir -p "$RUN/logs" "$RUN/experiments"

echo "=== validation experiment : $VAL_EXP"
echo "=== test experiment       : $TEST_EXP"
echo "=== test raw              : $DS/correction/raw"
echo "=== test gold             : $DS/correction/reference_private"
echo "=== manifest              : $MANIFEST"
echo "=== log                   : $LOG"
echo

cd /tmp
conda run -n py311agent2 --no-capture-output env PYTHONPATH="$PROJ" \
  python -m agent.pi_test_harness_cli \
    --validation-experiment "$VAL_EXP" \
    --test-experiment "$TEST_EXP" \
    --test-raw "$DS/correction/raw" \
    --test-gold "$DS/correction/reference_private" \
    --evaluation-manifest "$MANIFEST" \
    --replay-timeout "${REPLAY_TIMEOUT:-21600}" \
    --scoring-timeout "${SCORING_TIMEOUT:-28800}" > "$LOG" 2>&1
rc=$?
echo
echo "test harness exit=$rc   log=$LOG"
tail -40 "$LOG"
exit $rc
