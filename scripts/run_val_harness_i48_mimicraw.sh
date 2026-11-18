#!/bin/bash
# 在 I48「MIMIC 原生错误」增强版（mimicraw2）上跑 val harness：
# LLM 多回合 + host 每回合打分反馈 + 分数掉了回滚 workdir，全程不接触 test。
#
# 与 icu5000 的同名脚本同一套流程，只换 profile / 视图 / prompt：
#   * 数据集视图 = datasets/val_views/i48_mimicraw2_{16,47}（**密封位置**）
#   * prompt     = eval_handoff/prompts/i48_{1.6,4.7}pct_mimicraw_val_prompt_v5.md
#                  （多三节：「类 2 哨兵占位码」+「清单不是封闭的」+「itemid 别名注入」）
#   * profile    = i48_mimicraw2_{16,47}
#
# 视图里刻意没有：modification_log.csv（注入日志=答案键）、
#                  mimic_native_error_log.csv（原生错误答案键）、test/
set -euo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$WT/i48_agent_run}"
DRIVER="${DRIVER:-$WT/eval_handoff/scripts/run_val_harness_ft.py}"
SUFFIX="${1:-v1}"

TIER="${TIER:-16}"
case "$TIER" in
  16)
    TAG="1.6pct"; VIEW="i48_mimicraw2_16"
    FULLBASE="I48_atrial_fibrillation_v2_rawerror_mimicraw2_test1.6pct_411M"
    PROFILE="i48_mimicraw2_16"; EXPPREFIX="i48_mimicraw2_1p6_val_seed666"
    ;;
  47)
    TAG="4.7pct"; VIEW="i48_mimicraw2_47"
    FULLBASE="I48_atrial_fibrillation_v2_rawerror_mimicraw2_test4.7pct_411M"
    PROFILE="i48_mimicraw2_47"; EXPPREFIX="i48_mimicraw2_4p7_val_seed666"
    ;;
  *) echo "TIER 只能是 16 或 47" >&2; exit 2 ;;
esac

DS="${DS:-$PROJ/datasets/val_views/$VIEW}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/i48/$FULLBASE}"
EXP="${EXP:-$RUN/experiments/${EXPPREFIX}_$SUFFIX}"
PROMPT="${PROMPT:-$WT/eval_handoff/prompts/i48_${TAG}_mimicraw_val_prompt_v5.md}"
LOG="${LOG:-$RUN/logs/${EXPPREFIX}_$SUFFIX.log}"

for f in "$PROMPT" "$DRIVER" \
         "$FULL/v2_rawerror_i48_final_report.json" \
         "$PROJ/evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json"; do
  if [ ! -f "$f" ]; then
    echo "FATAL: 缺文件: $f" >&2
    exit 5
  fi
done
for d in "$DS/train/raw" "$DS/train/reference" "$DS/validation/raw" "$DS/validation/reference"; do
  if [ ! -d "$d" ]; then
    echo "FATAL: 缺目录: $d" >&2
    exit 5
  fi
done

mkdir -p "$RUN/logs" "$RUN/experiments"

if [ -e "$EXP" ]; then
  echo "FATAL: 实验目录已存在（harness 要求 new+empty）: $EXP" >&2
  exit 2
fi

# 泄漏自检 —— 视图里不该有答案
if find "$DS" \( -name 'modification_log*' -o -name 'mimic_native_error_log*' \) | grep -q .; then
  echo "FATAL: 视图里有答案文件，不能给 agent" >&2
  exit 4
fi

echo "=== experiment : $EXP"
echo "=== log        : $LOG"
echo "=== dataset    : $DS"
echo "=== prompt     : $PROMPT"
echo "=== driver     : $DRIVER"
echo

cd /tmp
conda run -n py311agent2 --no-capture-output env \
  PYTHONPATH="$PROJ" RUN="$RUN" PROMPT_DIR="$WT/eval_handoff/prompts" WT="$WT" \
  python "$DRIVER" "$PROFILE" "$SUFFIX" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo
echo "val harness exit=$rc   log=$LOG"
exit $rc
