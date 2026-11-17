#!/bin/bash
# 在 ICU5000 1.6pct「MIMIC 原生错误」增强版上跑 val harness：
# LLM 多回合 + host 每回合打分反馈 + 分数掉了回滚 workdir，全程不接触 test。
#
# 与 K29 的 run_val_harness_k29_mimicraw.sh 同一套流程，只换 profile：
#   * 数据集视图 = datasets/val_views/icu5000_mimicraw_16（**密封位置**，
#     父目录里只有别的视图，没有完整数据集 ⇒ agent 枚举父目录也看不到答案）
#   * prompt     = eval_handoff/prompts/icu5000_1.6pct_mimicraw_val_prompt_v1.md
#                  （多两节：「MIMIC 自身的原生错误」+「itemid 列上的 CareVue 别名注入」）
#   * profile    = icu5000_mimicraw_16
#
# 视图里刻意没有：modification_log.csv（注入日志=答案键）、
#                  mimic_native_error_log.csv（原生错误答案键）、test/
set -euo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$WT/icu5000_agent_run}"
DRIVER="${DRIVER:-$WT/eval_handoff/scripts/run_val_harness_ft.py}"
SUFFIX="${1:-v1}"

# 档位：TIER=16（默认，1.6pct）/ TIER=47（4.7pct）
TIER="${TIER:-16}"
case "$TIER" in
  16)
    TAG="1.6pct"; VIEW="icu5000_mimicraw_16"
    FULLBASE="icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror"
    PROFILE="icu5000_mimicraw_16"; EXPPREFIX="icu5000_mimicraw_1p6_val_seed666"
    ;;
  47)
    TAG="4.7pct"; VIEW="icu5000_mimicraw_47"
    FULLBASE="icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror"
    PROFILE="icu5000_mimicraw_47"; EXPPREFIX="icu5000_mimicraw_4p7_val_seed666"
    ;;
  *) echo "TIER 只能是 16 或 47" >&2; exit 2 ;;
esac

DS="${DS:-$PROJ/datasets/val_views/$VIEW}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/icu5000/$FULLBASE}"
EXP="${EXP:-$RUN/experiments/${EXPPREFIX}_$SUFFIX}"
PROMPT="${PROMPT:-$WT/eval_handoff/prompts/icu5000_${TAG}_mimicraw_val_prompt_v1.md}"
LOG="${LOG:-$RUN/logs/${EXPPREFIX}_$SUFFIX.log}"

for f in "$PROMPT" "$DRIVER" \
         "$FULL/v2_rawerror_icu5000_final_report.json" \
         "$PROJ/evaluation_manifests/icu_5000_v1.json"; do
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
