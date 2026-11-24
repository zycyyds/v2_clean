#!/bin/bash
# 在 N18「MIMIC 原生错误」增强版（mimicraw）上跑 val harness：
# LLM 多回合 + host 每回合打分反馈 + 分数掉了回滚 workdir，全程不接触 test。
#
# 与 i48 的同名脚本同一套流程，只换 profile / 视图 / prompt / manifest：
#   * 数据集视图 = datasets/val_views/n18_mimicraw_{16,47}（**密封位置**）
#   * prompt     = eval_handoff/prompts/n18_{1.6,4.7}pct_mimicraw_val_prompt_v2.md
#                  （多三节：「生理不可能区间」「哨兵占位码 + 清单不是封闭的」
#                    「itemid 别名注入」）
#   * profile    = n18_mimicraw_16 / n18_mimicraw_47
#
# 视图里刻意没有：modification_log.csv（注入日志=答案键）、
#                  mimic_native_error_log.csv（原生错误答案键）、test/
set -euo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$WT/n18_agent_run}"
DRIVER="${DRIVER:-$WT/eval_handoff/scripts/run_val_harness_ft.py}"
SUFFIX="${1:-v1}"

TIER="${TIER:-16}"
PROMPTV="${PROMPTV:-v5}"
case "$TIER" in
  16)
    TAG="1.6pct"; VIEW="n18_mimicraw_16"
    FULLBASE="N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
    PROFILE="n18_mimicraw_16"; EXPPREFIX="n18_mimicraw_1p6_val_seed666"
    ;;
  47)
    TAG="4.7pct"; VIEW="n18_mimicraw_47"
    FULLBASE="N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"
    PROFILE="n18_mimicraw_47"; EXPPREFIX="n18_mimicraw_4p7_val_seed666"
    ;;
  *) echo "TIER 只能是 16 或 47" >&2; exit 2 ;;
esac
# v6 的 prompt 多一节「显式 6 步顺序」（先 cohort 筛选、再修原生错误）
# v7 的 prompt 再多一节「上下文预算 + 续跑摘要格式」（防上下文压缩摘要的 schema 崩）
# ⇒ 用单独的 profile（driver 里 prompt 路径写死在 profile 上）。
case "$PROMPTV" in
  v6) PROFILE="${PROFILE}_v6" ;;
  v7) PROFILE="${PROFILE}_v7" ;;
  v8) PROFILE="${PROFILE}_v8" ;;
  v9) PROFILE="${PROFILE}_v9" ;;
  v2|v3|v4|v5) : ;;
  *) echo "PROMPTV 只能是 v2..v9" >&2; exit 2 ;;
esac

DS="${DS:-$PROJ/datasets/val_views/$VIEW}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/n18/$FULLBASE}"
EXP="${EXP:-$RUN/experiments/${EXPPREFIX}_$SUFFIX}"
PROMPT="${PROMPT:-$WT/eval_handoff/prompts/n18_${TAG}_mimicraw_val_prompt_${PROMPTV}.md}"
LOG="${LOG:-$RUN/logs/${EXPPREFIX}_$SUFFIX.log}"

for f in "$PROMPT" "$DRIVER" \
         "$FULL/v2_rawerror_n18_final_report.json" \
         "$PROJ/evaluation_manifests/n18_ckd_ED_notes_v1.json"; do
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
# ⚠️ SEED_FROM：续跑模式 —— 把上一轮实验的 best_snapshot 灌进新实验的 workdir，
#    让 agent 从自己上一轮的状态（analysis/ pipeline/ train_analysis.json）继续，
#    而不是从头重推一遍。换 prompt 版本做 A/B 时用得上。
# ⚠️ 未设 SEED_FROM 时必须**完全不出现**这个参数（空串会被 env 当成命令名）。
SEED_ARG=""
if [ -n "${SEED_FROM:-}" ]; then
  SEED_ARG="SEED_FROM=$SEED_FROM"
  echo "=== 续跑模式：seed = $SEED_FROM"
fi
conda run -n py311agent2 --no-capture-output env \
  PYTHONPATH="$PROJ" RUN="$RUN" PROMPT_DIR="$WT/eval_handoff/prompts" WT="$WT" \
  $SEED_ARG \
  python "$DRIVER" "$PROFILE" "$SUFFIX" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo
echo "val harness exit=$rc   log=$LOG"
exit $rc
