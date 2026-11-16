#!/bin/bash
# 在 K29 4.7pct「MIMIC 原生错误」增强版上跑我们的 val harness：
# LLM 多回合 + host 每回合打分反馈 + 分数掉了回滚 workdir，全程不接触 test。
#
# 与 I48 的 run_val_harness_i48_mimicraw.sh 同一套流程，只换 profile：
#   * 数据集视图 = k29_mimicraw_1（train/validation 的 reference 里，MIMIC 自身的
#     原生错误已经修好 ⇒ agent 在 train 对照里第一次能看见这类错误模式）
#   * prompt     = k29_4.7pct_mimicraw_val_prompt_v1.md（多一节「新增错误类：
#     MIMIC 自身的原生错误」）
#   * profile    = k29_mimicraw
#
# 视图里刻意没有：modification_log.csv（注入日志=答案键）、
#                  mimic_native_error_log.csv（原生错误答案键）、test/
set -euo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$WT/k29_agent_run}"
DRIVER="${DRIVER:-$WT/eval_handoff/scripts/run_val_harness_ft.py}"
SUFFIX="${1:-v1}"

# 档位：TIER=47（默认，4.7pct）/ TIER=16（1.6pct）。两个档位的**原生错误层完全相同**
# （同样 873 格），差别只在注入密度 ⇒ prompt 只有那 3 行密度数字不同。
TIER="${TIER:-47}"
case "$TIER" in
  47)
    TAG="4.7pct"; VIEW="k29_mimicraw_47"
    FULLBASE="K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror"
    PROFILE="k29_mimicraw"; EXPPREFIX="k29_mimicraw_4p7_val_seed666"
    ;;
  16)
    TAG="1.6pct"; VIEW="k29_mimicraw_16"
    FULLBASE="K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"
    PROFILE="k29_mimicraw_16"; EXPPREFIX="k29_mimicraw_1p6_val_seed666"
    ;;
  *) echo "TIER 只能是 47 或 16" >&2; exit 2 ;;
esac

DS="${DS:-$PROJ/datasets/注入错误数据集/$VIEW}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/k29/$FULLBASE}"
EXP="${EXP:-$RUN/experiments/${EXPPREFIX}_$SUFFIX}"
PROMPT="${PROMPT:-$RUN/prompts/k29_${TAG}_mimicraw_val_prompt_v1.md}"
LOG="${LOG:-$RUN/logs/${EXPPREFIX}_$SUFFIX.log}"

for f in "$PROMPT" "$DRIVER" \
         "$FULL/v2_rawerror_k29_rebuild_final_report.json" \
         "$PROJ/evaluation_manifests/k29_gastritis_duodenitis_v3_1.json"; do
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
  PYTHONPATH="$PROJ" RUN="$RUN" PROMPT_DIR="$RUN/prompts" WT="$WT" \
  python "$DRIVER" "$PROFILE" "$SUFFIX" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo
echo "val harness exit=$rc   log=$LOG"
exit $rc
