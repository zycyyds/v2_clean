#!/bin/bash
# 在「校正类（单表纠错）」切分上跑 val harness：
# LLM 多回合 + host 每回合打分反馈 + 分数掉了回滚 workdir，全程不接触 test。
#
# 这一族和 MIMIC cohort 那一族**完全不同**：
#   任务 = 给 10 行「脏 ↔ 干净」对照，归纳破坏规则，**修好整张单表**；
#   切分 = train(10~30 行) / validation(20 行) / correction(=test，970~199,950 行)
#   ⚠️ validation 的标准答案在 **validation/gold**（不是 reference）
#   ⚠️ correction/reference_private 与 host_private 对 agent **不可读**
#      （sandbox 只放行 train raw / train reference / validation raw 三个根）
#
# 用法:
#   TIER=hospital bash scripts/run_val_harness_correction.sh [suffix]
#   TIER=beers|rayyan|tax bash scripts/run_val_harness_correction.sh [suffix]
set -euo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# ⚠️ 校正类的实验单独放一个 RUN 目录（driver 默认是 PROJ/k29_agent_run，会跟 k29 混在一起）
RUN="${RUN:-$WT/correction_agent_run}"
DRIVER="${DRIVER:-$WT/eval_handoff/scripts/run_val_harness_ft.py}"
SUFFIX="${1:-v1}"

TIER="${TIER:-hospital}"
case "$TIER" in
  hospital)
    TAG="hospital"; PROFILE="hospital_correction"; EXPPREFIX="hospital_correction_val_seed666"
    SPLITDIR="raha_hospital_curated10_val20_dense_v1"
    PROMPT="$WT/eval_handoff/prompts/hospital_correction_val_prompt_v1.md"
    MANIFEST="$WT/evaluation_manifests/raha_correction_hospital_v1.json"
    ;;
  beers)
    TAG="beers"; PROFILE="beers_correction"; EXPPREFIX="beers_correction_val_seed666"
    SPLITDIR="beers_curated10_val20_v1"
    PROMPT="$WT/eval_handoff/prompts/beers_correction_val_prompt_v1.md"
    MANIFEST="$WT/evaluation_manifests/raha_correction_beers_v1.json"
    ;;
  rayyan)
    TAG="rayyan"; PROFILE="rayyan_correction"; EXPPREFIX="rayyan_correction_val_seed666"
    SPLITDIR="rayyan_curated10_val20_v1"
    PROMPT="$WT/eval_handoff/prompts/rayyan_correction_val_prompt_v1.md"
    MANIFEST="$WT/evaluation_manifests/raha_correction_rayyan_v1.json"
    ;;
  tax)
    TAG="tax"; PROFILE="tax_correction"; EXPPREFIX="tax_correction_val_seed666"
    SPLITDIR="tax_curated10_val20_v1"
    PROMPT="$WT/eval_handoff/prompts/tax_correction_val_prompt_v1.md"
    MANIFEST="$WT/evaluation_manifests/raha_correction_tax_v1.json"
    ;;
  *) echo "TIER 只能是 hospital / beers / rayyan / tax" >&2; exit 2 ;;
esac

DS="${DS:-$PROJ/datasets/correction/$SPLITDIR}"
# ⚠️ 实验目录名必须与 driver 拼出来的一致（exp_prefix + "_" + suffix），
#    否则「已存在就报错」这道保险会看错目录（driver 自己拼的是 exp_prefix_suffix）
EXP="${EXP:-$RUN/experiments/${EXPPREFIX}_$SUFFIX}"
LOG="${LOG:-$RUN/logs/${EXPPREFIX}_$SUFFIX.log}"

for f in "$PROMPT" "$MANIFEST" "$DRIVER" "$DS/split_manifest.json"; do
  if [ ! -f "$f" ]; then
    echo "FATAL: 缺文件: $f" >&2
    exit 5
  fi
done
for d in "$DS/train/raw" "$DS/train/reference" "$DS/validation/raw" "$DS/validation/gold"; do
  if [ ! -d "$d" ]; then
    echo "FATAL: 缺目录: $d" >&2
    exit 5
  fi
done

# ⚠️ 答案隔离自检：
#   ① validation/gold 必须**不在** validation/raw 里面（否则 agent 能读到答案）
#   ② correction/reference_private 与 host_private 不能落在 agent 可见的三个根里
case "$(cd "$DS/validation/raw" && pwd -P)" in
  "$(cd "$DS/validation/gold" && pwd -P)"*)
    echo "FATAL: validation/gold 在 validation/raw 里面 ⇒ 会漏答案" >&2; exit 4 ;;
esac
if [ -e "$DS/validation/raw/gold" ] || [ -e "$DS/train/raw/reference_private" ]; then
  echo "FATAL: agent 可见根里出现了答案文件" >&2; exit 4
fi

mkdir -p "$RUN/logs" "$RUN/experiments"

if [ -e "$EXP" ]; then
  echo "FATAL: 实验目录已存在（harness 要求 new+empty）: $EXP" >&2
  exit 2
fi

echo "=== experiment : $EXP"
echo "=== log        : $LOG"
echo "=== dataset    : $DS"
echo "=== prompt     : $PROMPT"
echo "=== manifest   : $MANIFEST"
echo "=== RUN        : $RUN"
echo

cd /tmp
conda run -n py311agent2 --no-capture-output env \
  PYTHONPATH="$PROJ" RUN="$RUN" PROMPT_DIR="$WT/eval_handoff/prompts" WT="$WT" \
  python "$DRIVER" "$PROFILE" "$SUFFIX" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo
echo "val harness exit=$rc   log=$LOG"
exit $rc
