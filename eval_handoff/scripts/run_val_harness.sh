#!/bin/bash
# val 阶段启动脚本（icu5000 4.7pct 范例）。
#
# 换成别的数据集时改这四个变量：
#   DS        —— 密封 val 视图目录（父目录只能有密封视图！）
#   FULL      —— 完整数据集目录（host 侧只用来读 report json，不进 agent 的 sandbox）
#   EXP       —— 实验目录（必须不存在，harness 要求 new+empty）
#   PROMPT    —— val prompt
#
# 用法: bash run_val_harness.sh
set -euo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
RUN="${RUN:-$PROJ/k29_agent_run}"
DS="${DS:-$PROJ/datasets/val_views/icu5000_more1}"
FULL="${FULL:-$PROJ/datasets/注入错误数据集/icu_5000_v2_rawerror_test4.7pct_614M}"
EXP="${EXP:-$RUN/experiments/icu5000_4p7_val_seed666_v1}"
PROMPT="${PROMPT:-$RUN/prompts/icu5000_4.7pct_val_prompt_v1.md}"

mkdir -p "$(dirname "$EXP")"

if [ -e "$EXP" ]; then
  echo "FATAL: 实验目录已存在（harness 要求 new+empty）: $EXP" >&2
  exit 2
fi

# 泄漏自检 —— 有 modification_log 就地失败，不要浪费一轮跑批
if find "$DS" -name 'modification_log*' | grep -q .; then
  echo "FATAL: 密封视图里有 modification_log（=答案），必须重建" >&2
  exit 4
fi
if [ -e "$DS/test" ]; then
  echo "WARN: 密封视图里有 test/ —— 确认 agent sandbox 没有放行它" >&2
fi

echo "=== experiment : $EXP"
echo "=== dataset    : $DS"
echo "=== prompt     : $PROMPT"
echo

# cd /tmp 是必须的：worktree 里也有一个叫 agent 的包，会在 sys.path 上遮蔽真正的 agent。
# PYTHONPATH 必须通过 conda run env 传，因为 conda run 不转发调用方的 PYTHONPATH。
cd /tmp
exec conda run -n py311agent2 --no-capture-output env PYTHONPATH="$PROJ" \
  python -m agent.pi_harness_cli \
    --experiment-dir "$EXP" \
    --train-raw "$DS/train/raw" \
    --train-reference "$DS/train/reference" \
    --validation-raw "$DS/validation/raw" \
    --validation-gold "$DS/validation/reference" \
    --evaluation-manifest "$PROJ/evaluation_manifests/icu_5000_v1.json" \
    --dataset-manifest "$FULL/v2_rawerror_icu5000_final_report.json" \
    --prompt-file "$PROMPT" \
    --max-rounds "${MAX_ROUNDS:-20}" \
    --patience "${PATIENCE:-3}" \
    --max-iters 10000
