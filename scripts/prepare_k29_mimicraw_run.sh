#!/bin/bash
# 准备 k29 mimicrawerror 的一次完整评测：
#   ① 把 dataset-manifest（v2_rawerror_k29_rebuild_final_report.json）复制进新数据集
#   ② 生成 agent 视图（train/validation 的 raw+reference，剥掉答案文件，无 test）
#   ③ 生成 prompt（含「新增错误类：MIMIC 自身的原生错误」一节）
#
# 用法: bash prepare_k29_mimicraw_run.sh <档位>   档位 ∈ {47, 16}
set -euo pipefail

WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
DSROOT="$PROJ/datasets/注入错误数据集"

case "${1:-47}" in
  47) BASE="K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror"; TAG="4.7pct" ;;
  16) BASE="K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"; TAG="1.6pct" ;;
  *) echo "档位只能是 47 或 16" >&2; exit 2 ;;
esac

FULL="$DSROOT/k29/$BASE"
VIEW="$DSROOT/k29_mimicraw_$1"
PROMPTDIR="$WT/k29_agent_run/prompts"
PROMPT="$PROMPTDIR/k29_${TAG}_mimicraw_val_prompt_v1.md"

[ -d "$FULL" ] || { echo "FATAL: 缺数据集 $FULL" >&2; exit 5; }

echo "=== ① dataset-manifest"
cp -f "$FULL/v2_rawerror_k29_rebuild_final_report.json" "$FULL/" 2>/dev/null || \
  cp -f "$DSROOT/k29/K29_v2_rawerror_test4.7pct_14.1M/v2_rawerror_k29_rebuild_final_report.json" "$FULL/"
ls -la "$FULL"/*.json

echo
echo "=== ② agent 视图"
python3 "$WT/reproduction/make_agent_view.py" --src "$FULL" --dst "$VIEW"

echo
echo "=== ③ prompt"
mkdir -p "$PROMPTDIR"
python3 "$WT/reproduction/make_k29_mimicraw_prompt.py" --dataset "$FULL" --out "$PROMPT"
echo "prompt: $PROMPT  ($(wc -c < "$PROMPT") bytes)"

echo
echo "=== ④ prompt 同步进数据集目录（数据集自带 prompt，避免与实际跑的版本漂移）"
mkdir -p "$FULL/prompt"
cp -f "$PROMPT" "$FULL/prompt/"
ls -la "$FULL/prompt/"

echo
echo "=== 产物清单"
echo "  完整数据集 : $FULL"
echo "  agent 视图 : $VIEW"
echo "  prompt     : $PROMPT"
echo
echo "下一步: bash $WT/scripts/run_val_harness_k29_mimicraw.sh v1"
