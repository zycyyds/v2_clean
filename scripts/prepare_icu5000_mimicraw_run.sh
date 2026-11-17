#!/bin/bash
# 准备 icu5000 mimicrawerror 的一次完整评测：
#   ① dataset-manifest（v2_rawerror_icu5000_final_report.json）
#   ② agent 视图（train/validation 的 raw+reference，剥掉答案文件，无 test）
#   ③ 实测统计（icu5000_native_stats.py ⇒ prompt 里的数字全部来自本档实测）
#   ④ prompt（基础 prompt + 「新增错误类：MIMIC 自身的原生错误」+「itemid 别名注入」两节）
#   ⑤ prompt 同步进数据集目录（数据集自带 prompt，避免与实际跑的版本漂移）
#
# 用法: bash prepare_icu5000_mimicraw_run.sh <档位>   档位 ∈ {47, 16}
set -euo pipefail

WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
DSROOT="$PROJ/datasets/注入错误数据集/icu5000"

case "${1:-47}" in
  47) TAG="4.7pct"; SRC="icu_5000_v2_rawerror_test4.7pct_614M" ;;
  16) TAG="1.6pct"; SRC="icu_5000_v2_rawerror_test1.6pct_614M" ;;
  *) echo "档位只能是 47 或 16" >&2; exit 2 ;;
esac

FULL="$DSROOT/${SRC}_mimicrawerror"
# ⚠️ 视图放在 **datasets/val_views/** 这个「密封位置」（与 icu5000 原版 less1/more1
#    一致），**不放** $DSROOT 下面 —— 那里的父目录里躺着完整数据集（含 test/reference
#    = 答案、modification_log.csv = 答案键），agent 枚举父目录就可能看到。
VIEW="$PROJ/datasets/val_views/icu5000_mimicraw_$1"
PROMPTDIR="$WT/eval_handoff/prompts"
BASE="$PROMPTDIR/icu5000_${TAG}_val_prompt_v1.md"
PROMPT="$PROMPTDIR/icu5000_${TAG}_mimicraw_val_prompt_v1.md"
STATS="$WT/reproduction/icu5000_native_stats_$1.json"
VALSTATS="$WT/reproduction/icu5000_value_reliability_$1.json"

[ -d "$FULL" ] || { echo "FATAL: 缺数据集 $FULL" >&2; exit 5; }
[ -f "$BASE" ] || { echo "FATAL: 缺基础 prompt $BASE" >&2; exit 5; }

echo "=== ① dataset-manifest"
# 两档都从权威来源重新生成（字段结构与单位统一：cell_level_error_rate 用百分数）
python3 "$WT/reproduction/make_icu5000_manifest.py" --preset "$1"
ls -la "$FULL"/*.json

echo
echo "=== ② agent 视图"
python3 "$WT/reproduction/make_agent_view.py" --src "$FULL" --dst "$VIEW"

echo
echo "=== ③ 实测统计（prompt 里引用的每个数字都来自这里）"
python3 "$WT/reproduction/icu5000_native_stats.py" --dataset "$FULL" --out "$STATS"
# value 文本列的可靠性分类（icu5000 有大量文本型 chart 行 ⇒ 「直接抄 value」只适用约三成）
python3 "$WT/reproduction/icu5000_value_reliability.py" \
  --dataset "$FULL" --sample 20000 --out "$VALSTATS"

echo
echo "=== ④ prompt"
python3 "$WT/reproduction/make_icu5000_mimicraw_prompt.py" \
  --dataset "$FULL" --base-prompt "$BASE" --stats "$STATS" \
  --value-stats "$VALSTATS" --out "$PROMPT"
echo "prompt: $PROMPT  ($(wc -c < "$PROMPT") bytes)"

echo
echo "=== ⑤ prompt 同步进数据集目录"
mkdir -p "$FULL/prompt"
cp -f "$PROMPT" "$FULL/prompt/"
ls -la "$FULL/prompt/"

echo
echo "=== 产物清单"
echo "  完整数据集 : $FULL"
echo "  agent 视图 : $VIEW"
echo "  prompt     : $PROMPT"
echo "  统计       : $STATS"
echo "  value 分类 : $VALSTATS"
