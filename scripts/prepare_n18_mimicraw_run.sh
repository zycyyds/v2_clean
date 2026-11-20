#!/bin/bash
# 准备 N18 mimicraw（MIMIC 原生错误增强版）的一次完整评测：
#   ① dataset-manifest（v2_rawerror_n18_final_report.json：注入层取自源数据集、
#      原生层从产物实测；cell_level_error_rate 统一成百分数）
#   ② agent 视图（train/validation 的 raw+reference，剥掉答案文件，无 test）
#   ③ prompt v2（基线 v1 + T1 哨兵码节 +「清单不是封闭的」+ itemid 别名节
#      + 原生层规模 + 重算的量级三行 + 硬区间 itemid 对照表 + train 例子）
#   ④ prompt 同步进数据集目录（数据集自带 prompt，避免与实际跑的版本漂移）
#   ⑤ 独立校验（从产物重算，不信构建器的自检）
#
# 用法: bash prepare_n18_mimicraw_run.sh <档位>   档位 ∈ {16, 47}
set -euo pipefail

WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
DSROOT="$PROJ/datasets/注入错误数据集/n18"

case "${1:-16}" in
  16) TAG="1.6pct"; SRC="N18_ckd_ED_v2_rawerror_test1.6pct_490M"
      BASEV="v1"; PROMPTV="v4" ;;
  47) TAG="4.7pct"; SRC="N18_ckd_ED_v2_rawerror_test4.7pct_490M"
      BASEV="v1"; PROMPTV="v4" ;;
  *) echo "档位只能是 16 或 47" >&2; exit 2 ;;
esac

FULL="$DSROOT/N18_ckd_ED_v2_rawerror_mimicraw_test${TAG}_490M"
# ⚠️ 视图放在 **datasets/val_views/** 这个「密封位置」（与 i48/icu5000 一致），
#    **不放** $DSROOT 下面 —— 那里的父目录里躺着完整数据集（含 test/reference
#    = 答案、modification_log.csv = 答案键），agent 枚举父目录就可能看到。
VIEW="$PROJ/datasets/val_views/n18_mimicraw_$1"
PROMPTDIR="$WT/eval_handoff/prompts"
BASE="$PROMPTDIR/n18_${TAG}_val_prompt_${BASEV}.md"
PROMPT="$PROMPTDIR/n18_${TAG}_mimicraw_val_prompt_${PROMPTV}.md"

[ -d "$FULL" ] || { echo "FATAL: 缺数据集 $FULL" >&2; exit 5; }
[ -f "$BASE" ] || { echo "FATAL: 缺基础 prompt $BASE" >&2; exit 5; }

echo "=== ① dataset-manifest"
python3 "$WT/reproduction/make_n18_manifest.py" --preset "$1"
ls -la "$FULL"/*.json

echo
echo "=== ② agent 视图"
python3 "$WT/reproduction/make_agent_view.py" --src "$FULL" --dst "$VIEW"

echo
echo "=== ③ prompt ${PROMPTV}"
python3 "$WT/reproduction/make_n18_mimicraw_prompt.py" \
  --dataset "$FULL" --src-dataset "$DSROOT/$SRC" \
  --base-prompt "$BASE" --out "$PROMPT"
echo "prompt: $PROMPT  ($(wc -c < "$PROMPT") bytes)"

echo
echo "=== ④ prompt 同步进数据集目录"
mkdir -p "$FULL/prompt"
cp -f "$PROMPT" "$FULL/prompt/"
ls -la "$FULL/prompt/"

echo
echo "=== ⑤ 独立校验（从产物重算）"
python3 "$WT/reproduction/verify_n18_mimic_raw_error.py" \
  --src "$DSROOT/$SRC" --dst "$FULL" --splits train,validation,test \
  --out "$FULL/verify_native_report.json" || {
  echo "!! 独立校验有失败项（详见上面的输出与 $FULL/verify_native_report.json）" >&2
  exit 1
}

echo
echo "=== 产物清单"
echo "  完整数据集 : $FULL"
echo "  agent 视图 : $VIEW"
echo "  prompt     : $PROMPT"
echo "  manifest   : $FULL/v2_rawerror_n18_final_report.json"
