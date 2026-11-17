#!/bin/bash
# 我们的 agent（icu5000 mimic-raw-error val harness 冻结的 pipeline）在
# **icu5000 mimic-raw-error 版 test** 上的错误级指标。
#
# v29 口径：recover_clean_key=true + exclude_rows_not_in_gold=true
#   —— raw 比 gold 多一截时，agent 正确删掉的行里的注入错误不能记成漏检。
#
# 关键：数据集是 mimic-raw-error 版 ⇒ test/modification_log.csv 里
#   多了 mimic_native_* 条目（1.6pct 2,450 条 / 4.7pct 2,174 条），
#   agent 对这些格的修复会被单独记成 b_clean / b_wrong / b_dirty，
#   而不是 natural_noise。
#
# ⚠️ 两个档位的 ds-key 必须不同 —— 评分缓存（agent 输出 + reference 的 sqlite）
#    是按 ds-key 命名空间存的。
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCORE_ONLY="${SCORE_ONLY:-$WT/score_only}"

TIER="${TIER:-16}"
case "$TIER" in
  16)
    FULLBASE="icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror"
    EXPPREFIX="icu5000_mimicraw_1p6"; DSKEY="icu5000_mimicraw_16_agent"
    OUTBASE="icu5000_mimicraw_16_agent_error_metrics_v29.json"
    ;;
  47)
    FULLBASE="icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror"
    EXPPREFIX="icu5000_mimicraw_4p7"; DSKEY="icu5000_mimicraw_47_agent"
    OUTBASE="icu5000_mimicraw_47_agent_error_metrics_v29.json"
    ;;
  *) echo "TIER 只能是 16 或 47" >&2; exit 2 ;;
esac

DS="${DS:-$PROJ/datasets/注入错误数据集/icu5000/$FULLBASE}"
PKG="${PKG:-$WT/icu5000_agent_run/experiments/${EXPPREFIX}_test_seed666_v1/host/test_result_package}"
# ⚠️ agent 可能把结果放在**子目录**里（submission.json 的 result_root）。
#    ⇒ 统一按 submission.json 解析，不要假定 CSV 平铺在包根下。
RESULT="${RESULT:-$PKG}"
if [ -f "$PKG/submission.json" ]; then
  RR=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('result_root',''))" "$PKG/submission.json" 2>/dev/null || true)
  [ -n "$RR" ] && [ -d "$PKG/$RR" ] && RESULT="$PKG/$RR"
fi
OUT="${OUT:-$PROJ/runs/$OUTBASE}"
LOG="${LOG:-$WT/icu5000_agent_run/logs/err_${DSKEY}.log}"

if [ ! -d "$RESULT" ]; then
  echo "!! 找不到结果包：$RESULT" >&2
  exit 3
fi
echo "dataset = $DS"
echo "result  = $RESULT"
echo "ds-key  = $DSKEY"
echo "out     = $OUT"
echo "log     = $LOG"

bash "$SCORE_ONLY/run_error_metrics.sh" "$DS" "$RESULT" "$DSKEY" "$OUT" "$LOG"
