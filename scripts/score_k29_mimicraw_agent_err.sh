#!/bin/bash
# 我们的 agent（4.7pct mimic-raw-error val harness 冻结的 pipeline）在
# **4.7pct mimic-raw-error 版 K29 test** 上的错误级指标。
#
# v29 口径：recover_clean_key=true + exclude_rows_not_in_gold=true
#   —— raw 比 gold 多一截时，agent 正确删掉的行里的注入错误不能记成漏检。
#
# 关键：数据集是 4.7pct mimic-raw-error 版 ⇒ test/modification_log.csv 里
#   多了 873 条 mimic_native_* 条目，agent 对这些格的修复会被单独记成
#   b_clean / b_wrong / b_dirty，而不是 natural_noise。
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCORE_ONLY="${SCORE_ONLY:-$WT/score_only}"

# 档位：TIER=47（默认，4.7pct）/ TIER=16（1.6pct）
# ⚠️ 两个档位的 ds-key 必须不同 —— 评分缓存（agent 输出 + reference 的 sqlite）
#    是按 ds-key 命名空间存的。
TIER="${TIER:-47}"
case "$TIER" in
  47)
    FULLBASE="K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror"
    EXPPREFIX="k29_mimicraw_4p7"; DSKEY="k29_mimicraw_47_agent"
    OUTBASE="k29_mimicraw_47_agent_error_metrics_v29.json"
    ;;
  16)
    FULLBASE="K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"
    EXPPREFIX="k29_mimicraw_1p6"; DSKEY="k29_mimicraw_16_agent"
    OUTBASE="k29_mimicraw_16_agent_error_metrics_v29.json"
    ;;
  *) echo "TIER 只能是 47 或 16" >&2; exit 2 ;;
esac

DS="${DS:-$PROJ/datasets/注入错误数据集/k29/$FULLBASE}"
RESULT="${RESULT:-$WT/k29_agent_run/experiments/${EXPPREFIX}_test_seed666_v1/host/test_result_package}"
OUT="${OUT:-$PROJ/runs/$OUTBASE}"
LOG="${LOG:-/Users/mac/.claude/jobs/448aad57/tmp/err_${DSKEY}.log}"

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
