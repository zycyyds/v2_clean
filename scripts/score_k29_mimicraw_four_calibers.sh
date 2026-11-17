#!/bin/bash
# K29 mimic-raw-error 的**四口径**错误级指标 + 原生格重判。
#
# 四口径 = 2 个正交开关 × 4 组合：
#   ① 值比较：numeric（数值感知，默认）/ string（--string-compare，纯字符串比对）
#   ② 旧开关：recover_clean_key + exclude_rows_not_in_gold（默认，新口径）
#            / --no-recover-clean-key --keep-rows-not-in-gold（旧开关）
#   组合：v29（numeric+新开关） / str（string+新开关） / oldsw（numeric+旧开关） / old（string+旧开关）
#
# ⚠️ 纯字符串口径的**绝对值**跨档仍不可比（两档的「写法税」机制已同源，
#    但注入错误的列构成不同 ⇒ 税的绝对量不同）。见报告 §6.10。
#
# 用法：
#   bash scripts/score_k29_mimicraw_four_calibers.sh 16 [suffix]
#   bash scripts/score_k29_mimicraw_four_calibers.sh 47 [suffix]
# 不给 suffix 会自动加时间戳（**不会**覆盖历史存档）。
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TIER="${1:-16}"
SUFFIX="${2:-}"
# ⚠️ 不带后缀会**覆盖历史存档**（如 k29_mimicraw_16_agent_error_metrics_v29.json
#    存的是「改 log 之前」的数字）⇒ 没给后缀就自动加时间戳，绝不静默覆盖。
if [ -z "$SUFFIX" ]; then
  SUFFIX="run$(date +%Y%m%d_%H%M%S)"
  echo "提示：未指定后缀，自动使用 '$SUFFIX'（避免覆盖历史存档）"
fi
case "$TIER" in
  47)
    FULLBASE="K29_v2_rawerror_test4.7pct_14.1M_mimicrawerror"
    EXPPREFIX="k29_mimicraw_4p7"; DSKEY="k29_mimicraw_47_agent"
    ;;
  16)
    FULLBASE="K29_v2_rawerror_test1.6pct_4.2M_mimicrawerror"
    EXPPREFIX="k29_mimicraw_1p6"; DSKEY="k29_mimicraw_16_agent"
    ;;
  *) echo "TIER 只能是 47 或 16" >&2; exit 2 ;;
esac
DS="${DS:-$PROJ/datasets/注入错误数据集/k29/$FULLBASE}"
RUNS="${RUNS:-$PROJ/runs}"
PKG="${PKG:-$WT/k29_agent_run/experiments/${EXPPREFIX}_test_seed666_v1/host/test_result_package}"

# agent 可能把结果放在**子目录**里（submission.json 的 result_root）：
#   1.6pct 档是 result_root="result"；4.7pct 档是平铺。⇒ 统一按 submission.json 解析。
RESULT="${RESULT:-$PKG}"
if [ -f "$PKG/submission.json" ]; then
  RR=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('result_root',''))" "$PKG/submission.json" 2>/dev/null || true)
  [ -n "$RR" ] && [ -d "$PKG/$RR" ] && RESULT="$PKG/$RR"
fi
if [ ! -d "$RESULT" ]; then
  echo "!! 找不到结果包：$RESULT" >&2; exit 3
fi
echo "tier=$TIER  dataset=$DS"
echo "        result=$RESULT"
echo "        ds-key=$DSKEY  suffix='$SUFFIX'"

SFX="_${SUFFIX}"           # 后缀（可为空）
SFX="${SFX//__/}"          # 折叠空后缀
OUTBASE="${DSKEY}_error_metrics"

run_cal() {   # $1=口径名  $2..=额外参数
  local cal="$1"; shift
  local out="$RUNS/${OUTBASE}_${cal}${SFX}.json"
  local log="/Users/mac/.claude/jobs/448aad57/tmp/err_${DSKEY}_${cal}${SFX}.log"
  echo "--- 口径 $cal ---"
  ( cd "$WT/score_only" && python3 -u error_metrics/compute_error_metrics.py \
      --dataset-dir "$DS" --ds-key "$DSKEY" --split test --out "$out" "$@" ) \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!! 口径 $cal 失败 (rc=$rc)，日志：$log" >&2
    tail -20 "$log" >&2
    return $rc
  fi
  python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
b = d["buckets"]
print("  Det F1=%.6f (P %.6f/R %.6f)  Rep F1=%.6f (P %.6f/R %.6f)"
      % (d["detection"]["F1"], d["detection"]["P"], d["detection"]["R"],
         d["repair"]["F1"], d["repair"]["P"], d["repair"]["R"]))
print("  b_clean=%s b_dirty=%s b_wrong=%s b_missing=%s excluded=%s fp_det=%s natural_noise=%s"
      % (f"{b.get('b_clean',0):,}", f"{b.get('b_dirty',0):,}", f"{b.get('b_wrong',0):,}",
         f"{b.get('b_missing',0):,}", f"{d.get('excluded_cells',0):,}",
         f"{b.get('fp_det',0):,}", f"{b.get('natural_noise',0):,}"))
PY
  return 0
}

run_cal v29
run_cal str    --string-compare
run_cal oldsw  --no-recover-clean-key --keep-rows-not-in-gold
run_cal old    --no-recover-clean-key --keep-rows-not-in-gold --string-compare

echo "--- 原生格重判（873 格）---"
for m in numeric string; do
  flag=""; [ "$m" = "string" ] && flag="--string-compare"
  out="$RUNS/${DSKEY}_native_cells${SFX}_${m}.json"
  python3 "$WT/reproduction/eval_native_cells.py" --dataset "$DS" --result "$RESULT" \
      --split test --out "$out" $flag 2>&1 | sed 's/^/    /'
done
echo "=== 四口径完成 ==="
