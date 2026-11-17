#!/bin/bash
# icu5000 mimic-raw-error 的**四口径**错误级指标 + 原生格重判。
#
# 四口径 = 2 个正交开关 × 4 组合：
#   ① 值比较：numeric（数值感知，默认）/ string（--string-compare，纯字符串比对）
#   ② 旧开关：recover_clean_key + exclude_rows_not_in_gold（默认，新口径）
#            / --no-recover-clean-key --keep-rows-not-in-gold（旧开关）
#   组合：v29（numeric+新开关） / str（string+新开关） / oldsw（numeric+旧开关） / old（string+旧开关）
#
# 另加三个「gold 判据」口径（判据改用 reference 的值 ⇒ 跨档绝对可比）：
#   goldv29 / goldstr / goldold（= B 完全老口径 + gold 判据）
#
# ⚠️ icu5000 两档的**原生层格数不同**（1.6pct 2,450 / 4.7pct 2,174）——
#    候选层相同（两档 reference 是同一份），但「itemid 也被注入器改过」的格
#    被划归注入层（141 vs 417）⇒ 跨档比原生修复率要用**交集 2,059 格**。
#
# 用法：
#   bash scripts/score_icu5000_mimicraw_four_calibers.sh 16 [suffix]
#   bash scripts/score_icu5000_mimicraw_four_calibers.sh 47 [suffix]
# 不给 suffix 会自动加时间戳（**不会**覆盖历史存档）。
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TIER="${1:-16}"
SUFFIX="${2:-}"
# ⚠️ 不带后缀会**覆盖历史存档** ⇒ 没给后缀就自动加时间戳，绝不静默覆盖。
if [ -z "$SUFFIX" ]; then
  SUFFIX="run$(date +%Y%m%d_%H%M%S)"
  echo "提示：未指定后缀，自动使用 '$SUFFIX'（避免覆盖历史存档）"
fi
case "$TIER" in
  16)
    FULLBASE="icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror"
    EXPPREFIX="icu5000_mimicraw_1p6"; DSKEY="icu5000_mimicraw_16_agent"
    NAT_N=2450
    ;;
  47)
    FULLBASE="icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror"
    EXPPREFIX="icu5000_mimicraw_4p7"; DSKEY="icu5000_mimicraw_47_agent"
    NAT_N=2174
    ;;
  *) echo "TIER 只能是 16 或 47" >&2; exit 2 ;;
esac
DS="${DS:-$PROJ/datasets/注入错误数据集/icu5000/$FULLBASE}"
RUNS="${RUNS:-$PROJ/runs}"
PKG="${PKG:-$WT/icu5000_agent_run/experiments/${EXPPREFIX}_test_seed666_v1/host/test_result_package}"

# agent 可能把结果放在**子目录**里（submission.json 的 result_root）。
#   ⇒ 统一按 submission.json 解析，不假定 CSV 平铺在包根下。
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
echo "        ds-key=$DSKEY  suffix='$SUFFIX'  native_cells=$NAT_N"

SFX="_${SUFFIX}"           # 后缀（可为空）
SFX="${SFX//__/}"          # 折叠空后缀
OUTBASE="${DSKEY}_error_metrics"

run_cal() {   # $1=口径名  $2..=额外参数
  local cal="$1"; shift
  local out="$RUNS/${OUTBASE}_${cal}${SFX}.json"
  local log="$WT/icu5000_agent_run/logs/err_${DSKEY}_${cal}${SFX}.log"
  mkdir -p "$(dirname "$log")"
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

# ⚠️ **必须先建缓存**：compute_error_metrics.py 依赖 build_eval_cache.py 建好的
#    sqlite 索引（agent 输出 + gold 各一份）；缓存不存在时它会把**每个文件都 SKIP**
#    ⇒ 指标全 0（看起来像「agent 一格没修」，其实是没读到东西）。
#    （K29 的同名脚本当初能跑，是因为前面先跑过走 run_error_metrics.sh 的那条路，
#      缓存已存在 —— 这里不依赖那个顺序。）
echo "--- 建缓存（agent 输出 + gold 的 sqlite 索引）---"
( cd "$WT/score_only" && python3 -u error_metrics/build_eval_cache.py \
    --result-dir "$RESULT" --ref-dir "$DS/test/reference" --ds-key "$DSKEY" ) \
  > "$WT/icu5000_agent_run/logs/cache_${DSKEY}${SFX}.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "!! build_eval_cache 失败 (rc=$rc)" >&2
  tail -20 "$WT/icu5000_agent_run/logs/cache_${DSKEY}${SFX}.log" >&2
  exit $rc
fi
tail -3 "$WT/icu5000_agent_run/logs/cache_${DSKEY}${SFX}.log"

run_cal v29
run_cal str    --string-compare
run_cal oldsw  --no-recover-clean-key --keep-rows-not-in-gold
run_cal old    --no-recover-clean-key --keep-rows-not-in-gold --string-compare

# ---- 「gold 判据」口径：判据改用 reference 的值 ⇒ 跨档绝对可比 ----
run_cal goldv29 --clean-from-gold
run_cal goldstr --clean-from-gold --string-compare
run_cal goldold --clean-from-gold --no-recover-clean-key --keep-rows-not-in-gold --string-compare

echo "--- 原生格重判（${NAT_N} 格）---"
for m in numeric string; do
  flag=""; [ "$m" = "string" ] && flag="--string-compare"
  out="$RUNS/${DSKEY}_native_cells${SFX}_${m}.json"
  python3 "$WT/reproduction/eval_native_cells.py" --dataset "$DS" --result "$RESULT" \
      --split test --out "$out" $flag 2>&1 | sed 's/^/    /'
done
echo "=== 四口径完成 ==="
