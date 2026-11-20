#!/bin/bash
# N18 mimicraw 的**七口径**错误级指标 + 原生格重判。
#
# 七口径 = 2 个正交开关 × 4 组合 + 3 个 gold 判据口径：
#   ① 值比较：numeric（默认）/ string（--string-compare）
#   ② 旧开关：recover_clean_key + exclude_rows_not_in_gold（默认，新口径）
#            / --no-recover-clean-key --keep-rows-not-in-gold（旧开关）
#   ③ gold 判据：--clean-from-gold（判据改用 reference 的值 ⇒ 跨档绝对可比）
#   v29 / str / oldsw / old / goldv29 / goldstr / goldold
#
# ⚠️ **索引空间**：评分器已改成「数据探测」判定 zero/bisect（N18 的 reference 是
#    cohort 筛选后的输出表，行数判据对它不成立）。
# ⚠️ **原生格重判**：N18 与 i48 同为 cohort 型 ⇒ 用 eval_n18_native_cells.py
#    （按 5 段 key 定位，不能用 icu5000 那版按行号对齐的）。
#
# 用法:
#   bash scripts/score_n18_mimicraw_four_calibers.sh 16 [suffix]
#   bash scripts/score_n18_mimicraw_four_calibers.sh 47 [suffix]
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TIER="${1:-16}"
SUFFIX="${2:-}"
if [ -z "$SUFFIX" ]; then
  SUFFIX="run$(date +%Y%m%d_%H%M%S)"
  echo "提示：未指定后缀，自动使用 '$SUFFIX'（避免覆盖历史存档）"
fi
case "$TIER" in
  16) FULLBASE="N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
       EXPPREFIX="n18_mimicraw_1p6"; DSKEY="n18_mimicraw_16_agent" ;;
  47) FULLBASE="N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"
       EXPPREFIX="n18_mimicraw_4p7"; DSKEY="n18_mimicraw_47_agent" ;;
  *) echo "TIER 只能是 16 或 47" >&2; exit 2 ;;
esac
DS="${DS:-$PROJ/datasets/注入错误数据集/n18/$FULLBASE}"
RUNS="${RUNS:-$PROJ/runs}"
PKG="${PKG:-$WT/n18_agent_run/experiments/${EXPPREFIX}_test_seed666_v1/host/test_result_package}"

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

SFX="_${SUFFIX}"
SFX="${SFX//__/}"
OUTBASE="${DSKEY}_error_metrics"

run_cal() {
  local cal="$1"; shift
  local out="$RUNS/${OUTBASE}_${cal}${SFX}.json"
  local log="$WT/n18_agent_run/logs/err_${DSKEY}_${cal}${SFX}.log"
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
print("  b_clean=%s b_wrong=%s b_dirty=%s b_missing=%s excluded=%s fp_det=%s natural_noise=%s"
      % (f"{b.get('b_clean',0):,}", f"{b.get('b_wrong',0):,}", f"{b.get('b_dirty',0):,}",
         f"{b.get('b_missing',0):,}", f"{d.get('excluded_cells',0):,}",
         f"{b.get('fp_det',0):,}", f"{b.get('natural_noise',0):,}"))
print("  索引空间: %s" % d.get("offset_mode"))
PY
  return 0
}

# ⚠️ **必须先建缓存**：compute_error_metrics.py 依赖 build_eval_cache.py 建好的
#    sqlite 索引（agent 输出 + gold 各一份）；缓存不存在时它会把**每个文件都 SKIP**
#    ⇒ 指标全 0（看起来像「agent 一格没修」，其实是没读到东西）。
echo "--- 建缓存（agent 输出 + gold 的 sqlite 索引）---"
( cd "$WT/score_only" && python3 -u error_metrics/build_eval_cache.py \
    --result-dir "$RESULT" --ref-dir "$DS/test/reference" --ds-key "$DSKEY" ) \
  > "$WT/n18_agent_run/logs/cache_${DSKEY}${SFX}.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "!! build_eval_cache 失败 (rc=$rc)" >&2
  tail -20 "$WT/n18_agent_run/logs/cache_${DSKEY}${SFX}.log" >&2
  exit $rc
fi
tail -3 "$WT/n18_agent_run/logs/cache_${DSKEY}${SFX}.log"

run_cal v29
run_cal str    --string-compare
run_cal oldsw  --no-recover-clean-key --keep-rows-not-in-gold
run_cal old    --no-recover-clean-key --keep-rows-not-in-gold --string-compare
run_cal goldv29 --clean-from-gold
run_cal goldstr --clean-from-gold --string-compare
run_cal goldold --clean-from-gold --no-recover-clean-key --keep-rows-not-in-gold --string-compare

echo "--- 原生格重判 ---"
out="$RUNS/${DSKEY}_native_cells${SFX}_string.json"
python3 "$WT/reproduction/eval_n18_native_cells.py" --dataset "$DS" --result "$RESULT" \
    --split test --out "$out" --string-compare 2>&1 | sed 's/^/    /'
echo "=== 七口径完成 ==="
