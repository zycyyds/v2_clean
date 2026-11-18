#!/bin/bash
# icu5000 mimicraw 两档的**原生错误格跨档对比**（交集口径）。
#
# 为什么必须用交集：两档的原生错误**候选层**相同（reference 是同一份干净 MIMIC，
# 注入器只改 raw），但**登记层格数不同**（1.6pct 2,450 / 4.7pct 2,174）——
# 差别来自「itemid 也被注入器改过」的格被划归注入层（141 vs 417）。
# ⇒ 直接比原生修复率是分母不同的比法；严格可比只看两档都登记的交集格。
#
# 前置：两档的 test 结果包都已存在（即两档 test harness 都跑完了）。
# 用法：
#   bash scripts/score_icu5000_mimicraw_cross_tier.sh [suffix]
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SUFFIX="${1:-}"
if [ -z "$SUFFIX" ]; then
  SUFFIX="run$(date +%Y%m%d_%H%M%S)"
  echo "提示：未指定后缀，自动使用 '$SUFFIX'（避免覆盖历史存档）"
fi

DS16="$PROJ/datasets/注入错误数据集/icu5000/icu_5000_v2_rawerror_test1.6pct_614M_mimicrawerror"
DS47="$PROJ/datasets/注入错误数据集/icu5000/icu_5000_v2_rawerror_test4.7pct_614M_mimicrawerror"
RES16="${RES16:-$WT/icu5000_agent_run/experiments/icu5000_mimicraw_1p6_test_seed666_v1/host/test_result_package}"
RES47="${RES47:-$WT/icu5000_agent_run/experiments/icu5000_mimicraw_4p7_test_seed666_v1/host/test_result_package}"

for d in "$DS16" "$DS47" "$RES16" "$RES47"; do
  if [ ! -d "$d" ]; then
    echo "FATAL: 缺目录: $d" >&2
    exit 5
  fi
done

OUT="$PROJ/runs/icu5000_mimicraw_native_crosstier_${SUFFIX}.json"

python3 "$WT/reproduction/icu5000_native_cross_tier.py" \
    --a-ds "$DS16" --a-result "$RES16" \
    --b-ds "$DS47" --b-result "$RES47" \
    --split test --out "$OUT"
rc=$?
echo
echo "跨档对比完成 rc=$rc   存档=$OUT"
exit $rc
