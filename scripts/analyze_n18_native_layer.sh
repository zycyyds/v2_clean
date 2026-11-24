#!/bin/bash
# N18 mimicraw 的**原生层深挖**（一条命令复现台账 ㉖ 里那节分析）。
#
# 回答的问题：「原生层 99.4% / 98.6%」这个数字**低估**了多少？
# 拆开：① raw 里这一行的 key 对不对得上  ② agent 定位到了没有  ③ 值修对了没有
#
# 只输出聚合统计（不输出病人级标识与逐格医学值）。
#
# 用法:
#   bash scripts/analyze_n18_native_layer.sh [16|47]      # 默认两档都跑
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN="${RUN:-$WT/n18_agent_run}"

TIERS="${*:-16 47}"
for TIER in $TIERS; do
  case "$TIER" in
    16) FULLBASE="N18_ckd_ED_v2_rawerror_mimicraw_test1.6pct_490M"
         PKG="$RUN/experiments/n18_mimicraw_1p6_test_seed666_v1/host/test_result_package" ;;
    47) FULLBASE="N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M"
         PKG="$RUN/experiments/n18_mimicraw_4p7_test_seed666_v1/host/test_result_package" ;;
    *) echo "TIER 只能是 16 或 47" >&2; exit 2 ;;
  esac
  DS="$PROJ/datasets/注入错误数据集/n18/$FULLBASE"

  echo "################################################################"
  echo "# N18 ${TIER}pct —— 原生层深挖"
  echo "################################################################"
  echo
  echo "--- ① key 命中交叉表（raw 对不对得上 × agent 定位到没有 × 值修对没有）---"
  python3 "$WT/reproduction/n18_native_key_vs_reference.py" \
    --dataset "$DS" --result "$PKG" --split test
  echo
  echo "--- ② 没修对的格：还能不能修（缺行 / 值错）---"
  python3 "$WT/reproduction/n18_native_missed_cells.py" \
    --dataset "$DS" --result "$PKG" --split test
  echo
  echo "--- ③ 缺行的格在 raw 里的形态（别名形态 / charttime 被挪 / 整行没有）---"
  python3 "$WT/reproduction/n18_native_missing_rows.py" \
    --dataset "$DS" --result "$PKG" --split test
  echo
  echo "--- ④ itemid 别名还原规则的正确率（prompt 那条「2+a[3]+a[-4:]」）---"
  python3 "$WT/reproduction/n18_alias_rule_audit.py" \
    --dataset "$DS" --result "$PKG" --split test
  echo
done

echo "=== 注入器对 charttime 做了什么（与档无关，查一遍）==="
python3 "$WT/reproduction/n18_charttime_injection_profile.py" \
  --dataset "$PROJ/datasets/注入错误数据集/n18/N18_ckd_ED_v2_rawerror_mimicraw_test4.7pct_490M" \
  --split test
