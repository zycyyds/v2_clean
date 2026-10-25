#!/bin/bash
# 错误级指标（cell 级明细）。
#
# 用法:
#   bash score_error_metrics.sh <完整数据集目录> <结果包目录> <ds_key> <输出json>
#
# 例:
#   bash score_error_metrics.sh \
#     "$PROJ/datasets/注入错误数据集/icu_5000_v2_rawerror_test4.7pct_614M" \
#     "$PROJ/k29_agent_run/experiments/icu5000_4p7_test_seed666_v1/host/test_result_package" \
#     icu5000_more_agent \
#     "$PROJ/runs/icu5000_more_agent_error_metrics.json"
#
# 内部两步：
#   1. build_eval_cache.py      —— 结果包 + gold → 可逐 cell 比对的缓存
#   2. compute_error_metrics.py —— 逐 cell 分类记账，出 P/R/F1 + 分桶
#
# 耗时：4.7pct 约 1.5 小时（扫 3,200 万条注入日志 + 逐 cell 比对）。纯 CPU，
#      可放后台跑，不阻塞 Raha 总分的结论。
#
# ⚠️ 口径：默认是新口径（recover_clean_key=true）。要旧口径加 --no-recover-clean-key
#    —— 但旧口径的 b_missing 会爆炸（key 被破坏的行整行记 b_missing），
#    只在跟更早的历史 run 横比时才用。
set -uo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
C="conda run -n py311agent2 --no-capture-output"

if [ $# -lt 4 ]; then
  echo "用法: $0 <完整数据集目录> <结果包目录> <ds_key> <输出json>" >&2
  exit 2
fi
DS="$1"
RESULT="$2"
DS_KEY="$3"
OUT="$4"
GOLD="$DS/test/reference"
LOG="${LOG:-/tmp/err_metrics_${DS_KEY}.log}"

if [ ! -d "$RESULT" ]; then
  echo "!! 找不到结果包：$RESULT" >&2
  exit 3
fi
if [ ! -d "$GOLD" ]; then
  echo "!! 找不到 gold：$GOLD" >&2
  exit 3
fi
echo "result = $RESULT"
echo "gold   = $GOLD"
echo "log    = $LOG"

# reproduction/ 下的两个脚本不在 git 里（仓库只跟踪 125 个文件），
# 所以从交付包内的 reference/ 找，找不到再退回项目根
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO=""
for cand in "$SCRIPT_DIR/../reference" "$PROJ/reproduction" "$SCRIPT_DIR/../reproduction"; do
  if [ -f "$cand/compute_error_metrics.py" ]; then REPRO="$cand"; break; fi
done
if [ -z "$REPRO" ]; then
  echo "!! 找不到 compute_error_metrics.py" >&2
  exit 4
fi
echo "repro  = $REPRO"

$C python -u "$REPRO/build_eval_cache.py" \
  --result-dir "$RESULT" --ref-dir "$GOLD" --ds-key "$DS_KEY" >> "$LOG" 2>&1
echo "cache 建完" >> "$LOG"
$C python -u "$REPRO/compute_error_metrics.py" \
  --dataset-dir "$DS" --ds-key "$DS_KEY" --split test \
  --out "$OUT" >> "$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "=== ERR_METRICS_DONE ===" >> "$LOG"
  python3 -c "
import json
d = json.load(open('$OUT'))
print('recover_clean_key:', d.get('recover_clean_key'))
print('detection F1:', d['detection']['F1'])
print('repair    F1:', d['repair']['F1'])
"
fi
exit $rc
