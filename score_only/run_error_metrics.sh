#!/bin/bash
# 错误级指标（cell 级明细）—— 独立版，不依赖项目环境。
#
# 用法:
#   bash run_error_metrics.sh <完整数据集目录> <结果包目录> <ds_key> <输出json> [日志路径]
#
# 例:
#   bash run_error_metrics.sh \
#     "/path/to/dataset" \
#     "/path/to/result_root" \
#     my_run \
#     /tmp/err_metrics.json
#
# ⚠️ 口径：默认新口径（recover_clean_key=true）。旧口径在第二步加
#    --no-recover-clean-key —— b_missing 会爆炸，只在跟更早历史 run 横比时用。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 4 ]; then
  echo "用法: $0 <完整数据集目录> <结果包目录> <ds_key> <输出json> [日志路径]" >&2
  exit 2
fi
DS="$1"
RESULT="$2"
DS_KEY="$3"
OUT="$4"
LOG="${5:-/tmp/err_metrics_${DS_KEY}.log}"
GOLD="$DS/test/reference"

for d in "$DS" "$RESULT" "$GOLD"; do
  if [ ! -d "$d" ]; then
    echo "!! 目录不存在: $d" >&2
    exit 3
  fi
done
echo "dataset = $DS"
echo "result  = $RESULT"
echo "gold    = $GOLD"
echo "log     = $LOG"

cd "$HERE"
PY="python3"
if command -v conda >/dev/null 2>&1; then
  # pandas 可能不在系统 python 里，优先用项目环境
  if conda env list 2>/dev/null | grep -q "py311agent2"; then
    PY="conda run -n py311agent2 --no-capture-output python3"
  fi
fi

$PY -u error_metrics/build_eval_cache.py \
  --result-dir "$RESULT" --ref-dir "$GOLD" --ds-key "$DS_KEY" >> "$LOG" 2>&1
rc1=$?
if [ $rc1 -ne 0 ]; then
  echo "!! build_eval_cache 失败 (rc=$rc1)，看 $LOG" >&2
  exit $rc1
fi
echo "cache 建完" >> "$LOG"

$PY -u error_metrics/compute_error_metrics.py \
  --dataset-dir "$DS" --ds-key "$DS_KEY" --split test \
  --out "$OUT" >> "$LOG" 2>&1
rc2=$?
if [ $rc2 -eq 0 ]; then
  echo "=== ERR_METRICS_DONE ===" >> "$LOG"
  python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  recover_clean_key: {d.get('recover_clean_key')}")
print(f"  detection F1: {d['detection']['F1']:.6f}  (P={d['detection']['P']:.6f} R={d['detection']['R']:.6f})")
print(f"  repair    F1: {d['repair']['F1']:.6f}  (P={d['repair']['P']:.6f} R={d['repair']['R']:.6f})")
b = d["buckets"]
for k in ("b_clean", "b_dirty", "b_wrong", "b_missing", "fp_det",
          "dup_removed", "dup_kept", "leak_removed", "leak_kept",
          "crow_recovered", "natural_noise"):
    print(f"    {k:16s} {b.get(k):,}")
PY
else
  echo "!! compute_error_metrics 失败 (rc=$rc2)，看 $LOG" >&2
fi
exit $rc2
