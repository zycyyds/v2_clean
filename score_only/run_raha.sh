#!/bin/bash
# 独立 Raha 打分 —— 不依赖项目环境，只要 python3 + pandas。
#
# 用法:
#   bash run_raha.sh <结果包目录> <gold目录> [manifest] [输出json]
#
# 例:
#   bash run_raha.sh \
#     /path/to/result_root \
#     /path/to/dataset/test/reference \
#     manifests/icu_5000_v1.json \
#     /tmp/score.json
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 2 ]; then
  echo "用法: $0 <结果包目录> <gold目录> [manifest] [输出json]" >&2
  exit 2
fi
RESULT="$1"
GOLD="$2"
MANIFEST="${3:-$HERE/manifests/icu_5000_v1.json}"
OUT="${4:-$HERE/score_result.json}"

for d in "$RESULT" "$GOLD"; do
  if [ ! -d "$d" ]; then
    echo "!! 目录不存在: $d" >&2
    exit 3
  fi
done
if [ ! -f "$MANIFEST" ]; then
  echo "!! manifest 不存在: $MANIFEST" >&2
  exit 3
fi

# 评分器 import workflow.reference_evaluation，所以本包自带 workflow/ 目录，
# 把包根放进 PYTHONPATH 即可（不依赖项目环境）
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
cd "$HERE"

python3 "$HERE/pi_test_scorer.py" \
  --result-root "$RESULT" \
  --gold-root "$GOLD" \
  --evaluation-manifest "$MANIFEST" > "$OUT"

echo "=== 打分完成 → $OUT"
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d.get("metrics", {})
print(f"  status         : {d.get('status')}")
print(f"  composite_score: {m.get('composite_score')}")
for k in ("schema_f1", "key_structure_f1", "row_aligned_cell_f1", "exact_row_f1"):
    print(f"  {k:16s}: {m.get(k)}")
print("  逐文件:")
for fr in sorted(d.get("file_reports", []), key=lambda x: x.get("metrics", {}).get("file_score") or 0):
    mm = fr.get("metrics", {})
    print(f"    {fr.get('relative_path',''):30s} {mm.get('file_score', 0):.4f}  "
          f"(ref {fr.get('reference_rows')} / res {fr.get('result_rows')} / "
          f"missing {fr.get('missing_reference_rows')} / extra {fr.get('extra_result_rows')})")
PY
