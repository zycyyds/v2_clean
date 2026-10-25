#!/bin/bash
# 给新数据集的 agent 跑批打分：Raha 总分 + 错误级 detection/repair 指标。
#
# 用法: score_new_runs.sh <key> <preset>       preset ∈ {more, less}
set -uo pipefail
cd /Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning
PROJ=/Users/mac/PycharmProjects/v2_clean
RUN="$PROJ/.claude/worktrees/eval-movies-cleaning/k29_agent_run"
LOGS=/Users/mac/.claude/jobs/910902cd/tmp
RUNS="$PROJ/runs"
KEY="${1:?}"; PRESET="${2:?}"

case "$KEY" in
  n18)     DS="$PROJ/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_test${PRESET/less/1.6pct}"
          [[ "$PRESET" == less ]] && DS="$PROJ/datasets/注入错误数据集/N18_ckd_ED_v2_rawerror_test1.6pct_490M" ;;
  i48)     [[ "$PRESET" == less ]] && DS="$PROJ/datasets/注入错误数据集/I48_atrial_fibrillation_v2_rawerror_test1.6pct_411M" \
          || DS="$PROJ/datasets/注入错误数据集/I48_atrial_fibrillation_v2_rawerror_test4.7pct_411M" ;;
  icu5000) [[ "$PRESET" == less ]] && DS="$PROJ/datasets/注入错误数据集/icu_5000_v2_rawerror_test1.6pct_614M" \
          || DS="$PROJ/datasets/注入错误数据集/icu_5000_v2_rawerror_test4.7pct_614M" ;;
  k29)     [[ "$PRESET" == less ]] && DS="$PROJ/datasets/注入错误数据集/K29_v2_rawerror_test1.6pct_4.2M" \
          || DS="$PROJ/datasets/注入错误数据集/K29_v2_rawerror_test4.7pct_14.1M" ;;
  *) echo "未知 key $KEY"; exit 2 ;;
esac
case "$KEY" in
  n18)     MANIFEST="$PROJ/evaluation_manifests/n18_ckd_ED_notes_v1.json" ;;
  i48)     MANIFEST="$PROJ/evaluation_manifests/i48_atrial_fibrillation_ECG_v1.json" ;;
  icu5000) MANIFEST="$PROJ/evaluation_manifests/icu_5000_v1.json" ;;
  k29)     MANIFEST="$PROJ/evaluation_manifests/k29_gastritis_duodenitis_v3_1.json" ;;
esac

RESULT="$RUN/agent_output_${KEY}_${PRESET}"
# agent 可能把结果包直接写在 output_dir，也可能写在 output_dir/result
if [ ! -d "$RESULT" ] || [ -z "$(ls -A "$RESULT" 2>/dev/null)" ]; then
  echo "!! $KEY/$PRESET 没有产出（$RESULT）"; exit 4
fi
if [ -d "$RESULT/result" ]; then RESULT="$RESULT/result"; fi

GOLD="$DS/test/reference"
LOG="$LOGS/score_${KEY}_${PRESET}.log"
mkdir -p "$RUNS"

echo "=== $KEY / $PRESET ==="
echo "  result = $RESULT"
echo "  gold   = $GOLD"
echo "  log    = $LOG"

# ---- 1. Raha 总分 ----
( cd /tmp && conda run -n py311agent2 env PYTHONPATH="$PROJ" \
  python -m agent.pi_test_scorer \
  --result-root "$RESULT" \
  --gold-root "$GOLD" \
  --evaluation-manifest "$MANIFEST" ) > "$LOGS/score_${KEY}_${PRESET}_raha.json" 2>"$LOG"
rc=$?
if [ $rc -ne 0 ]; then
  echo "  Raha 评分失败 rc=$rc"; tail -20 "$LOG"
else
  python3 -c "
import json
d=json.load(open('$LOGS/score_${KEY}_${PRESET}_raha.json'))
m=d.get('metrics',{})
print('  Raha: composite=%s schema=%s key=%s cell=%s exact=%s cov=%s' % (
 m.get('composite_score'),m.get('schema_f1'),m.get('key_structure_f1'),
 m.get('row_aligned_cell_f1'),m.get('exact_row_f1'),m.get('file_coverage')))
" || { echo "  Raha 输出解析失败"; head -20 "$LOGS/score_${KEY}_${PRESET}_raha.json"; }
  cp "$LOGS/score_${KEY}_${PRESET}_raha.json" "$RUNS/${KEY}_${PRESET}_overall_score.json"
fi

# ---- 2. 错误级指标 ----
DSKEY="${KEY}_${PRESET}"
conda run -n py311agent2 --no-capture-output \
  python reproduction/build_eval_cache.py \
  --result-dir "$RESULT" --ref-dir "$GOLD" --ds-key "$DSKEY" \
  >> "$LOG" 2>&1
conda run -n py311agent2 --no-capture-output \
  python reproduction/compute_error_metrics.py \
  --dataset-dir "$DS" --ds-key "$DSKEY" --split test \
  --out "$RUNS/${KEY}_${PRESET}_error_metrics.json" \
  >> "$LOG" 2>&1
if [ -f "$RUNS/${KEY}_${PRESET}_error_metrics.json" ]; then
  python3 -c "
import json
d=json.load(open('$RUNS/${KEY}_${PRESET}_error_metrics.json'))
t=d.get('totals',d)
print('  错误级: det P=%s R=%s F1=%s | rep P=%s R=%s F1=%s' % (
 t.get('detect_P'),t.get('detect_R'),t.get('detect_F1'),
 t.get('repair_P'),t.get('repair_R'),t.get('repair_F1')))
" 2>/dev/null || echo "  错误级输出解析失败（看 $LOG）"
else
  echo "  错误级指标失败（看 $LOG）"
fi
echo "  done"
