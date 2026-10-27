#!/bin/bash
# 逐个 split 跑 verify_rawerror_dataset.py，串行（内存峰值叠加会 OOM）。
# 已经有 modification_log.csv 的 split 才跑 —— 注入还没完成的跳过。
set -uo pipefail
cd /Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning
LOGDIR=/Users/mac/.claude/jobs/910902cd/tmp
OUTROOT=/Users/mac/PycharmProjects/v2_clean/datasets/注入错误数据集
RATES="${1:-4.7pct}"
shift || true
DS_LIST=("$@")
if [ ${#DS_LIST[@]} -eq 0 ]; then
  DS_LIST=(n18 i48 icu5000)
fi
mkdir -p "$LOGDIR"

dir_name() {
  case "$1" in
    n18)     echo "N18_ckd_ED_v2_rawerror_test${RATES}_490M" ;;
    i48)     echo "I48_atrial_fibrillation_v2_rawerror_test${RATES}_411M" ;;
    icu5000) echo "icu_5000_v2_rawerror_test${RATES}_614M" ;;
    # K29 的低密度档目录名不带错误率（固定 10train_400_800_{tag}），
    # 只有 1800 这一档存在。
    k29)     echo "K29_clean_v2_rawerror_10train_400_800_${RATES}" ;;
  esac
}

for DS in "${DS_LIST[@]}"; do
  for SP in train validation test; do
    D="$OUTROOT/$(dir_name "$DS")/$SP"
    if [ ! -f "$D/modification_log.csv" ]; then
      echo "=== $DS / $SP  跳过（还没注入）"
      continue
    fi
    LOG="$LOGDIR/verify_${DS}_${SP}.log"
    echo "=== $DS / $SP -> $LOG"
    conda run -n py311agent2 --no-capture-output \
      python -u reproduction/verify_rawerror_dataset.py "$DS" "$SP" --rates "$RATES" \
      > "$LOG" 2>&1
    rc=$?
    echo "    exit=$rc"
    grep -E "^  \[[0-9]|→ |总判定" "$LOG" | tail -20
  done
done
echo "全部验证完成"
