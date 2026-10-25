#!/bin/bash
# 跑完 N18 / I48 / ICU5000 三个数据集 × 三个 split 的错误注入。
# 串行执行 —— 内存峰值叠加会 OOM（16 GB 机器）。
set -uo pipefail
cd /Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning
LOGDIR=/Users/mac/.claude/jobs/910902cd/tmp
mkdir -p "$LOGDIR"

for DS in n18 i48 icu5000; do
  for SP in train validation test; do
    LOG="$LOGDIR/inject_${DS}_${SP}.log"
    echo "=== $DS / $SP -> $LOG"
    /usr/bin/time -l conda run -n py311agent2 --no-capture-output \
      python -u reproduction/build_rawerror_by_ratio.py "$DS" "$SP" \
      > "$LOG" 2>&1
    rc=$?
    echo "    exit=$rc"
    tail -6 "$LOG"
    if [ $rc -ne 0 ]; then
      echo "!!! $DS/$SP 失败，停止"
      exit 1
    fi
  done
done
echo "全部完成"
