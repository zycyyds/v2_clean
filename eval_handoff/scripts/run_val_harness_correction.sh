#!/bin/bash
# 在校正类（单表纠错）任务上跑我们 agent 的 val harness（多轮 + 每轮反馈）。
#
#   bash eval_handoff/scripts/run_val_harness_correction.sh <profile> [suffix]
#     profile: flights_correction | hospital_correction
#     suffix : 实验目录后缀，默认 v1（实验目录必须全新且为空）
#
# 用的是 run_val_harness_ft.py 这个**带容错**的 driver：
#   ① run_turn 失败（ReadTimeout / 429 / 畸形响应）自动重试
#   ② harness 整体抛异常时兜底收尾（独立重放 best_snapshot → 冻结 reproducible_snapshot）
#      ⇒ 「第 1 轮就崩」也能留下第 1 轮之前的成果
set -uo pipefail

PROFILE="${1:-flights_correction}"
SUFFIX="${2:-v1}"

PROJ=/Users/mac/PycharmProjects/v2_clean
WT=/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning
RUN="$WT/k29_agent_run"

export RUN
export PROJ
export WT
export PYTHONPATH="$PROJ"

LOG="$RUN/logs/${PROFILE}_${SUFFIX}.log"
mkdir -p "$RUN/logs" "$RUN/experiments"

echo "=== profile   : $PROFILE"
echo "=== suffix    : $SUFFIX"
echo "=== log       : $LOG"
echo "=== experiment: $RUN/experiments/${PROFILE}_${SUFFIX}"
echo

# `cd /tmp` 是必须的：worktree 里也有一份叫 `agent/` 的包会遮蔽真正的那个；
# PYTHONPATH 必须用 `conda run env` 传（conda run 自己不转发）。
cd /tmp
exec conda run -n py311agent2 --no-capture-output python \
  "$WT/eval_handoff/scripts/run_val_harness_ft.py" "$PROFILE" "$SUFFIX" 2>&1 | tee "$LOG"
