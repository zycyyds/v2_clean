#!/bin/bash
# 收尾链：等当前注入跑完 → 补 ICU5000 的 1.6pct → 验证 → 报告 → 扣留视图
set -uo pipefail
cd /Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning
LOGDIR=/Users/mac/.claude/jobs/910902cd/tmp
CONDA="conda run -n py311agent2 --no-capture-output"

step() { echo; echo "########## $* ##########"; }

# ---- 1. 等当前两个批次结束 ----
step "等 ICU5000 4.7pct + n18/i48 1.6pct 跑完"
while true; do
  n=$(ps -axo command | grep -c "[b]uild_rawerror_by_ratio.py")
  icu=$(grep -c "ICU5000 完成" "$LOGDIR/run_icu5000.log" 2>/dev/null || echo 0)
  p16=$(grep -c "1.6pct 完成" "$LOGDIR/run_1p6_n18_i48.log" 2>/dev/null || echo 0)
  if [ "$n" -eq 0 ] && [ "$icu" -ge 1 ] && [ "$p16" -ge 1 ]; then break; fi
  sleep 20
done
echo "两个批次都完成了"

# ---- 2. ICU5000 的 1.6pct ----
step "ICU5000 1.6pct 注入"
bash "$LOGDIR/run_1p6.sh" icu5000

# ---- 3. 验证 ----
step "验证 4.7pct（三个数据集全部 split）"
bash reproduction/run_all_verifications.sh 4.7pct n18 i48 icu5000

step "验证 1.6pct（三个数据集全部 split）"
bash reproduction/run_all_verifications.sh 1.6pct n18 i48 icu5000

# ---- 4. 报告 ----
step "重建每数据集独立报告"
for DS in n18 i48 icu5000; do
  $CONDA python -u reproduction/rebuild_dataset_report.py "$DS" --rates 4.7pct
  $CONDA python -u reproduction/rebuild_dataset_report.py "$DS" --rates 1.6pct
done

step "重建共享报告"
$CONDA python -u reproduction/rebuild_shared_report.py --rates 4.7pct
$CONDA python -u reproduction/rebuild_shared_report.py --rates 1.6pct

# ---- 5. 扣留视图 ----
step "建 agent 扣留视图"
$CONDA python -u reproduction/build_agent_views.py --rates 4.7pct
$CONDA python -u reproduction/build_agent_views.py --rates 1.6pct

step "全部完成"
