#!/bin/bash
# 建 K29 1800（低密度档，fixed-v1）的 **codex 扣留视图**：
#     datasets/注入错误数据集/k29_1800/
#
# codex 是一次性跑（无 val 阶段、无 harness、无多轮反馈），所以它需要的东西只有三样：
#     train/raw       —— 带错的输入
#     train/reference —— 干净金标准（它自己 train 自检用）
#     test/raw        —— 唯一的目标处理集
# 扣留两样：
#     test/reference      —— test 的标准答案
#     modification_log.csv —— 每个被注入 cell 的 clean_value 真值
#
# 目录约定跟别的 codex 档一致（`k29_less` / `k29_more` / `i48_less` …）：
# 视图跟全量数据集同在 `datasets/注入错误数据集/` 下，raw/ 和 reference/ 是软链。
# ⇒ **同目录可枚举风险跟别的 codex 档完全一样**（全量数据集的 modification_log.csv
#   在兄弟目录里），不是这一档特有的。任务书里那句「不要去父目录里找同名文件」
#   就是这里的护栏，历次 codex/cc 跑都是这么过来的。
#   ⚠️ 别以为换个父目录就能隔离 —— `datasets/` 在项目根下面，
#   codex 照样能 `ls /Users/mac/PycharmProjects/v2_clean/datasets/`，
#   所以隔离是**假保护**，不做这种自欺欺人的目录设计。
#
# raw/ 和 reference/ 用 symlink（不占磁盘）；keys.csv 拷贝。
#
# 用法: bash build_codex_view_k29_1800.sh [--force]
set -euo pipefail

PROJ="${PROJ:-/Users/mac/PycharmProjects/v2_clean}"
SRC="$PROJ/datasets/注入错误数据集/K29_clean_v2_rawerror_10train_400_800_1800"
DST="$PROJ/datasets/注入错误数据集/k29_1800"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ -e "$DST" ]; then
  if [ "$FORCE" = "1" ]; then
    echo "=== --force：删掉旧的 $DST"
    rm -rf "$DST"
  else
    echo "FATAL: $DST 已存在（要重建就加 --force）" >&2
    exit 2
  fi
fi

for d in "$SRC/train/raw" "$SRC/train/reference" "$SRC/test/raw"; do
  [ -d "$d" ] || { echo "FATAL: 源缺目录: $d" >&2; exit 3; }
done

mkdir -p "$DST"

# --- train: raw + reference（codex 自己 train 自检要用）---
mkdir -p "$DST/train"
cp "$SRC/train/keys.csv" "$DST/train/keys.csv"
ln -s "$SRC/train/raw"       "$DST/train/raw"
ln -s "$SRC/train/reference" "$DST/train/reference"

# --- test: raw ONLY。reference 永远不建、不链。---
mkdir -p "$DST/test"
cp "$SRC/test/keys.csv" "$DST/test/keys.csv"
ln -s "$SRC/test/raw" "$DST/test/raw"

# ---------------- 泄漏自检 ----------------
fail=0

# 1) 视图里任何位置都不能有 modification_log.csv / verify_report.json
leak=$(find "$DST" \( -name 'modification_log*' -o -name 'verify_report.json' \) 2>/dev/null || true)
if [ -n "$leak" ]; then
  echo "FAIL: 视图里出现注入日志/验证报告："; echo "$leak"; fail=1
fi

# 2) test/reference 不能存在（也不能是软链）
if [ -e "$DST/test/reference" ] || [ -L "$DST/test/reference" ]; then
  echo "FAIL: test/reference 存在 —— test gold 会泄漏"; fail=1
fi

# 3) train/reference 必须存在（codex 自检要用）
[ -d "$DST/train/reference" ] || { echo "FAIL: train/reference 缺失"; fail=1; }

# 4) 三个入口路径都在
for p in train/raw train/reference test/raw; do
  [ -e "$DST/$p" ] || { echo "FAIL: $p 缺失"; fail=1; }
done

# 5) 软链必须指向源数据集的 train/test raw 或 train reference，不能指向 test/reference
for l in train/raw train/reference test/raw; do
  tgt=$(readlink "$DST/$l")
  case "$tgt" in
    */test/reference*) echo "FAIL: $l 指向了 test/reference"; fail=1 ;;
  esac
done

# 6) 视图里不能有 validation/（codex 没有 val 阶段，给了反而是误导）
if [ -e "$DST/validation" ]; then
  echo "FAIL: 视图里有 validation/ —— codex 没有 val 阶段"; fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "BUILD FAILED —— 不可用。删掉 $DST 并排查。"
  exit 1
fi

echo "=== built $DST (verified) ==="
find "$DST" -not -name '.DS_Store' | sort
echo
echo "withheld as intended:"
echo "  OK  任何 split 下都没有 modification_log.csv / verify_report.json"
echo "  OK  test/reference 不存在"
echo "  OK  没有 validation/（codex 无 val 阶段）"
