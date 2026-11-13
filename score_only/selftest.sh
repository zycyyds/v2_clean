#!/bin/bash
# 评分包自检 —— 拿到包先跑这个，确认能用。
#
# 用法: bash selftest.sh
#
# 检查:
#   1. python + pandas 可用
#   2. 脚本语法正确
#   3. 评分器能 import（workflow 包布局对不对）
#   4. --help 能跑起来
#   5. manifest 里的 13 个文件都有主键定义
#   6. 包里不含密钥
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
fail=0

echo "=== 1. 运行环境 ==="
if python3 -c "import pandas, sys; print(f'  python {sys.version.split()[0]}  pandas {pandas.__version__}')" 2>/dev/null; then
  :
else
  echo "  FAIL python3 或 pandas 不可用"
  echo "       装：pip3 install pandas  （或用 conda 环境 py311agent2）"
  fail=1
fi

echo
echo "=== 2. 脚本语法 ==="
for f in pi_test_scorer.py workflow/*.py error_metrics/*.py; do
  if python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" 2>/dev/null; then
    echo "  OK   $f"
  else
    echo "  FAIL $f"; fail=1
  fi
done
for f in *.sh; do
  if bash -n "$f" 2>/dev/null; then echo "  OK   $f"; else echo "  FAIL $f"; fail=1; fi
done

echo
echo "=== 3. 评分器 import 链 ==="
if PYTHONPATH="$HERE" python3 -c "
from workflow.pi_harness_evaluation import load_evaluation_manifest, score_equal_weight_reference_directory
print('  OK   workflow.pi_harness_evaluation 可 import')
" 2>/dev/null; then
  :
else
  echo "  FAIL workflow 包 import 失败"
  echo "       检查：workflow/__init__.py 与 workflow/*.py 是否齐全"
  fail=1
fi

echo
echo "=== 4. CLI --help ==="
for cmd in "pi_test_scorer.py --help" "error_metrics/build_eval_cache.py --help" "error_metrics/compute_error_metrics.py --help"; do
  if PYTHONPATH="$HERE" python3 $cmd >/dev/null 2>&1; then
    echo "  OK   python3 $cmd"
  else
    echo "  FAIL python3 $cmd"; fail=1
  fi
done

echo
echo "=== 5. manifest 的每个文件都有主键 ==="
python3 - <<'PY'
import json, sys, glob, os
bad = 0
for path in sorted(glob.glob("manifests/*.json")):
    d = json.load(open(path))
    files = d.get("files", {})
    missing = [k for k, v in files.items() if not v.get("key_columns")]
    name = os.path.basename(path)
    print(f"  {name} 声明 {len(files)} 个文件")
    if missing:
        print(f"  FAIL 这些文件没有 key_columns: {missing}")
        bad = 1
        continue
    print(f"  OK   全部有 key_columns")
if bad:
    sys.exit(1)
PY
[ $? -ne 0 ] && fail=1

echo
echo "=== 6. 包内不含密钥 ==="
if grep -rniE "api[_-]?key *=|sk-[a-zA-Z0-9]{16}|secret *=|password *=" \
     . --include='*.py' --include='*.sh' --include='*.json' --include='*.md' 2>/dev/null; then
  echo "  FAIL 发现疑似密钥，人工确认"
  fail=1
else
  echo "  OK   无密钥"
fi

echo
if [ $fail -eq 0 ]; then
  echo "=== 自检通过，可以开始评测 ==="
  echo
  echo "下一步：读 评测规范.md §2，按命令跑"
else
  echo "=== 有 FAIL 项，先修好再评 ==="
fi
exit $fail
