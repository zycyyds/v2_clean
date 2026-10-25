#!/bin/bash
# 交付包自检：语法 + 参数 + 路径解析
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || exit 1
fail=0

echo "=== 1. python 语法 ==="
for f in scripts/*.py reference/*.py; do
  if python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" 2>/dev/null; then
    echo "  OK   $f"
  else
    echo "  FAIL $f"; fail=1
  fi
done

echo
echo "=== 2. bash 语法 ==="
for f in scripts/*.sh; do
  if bash -n "$f" 2>/dev/null; then echo "  OK   $f"; else echo "  FAIL $f"; fail=1; fi
done

echo
echo "=== 3. CLI --help 能起来 ==="
if python3 scripts/build_sealed_view.py --help >/dev/null 2>&1; then
  echo "  OK   build_sealed_view.py --help"
else
  echo "  FAIL build_sealed_view.py --help"; fail=1
fi

echo
echo "=== 4. 交付包引用的文件都在 ==="
for f in reference/compute_error_metrics.py reference/build_eval_cache.py \
         reference/icu_5000_v1.json \
         prompts/icu5000_4.7pct_val_prompt_v1.md \
         prompts/icu5000_1.6pct_val_prompt_v1.md \
         scripts/build_sealed_view.py scripts/run_val_harness_ft.py \
         scripts/run_val_harness.sh scripts/run_test_harness.sh \
         scripts/score_error_metrics.sh scripts/final_table.py \
         README.md; do
  if [ -f "$f" ]; then echo "  OK   $f"; else echo "  MISS $f"; fail=1; fi
done

echo
echo "=== 5. 脚本里没有硬编码交付者的工作区路径 ==="
# 排除自检脚本本身（它要 grep 这个模式）
HITS=$(grep -rn "worktrees/eval-movies-cleaning" scripts/ 2>/dev/null | grep -v "selftest.sh" || true)
if [ -n "$HITS" ]; then
  echo "$HITS"
  echo "  FAIL 仍有硬编码路径"; fail=1
else
  echo "  OK   无硬编码"
fi

echo
echo "=== 6. 交付包不含任何密钥 ==="
if grep -rniE "api[_-]?key|sk-[a-z0-9]{10}|token *=|secret" scripts/ reference/ prompts/ README.md 2>/dev/null | grep -v "OPENAI_API_KEYS_JSON" | grep -v "不要把 key"; then
  echo "  WARN  上面这些行需要人工确认"
else
  echo "  OK   无密钥"
fi

echo
if [ $fail -eq 0 ]; then echo "=== 自检全部通过 ==="; else echo "=== 有 FAIL 项 ==="; fi
exit $fail
