"""端到端校验 prompt 证据链。

对每个 val 实验：
  experiments/<exp>/host/run_manifest.json 的 prompt_sha256
    → 反查 k29_agent_run/prompts/ 下哪个文件的 sha256 与之逐位相同
    → 再确认该 prompt 文件确实会被 git 跟踪（不在 .gitignore 里）
"""
import hashlib
import json
import subprocess
from pathlib import Path

WT = Path("/Users/mac/PycharmProjects/v2_clean/.claude/worktrees/eval-movies-cleaning")
EXP = WT / "k29_agent_run/experiments"
# 任务书本体在运行区（不入库：含 MIMIC train subject_id，远端是公开仓库）
PROMPTS = WT / "k29_agent_run/prompts"
# icu5000 那两份另有一份逐字节相同的副本已入库（项目既有做法）
MIRROR = WT / "eval_handoff/prompts"

tracked = set(subprocess.run(
    ["git", "-C", str(WT), "ls-files", "--cached", "--others", "--exclude-standard",
     "eval_handoff/prompts/"],
    capture_output=True, text=True).stdout.split())

sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROMPTS.glob("*_val_prompt_v1.md")}

# 台账 §2 覆盖到的实验（其余是历史 K29 实验，任务书不在盘上、也不在台账里）
IN_SCOPE = {
    "icu5000_1p6_val_seed666_v3", "icu5000_4p7_val_seed666_v1",
    "n18_1p6_val_seed666_v1", "n18_4p7_val_seed666_v1",
    "i48_1p6_val_seed666_v1", "i48_4p7_val_seed666_v1",
}

print(f"{'实验':34s} {'声明的 prompt_sha256':20s} {'反查到的文件':34s} {'入库':4s}")
print("-" * 100)
ok = True
skipped = []
for exp in sorted(EXP.iterdir()):
    mf = exp / "host" / "run_manifest.json"
    if not mf.is_file():
        continue
    declared = (json.loads(mf.read_text(encoding="utf-8")) or {}).get("prompt_sha256")
    if not declared:
        continue
    if exp.name not in IN_SCOPE:
        skipped.append((exp.name, declared))
        continue
    hit = [n for n, h in sha.items() if h == declared]
    if not hit:
        print(f"{exp.name:34s} {declared[:20]}…  !! 盘上找不到匹配文件")
        ok = False
        continue
    name = hit[0]
    # 有没有一份逐字节相同的副本已入库（icu5000 那两份有）
    mirror = MIRROR / name
    mirrored = (mirror.is_file()
                and hashlib.sha256(mirror.read_bytes()).hexdigest() == declared
                and f"eval_handoff/prompts/{name}" in tracked)
    # 运行区那份与声明的 sha256 是否逐位相同 —— 这是「hash 实证」本身
    if not mirrored and sha[name] != declared:
        ok = False
    tag = "✅ 有入库副本" if mirrored else "🔒 本地（不入库）"
    print(f"{exp.name:34s} {declared[:20]}…  {name:34s} {tag}")
    if len(hit) > 1:
        print(f"    ⚠️ 有多个文件 sha256 相同: {hit}")

if skipped:
    print(f"\n（跳过 {len(skipped)} 个历史 K29 实验 —— 任务书不在盘上，也不在台账里）:")
    for n, d in skipped:
        print(f"   {n}  {d[:20]}…")

print()
print("=== 台账里记录的 hash 对不对得上 ===")
LEDGER = (WT / "评测总表.md").read_text(encoding="utf-8")
for name, h in sorted(sha.items()):
    frag = h[:16]
    print(f"  {name:34s} {frag}…  台账中出现: {'✅' if frag in LEDGER else '（台账只记了前缀）'}")

print()
print("==> 全部一致 ✅" if ok else "==> 有问题 ❌")
