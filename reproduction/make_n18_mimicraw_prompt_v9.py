#!/usr/bin/env python3
"""make_n18_mimicraw_prompt_v9.py —— 由 **v8** prompt 生成 **v9**（N18 mimicraw）。

v9 相对 v8 只多**一节**：`REPLAY_CONTRACT_SECTION`（`submission.json` 的 `replay.argv` 契约）。

## 为什么加这一节

**独立重放这一关会把整轮判 0。** N18 1.6pct v7 run 实测：
第 1 轮 val 打分 0.878045（promoted），结果在**独立重放**这关挂掉：

```
[Harness] independent replay: status=EXECUTION_FAILED score=0.000000
[Harness] replay repair 1: status=INVALID_SUBMISSION score=0.000000
```

agent 为此烧了两轮去猜 harness 的执行契约（cwd 是不是 workdir、`/usr/bin/env` 在不在、
相对路径行不行……）。**契约其实是可以写死的**，源码在 `agent/pi_harness.py`：

| harness 实际做的事 | 源码位置 |
|---|---|
| `cwd=str(replay_workdir)` —— 重放的 cwd **就是** workdir（相对路径能解析） | pi_harness.py:896 |
| `argv[0] in {"python","python3"} ⇒ argv[0] = sys.executable` | pi_harness.py:883 |
| `command = ["/usr/bin/sandbox-exec", "-f", profile, *argv]` | pi_harness.py:885 |
| sandbox **只放行 harness 自己那个 python 可执行**（`allow process-exec (literal ...)`） | pi_harness_sandbox.py |
| sandbox 可读：workdir / `{raw_root}` / `{train_reference}`；可写：workdir / HOME / TMPDIR；**无网络** | 同上 |
| `{workdir}` `{output_dir}` `{raw_root}` `{train_reference}` 会被替换成**绝对路径** | pi_harness.py:190-209 |
| 校验器：`argv[0]` 不能是 shell（sh/bash/zsh/env…）、必须含 `{raw_root}` 与 `{output_dir}`、**任何不含占位符的项不得是绝对路径** | pi_harness.py:154-172 |

⇒ **两条致命细节**（agent 猜的就是这两条）：
1. **`/usr/bin/env` 和 `/bin/sh` 在 sandbox 里都不可执行**
   ⇒ `#!/usr/bin/env python3` 的 shebang 启动脚本、`sh -c "..."` 一律 **ENOENT**；
2. **「静态绝对路径」被校验器直接拒**
   ⇒ `submission replay.argv cannot contain static absolute paths`。

⇒ v9 把正确写法直接给出来。

## 用法

    python3 make_n18_mimicraw_prompt_v9.py --in <v8 prompt> --out <v9 prompt>
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 插入锚点：prompt 末尾的「限制：」块之前
A_LIMITS = "\n限制：\n"

REPLAY_CONTRACT_SECTION = """
### ▶️ `submission.json` 的 `replay.argv` 契约（**写错这一关直接 0 分**）

harness 会**独立重放**你的 pipeline：拿你 `submission.json` 里的 `replay.argv`，
在一个 **sandbox**（`sandbox-exec`）里真的跑一遍，跑法固定：

```
cd <workdir>                                # cwd = workdir
/usr/bin/sandbox-exec -f <profile> <argv…>  # 只放行 workdir / {raw_root} / {train_reference}，无网络
```

⇒ **直接照抄这两种之一**（这两种都是本仓库**已经跑通**的形状）：

```json
// A：脚本路径写相对（cwd 就是 workdir）
"replay": {"argv": ["python3", "scripts/n18_clean.py",
                    "--raw_root", "{raw_root}",
                    "--output_dir", "{output_dir}",
                    "--workdir", "{workdir}"]}
```
```json
// B：脚本路径用 {workdir} 占位符
"replay": {"argv": ["python3", "{workdir}/scripts/run_pipeline.py",
                    "--raw-root", "{raw_root}",
                    "--output-dir", "{output_dir}",
                    "--train-reference", "{train_reference}",
                    "--workdir", "{workdir}"]}
```

⇒ **四条硬规则**（违反任何一条都会被 harness 直接拒）：

1. **`argv[0]` 必须是 `"python3"` 或 `"python"`** —— harness 会把它换成它自己那个 python
   （`argv[0] in {"python","python3"} ⇒ argv[0] = sys.executable`）。
   ⚠️ **不要把「启动脚本」当 argv[0]**（`"run_pipeline"` / `"{workdir}/run_pipeline"` /
   `"sh"` / `"bash"` 都**不行**）：sandbox 里执行启动脚本会被拒 ⇒
   `independent replay: status=EXECUTION_FAILED score=0.000000` ⇒ **整轮 0 分**。
2. **`argv[0]` 不能是 shell 命令字符串**：`sh` / `bash` / `zsh` / `dash` / `env` / `fish` / `ksh`
   ⇒ harness 报 `submission replay.argv cannot execute shell command strings`。
3. **必须含 `{raw_root}` 和 `{output_dir}` 两个占位符**（缺一个直接被拒）。
   它们会被替换成**绝对路径**：`{output_dir}` = workdir 下的结果子目录，
   `{raw_root}` = 本次重放的 raw 根。
4. **不许出现「静态绝对路径」**：任何一项如果**不含占位符**，它就必须是**相对路径**
   —— 否则 harness 报 `submission replay.argv cannot contain static absolute paths`。
   （所以脚本路径要么相对、要么写成 `{workdir}/xxx.py`，不能写 `/Users/…/xxx.py`。）

⚠️ **`argv` 是 JSON 字符串数组**，不是对象、不是 shell 字符串。

⇒ **交付前自己验一遍**（比事后猜省两轮）：

```bash
cd <workdir> && python3 -c "import json;print(json.load(open('submission.json'))['replay']['argv'])"
# 用「同样的 cwd + 最小 PATH」实跑一遍（模拟 sandbox 的环境）
( cd <workdir> && env -i PATH=/usr/bin:/bin HOME=$HOME TMPDIR=/tmp \
    python3 scripts/n18_clean.py --raw_root <raw> --output_dir <out> )
```
⚠️ 这一行**必须写成 `python3 <脚本路径>`**（不要靠 shebang 启动），
这样才是 harness 真正的执行形态。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="v8 prompt")
    ap.add_argument("--out", dest="dst", required=True, help="v9 prompt 输出")
    args = ap.parse_args()

    text = Path(args.src).read_text(encoding="utf-8")
    if "replay.argv` 契约" in text:
        raise SystemExit("FATAL: 输入里已经有「replay.argv 契约」一节了，别重复生成")
    if A_LIMITS not in text:
        raise SystemExit("FATAL: 找不到锚点 %r" % A_LIMITS)

    text = text.replace(A_LIMITS, "\n" + REPLAY_CONTRACT_SECTION + A_LIMITS, 1)
    Path(args.dst).write_text(text, encoding="utf-8")
    print("已写出:", args.dst, "(%d bytes)" % len(text.encode("utf-8")))
    print("  新增一节：replay.argv 契约（%d 字符）" % len(REPLAY_CONTRACT_SECTION))


if __name__ == "__main__":
    main()
