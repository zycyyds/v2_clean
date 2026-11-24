#!/usr/bin/env python3
"""verify_worker_stream_limit.py —— 验证「传输层行上限」补丁。

假 worker 吐一条 200 KB 的 JSONL 单行。

不补丁 ⇒ asyncio 默认 64 KiB 行上限 ⇒ readline() 抛
       ValueError: Separator is not found, and chunk exceed the limit
打补丁 ⇒ limit=64 MiB ⇒ 正常读到，id 不错位。

跑法：python3 test_stream_limit.py
"""
import asyncio
import json
import os
import sys
import tempfile
import textwrap

PROJ = "/Users/mac/PycharmProjects/v2_clean"
sys.path.insert(0, PROJ)
from agent.pi_worker_client import JsonlWorkerClient  # noqa: E402

# ── 复刻 driver 里的补丁 ────────────────────────────────────────────────
WORKER_STREAM_LIMIT = 64 * 1024 * 1024
_orig_start = JsonlWorkerClient.start


async def _start_with_large_stream_limit(self):
    orig_exec = asyncio.create_subprocess_exec

    async def _exec(*args, **kwargs):
        kwargs.setdefault("limit", WORKER_STREAM_LIMIT)
        return await orig_exec(*args, **kwargs)

    asyncio.create_subprocess_exec = _exec
    try:
        return await _orig_start(self)
    finally:
        asyncio.create_subprocess_exec = orig_exec


JsonlWorkerClient.start = _start_with_large_stream_limit
if os.environ.get("NO_PATCH") == "1":
    JsonlWorkerClient.start = _orig_start
    print("  (已禁用补丁 ⇒ 期望复现原 bug)")


FAKE_WORKER = textwrap.dedent(
    """
    import json, sys
    def w(p):
        sys.stdout.write(json.dumps(p, ensure_ascii=False) + "\\n")
        sys.stdout.flush()
    w({"event": "ready"})
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("command") == "run_turn":
            # 造一条 200 KB 的超长回复（> asyncio 默认 64 KiB 行上限）
            w({"id": req["id"], "result": {"status": "SUCCESS",
                                          "text": "X" * 200000}})
        else:
            w({"id": req["id"], "result": {}})
    """
)


async def main():
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "fake_worker.py")
        with open(script, "w") as f:
            f.write(FAKE_WORKER)
        c = JsonlWorkerClient(
            command=[sys.executable, script],
            cwd=td,
            env=dict(os.environ),
            output=open(os.devnull, "w"),
        )
        await c.start()
        try:
            r = await c.run_turn("hi")
            n = len(r.get("text", ""))
            print("  ✅ 读到超长行，text 长度 = %d，status=%s" % (n, r.get("status")))
            ok = n == 200000
        except Exception as exc:
            print("  ❌ 失败: %s: %s" % (type(exc).__name__, exc))
            ok = False
        # 再连发两轮，确认没有错位
        try:
            r2 = await c.run_turn("hi2")
            r3 = await c.run_turn("hi3")
            print("  ✅ 后续两轮也正常：%d / %d" % (len(r2.get("text", "")), len(r3.get("text", ""))))
            ok = ok and len(r2.get("text", "")) == 200000 and len(r3.get("text", "")) == 200000
        except Exception as exc:
            print("  ❌ 后续轮失败: %s: %s" % (type(exc).__name__, exc))
            ok = False
        await c.close()
        print("  ⇒", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
