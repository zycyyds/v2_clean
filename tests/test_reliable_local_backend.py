from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from agentscope.message import ToolResultState

from agent.pi_harness_sandbox import build_macos_sandbox_profile
from agent.pi_runtime import build_pi_toolkit


def test_bash_does_not_wait_for_descendant_holding_stdio(tmp_path: Path) -> None:
    async def exercise() -> tuple[float, ToolResultState]:
        toolkit = build_pi_toolkit(tmp_path)
        bash = await toolkit.get_tool("Bash")
        started = time.monotonic()
        chunks = [
            chunk
            async for chunk in bash.call(
                command="sleep 2 & exit 7",
                timeout=100,
            )
        ]
        return time.monotonic() - started, chunks[-1].state

    elapsed, state = asyncio.run(exercise())

    assert elapsed < 1.0
    assert state is ToolResultState.ERROR


def test_bash_timeout_does_not_wait_for_shell_descendant(tmp_path: Path) -> None:
    async def exercise() -> tuple[float, ToolResultState, str]:
        toolkit = build_pi_toolkit(tmp_path)
        bash = await toolkit.get_tool("Bash")
        started = time.monotonic()
        chunks = [
            chunk
            async for chunk in bash.call(
                command="sleep 2",
                timeout=100,
            )
        ]
        text = "".join(block.text for block in chunks[-1].content)
        return time.monotonic() - started, chunks[-1].state, text

    elapsed, state, text = asyncio.run(exercise())

    assert elapsed < 1.0
    assert state is ToolResultState.ERROR
    assert "timed out" in text.lower()


def test_cancelling_bash_kills_the_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived.txt"

    async def exercise() -> bool:
        toolkit = build_pi_toolkit(tmp_path)
        bash = await toolkit.get_tool("Bash")
        command = f"sleep 0.5; printf survived > {shlex.quote(str(marker))}"

        async def consume() -> None:
            async for _chunk in bash.call(command=command, timeout=10_000):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.6)
        return marker.exists()

    assert asyncio.run(exercise()) is False


def test_bash_preserves_stdout_stderr_and_nonzero_exit(tmp_path: Path) -> None:
    program = (
        "import sys; print('stdout-ok'); "
        "print('stderr-ok', file=sys.stderr); sys.exit(9)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"

    async def exercise() -> tuple[ToolResultState, str]:
        toolkit = build_pi_toolkit(tmp_path)
        bash = await toolkit.get_tool("Bash")
        chunks = [chunk async for chunk in bash.call(command=command)]
        text = "".join(block.text for block in chunks[-1].content)
        return chunks[-1].state, text

    state, text = asyncio.run(exercise())

    assert state is ToolResultState.ERROR
    assert "stdout-ok" in text
    assert "stderr-ok" in text


def test_bash_keeps_draining_active_output_after_parent_exit(
    tmp_path: Path,
) -> None:
    command = (
        "(printf first; sleep 0.05; printf second; "
        "sleep 0.05; printf third) & exit 0"
    )

    async def exercise() -> tuple[ToolResultState, str]:
        toolkit = build_pi_toolkit(tmp_path)
        bash = await toolkit.get_tool("Bash")
        chunks = [chunk async for chunk in bash.call(command=command)]
        text = "".join(block.text for block in chunks[-1].content)
        return chunks[-1].state, text

    state, text = asyncio.run(exercise())

    assert state is ToolResultState.RUNNING
    assert text == "firstsecondthird"


def test_python_failure_piped_to_head_finishes(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("raise KeyError('diagnoses_icd')\n", encoding="utf-8")
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} "
        "2>&1 | head -100"
    )

    async def exercise() -> tuple[float, ToolResultState, str]:
        toolkit = build_pi_toolkit(tmp_path)
        bash = await toolkit.get_tool("Bash")
        started = time.monotonic()
        chunks = [chunk async for chunk in bash.call(command=command)]
        text = "".join(block.text for block in chunks[-1].content)
        return time.monotonic() - started, chunks[-1].state, text

    elapsed, state, text = asyncio.run(exercise())

    assert elapsed < 1.0
    assert state is ToolResultState.RUNNING
    assert "KeyError" in text
    assert "diagnoses_icd" in text


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="requires macOS sandbox-exec",
)
def test_sandboxed_timeout_kills_the_entire_process_group(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    pid_file = tmp_path / "timed-out-process-group.txt"
    command = f"echo $$ > {shlex.quote(str(pid_file))}; sleep 30"
    program = (
        "import asyncio\n"
        "from lib.reliable_local_backend import ReliableLocalBackend\n"
        "async def main():\n"
        "    result = await ReliableLocalBackend().exec_shell(\n"
        f"        ['/bin/sh', '-c', {command!r}], timeout=0.2,\n"
        "    )\n"
        "    print(result.stderr.decode())\n"
        "asyncio.run(main())\n"
    )
    profile = tmp_path / "timeout.sb"
    profile.write_text(
        build_macos_sandbox_profile(
            executable=Path(sys.executable),
            read_roots=[
                project_root,
                Path(sys.prefix),
                Path(sys.base_prefix),
                Path("/System"),
                Path("/usr"),
                Path("/bin"),
                Path("/private/etc"),
                tmp_path,
            ],
            write_roots=[tmp_path, Path("/dev/null")],
            allow_network=False,
        ),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(project_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    process_group: int | None = None

    try:
        completed = subprocess.run(
            [
                "/usr/bin/sandbox-exec",
                "-f",
                str(profile),
                sys.executable,
                "-c",
                program,
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        process_group = int(pid_file.read_text(encoding="utf-8").strip())

        assert completed.returncode == 0, completed.stderr
        assert "timed out" in completed.stdout.lower()
        with pytest.raises(ProcessLookupError):
            os.killpg(process_group, 0)
    finally:
        if process_group is not None:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
