from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from agent.pi_harness_sandbox import build_macos_sandbox_profile


@dataclass(frozen=True)
class SandboxedWorkerConfig:
    project_root: Path
    agent_workdir: Path
    runtime_root: Path
    public_read_roots: tuple[Path, ...]
    skill_dirs: tuple[Path, ...]
    max_iters: int
    model_environment: dict[str, str]
    tool_profile: str = "full"
    runner_spec_path: Path | None = None


@dataclass(frozen=True)
class WorkerLaunch:
    command: list[str]
    cwd: Path
    env: dict[str, str]
    profile_path: Path


def build_sandboxed_worker_launch(config: SandboxedWorkerConfig) -> WorkerLaunch:
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("Pi Validation Harness requires macOS sandbox-exec")
    project = config.project_root.expanduser().resolve()
    workdir = config.agent_workdir.expanduser().resolve()
    runtime = config.runtime_root.expanduser().resolve()
    home = runtime / "home"
    scratch = runtime / "tmp"
    for directory in (workdir, runtime, home, scratch):
        directory.mkdir(parents=True, exist_ok=True)

    read_roots = [
        project / "agent",
        project / "lib",
        project / "config_loader.py",
        project / "model_config.yaml",
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path("/System"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/private/etc"),
        Path("/private/var/select"),
        workdir,
        *config.public_read_roots,
        *config.skill_dirs,
    ]
    profile_path = runtime / "agent.sb"
    profile_path.write_text(
        build_macos_sandbox_profile(
            executable=Path(sys.executable),
            read_roots=[path for path in read_roots if path.exists()],
            write_roots=[workdir, home, scratch, Path("/dev/null")],
            allow_network=True,
            traversal_roots=[project],
        ),
        encoding="utf-8",
    )

    environment = {
        "HOME": str(home),
        "TMPDIR": str(scratch),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(project),
        "PI_HARNESS_SANDBOX": "1",
        **config.model_environment,
        "V2_SKIP_LOCAL_MODEL_CONFIG": "1",
    }
    command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile_path),
        sys.executable,
        "-u",
        "-m",
        "agent.pi_worker",
        "--workdir",
        str(workdir),
        "--max-iters",
        str(config.max_iters),
    ]
    for skill_dir in config.skill_dirs:
        command.extend(["--skills-dir", str(skill_dir.resolve())])
    command.extend(["--tool-profile", config.tool_profile])
    for read_root in config.public_read_roots:
        command.extend(["--read-root", str(read_root.resolve())])
    if config.runner_spec_path is not None:
        command.extend(["--runner-spec", str(config.runner_spec_path.resolve())])
    return WorkerLaunch(
        command=command,
        cwd=workdir,
        env=environment,
        profile_path=profile_path,
    )


class JsonlWorkerClient:
    def __init__(
        self,
        *,
        command: list[str],
        cwd: str | Path,
        env: dict[str, str] | None,
        output: TextIO | None = None,
        startup_timeout: float = 60.0,
    ) -> None:
        self.command = list(command)
        self.cwd = Path(cwd).resolve()
        self.env = env
        self.output = output or sys.stderr
        self.startup_timeout = startup_timeout
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.process is not None:
            return
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=str(self.cwd),
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._stderr_task = asyncio.create_task(self._pump_stderr())
        ready = await asyncio.wait_for(self._read_message(), timeout=self.startup_timeout)
        if ready != {"event": "ready"}:
            raise RuntimeError(f"unexpected Pi worker startup message: {ready}")

    async def request(self, command: str, **payload: Any) -> dict[str, Any]:
        async with self._lock:
            if self.process is None or self.process.returncode is not None:
                raise RuntimeError("Pi worker is not running")
            assert self.process.stdin is not None
            self._request_id += 1
            request_id = self._request_id
            message = {"id": request_id, "command": command, **payload}
            self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
            await self.process.stdin.drain()
            response = await self._read_message()
            if response.get("id") != request_id:
                raise RuntimeError(f"Pi worker response id mismatch: {response}")
            if response.get("error"):
                error = response["error"]
                raise RuntimeError(
                    f"Pi worker {error.get('type', 'error')}: {error.get('message', '')}",
                )
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"Pi worker returned an invalid result: {response}")
            return result

    async def run_turn(self, prompt: str) -> dict[str, Any]:
        return await self.request("run_turn", prompt=prompt)

    async def clear_file_cache(self) -> None:
        await self.request("clear_file_cache")

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            try:
                await self.request("close")
            except (BrokenPipeError, ConnectionError, RuntimeError):
                process.terminate()
            await process.wait()
        if self._stderr_task is not None:
            await self._stderr_task
        self.process = None

    async def interrupt(self, *, force: bool = False) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        if force:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            os.killpg(process.pid, signal.SIGINT)
        await process.wait()

    async def _read_message(self) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Pi worker is not running")
        line = await self.process.stdout.readline()
        if not line:
            code = await self.process.wait()
            raise RuntimeError(f"Pi worker exited before responding: returncode={code}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Pi worker polluted the JSONL control channel: {line!r}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Pi worker response must be a JSON object")
        return payload

    async def _pump_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        while True:
            line = await self.process.stderr.readline()
            if not line:
                return
            self.output.write(line.decode(errors="replace"))
            self.output.flush()
