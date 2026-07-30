from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Sequence

from agentscope.tool import ExecResult, LocalBackend


class ReliableLocalBackend(LocalBackend):
    """Local backend that cannot hang on inherited subprocess pipes."""

    def __init__(
        self,
        *,
        exit_stdio_grace_seconds: float = 0.1,
        cleanup_timeout_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        self.exit_stdio_grace_seconds = exit_stdio_grace_seconds
        self.cleanup_timeout_seconds = cleanup_timeout_seconds

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        kwargs: dict[str, object] = {}
        if cwd is not None:
            kwargs["cwd"] = cwd
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **kwargs,
            )
        except (FileNotFoundError, NotADirectoryError, OSError) as exc:
            return ExecResult(
                exit_code=127,
                stdout=b"",
                stderr=str(exc).encode("utf-8"),
            )

        stdout = bytearray()
        stderr = bytearray()
        output_activity = asyncio.Event()
        last_output_at = [asyncio.get_running_loop().time()]
        readers = [
            asyncio.create_task(
                self._drain_stream(
                    process.stdout,
                    stdout,
                    output_activity,
                    last_output_at,
                ),
            ),
            asyncio.create_task(
                self._drain_stream(
                    process.stderr,
                    stderr,
                    output_activity,
                    last_output_at,
                ),
            ),
        ]
        wait_task = asyncio.create_task(process.wait())
        timed_out = False

        try:
            done, _ = await asyncio.wait({wait_task}, timeout=timeout)
            if not done:
                timed_out = True
                await self._kill_process_group(process)
            else:
                await wait_task
                await self._finish_readers_after_exit(
                    process,
                    readers,
                    output_activity,
                    last_output_at,
                )
        except asyncio.CancelledError:
            await self._kill_process_group(process)
            await self._settle_tasks(readers)
            raise
        finally:
            if not wait_task.done():
                wait_task.cancel()
            await self._settle_tasks([wait_task])

        if timed_out:
            await self._settle_tasks(readers)
            return ExecResult(
                exit_code=-1,
                stdout=b"",
                stderr=b"timed out",
            )

        await self._settle_tasks(readers)
        return ExecResult(
            exit_code=process.returncode or 0,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
        )

    @staticmethod
    async def _drain_stream(
        stream: asyncio.StreamReader | None,
        destination: bytearray,
        activity: asyncio.Event,
        last_output_at: list[float],
    ) -> None:
        if stream is None:
            return
        loop = asyncio.get_running_loop()
        while chunk := await stream.read(64 * 1024):
            destination.extend(chunk)
            last_output_at[0] = loop.time()
            activity.set()

    async def _finish_readers_after_exit(
        self,
        process: asyncio.subprocess.Process,
        readers: Sequence[asyncio.Task[None]],
        activity: asyncio.Event,
        last_output_at: list[float],
    ) -> None:
        loop = asyncio.get_running_loop()
        exited_at = loop.time()
        while any(not task.done() for task in readers):
            remaining = self.exit_stdio_grace_seconds - (
                loop.time() - max(exited_at, last_output_at[0])
            )
            if remaining <= 0:
                await self._kill_process_group(process)
                return

            activity.clear()
            activity_task = asyncio.create_task(activity.wait())
            pending_readers = [task for task in readers if not task.done()]
            done, _ = await asyncio.wait(
                [activity_task, *pending_readers],
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not activity_task.done():
                activity_task.cancel()
            await self._settle_tasks([activity_task])
            if not done and loop.time() - max(exited_at, last_output_at[0]) >= (
                self.exit_stdio_grace_seconds
            ):
                await self._kill_process_group(process)
                return

    async def _kill_process_group(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        if os.name == "posix" and process.pid is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
        elif process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

        if process.returncode is None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(process.wait()),
                    timeout=self.cleanup_timeout_seconds,
                )
            except (asyncio.TimeoutError, ProcessLookupError):
                pass

    async def _settle_tasks(
        self,
        tasks: Sequence[asyncio.Task[object]],
    ) -> None:
        pending = [task for task in tasks if not task.done()]
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=self.cleanup_timeout_seconds,
                )
            except asyncio.TimeoutError:
                for task in pending:
                    task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
