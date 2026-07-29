from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

from agentscope.tool import ExecResult, LocalBackend


def _roots(values: Iterable[str | Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if path not in result:
            result.append(path)
    return tuple(result)


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class RestrictedLocalBackend(LocalBackend):
    """AgentScope native-tool backend constrained to explicit path roots."""

    def __init__(
        self,
        *,
        read_roots: Iterable[str | Path],
        denied_read_roots: Iterable[str | Path] = (),
        write_roots: Iterable[str | Path],
        cwd: str | Path,
    ) -> None:
        super().__init__()
        self.read_roots = _roots(read_roots)
        self.denied_read_roots = _roots(denied_read_roots)
        self.write_roots = _roots(write_roots)
        self.cwd = Path(cwd).expanduser().resolve()
        if not any(_within(self.cwd, root) for root in self.read_roots):
            raise ValueError("cwd must be inside an authorized read root")

    def _resolve(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.cwd / path
        return path.resolve()

    def _require_read(self, value: str | os.PathLike[str]) -> Path:
        path = self._resolve(value)
        if any(_within(path, root) for root in self.denied_read_roots):
            raise PermissionError(f"read is within a denied read root: {path}")
        if not any(_within(path, root) for root in self.read_roots):
            raise PermissionError(f"read outside authorized roots: {path}")
        return path

    def _require_write(self, value: str | os.PathLike[str]) -> Path:
        path = self._resolve(value)
        if not any(_within(path, root) for root in self.write_roots):
            raise PermissionError(f"write outside authorized roots: {path}")
        return path

    async def getcwd(self) -> str:
        return str(self.cwd)

    async def read_file(self, path: str) -> bytes:
        return await super().read_file(str(self._require_read(path)))

    async def write_file(self, path: str, data: bytes) -> None:
        await super().write_file(str(self._require_write(path)), data)

    async def file_exists(self, path: str) -> bool:
        return await super().file_exists(str(self._require_read(path)))

    async def is_dir(self, path: str) -> bool:
        return await super().is_dir(str(self._require_read(path)))

    async def list_dir(self, path: str, *, recursive: bool = False) -> list[str]:
        return await super().list_dir(str(self._require_read(path)), recursive=recursive)

    async def stat_mtime(self, path: str) -> float | None:
        return await super().stat_mtime(str(self._require_read(path)))

    async def delete_path(self, path: str) -> None:
        await super().delete_path(str(self._require_write(path)))

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        checked_cwd = str(self._require_read(cwd or self.cwd))
        if not command:
            raise PermissionError("empty backend command")
        executable = Path(command[0]).name
        if executable == "rg":
            self._require_read(command[-1])
        elif executable == "mkdir":
            if len(command) != 3 or command[1] != "-p":
                raise PermissionError("only mkdir -p <authorized-write-path> is allowed")
            self._require_write(command[2])
        elif Path(command[0]).resolve() == Path(sys.executable).resolve():
            if len(command) < 2 or Path(command[1]).name != "_glob_helper.py":
                raise PermissionError("only the AgentScope Glob helper is allowed")
            if "--base-dir" not in command:
                raise PermissionError("Glob helper command is missing --base-dir")
            index = command.index("--base-dir")
            if index + 1 >= len(command):
                raise PermissionError("Glob helper command has no base directory")
            self._require_read(command[index + 1])
        else:
            raise PermissionError(f"backend command is not allowed: {command[0]}")
        return await super().exec_shell(command, cwd=checked_cwd, timeout=timeout)
