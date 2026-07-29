from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest

from agent.pi_harness_sandbox import build_macos_sandbox_profile
from agent.pi_worker import require_sandbox_launcher, serve_worker


@dataclass
class _Turn:
    status: str = "SUCCESS"
    text: str = "done"
    finished_reason: str = "completed"
    model_calls: int = 1
    input_tokens: int = 2
    output_tokens: int = 3
    react_iterations: int = 4
    duration_seconds: float = 0.1


class _FakeToolContext:
    def __init__(self) -> None:
        self.cleared = 0

    async def clean_file_cache(self) -> None:
        self.cleared += 1


class _FakeState:
    def __init__(self) -> None:
        self.tool_context = _FakeToolContext()


class _FakeAgent:
    def __init__(self) -> None:
        self.state = _FakeState()


class _FakeRuntime:
    def __init__(self) -> None:
        self.agent = _FakeAgent()
        self.prompts: list[str] = []
        self.initialized = 0
        self.closed = 0

    async def initialize(self) -> None:
        self.initialized += 1

    async def run_turn(self, prompt: str) -> _Turn:
        self.prompts.append(prompt)
        return _Turn(text=f"done:{prompt}")

    async def close(self) -> None:
        self.closed += 1


def test_worker_reuses_runtime_and_supports_cache_clear() -> None:
    runtime = _FakeRuntime()
    reader = StringIO(
        "\n".join(
            [
                json.dumps({"id": 1, "command": "run_turn", "prompt": "first"}),
                json.dumps({"id": 2, "command": "clear_file_cache"}),
                json.dumps({"id": 3, "command": "run_turn", "prompt": "second"}),
                json.dumps({"id": 4, "command": "close"}),
            ],
        )
        + "\n",
    )
    writer = StringIO()

    asyncio.run(serve_worker(runtime, reader=reader, writer=writer))

    messages = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert messages[0] == {"event": "ready"}
    assert messages[1]["id"] == 1
    assert messages[1]["result"]["text"] == "done:first"
    assert messages[2] == {"id": 2, "result": {"status": "SUCCESS"}}
    assert messages[3]["result"]["text"] == "done:second"
    assert messages[4] == {"id": 4, "result": {"status": "CLOSED"}}
    assert runtime.initialized == 1
    assert runtime.prompts == ["first", "second"]
    assert runtime.agent.state.tool_context.cleared == 1
    assert runtime.closed == 1


def test_worker_refuses_direct_unsandboxed_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_HARNESS_SANDBOX", raising=False)
    with pytest.raises(RuntimeError, match="trusted Harness launcher"):
        require_sandbox_launcher()

    monkeypatch.setenv("PI_HARNESS_SANDBOX", "1")
    require_sandbox_launcher()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec test")
def test_macos_sandbox_allows_public_and_denies_gold_and_symlink(tmp_path: Path) -> None:
    public = tmp_path / "public"
    gold = tmp_path / "gold"
    workdir = tmp_path / "workdir"
    home = tmp_path / "home"
    scratch = tmp_path / "tmp"
    for directory in (public, gold, workdir, home, scratch):
        directory.mkdir()
    (public / "visible.txt").write_text("visible", encoding="utf-8")
    (gold / "secret.txt").write_text("secret", encoding="utf-8")
    (workdir / "gold-link").symlink_to(gold, target_is_directory=True)

    profile = tmp_path / "agent.sb"
    profile.write_text(
        build_macos_sandbox_profile(
            executable=Path(sys.executable),
            read_roots=[public, workdir, Path(sys.prefix), Path("/System"), Path("/usr/lib")],
            write_roots=[workdir, home, scratch, Path("/dev/null")],
            allow_network=False,
        ),
        encoding="utf-8",
    )

    script = (
        "from pathlib import Path; import json; "
        f"paths={[str(public / 'visible.txt'), str(gold / 'secret.txt'), str(workdir / 'gold-link/secret.txt')]!r}; "
        "out=[]; "
        "\nfor p in paths:\n"
        " try: out.append(Path(p).read_text())\n"
        " except Exception as e: out.append(type(e).__name__)\n"
        "print(json.dumps(out))"
    )
    completed = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable, "-c", script],
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == ["visible", "PermissionError", "PermissionError"]
