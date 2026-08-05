from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent.pi_worker_client import (
    JsonlWorkerClient,
    SandboxedWorkerConfig,
    build_sandboxed_worker_launch,
)
from agent.pi_harness import PiValidationHarnessConfig, verify_agent_sandbox_access


def test_restricted_backend_import_does_not_load_pipeline_workflow() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import agent_tools.restricted_backend; "
                "assert 'workflow.skill_adapter' not in sys.modules"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_pi_runtime_does_not_import_agent_tools_compatibility_package() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import agent.pi_runtime; "
                "assert not any(name == 'agent_tools' or "
                "name.startswith('agent_tools.') for name in sys.modules)"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_agent_tools_package_keeps_lazy_public_exports() -> None:
    import agent_tools

    assert agent_tools.EngineerTools.__name__ == "EngineerTools"
    assert agent_tools.ReferenceVariantTools.__name__ == "ReferenceVariantTools"
    assert callable(agent_tools.register_engineer_atomic_tools)
    assert callable(agent_tools.register_reference_variant_tools)


def test_jsonl_worker_client_reuses_one_process(tmp_path: Path) -> None:
    script = tmp_path / "dummy_worker.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'event': 'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    if req['command'] == 'close':\n"
        "        print(json.dumps({'id': req['id'], 'result': {'status': 'CLOSED'}}), flush=True)\n"
        "        break\n"
        "    print(json.dumps({'id': req['id'], 'result': {'prompt': req.get('prompt', '')}}), flush=True)\n",
        encoding="utf-8",
    )

    async def exercise() -> None:
        client = JsonlWorkerClient(
            command=[sys.executable, "-u", str(script)],
            cwd=tmp_path,
            env={},
        )
        await client.start()
        process = client.process
        assert (await client.request("run_turn", prompt="first"))["prompt"] == "first"
        assert (await client.request("run_turn", prompt="second"))["prompt"] == "second"
        assert client.process is process
        await client.close()
        await client.close()
        assert process is not None and process.returncode == 0
        assert client.process is None

    asyncio.run(exercise())


def test_worker_interrupt_escalates_to_sigterm_and_reaps(tmp_path: Path) -> None:
    marker = tmp_path / "sigterm.txt"
    script = tmp_path / "ignore_sigint.py"
    script.write_text(
        "import json, signal, sys\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        f"def stop(*_args): Path({str(marker)!r}).write_text('term'); raise SystemExit(143)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print(json.dumps({'event': 'ready'}), flush=True)\n"
        "for _line in sys.stdin: pass\n",
        encoding="utf-8",
    )

    async def exercise() -> tuple[float, int | None]:
        client = JsonlWorkerClient(
            command=[sys.executable, "-u", str(script)],
            cwd=tmp_path,
            env={},
            shutdown_timeout=0.05,
        )
        await client.start()
        process = client.process
        assert process is not None
        started = time.monotonic()
        try:
            await asyncio.wait_for(client.interrupt(), timeout=1.0)
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        return time.monotonic() - started, process.returncode

    elapsed, returncode = asyncio.run(exercise())

    assert elapsed < 1.0
    assert marker.read_text(encoding="utf-8") == "term"
    assert returncode is not None


def test_worker_close_escalates_to_sigkill_and_is_idempotent(tmp_path: Path) -> None:
    script = tmp_path / "ignore_shutdown.py"
    script.write_text(
        "import json, signal, sys, time\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print(json.dumps({'event': 'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        " request = json.loads(line)\n"
        " if request.get('command') == 'close': time.sleep(60)\n",
        encoding="utf-8",
    )

    async def exercise() -> tuple[float, int | None]:
        client = JsonlWorkerClient(
            command=[sys.executable, "-u", str(script)],
            cwd=tmp_path,
            env={},
            shutdown_timeout=0.05,
        )
        await client.start()
        process = client.process
        assert process is not None
        started = time.monotonic()
        try:
            await asyncio.wait_for(client.close(), timeout=1.0)
            await client.close()
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        return time.monotonic() - started, process.returncode

    elapsed, returncode = asyncio.run(exercise())

    assert elapsed < 1.0
    assert returncode == -signal.SIGKILL


def test_worker_keeps_process_handle_when_sigkill_cannot_be_reaped(tmp_path: Path) -> None:
    async def exercise() -> None:
        client = JsonlWorkerClient(
            command=[sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env={},
        )

        class FakeProcess:
            pid = 12345
            returncode = None

        process = FakeProcess()
        client.process = process  # type: ignore[assignment]
        client._signal_process_group = lambda *_args: None  # type: ignore[method-assign]

        async def never_exits(_process) -> bool:
            return False

        client._wait_for_exit = never_exits  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="did not exit after SIGKILL"):
            await client.interrupt(force=True)
        assert client.process is process

    asyncio.run(exercise())


def test_sandboxed_launch_uses_marker_and_never_authorizes_gold(tmp_path: Path) -> None:
    project = tmp_path / "project"
    workdir = tmp_path / "experiment/agent_workdir"
    runtime = tmp_path / "experiment/host/runtime"
    train_raw = tmp_path / "dataset/train/raw"
    train_reference = tmp_path / "dataset/train/reference"
    validation_raw = tmp_path / "dataset/validation/raw"
    gold = tmp_path / "dataset/validation/reference_private"
    for path in (
        project / "agent",
        project / "agent_tools",
        project / "lib",
        project / "workflow",
        workdir,
        runtime,
        train_raw,
        train_reference,
        validation_raw,
        gold,
    ):
        path.mkdir(parents=True, exist_ok=True)
    (project / "config_loader.py").write_text("", encoding="utf-8")
    (project / "model_config.yaml").write_text("", encoding="utf-8")

    launch = build_sandboxed_worker_launch(
        SandboxedWorkerConfig(
            project_root=project,
            agent_workdir=workdir,
            runtime_root=runtime,
            public_read_roots=(train_raw, train_reference, validation_raw),
            skill_dirs=(),
            max_iters=25,
            model_environment={"OPENAI_API_KEYS_JSON": '["secret"]'},
        ),
    )

    assert launch.command[:3] == ["/usr/bin/sandbox-exec", "-f", str(launch.profile_path)]
    assert launch.env["PI_HARNESS_SANDBOX"] == "1"
    assert launch.env["V2_SKIP_LOCAL_MODEL_CONFIG"] == "1"
    assert launch.env["OPENAI_API_KEYS_JSON"] == '["secret"]'
    assert "--max-iters" in launch.command
    profile = launch.profile_path.read_text(encoding="utf-8")
    assert f'(allow file-read* (literal "{project.resolve()}"))' in profile
    assert f'(subpath "{project.resolve()}")' not in profile
    assert str((project / "agent_tools").resolve()) not in profile
    assert str((project / "workflow").resolve()) not in profile
    assert str(train_raw.resolve()) in profile
    assert str(train_reference.resolve()) in profile
    assert str(validation_raw.resolve()) in profile
    assert str(gold.resolve()) not in profile
    assert "model_config.local.yaml" not in profile
    assert '(subpath "/private/var/select")' in profile
    for literal_file in (project / "config_loader.py", project / "model_config.yaml"):
        assert f'(allow file-read* (literal "{literal_file.resolve()}"))' in profile
        assert f'(subpath "{literal_file.resolve()}")' not in profile


def test_real_sandbox_probe_denies_gold_python_shell_and_symlink(tmp_path: Path) -> None:
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        return
    project = tmp_path / "project"
    workdir = tmp_path / "experiment/agent_workdir"
    runtime = tmp_path / "experiment/host/runtime"
    train_raw = tmp_path / "dataset/train/raw"
    train_reference = tmp_path / "dataset/train/reference"
    validation_raw = tmp_path / "dataset/validation/raw"
    gold = tmp_path / "dataset/validation/reference_private"
    for path in (project / "agent", project / "lib", workdir, runtime, train_raw, train_reference, validation_raw, gold):
        path.mkdir(parents=True, exist_ok=True)
    (project / "config_loader.py").write_text("", encoding="utf-8")
    (project / "model_config.yaml").write_text("", encoding="utf-8")
    (train_raw / "raw.csv").write_text("dirty", encoding="utf-8")
    (train_reference / "reference.csv").write_text("clean", encoding="utf-8")
    (validation_raw / "public.csv").write_text("public", encoding="utf-8")
    (gold / "gold.csv").write_text("secret", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    launch = build_sandboxed_worker_launch(
        SandboxedWorkerConfig(
            project_root=project,
            agent_workdir=workdir,
            runtime_root=runtime,
            public_read_roots=(train_raw.parent, validation_raw),
            skill_dirs=(),
            max_iters=1,
            model_environment={},
        ),
    )
    config = PiValidationHarnessConfig(
        project_root=project,
        experiment_dir=tmp_path / "experiment",
        train_raw=train_raw,
        train_reference=train_reference,
        validation_raw=validation_raw,
        validation_gold=gold,
        evaluation_manifest=manifest,
        dataset_manifest=manifest,
    )

    asyncio.run(verify_agent_sandbox_access(config, launch))


def test_real_project_worker_import_respects_runtime_code_boundary(tmp_path: Path) -> None:
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        return
    project = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "experiment/agent_workdir"
    runtime = tmp_path / "experiment/host/runtime"
    public = tmp_path / "dataset/public"
    public.mkdir(parents=True)
    launch = build_sandboxed_worker_launch(
        SandboxedWorkerConfig(
            project_root=project,
            agent_workdir=workdir,
            runtime_root=runtime,
            public_read_roots=(public,),
            skill_dirs=(),
            max_iters=1,
            model_environment={},
        ),
    )
    profile = launch.profile_path.read_text(encoding="utf-8")
    assert str((project / "agent_tools").resolve()) not in profile
    assert str((project / "workflow").resolve()) not in profile
    completed = subprocess.run(
        [
            "/usr/bin/sandbox-exec",
            "-f",
            str(launch.profile_path),
            sys.executable,
            "-c",
            (
                "import sys; "
                "import agent.pi_worker; "
                "assert not any(name == 'agent_tools' or "
                "name.startswith('agent_tools.') or name == 'workflow' or "
                "name.startswith('workflow.') for name in sys.modules); "
                "print('PI_WORKER_SANDBOX_IMPORT_OK')"
            ),
        ],
        cwd=workdir,
        env=launch.env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PI_WORKER_SANDBOX_IMPORT_OK"

    shell = subprocess.run(
        [
            "/usr/bin/sandbox-exec",
            "-f",
            str(launch.profile_path),
            "/bin/sh",
            "-c",
            "printf PI_WORKER_SANDBOX_SHELL_OK",
        ],
        cwd=workdir,
        env=launch.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert shell.returncode == 0, shell.stderr
    assert shell.stdout == "PI_WORKER_SANDBOX_SHELL_OK"
    assert "Error opening /private/var/select/sh" not in shell.stderr
