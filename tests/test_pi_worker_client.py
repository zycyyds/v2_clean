from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from agent.pi_worker_client import (
    JsonlWorkerClient,
    SandboxedWorkerConfig,
    build_sandboxed_worker_launch,
)
from agent.pi_harness import PiValidationHarnessConfig, verify_agent_sandbox_access


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
        assert process is not None and process.returncode == 0

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
        project / "lib",
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
    assert launch.env["OPENAI_API_KEYS_JSON"] == '["secret"]'
    assert "--max-iters" in launch.command
    profile = launch.profile_path.read_text(encoding="utf-8")
    assert f'(allow file-read* (literal "{project.resolve()}"))' in profile
    assert f'(subpath "{project.resolve()}")' not in profile
    assert str(train_raw.resolve()) in profile
    assert str(train_reference.resolve()) in profile
    assert str(validation_raw.resolve()) in profile
    assert str(gold.resolve()) not in profile
    assert "model_config.local.yaml" not in profile


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
    (validation_raw / "public.csv").write_text("public", encoding="utf-8")
    (gold / "gold.csv").write_text("secret", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    launch = build_sandboxed_worker_launch(
        SandboxedWorkerConfig(
            project_root=project,
            agent_workdir=workdir,
            runtime_root=runtime,
            public_read_roots=(train_raw, train_reference, validation_raw),
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
    )

    asyncio.run(verify_agent_sandbox_access(config, launch))
