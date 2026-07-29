"""Independent frozen-pipeline Test Harness for the Pi-style validation runtime."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.pi_harness import (
    _RedactingTee,
    directory_sha256,
    load_submission,
    render_replay_argv,
    snapshot_agent_workdir,
    validate_plain_directory_tree,
)
from agent.pi_harness_sandbox import build_macos_sandbox_profile
from agent.pi_worker_client import (
    JsonlWorkerClient,
    SandboxedWorkerConfig,
    WorkerLaunch,
    build_sandboxed_worker_launch,
)
from lib.agent_runtime import build_worker_model_environment
from workflow.pi_harness_evaluation import (
    SCORER_VERSION,
    load_evaluation_manifest,
    score_equal_weight_reference_directory,
)


RUNNER_PLACEHOLDERS = {"frozen_root", "raw_root", "output_dir", "train_reference"}
REQUIRED_RUNNER_PLACEHOLDERS = {"frozen_root", "raw_root", "output_dir"}
SHELL_LAUNCHERS = {"bash", "dash", "fish", "ksh", "sh", "zsh"}
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True)
class PiTestHarnessConfig:
    project_root: Path
    validation_experiment: Path
    test_experiment: Path
    test_raw: Path
    test_gold: Path
    evaluation_manifest: Path
    max_declaration_repairs: int = 3
    replay_timeout_seconds: float = 1_800.0
    max_iters: int = 10_000


@dataclass(frozen=True)
class RunnerSpec:
    path: Path
    frozen_snapshot_sha256: str
    replay_argv: tuple[str, ...]


@dataclass(frozen=True)
class TestReplayExecution:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    result_root: Path | None


@dataclass(frozen=True)
class PiTestHarnessResult:
    status: str
    score: float
    declaration_agent_used: bool
    declaration_rounds: int
    replay_status: str
    frozen_snapshot: Path
    result_package: Path | None


ReplayRunner = Callable[[Path, Path | None], Awaitable[TestReplayExecution]]
Scorer = Callable[[str | Path, str | Path, dict[str, tuple[str, ...]]], dict[str, Any]]


def load_runner_spec(path: str | Path, frozen_snapshot: str | Path) -> RunnerSpec:
    spec_path = Path(path).expanduser().resolve()
    frozen = Path(frozen_snapshot).expanduser().resolve()
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("runner_spec.json is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"runner_spec.json is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("runner_spec schema_version must be 1")
    unknown_fields = sorted(
        set(payload) - {"schema_version", "frozen_snapshot_sha256", "replay_argv"}
    )
    if unknown_fields:
        raise ValueError("runner_spec contains unsupported fields: " + ", ".join(unknown_fields))
    expected_hash = directory_sha256(frozen)
    actual_hash = str(payload.get("frozen_snapshot_sha256") or "")
    if actual_hash != expected_hash:
        raise ValueError(
            f"frozen snapshot hash mismatch: expected={expected_hash}, actual={actual_hash}"
        )
    argv = payload.get("replay_argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("runner_spec replay_argv must be a non-empty string array")
    if Path(argv[0]).name.lower() in SHELL_LAUNCHERS:
        raise ValueError("runner_spec cannot use a shell launcher")
    if argv[0] not in {"python", "python3"}:
        raise ValueError("runner_spec must invoke the frozen snapshot with Python")
    placeholders = {name for item in argv for name in PLACEHOLDER.findall(item)}
    unknown = sorted(placeholders - RUNNER_PLACEHOLDERS)
    if unknown:
        raise ValueError("runner_spec contains unknown placeholder: " + ", ".join(unknown))
    missing = sorted(REQUIRED_RUNNER_PLACEHOLDERS - placeholders)
    if missing:
        raise ValueError("runner_spec is missing required placeholder: " + ", ".join(missing))
    if len(argv) < 2 or set(PLACEHOLDER.findall(argv[1])) != {"frozen_root"}:
        raise ValueError("runner_spec Python entrypoint must be inside the frozen snapshot")
    entrypoint = Path(argv[1].format_map({"frozen_root": str(frozen)})).resolve()
    if entrypoint != frozen and frozen not in entrypoint.parents:
        raise ValueError("runner_spec Python entrypoint escapes the frozen snapshot")
    if not entrypoint.is_file():
        raise ValueError(f"runner_spec frozen entrypoint does not exist: {entrypoint}")
    for item in argv:
        static = PLACEHOLDER.sub("placeholder", item)
        if Path(static).is_absolute() and not PLACEHOLDER.search(item):
            raise ValueError("runner_spec cannot contain static absolute paths")
    return RunnerSpec(spec_path, actual_hash, tuple(argv))


def render_runner_argv(
    spec: RunnerSpec,
    *,
    frozen_root: Path,
    raw_root: Path,
    output_dir: Path,
    train_reference: Path,
) -> list[str]:
    values = {
        "frozen_root": str(frozen_root.resolve()),
        "raw_root": str(raw_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "train_reference": str(train_reference.resolve()),
    }
    return [item.format_map(values) for item in spec.replay_argv]


class PiTestHarness:
    def __init__(
        self,
        config: PiTestHarnessConfig,
        *,
        replay_runner: ReplayRunner | None = None,
        scorer: Scorer = score_equal_weight_reference_directory,
        worker_factory: Callable[..., Any] | None = None,
        sandbox_probe: Callable[..., Any] | None = None,
        output: Any | None = None,
    ) -> None:
        self.config = config
        self.output = output or sys.stderr
        self.experiment = config.test_experiment.expanduser().resolve()
        self.host_dir = self.experiment / "host"
        self.frozen_snapshot = self.host_dir / "frozen_snapshot"
        self.test_agent_workdir = self.experiment / "test_agent_workdir"
        self.replay_runner = replay_runner or self._run_replay
        self.scorer = scorer
        self.worker_factory = worker_factory
        self.sandbox_probe = sandbox_probe or verify_test_agent_sandbox_access
        self._worker_log_handle: Any | None = None
        self.train_reference = Path()

    async def run(self) -> PiTestHarnessResult:
        frozen_hash = self._prepare()
        declaration_used = False
        declaration_rounds = 0
        replay = await self.replay_runner(self.frozen_snapshot, None)
        worker: Any | None = None
        launch: WorkerLaunch | None = None
        try:
            if replay.status != "SUCCESS":
                declaration_used = True
                worker, launch = self._create_worker(frozen_hash)
                probe = self.sandbox_probe(self.config, launch, self.frozen_snapshot, self.test_agent_workdir)
                if inspect.isawaitable(probe):
                    await probe
                await worker.start()
                feedback = self._initial_prompt(frozen_hash, replay)
                for declaration_rounds in range(1, self.config.max_declaration_repairs + 1):
                    await worker.run_turn(feedback)
                    try:
                        self._validate_test_agent_files()
                        spec = load_runner_spec(
                            self.test_agent_workdir / "runner_spec.json",
                            self.frozen_snapshot,
                        )
                    except Exception as exc:
                        feedback = self._repair_prompt(f"{type(exc).__name__}: {exc}")
                        continue
                    replay = await self.replay_runner(self.frozen_snapshot, spec.path)
                    if replay.status == "SUCCESS":
                        break
                    feedback = self._repair_prompt(
                        f"replay status={replay.status}; returncode={replay.returncode}; "
                        f"stderr={replay.stderr[-2000:]}"
                    )
        finally:
            if worker is not None:
                await worker.close()
            if self._worker_log_handle is not None:
                self._worker_log_handle.close()
                self._worker_log_handle = None

        observed_frozen_hash = directory_sha256(self.frozen_snapshot)
        if observed_frozen_hash != frozen_hash:
            raise RuntimeError(
                "frozen Test snapshot changed during execution: "
                f"expected={frozen_hash}, actual={observed_frozen_hash}"
            )

        if replay.status != "SUCCESS" or replay.result_root is None:
            result = PiTestHarnessResult(
                status="REPLAY_FAILED",
                score=0.0,
                declaration_agent_used=declaration_used,
                declaration_rounds=declaration_rounds,
                replay_status=replay.status,
                frozen_snapshot=self.frozen_snapshot,
                result_package=replay.result_root,
            )
            self._write_json(self.host_dir / "run_report.json", _result_payload(result))
            return result

        # Gold is first opened here, after the declaration Agent has closed.
        key_columns = load_evaluation_manifest(self.config.evaluation_manifest, self.config.test_gold)
        score_report = self.scorer(replay.result_root, self.config.test_gold, key_columns)
        self._write_json(self.host_dir / "score_report.json", score_report)
        score = float((score_report.get("metrics") or {}).get("composite_score") or 0.0)
        result = PiTestHarnessResult(
            status="SUCCESS",
            score=score,
            declaration_agent_used=declaration_used,
            declaration_rounds=declaration_rounds,
            replay_status=replay.status,
            frozen_snapshot=self.frozen_snapshot,
            result_package=replay.result_root,
        )
        self._write_json(self.host_dir / "run_report.json", _result_payload(result))
        return result

    def _prepare(self) -> str:
        validation = self.config.validation_experiment.expanduser().resolve()
        source = validation / "host/reproducible_snapshot"
        validate_plain_directory_tree(source)
        if self.experiment.exists() and any(self.experiment.iterdir()):
            raise ValueError("test experiment directory must be new and empty")
        self.experiment.mkdir(parents=True, exist_ok=True)
        self.host_dir.mkdir()
        self.test_agent_workdir.mkdir()
        snapshot_agent_workdir(source, self.frozen_snapshot)
        source_hash = directory_sha256(source)
        frozen_hash = directory_sha256(self.frozen_snapshot)
        if source_hash != frozen_hash:
            raise RuntimeError("frozen Test snapshot hash verification failed")
        validation_manifest = json.loads(
            (validation / "host/run_manifest.json").read_text(encoding="utf-8")
        )
        self.train_reference = Path(str(validation_manifest.get("train_reference") or "")).expanduser().resolve()
        for path, label in (
            (self.train_reference, "Train reference"),
            (self.config.test_raw, "Test raw"),
            (self.config.test_gold, "Test gold"),
        ):
            if not path.is_dir():
                raise ValueError(f"{label} directory does not exist: {path}")
        if not self.config.evaluation_manifest.is_file():
            raise ValueError("evaluation manifest does not exist")
        if self.config.max_declaration_repairs < 1 or self.config.max_iters < 1:
            raise ValueError("Test declaration limits must be positive")
        self._write_json(
            self.host_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "validation_experiment": str(validation),
                "frozen_snapshot_sha256": frozen_hash,
                "test_raw": str(self.config.test_raw.resolve()),
                "evaluation_manifest": str(self.config.evaluation_manifest.resolve()),
                "evaluation_manifest_sha256": _sha256(self.config.evaluation_manifest),
                "max_declaration_repairs": self.config.max_declaration_repairs,
                "scorer_version": SCORER_VERSION,
            },
        )
        return frozen_hash

    def _create_worker(self, frozen_hash: str) -> tuple[Any, WorkerLaunch | None]:
        if self.worker_factory is not None:
            value = self.worker_factory(self.test_agent_workdir, frozen_hash)
            return value, None
        model_environment = build_worker_model_environment("react_planner", "MiniMax-M3")
        launch = build_sandboxed_worker_launch(
            SandboxedWorkerConfig(
                project_root=self.config.project_root,
                agent_workdir=self.test_agent_workdir,
                runtime_root=self.host_dir / "agent_runtime",
                public_read_roots=(self.frozen_snapshot, self.config.test_raw, self.train_reference),
                skill_dirs=(),
                max_iters=self.config.max_iters,
                model_environment=model_environment,
                tool_profile="test_declaration",
                runner_spec_path=self.test_agent_workdir / "runner_spec.json",
            )
        )
        log = (self.host_dir / "test_agent.log").open("a", encoding="utf-8")
        self._worker_log_handle = log
        keys = json.loads(model_environment.get("OPENAI_API_KEYS_JSON", "[]"))
        stream = _RedactingTee((self.output, log), tuple(str(item) for item in keys))
        return JsonlWorkerClient(command=launch.command, cwd=launch.cwd, env=launch.env, output=stream), launch

    def _initial_prompt(self, frozen_hash: str, replay: TestReplayExecution) -> str:
        return f"""\
The frozen Validation pipeline could not be invoked on public Test raw data.
Frozen snapshot: {self.frozen_snapshot}
Frozen SHA-256: {frozen_hash}
Test raw: {self.config.test_raw.resolve()}
Train reference: {self.train_reference}
Write exactly {self.test_agent_workdir / 'runner_spec.json'} with schema_version=1,
frozen_snapshot_sha256, and replay_argv. Use only placeholders {{frozen_root}},
{{raw_root}}, {{output_dir}}, and optionally {{train_reference}}. Do not write scripts
or result files and do not modify the frozen snapshot.
Initial replay status: {replay.status}; returncode={replay.returncode}; stderr={replay.stderr[-2000:]}
"""

    def _repair_prompt(self, error: str) -> str:
        return (
            "The runner declaration is still invalid. Fix only runner_spec.json. "
            "Do not write scripts or modify the frozen snapshot.\nError: " + error
        )

    def _validate_test_agent_files(self) -> None:
        extras = [
            path.relative_to(self.test_agent_workdir).as_posix()
            for path in self.test_agent_workdir.rglob("*")
            if path.is_file()
            and path != self.test_agent_workdir / "runner_spec.json"
            and (not path.relative_to(self.test_agent_workdir).parts or path.relative_to(self.test_agent_workdir).parts[0] != ".agent_runs")
        ]
        if extras:
            raise ValueError("Test Agent wrote forbidden files: " + ", ".join(sorted(extras)))

    async def _run_replay(self, frozen: Path, runner_spec_path: Path | None) -> TestReplayExecution:
        runtime_root = self.host_dir / "replay_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        replay_root = Path(tempfile.mkdtemp(prefix="test-replay-", dir=runtime_root))
        started = time.monotonic()
        try:
            workdir = replay_root / "frozen"
            snapshot_agent_workdir(frozen, workdir)
            result_root = workdir / "result_package"
            if result_root.exists():
                shutil.rmtree(result_root)
            if runner_spec_path is None:
                try:
                    submission = load_submission(workdir)
                    result_root = submission.result_root
                    if result_root.exists():
                        if result_root.is_dir() and not result_root.is_symlink():
                            shutil.rmtree(result_root)
                        else:
                            result_root.unlink()
                    argv = render_replay_argv(
                        submission,
                        raw_root=self.config.test_raw,
                        train_reference=self.train_reference,
                        output_dir=result_root,
                        workdir=workdir,
                    )
                except Exception as exc:
                    return TestReplayExecution(
                        "INVALID_SUBMISSION", None, "", f"{type(exc).__name__}: {exc}",
                        time.monotonic() - started, None,
                    )
            else:
                try:
                    spec = load_runner_spec(runner_spec_path, frozen)
                    argv = render_runner_argv(
                        spec,
                        frozen_root=workdir,
                        raw_root=self.config.test_raw,
                        output_dir=result_root,
                        train_reference=self.train_reference,
                    )
                except Exception as exc:
                    return TestReplayExecution(
                        "INVALID_RUNNER_SPEC", None, "", f"{type(exc).__name__}: {exc}",
                        time.monotonic() - started, None,
                    )
            if argv[0] in {"python", "python3"}:
                argv[0] = sys.executable
            home = replay_root / "home"
            scratch = replay_root / "tmp"
            home.mkdir()
            scratch.mkdir()
            profile = replay_root / "test_replay.sb"
            profile.write_text(
                build_macos_sandbox_profile(
                    executable=Path(sys.executable),
                    read_roots=[
                        workdir,
                        self.config.test_raw,
                        self.train_reference,
                        Path(sys.prefix),
                        Path(sys.base_prefix),
                        Path("/System"),
                        Path("/usr"),
                        Path("/bin"),
                        Path("/sbin"),
                        Path("/private/etc"),
                    ],
                    write_roots=[workdir, home, scratch, Path("/dev/null")],
                    allow_network=False,
                ),
                encoding="utf-8",
            )
            process = await asyncio.create_subprocess_exec(
                "/usr/bin/sandbox-exec", "-f", str(profile), *argv,
                cwd=workdir,
                env={
                    "HOME": str(home),
                    "TMPDIR": str(scratch),
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
                    "LANG": os.environ.get("LANG", "en_US.UTF-8"),
                    "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr, timed_out = await _communicate_with_cleanup(
                process,
                timeout=self.config.replay_timeout_seconds,
            )
            if timed_out:
                return TestReplayExecution(
                    "TIMEOUT", process.returncode, stdout.decode(errors="replace")[-8000:],
                    stderr.decode(errors="replace")[-8000:], time.monotonic() - started, None,
                )
            if process.returncode != 0:
                return TestReplayExecution(
                    "EXECUTION_FAILED", process.returncode, stdout.decode(errors="replace")[-8000:],
                    stderr.decode(errors="replace")[-8000:], time.monotonic() - started, None,
                )
            try:
                validate_plain_directory_tree(result_root)
            except ValueError as exc:
                return TestReplayExecution(
                    "INVALID_OUTPUT",
                    0,
                    stdout.decode(errors="replace")[-8000:],
                    str(exc),
                    time.monotonic() - started,
                    None,
                )
            retained = self.host_dir / "test_result_package"
            if retained.exists():
                shutil.rmtree(retained)
            shutil.copytree(result_root, retained)
            return TestReplayExecution(
                "SUCCESS", 0, stdout.decode(errors="replace")[-8000:],
                stderr.decode(errors="replace")[-8000:], time.monotonic() - started, retained,
            )
        finally:
            shutil.rmtree(replay_root, ignore_errors=True)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def verify_test_agent_sandbox_access(
    config: PiTestHarnessConfig,
    launch: WorkerLaunch | None,
    frozen_snapshot: Path,
    agent_workdir: Path,
) -> None:
    if launch is None:
        raise RuntimeError("sandbox launch is required for the Test Agent probe")
    public_file = _first_file(config.test_raw)
    frozen_file = _first_file(frozen_snapshot)
    gold_file = _first_file(config.test_gold)
    link = agent_workdir / ".test_gold_probe"
    link.symlink_to(gold_file)
    script = (
        "import json,pathlib,sys\n"
        "def readable(p):\n"
        " try: pathlib.Path(p).open('rb').read(1); return True\n"
        " except OSError: return False\n"
        "print(json.dumps([readable(p) for p in sys.argv[1:]]))\n"
    )
    env = {key: value for key, value in launch.env.items() if key != "OPENAI_API_KEYS_JSON"}
    try:
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/sandbox-exec", "-f", str(launch.profile_path), sys.executable, "-c", script,
            str(public_file), str(frozen_file), str(gold_file), str(link),
            cwd=agent_workdir, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Test Agent sandbox probe failed: {stderr.decode(errors='replace')[-1000:]}")
        if json.loads(stdout) != [True, True, False, False]:
            raise RuntimeError(f"unsafe Test Agent sandbox access: {stdout.decode(errors='replace')}")
    finally:
        link.unlink(missing_ok=True)


def _first_file(root: Path) -> Path:
    for path in sorted(root.resolve().rglob("*")):
        if path.is_file() and not path.is_symlink():
            return path
    raise ValueError(f"directory contains no regular files: {root}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_payload(result: PiTestHarnessResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "score": result.score,
        "declaration_agent_used": result.declaration_agent_used,
        "declaration_rounds": result.declaration_rounds,
        "replay_status": result.replay_status,
        "frozen_snapshot": str(result.frozen_snapshot),
        "result_package": str(result.result_package) if result.result_package else "",
    }


async def _communicate_with_cleanup(
    process: Any,
    *,
    timeout: float,
) -> tuple[bytes, bytes, bool]:
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return stdout, stderr, False
    except asyncio.TimeoutError:
        _kill_process_group(process)
        stdout, stderr = await process.communicate()
        return stdout, stderr, True
    except BaseException:
        _kill_process_group(process)
        await process.wait()
        raise


def _kill_process_group(process: Any) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
