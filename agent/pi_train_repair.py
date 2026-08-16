from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.pi_harness import (
    _RedactingTee,
    _close_worker_without_losing_cancellation,
    _directory_bundle_sha256,
    _file_sha256,
    _git_identity,
    _public_model_identity,
    _text_sha256,
    _kill_and_reap_process_group,
    directory_sha256,
    snapshot_agent_workdir,
    validate_plain_directory_tree,
)
from agent.pi_harness_sandbox import (
    build_macos_sandbox_profile,
    require_path_isolated,
    sandbox_visible_recursive_roots,
)
from agent.pi_worker_client import (
    JsonlWorkerClient,
    SandboxedWorkerConfig,
    WorkerLaunch,
    build_sandboxed_worker_launch,
)
from lib.agent_runtime import build_worker_model_environment
from workflow.pi_repair_audit import (
    PiRepairAuditError,
    load_repair_submission,
    render_repair_replay_argv,
    score_repair_output,
)


@dataclass(frozen=True)
class PiTrainRepairConfig:
    project_root: Path
    experiment_dir: Path
    public_train_root: Path
    train_gold_log: Path
    train_row_gold_log: Path
    max_iters: int = 10_000
    max_repair_turns: int = 2
    skill_dirs: tuple[Path, ...] = ()
    replay_timeout_seconds: float = 1_800.0


@dataclass(frozen=True)
class RepairReplayExecution:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    output_root: Path | None
    score_report: dict[str, Any] | None


@dataclass(frozen=True)
class PiTrainRepairResult:
    status: str
    repair_turns: int
    frozen_snapshot: Path
    frozen_snapshot_sha256: str
    output_root: Path | None
    train_report: Path | None


ReplayRunner = Callable[[Path, Path], Awaitable[RepairReplayExecution]]

_FROZEN_SOURCE_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def default_repair_skill_dirs(project_root: str | Path) -> tuple[Path, ...]:
    root = Path(project_root).expanduser().resolve() / "skills"
    return tuple(
        root / name
        for name in (
            "correction_intra_table_errors",
            "correction_entity_alignment_errors",
            "correction_cross_table_errors",
            "correction_task_oriented_errors",
        )
    )


class PiTrainRepairHarness:
    """One Train-only Agent session followed by a reproducible frozen replay."""

    def __init__(
        self,
        config: PiTrainRepairConfig,
        *,
        worker: Any | None = None,
        replay_runner: ReplayRunner | None = None,
        sandbox_probe: Callable[..., Any] | None = None,
        output: Any | None = None,
    ) -> None:
        self.config = config
        self.worker = worker
        self.replay_runner = replay_runner or self._run_replay
        self.sandbox_probe = sandbox_probe or verify_train_repair_sandbox_access
        self.output = output or sys.stderr
        self.experiment = config.experiment_dir.expanduser().resolve()
        self.agent_workdir = self.experiment / "agent_workdir"
        self.host_dir = self.experiment / "host"
        self.frozen_snapshot = self.host_dir / "reproducible_snapshot"
        self.train_output = self.host_dir / "train_replay_output"
        self._worker_launch: WorkerLaunch | None = None
        self._worker_log_handle: Any | None = None
        self._model_environment: dict[str, str] | None = None
        self._turn_metrics: list[dict[str, Any]] = []

    async def run(self, task_prompt: str = "") -> PiTrainRepairResult:
        self._prepare(task_prompt)
        repair_turns = 0
        replay: RepairReplayExecution | None = None
        try:
            if self.worker is None:
                self.worker, self._worker_launch = self._create_worker()
            probe = self.sandbox_probe(self.config, self._worker_launch, self.agent_workdir)
            if asyncio.iscoroutine(probe):
                await probe
            await self.worker.start()
            prompt = self._initial_prompt(task_prompt)
            while True:
                self._host_line(
                    "Agent Train turn started"
                    if repair_turns == 0
                    else f"Train replay repair turn {repair_turns} started"
                )
                started = time.monotonic()
                turn = await self.worker.run_turn(prompt)
                self._turn_metrics.append(
                    {
                        "duration_seconds": round(time.monotonic() - started, 6),
                        **_public_turn_metrics(turn),
                    }
                )
                replay = await self.replay_runner(self.agent_workdir, self.train_output)
                self._write_replay_record(repair_turns, replay)
                if replay.status == "SUCCESS" and replay.score_report is not None:
                    break
                if repair_turns >= self.config.max_repair_turns:
                    result = PiTrainRepairResult(
                        status="TRAIN_REPLAY_FAILED",
                        repair_turns=repair_turns,
                        frozen_snapshot=self.frozen_snapshot,
                        frozen_snapshot_sha256="",
                        output_root=replay.output_root,
                        train_report=None,
                    )
                    self._write_run_report(result, replay)
                    return result
                repair_turns += 1
                prompt = self._repair_prompt(replay)

            freeze_repair_source(self.agent_workdir, self.frozen_snapshot)
            frozen_hash = directory_sha256(self.frozen_snapshot)
            report_path = self.train_output / "train_repair_report.json"
            _atomic_json(report_path, replay.score_report)
            result = PiTrainRepairResult(
                status="SUCCESS_REPRODUCIBLE",
                repair_turns=repair_turns,
                frozen_snapshot=self.frozen_snapshot,
                frozen_snapshot_sha256=frozen_hash,
                output_root=self.train_output,
                train_report=report_path,
            )
            self._write_run_report(result, replay)
            return result
        except asyncio.CancelledError:
            if self.worker is not None and hasattr(self.worker, "interrupt"):
                try:
                    await self.worker.interrupt()
                except Exception:
                    pass
            result = PiTrainRepairResult(
                status="INTERRUPTED",
                repair_turns=repair_turns,
                frozen_snapshot=self.frozen_snapshot,
                frozen_snapshot_sha256="",
                output_root=replay.output_root if replay else None,
                train_report=None,
            )
            self._write_run_report(result, replay)
            return result
        finally:
            if self.worker is not None:
                await _close_worker_without_losing_cancellation(self.worker)
            if self._worker_log_handle is not None:
                self._worker_log_handle.close()
                self._worker_log_handle = None

    def _prepare(self, task_prompt: str) -> None:
        if self.experiment.exists() and (
            not self.experiment.is_dir() or any(self.experiment.iterdir())
        ):
            raise ValueError("experiment directory must be new and empty")
        self.experiment.mkdir(parents=True, exist_ok=True)
        public = self.config.public_train_root.expanduser().resolve()
        manifest_path = public / "public_train_manifest.json"
        for relative in ("dirty_raw", "clean_raw", "evidence"):
            _require_directory(public / relative, f"public Train {relative}")
        manifest = json.loads(_require_file(manifest_path, "public Train manifest").read_text())
        supervision = manifest.get("supervision") or {}
        if manifest.get("status") != "SUCCESS" or (
            int(supervision.get("dirty_count") or 0),
            int(supervision.get("clean_count") or 0),
        ) != (20_000, 10_000):
            raise ValueError(
                "public Train manifest does not contain 20,000 dirty and 10,000 clean Cells"
            )
        for path, label in (
            (self.config.train_gold_log, "Train Gold log"),
            (self.config.train_row_gold_log, "Train row Gold log"),
        ):
            _require_file(path, label)
        if self.config.max_iters < 1 or self.config.max_repair_turns < 0:
            raise ValueError("max_iters must be positive and max_repair_turns non-negative")
        if (
            not math.isfinite(self.config.replay_timeout_seconds)
            or self.config.replay_timeout_seconds <= 0
        ):
            raise ValueError("replay timeout must be positive and finite")
        for skill in self.config.skill_dirs:
            _require_file(skill / "SKILL.md", "repair skill")

        self.host_dir.mkdir()
        (self.host_dir / "replays").mkdir()
        self.agent_workdir.mkdir()
        self._model_environment = build_worker_model_environment(
            "react_planner", "MiniMax-M3"
        )
        _atomic_json(
            self.host_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "mode": "pi_agent_train_only_repair",
                "public_train_manifest_sha256": _file_sha256(manifest_path),
                "train_gold_log_sha256": _file_sha256(self.config.train_gold_log),
                "train_row_gold_log_sha256": _file_sha256(
                    self.config.train_row_gold_log
                ),
                "prompt_sha256": _text_sha256(task_prompt),
                "git": _git_identity(self.config.project_root),
                "model": _public_model_identity(self._model_environment),
                "skills": {
                    "directories": [skill.name for skill in self.config.skill_dirs],
                    "bundle_sha256": _directory_bundle_sha256(self.config.skill_dirs),
                },
                "max_iters": self.config.max_iters,
                "max_repair_turns": self.config.max_repair_turns,
                "replay_timeout_seconds": self.config.replay_timeout_seconds,
                "test_data_visible_to_agent": False,
                "validation_used": False,
            },
        )

    def _create_worker(self) -> tuple[JsonlWorkerClient, WorkerLaunch]:
        environment = self._model_environment or build_worker_model_environment(
            "react_planner", "MiniMax-M3"
        )
        launch = build_sandboxed_worker_launch(
            SandboxedWorkerConfig(
                project_root=self.config.project_root,
                agent_workdir=self.agent_workdir,
                runtime_root=self.host_dir / "runtime",
                public_read_roots=(self.config.public_train_root,),
                skill_dirs=self.config.skill_dirs,
                max_iters=self.config.max_iters,
                model_environment=environment,
            )
        )
        worker_log = (self.host_dir / "worker.log").open("a", encoding="utf-8")
        self._worker_log_handle = worker_log
        try:
            secrets = json.loads(environment.get("OPENAI_API_KEYS_JSON", "[]"))
        except json.JSONDecodeError:
            secrets = []
        return (
            JsonlWorkerClient(
                command=launch.command,
                cwd=launch.cwd,
                env=launch.env,
                output=_RedactingTee(
                    (self.output, worker_log), tuple(str(item) for item in secrets)
                ),
            ),
            launch,
        )

    def _initial_prompt(self, task_prompt: str) -> str:
        public = self.config.public_train_root.resolve()
        prefix = task_prompt.strip()
        return f"""{prefix}

[Pi Train-only repair contract]
This is Agent-only raw-repair development. There is no Validation loop, graph detector, or
downstream formatting task.
You may read only this de-identified Train view:
- dirty raw: {public / 'dirty_raw'}
- clean raw: {public / 'clean_raw'}
- 20,000 dirty pairs plus 10,000 sampled clean examples: {public / 'evidence'}
- Agent workdir: {self.agent_workdir}

Use the loaded correction skills. Compare files with local scripts instead of pasting whole files
into model context. Build deterministic detection, candidate generation, candidate selection, and
raw correction. Do not build the 17-file formatting pipeline. Do not memorize row numbers,
patient/stay/admission tokens, or Train-specific mappings. The frozen pipeline must work without
model/network access and must not read clean raw, evidence, hidden Gold, or Test data at replay.

Create pipeline/run.py with this interface:
python pipeline/run.py --raw-root <raw> --output-dir <output> --split-mode train|test

Each output directory must contain corrected_raw/ and repair_candidates.jsonl.
repair_candidates.jsonl must be sorted by (table,row_index,column), use zero-based CSV data-row
indices, contain one record per selected repair, and use exactly these fields:
{{"table":"icu/inputevents.csv","row_index":0,"column":"amount",
 "original_value":"-999","candidates":[{{"value":"42.9","rule_id":"...",
 "evidence":"..."}}],"selected_value":"42.9"}}
Use column="", selected_value="__DELETE_ROW__", and a matching candidate for row deletion.
Return 1-5 unique candidates. Every actual raw change must be declared; undeclared files and Cells
must remain byte-identical.

Before ending, run the pipeline on public dirty raw and compare corrected_raw against public clean
raw. Write repair_submission.json:
{{
  "schema_version": 1,
  "current_output_root": "outputs/train",
  "outputs": {{
    "corrected_raw": "corrected_raw",
    "repair_candidates": "repair_candidates.jsonl"
  }},
  "replay": {{"argv": ["python", "pipeline/run.py", "--raw-root", "{{raw_root}}",
    "--output-dir", "{{output_dir}}", "--split-mode", "test"]}}
}}
The host will independently replay and audit Train after this turn. Test is not available to you.
""".strip()

    def _repair_prompt(self, replay: RepairReplayExecution) -> str:
        payload: dict[str, Any] = {
            "phase": "independent_train_replay",
            "status": replay.status,
            "returncode": replay.returncode,
            "stderr": replay.stderr[-4000:],
            "message": "Fix the replayable detector/repair pipeline using public Train only.",
        }
        if replay.score_report is not None:
            payload["metrics"] = replay.score_report.get("metrics") or {}
        return (
            "[Train replay feedback]\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\nContinue from current files, rerun public Train checks, and finish the contract."
        )

    async def _run_replay(
        self, source: Path, retained_output: Path
    ) -> RepairReplayExecution:
        public = self.config.public_train_root.resolve()
        execution = await run_frozen_repair_replay(
            source=source,
            raw_root=public / "dirty_raw",
            retained_output=retained_output,
            runtime_root=self.host_dir / "replay_runtime",
            timeout_seconds=self.config.replay_timeout_seconds,
        )
        if execution.status != "SUCCESS":
            return execution
        try:
            report = score_repair_output(
                output_root=retained_output,
                raw_root=public / "dirty_raw",
                gold_log=self.config.train_gold_log,
                row_gold_log=self.config.train_row_gold_log,
                pseudonymize_locator_values=True,
            )
            return RepairReplayExecution(
                "SUCCESS",
                execution.returncode,
                execution.stdout,
                execution.stderr,
                execution.duration_seconds,
                retained_output,
                report,
            )
        except Exception as exc:
            return RepairReplayExecution(
                "INVALID_OUTPUT" if isinstance(exc, PiRepairAuditError) else "REPLAY_FAILED",
                None,
                "",
                f"{type(exc).__name__}: {exc}",
                execution.duration_seconds,
                retained_output,
                None,
            )

    def _write_replay_record(
        self, repair_turn: int, replay: RepairReplayExecution
    ) -> None:
        _atomic_json(
            self.host_dir / "replays" / f"replay_{repair_turn:04d}.json",
            {
                "status": replay.status,
                "returncode": replay.returncode,
                "duration_seconds": round(replay.duration_seconds, 6),
                "stderr": replay.stderr[-4000:],
                "metrics": (replay.score_report or {}).get("metrics") or {},
            },
        )

    def _write_run_report(
        self,
        result: PiTrainRepairResult,
        replay: RepairReplayExecution | None,
    ) -> None:
        totals = {
            key: sum(int(turn.get(key) or 0) for turn in self._turn_metrics)
            for key in (
                "model_calls",
                "input_tokens",
                "output_tokens",
                "react_iterations",
                "tool_calls",
                "tool_errors",
                "api_failovers",
                "compactions",
            )
        }
        _atomic_json(
            self.host_dir / "run_report.json",
            {
                "status": result.status,
                "repair_turns": result.repair_turns,
                "frozen_snapshot_sha256": result.frozen_snapshot_sha256,
                "output_root": str(result.output_root) if result.output_root else "",
                "train_report": str(result.train_report) if result.train_report else "",
                "replay_status": replay.status if replay else "",
                "metrics": (replay.score_report or {}).get("metrics") if replay else {},
                "usage": totals,
                "turns": self._turn_metrics,
            },
        )

    def _host_line(self, message: str) -> None:
        print(f"[Pi Train Repair] {message}", file=self.output, flush=True)


async def verify_train_repair_sandbox_access(
    config: PiTrainRepairConfig,
    launch: WorkerLaunch | None,
    agent_workdir: Path,
) -> None:
    if launch is None:
        raise RuntimeError("sandbox launch is required for the Train repair probe")
    private_paths = (config.train_gold_log, config.train_row_gold_log)
    require_path_isolated(
        config.train_gold_log,
        recursive_roots=(config.public_train_root, agent_workdir),
        error_message="unsafe Train Gold path overlap",
    )
    links: list[Path] = []
    try:
        for index, private in enumerate(private_paths):
            link = agent_workdir / f".private_gold_probe_{index}"
            link.symlink_to(private.resolve())
            links.append(link)
        script = (
            "import json,pathlib,sys\n"
            "def readable(p):\n"
            "  try: pathlib.Path(p).open('rb').read(1); return True\n"
            "  except OSError: return False\n"
            "print(json.dumps([readable(p) for p in sys.argv[1:]]))\n"
        )
        environment = {
            key: value
            for key, value in launch.env.items()
            if key not in {"OPENAI_API_KEY", "OPENAI_API_KEYS_JSON"}
        }
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/sandbox-exec",
            "-f",
            str(launch.profile_path),
            sys.executable,
            "-c",
            script,
            *(str(path) for path in (*private_paths, *links)),
            cwd=str(agent_workdir),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                "Train repair sandbox probe failed: "
                + stderr.decode(errors="replace")[-1000:]
            )
        if json.loads(stdout) != [False, False, False, False]:
            raise RuntimeError("Train repair sandbox exposes private Gold")
    finally:
        for link in links:
            link.unlink(missing_ok=True)


def _public_turn_metrics(turn: Any) -> dict[str, Any]:
    if not isinstance(turn, dict):
        return {}
    return {
        key: turn.get(key, 0)
        for key in (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "react_iterations",
            "tool_calls",
            "tool_errors",
            "api_failovers",
            "compactions",
        )
    }


def freeze_repair_source(source: str | Path, destination: str | Path) -> None:
    """Freeze only replayable repair source, never Train outputs or run history."""

    root = Path(source).expanduser().resolve()
    validate_plain_directory_tree(root)
    load_repair_submission(root)
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise ValueError("frozen repair source destination already exists")
    target.mkdir(parents=True)
    pipeline = _require_directory(root / "pipeline", "repair pipeline")
    copied = 0
    for path in sorted(pipeline.rglob("*")):
        relative = path.relative_to(pipeline)
        if any(part == "__pycache__" or part.startswith(".") for part in relative.parts):
            continue
        output = target / "pipeline" / relative
        if path.is_dir():
            output.mkdir(parents=True, exist_ok=True)
            continue
        if path.suffix.lower() not in _FROZEN_SOURCE_SUFFIXES:
            raise ValueError(f"repair pipeline contains a non-source artifact: {relative}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
        copied += 1
    if copied == 0:
        raise ValueError("repair pipeline contains no source files")
    shutil.copyfile(root / "repair_submission.json", target / "repair_submission.json")
    load_repair_submission(target)


async def run_frozen_repair_replay(
    *,
    source: str | Path,
    raw_root: str | Path,
    retained_output: str | Path,
    runtime_root: str | Path,
    timeout_seconds: float,
) -> RepairReplayExecution:
    source_path = Path(source).expanduser().resolve()
    raw_path = Path(raw_root).expanduser().resolve()
    retained = Path(retained_output).expanduser().resolve()
    runtime = Path(runtime_root).expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    replay_root = Path(tempfile.mkdtemp(prefix="repair-replay-", dir=runtime))
    started = time.monotonic()
    try:
        validate_plain_directory_tree(source_path)
        replay_workdir = replay_root / "workdir"
        snapshot_agent_workdir(source_path, replay_workdir)
        submission = load_repair_submission(replay_workdir)
        if retained.exists():
            shutil.rmtree(retained)
        retained.mkdir(parents=True)
        home = replay_root / "home"
        scratch = replay_root / "tmp"
        home.mkdir()
        scratch.mkdir()
        profile = replay_root / "replay.sb"
        profile.write_text(
            build_macos_sandbox_profile(
                executable=Path(sys.executable),
                read_roots=sandbox_visible_recursive_roots(
                    replay_workdir,
                    raw_path,
                ),
                write_roots=[replay_workdir, retained, home, scratch, Path("/dev/null")],
                allow_network=False,
            ),
            encoding="utf-8",
        )
        argv = render_repair_replay_argv(
            submission,
            raw_root=raw_path,
            output_dir=retained,
            workdir=replay_workdir,
        )
        if argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile),
            *argv,
            cwd=str(replay_workdir),
            env=_offline_environment(home, scratch),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            stdout, stderr = await _kill_and_reap_process_group(process)
            return RepairReplayExecution(
                "TIMEOUT",
                process.returncode,
                stdout.decode(errors="replace")[-8000:],
                stderr.decode(errors="replace")[-8000:],
                time.monotonic() - started,
                retained,
                None,
            )
        except asyncio.CancelledError:
            await _kill_and_reap_process_group(process)
            raise
        stdout_text = stdout.decode(errors="replace")[-8000:]
        stderr_text = stderr.decode(errors="replace")[-8000:]
        if process.returncode != 0:
            return RepairReplayExecution(
                "EXECUTION_FAILED",
                process.returncode,
                stdout_text,
                stderr_text,
                time.monotonic() - started,
                retained,
                None,
            )
        for required in ("corrected_raw",):
            validate_plain_directory_tree(retained / required)
        if not (retained / "repair_candidates.jsonl").is_file():
            raise PiRepairAuditError("repair_candidates.jsonl is missing")
        return RepairReplayExecution(
            "SUCCESS",
            0,
            stdout_text,
            stderr_text,
            time.monotonic() - started,
            retained,
            None,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return RepairReplayExecution(
            (
                "INVALID_OUTPUT"
                if isinstance(exc, (PiRepairAuditError, ValueError))
                else "REPLAY_FAILED"
            ),
            None,
            "",
            f"{type(exc).__name__}: {exc}",
            time.monotonic() - started,
            retained,
            None,
        )
    finally:
        shutil.rmtree(replay_root, ignore_errors=True)


def _offline_environment(home: Path, scratch: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "TMPDIR": str(scratch),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _require_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} directory does not exist: {resolved}")
    return resolved


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
