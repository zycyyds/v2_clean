"""Strict no-Agent preflight and one-shot Test harness for frozen Pi pipelines."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.pi_harness import (
    ReplayProcessReapError,
    directory_sha256,
    load_submission,
    render_replay_argv,
    snapshot_agent_workdir,
    validate_plain_directory_tree,
)
from agent.pi_harness_sandbox import (
    build_macos_sandbox_profile,
    require_test_gold_isolated,
    sandbox_visible_recursive_roots,
)
from workflow.pi_harness_evaluation import SCORER_VERSION
from workflow.reference_evaluation import IGNORED_PACKAGE_FILES, STRUCTURED_SUFFIXES


@dataclass(frozen=True)
class PiTestPreflightConfig:
    project_root: Path
    validation_experiment: Path
    preflight_experiment: Path
    preflight_raw: Path
    replay_timeout_seconds: float = 1_800.0


@dataclass(frozen=True)
class PiTestHarnessConfig:
    project_root: Path
    validation_experiment: Path
    preflight_attestation: Path | None
    test_experiment: Path
    test_raw: Path
    test_gold: Path
    evaluation_manifest: Path
    replay_timeout_seconds: float = 1_800.0
    scoring_timeout_seconds: float = 3_600.0


@dataclass(frozen=True)
class ReplayRequest:
    phase: str
    frozen_snapshot: Path
    raw_root: Path
    train_reference: Path
    runtime_root: Path
    retained_result_root: Path
    timeout_seconds: float


@dataclass(frozen=True)
class TestReplayExecution:
    __test__ = False

    status: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    result_root: Path | None
    output_file_count: int
    output_bytes: int


@dataclass(frozen=True)
class ScoreExecution:
    status: str
    returncode: int | None
    stderr: str
    duration_seconds: float
    report: dict[str, Any] | None


@dataclass(frozen=True)
class PiTestHarnessResult:
    status: str
    phase: str
    scored: bool
    score: float | None
    test_execution_count: int
    replay_status: str
    frozen_snapshot_sha256: str
    frozen_snapshot: Path
    result_package: Path | None
    preflight_status: str = ""
    attestation: Path | None = None
    replay_duration_seconds: float = 0.0
    scoring_duration_seconds: float = 0.0


ReplayRunner = Callable[[ReplayRequest], Awaitable[TestReplayExecution]]
ScoreRunner = Callable[[Path, Path, Path, float], Awaitable[ScoreExecution]]

class PiTestPreflight:
    """Run a frozen Validation pipeline on large public Validation raw data."""

    def __init__(
        self,
        config: PiTestPreflightConfig,
        *,
        replay_runner: ReplayRunner | None = None,
        output: Any | None = None,
    ) -> None:
        self.config = config
        self.output = output or sys.stderr
        self.experiment = config.preflight_experiment.expanduser().resolve()
        self.host_dir = self.experiment / "host"
        self.frozen_snapshot = self.host_dir / "frozen_snapshot"
        self.replay_runner = replay_runner or _run_frozen_replay

    async def run(self) -> PiTestHarnessResult:
        source, train_reference, frozen_hash = _prepare_frozen_experiment(
            validation_experiment=self.config.validation_experiment,
            experiment=self.experiment,
            frozen_snapshot=self.frozen_snapshot,
        )
        preflight_raw = self.config.preflight_raw.expanduser().resolve()
        _require_directory(preflight_raw, "preflight raw")
        _require_positive_timeout(self.config.replay_timeout_seconds, "replay timeout")
        _atomic_json(
            self.host_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "mode": "strict_test_preflight",
                "validation_experiment": str(self.config.validation_experiment.resolve()),
                "preflight_raw": str(preflight_raw),
                "frozen_snapshot_sha256": frozen_hash,
                "replay_timeout_seconds": self.config.replay_timeout_seconds,
            },
        )
        request = ReplayRequest(
            phase="preflight",
            frozen_snapshot=self.frozen_snapshot,
            raw_root=preflight_raw,
            train_reference=train_reference,
            runtime_root=self.host_dir / "replay_runtime",
            retained_result_root=self.host_dir / "preflight_result_package",
            timeout_seconds=self.config.replay_timeout_seconds,
        )
        self._host_line("replay started")
        try:
            replay = await self.replay_runner(request)
        except asyncio.CancelledError:
            return self._finish(
                status="INTERRUPTED",
                phase="preflight",
                replay_status="INTERRUPTED",
                frozen_hash=frozen_hash,
            )
        except ReplayProcessReapError as exc:
            return self._finish(
                status="CLEANUP_FAILED",
                phase="preflight",
                replay_status="CLEANUP_FAILED",
                frozen_hash=frozen_hash,
                error=str(exc),
            )
        except Exception as exc:
            replay = TestReplayExecution(
                "EXECUTION_FAILED",
                None,
                "",
                f"{type(exc).__name__}: {exc}",
                0.0,
                None,
                0,
                0,
            )
            return self._finish(
                status="PREFLIGHT_FAILED",
                phase="preflight",
                replay_status=replay.status,
                frozen_hash=frozen_hash,
                replay=replay,
            )

        _verify_frozen_unchanged(source, self.frozen_snapshot, frozen_hash)
        replay = _validate_structured_execution(replay)
        if replay.status != "SUCCESS":
            _remove_owned_path(replay.result_root, self.experiment)
            return self._finish(
                status="PREFLIGHT_FAILED",
                phase="preflight",
                replay_status=replay.status,
                frozen_hash=frozen_hash,
                replay=replay,
            )

        attestation = self.host_dir / "preflight_attestation.json"
        _atomic_json(
            attestation,
            {
                "schema_version": 1,
                "status": "SUCCESS",
                "frozen_snapshot_sha256": frozen_hash,
                "preflight_raw_identity": _directory_identity(preflight_raw),
                "duration_seconds": replay.duration_seconds,
                "output_file_count": replay.output_file_count,
                "output_bytes": replay.output_bytes,
            },
        )
        _remove_owned_path(replay.result_root, self.experiment)
        _remove_owned_path(request.retained_result_root, self.experiment)
        return self._finish(
            status="SUCCESS",
            phase="complete",
            replay_status="SUCCESS",
            frozen_hash=frozen_hash,
            replay=replay,
            attestation=attestation,
        )

    def _finish(
        self,
        *,
        status: str,
        phase: str,
        replay_status: str,
        frozen_hash: str,
        replay: TestReplayExecution | None = None,
        attestation: Path | None = None,
        error: str = "",
    ) -> PiTestHarnessResult:
        result = PiTestHarnessResult(
            status=status,
            phase=phase,
            scored=False,
            score=None,
            test_execution_count=0,
            replay_status=replay_status,
            frozen_snapshot_sha256=frozen_hash,
            frozen_snapshot=self.frozen_snapshot,
            result_package=None,
            preflight_status="SUCCESS" if status == "SUCCESS" else replay_status,
            attestation=attestation,
            replay_duration_seconds=replay.duration_seconds if replay else 0.0,
        )
        payload = _result_payload(result)
        if replay is not None:
            payload["replay"] = _replay_payload(replay)
        if error:
            payload["error"] = error
        _atomic_json(self.host_dir / "run_report.json", payload)
        self._host_line(f"status={status} replay_status={replay_status}")
        return result

    def _host_line(self, message: str) -> None:
        print(f"[Test Preflight] {message}", file=self.output, flush=True)


class PiTestHarness:
    """Execute Test raw once and score once without creating an Agent."""

    def __init__(
        self,
        config: PiTestHarnessConfig,
        *,
        replay_runner: ReplayRunner | None = None,
        score_runner: ScoreRunner | None = None,
        output: Any | None = None,
    ) -> None:
        self.config = config
        self.output = output or sys.stderr
        self.experiment = config.test_experiment.expanduser().resolve()
        self.host_dir = self.experiment / "host"
        self.frozen_snapshot = self.host_dir / "frozen_snapshot"
        self.replay_runner = replay_runner or _run_frozen_replay
        self.score_runner = score_runner or self._run_score

    async def run(self) -> PiTestHarnessResult:
        validation = self.config.validation_experiment.expanduser().resolve(strict=False)
        source_snapshot = validation / "host/reproducible_snapshot"
        require_test_gold_isolated(
            self.config.test_gold,
            recursive_roots=sandbox_visible_recursive_roots(
                self.config.test_raw,
                source_snapshot,
                self.config.test_experiment,
                include_shell_state=True,
            ),
            literal_paths=(Path(sys.executable), Path("/dev/null")),
        )
        source, train_reference, source_hash = _load_validation_bundle(
            self.config.validation_experiment,
        )
        require_test_gold_isolated(
            self.config.test_gold,
            recursive_roots=(train_reference,),
        )
        attestation = (
            _load_attestation(self.config.preflight_attestation, source_hash)
            if self.config.preflight_attestation is not None
            else {}
        )
        _require_directory(self.config.test_raw, "Test raw")
        _require_directory(self.config.test_gold, "Test Gold")
        _require_file(self.config.evaluation_manifest, "evaluation manifest")
        _require_positive_timeout(self.config.replay_timeout_seconds, "replay timeout")
        _require_positive_timeout(self.config.scoring_timeout_seconds, "scoring timeout")
        _prepare_new_experiment(self.experiment)
        self.host_dir.mkdir()
        snapshot_agent_workdir(source, self.frozen_snapshot)
        frozen_hash = directory_sha256(self.frozen_snapshot)
        if frozen_hash != source_hash:
            raise RuntimeError("frozen Test snapshot hash verification failed")

        _atomic_json(
            self.host_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "mode": "strict_one_shot_test",
                "validation_experiment": str(self.config.validation_experiment.resolve()),
                "preflight_attestation_sha256": (
                    _file_sha256(self.config.preflight_attestation)
                    if self.config.preflight_attestation is not None
                    else ""
                ),
                "preflight_raw_identity": attestation.get("preflight_raw_identity", {}),
                "test_raw": str(self.config.test_raw.resolve()),
                "evaluation_manifest": str(self.config.evaluation_manifest.resolve()),
                "evaluation_manifest_sha256": _file_sha256(self.config.evaluation_manifest),
                "frozen_snapshot_sha256": frozen_hash,
                "replay_timeout_seconds": self.config.replay_timeout_seconds,
                "scoring_timeout_seconds": self.config.scoring_timeout_seconds,
                "scorer_version": SCORER_VERSION,
            },
        )

        request = ReplayRequest(
            phase="test_replay",
            frozen_snapshot=self.frozen_snapshot,
            raw_root=self.config.test_raw.resolve(),
            train_reference=train_reference,
            runtime_root=self.host_dir / "replay_runtime",
            retained_result_root=self.host_dir / "test_result_package",
            timeout_seconds=self.config.replay_timeout_seconds,
        )
        _atomic_json(
            self.host_dir / "test_started.json",
            {
                "schema_version": 1,
                "frozen_snapshot_sha256": frozen_hash,
                "started_at_unix": time.time(),
            },
        )
        self._host_line("Test replay started")
        try:
            replay = await self.replay_runner(request)
        except asyncio.CancelledError:
            return self._finish_interrupted("test_replay", frozen_hash, 1)
        except ReplayProcessReapError as exc:
            return self._finish_cleanup_failed("test_replay", frozen_hash, 1, str(exc))
        except Exception as exc:
            replay = TestReplayExecution(
                "EXECUTION_FAILED",
                None,
                "",
                f"{type(exc).__name__}: {exc}",
                0.0,
                None,
                0,
                0,
            )
            return self._finish(
                status="REPLAY_FAILED",
                phase="test_replay",
                scored=False,
                score=None,
                test_execution_count=1,
                replay_status=replay.status,
                frozen_hash=frozen_hash,
                result_package=None,
                replay=replay,
            )

        _verify_frozen_unchanged(source, self.frozen_snapshot, frozen_hash)
        replay = _validate_structured_execution(replay)
        if replay.status != "SUCCESS" or replay.result_root is None:
            return self._finish(
                status="REPLAY_FAILED",
                phase="test_replay",
                scored=False,
                score=None,
                test_execution_count=1,
                replay_status=replay.status,
                frozen_hash=frozen_hash,
                result_package=replay.result_root,
                replay=replay,
            )

        self._host_line("hidden scoring started")
        _atomic_json(
            self.host_dir / "scoring_started.json",
            {
                "schema_version": 1,
                "frozen_snapshot_sha256": frozen_hash,
                "started_at_unix": time.time(),
            },
        )
        try:
            score_execution = await self.score_runner(
                replay.result_root,
                self.config.test_gold.resolve(),
                self.config.evaluation_manifest.resolve(),
                self.config.scoring_timeout_seconds,
            )
        except asyncio.CancelledError:
            return self._finish_interrupted("scoring", frozen_hash, 1, replay)
        except ReplayProcessReapError as exc:
            return self._finish_cleanup_failed("scoring", frozen_hash, 1, str(exc), replay)
        except Exception as exc:
            scoring = ScoreExecution(
                "EXECUTION_FAILED",
                None,
                _redact_hidden_path(
                    f"{type(exc).__name__}: {exc}",
                    self.config.test_gold.resolve(),
                ),
                0.0,
                None,
            )
            return self._finish(
                status="SCORING_FAILED",
                phase="scoring",
                scored=False,
                score=None,
                test_execution_count=1,
                replay_status=replay.status,
                frozen_hash=frozen_hash,
                result_package=replay.result_root,
                replay=replay,
                scoring=scoring,
            )

        _verify_frozen_unchanged(source, self.frozen_snapshot, frozen_hash)
        if score_execution.status != "SUCCESS" or score_execution.report is None:
            return self._finish(
                status="SCORING_FAILED",
                phase="scoring",
                scored=False,
                score=None,
                test_execution_count=1,
                replay_status=replay.status,
                frozen_hash=frozen_hash,
                result_package=replay.result_root,
                replay=replay,
                scoring=score_execution,
            )

        public_report = _sanitize_score_report(
            score_execution.report,
            hidden_roots=(self.config.test_gold.resolve(), replay.result_root.resolve()),
        )
        _atomic_json(self.host_dir / "score_report.json", public_report)
        score = float((public_report.get("metrics") or {}).get("composite_score") or 0.0)
        return self._finish(
            status="SUCCESS",
            phase="complete",
            scored=True,
            score=score,
            test_execution_count=1,
            replay_status=replay.status,
            frozen_hash=frozen_hash,
            result_package=replay.result_root,
            replay=replay,
            scoring=replace(score_execution, report=None),
        )

    async def _run_score(
        self,
        result_root: Path,
        gold_root: Path,
        evaluation_manifest: Path,
        timeout: float,
    ) -> ScoreExecution:
        runtime_parent = self.host_dir / "scoring_runtime"
        runtime_parent.mkdir(parents=True, exist_ok=True)
        runtime = Path(tempfile.mkdtemp(prefix="score-", dir=runtime_parent))
        remove_runtime = True
        started = time.monotonic()
        try:
            home = runtime / "home"
            scratch = runtime / "tmp"
            home.mkdir()
            scratch.mkdir()
            profile = runtime / "scorer.sb"
            profile.write_text(
                build_macos_sandbox_profile(
                    executable=Path(sys.executable),
                    read_roots=sandbox_visible_recursive_roots(
                        self.config.project_root.resolve() / "agent",
                        self.config.project_root.resolve() / "workflow",
                        result_root,
                        gold_root,
                        evaluation_manifest,
                    ),
                    write_roots=[home, scratch, Path("/dev/null")],
                    allow_network=False,
                    traversal_roots=[self.config.project_root.resolve()],
                ),
                encoding="utf-8",
            )
            process = await asyncio.create_subprocess_exec(
                "/usr/bin/sandbox-exec",
                "-f",
                str(profile),
                sys.executable,
                "-u",
                "-m",
                "agent.pi_test_scorer",
                "--result-root",
                str(result_root),
                "--gold-root",
                str(gold_root),
                "--evaluation-manifest",
                str(evaluation_manifest),
                cwd=str(runtime),
                env=_restricted_python_environment(self.config.project_root, home, scratch),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr, timed_out = await _communicate_with_escalation(
                process,
                timeout=timeout,
            )
            stderr_text = _redact_hidden_path(stderr.decode(errors="replace")[-8000:], gold_root)
            if timed_out:
                return ScoreExecution(
                    "TIMEOUT",
                    process.returncode,
                    stderr_text,
                    time.monotonic() - started,
                    None,
                )
            if process.returncode != 0:
                return ScoreExecution(
                    "EXECUTION_FAILED",
                    process.returncode,
                    stderr_text,
                    time.monotonic() - started,
                    None,
                )
            try:
                report = json.loads(stdout.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return ScoreExecution(
                    "INVALID_OUTPUT",
                    process.returncode,
                    f"invalid scorer output: {exc}",
                    time.monotonic() - started,
                    None,
                )
            return ScoreExecution(
                "SUCCESS",
                process.returncode,
                stderr_text,
                time.monotonic() - started,
                report,
            )
        except ReplayProcessReapError:
            remove_runtime = False
            raise
        finally:
            if remove_runtime:
                shutil.rmtree(runtime, ignore_errors=True)

    def _finish_interrupted(
        self,
        phase: str,
        frozen_hash: str,
        execution_count: int,
        replay: TestReplayExecution | None = None,
    ) -> PiTestHarnessResult:
        return self._finish(
            status="INTERRUPTED",
            phase=phase,
            scored=False,
            score=None,
            test_execution_count=execution_count,
            replay_status=replay.status if replay else "INTERRUPTED",
            frozen_hash=frozen_hash,
            result_package=replay.result_root if replay else None,
            replay=replay,
        )

    def _finish_cleanup_failed(
        self,
        phase: str,
        frozen_hash: str,
        execution_count: int,
        error: str,
        replay: TestReplayExecution | None = None,
    ) -> PiTestHarnessResult:
        return self._finish(
            status="CLEANUP_FAILED",
            phase=phase,
            scored=False,
            score=None,
            test_execution_count=execution_count,
            replay_status=replay.status if replay else "CLEANUP_FAILED",
            frozen_hash=frozen_hash,
            result_package=replay.result_root if replay else None,
            replay=replay,
            error=error,
        )

    def _finish(
        self,
        *,
        status: str,
        phase: str,
        scored: bool,
        score: float | None,
        test_execution_count: int,
        replay_status: str,
        frozen_hash: str,
        result_package: Path | None,
        replay: TestReplayExecution | None = None,
        scoring: ScoreExecution | None = None,
        error: str = "",
    ) -> PiTestHarnessResult:
        result = PiTestHarnessResult(
            status=status,
            phase=phase,
            scored=scored,
            score=score,
            test_execution_count=test_execution_count,
            replay_status=replay_status,
            frozen_snapshot_sha256=frozen_hash,
            frozen_snapshot=self.frozen_snapshot,
            result_package=result_package,
            preflight_status=(
                "SUCCESS"
                if self.config.preflight_attestation is not None
                else "NOT_RUN"
            ),
            replay_duration_seconds=replay.duration_seconds if replay else 0.0,
            scoring_duration_seconds=scoring.duration_seconds if scoring else 0.0,
        )
        payload = _result_payload(result)
        if replay is not None:
            payload["replay"] = _replay_payload(
                replay,
                hidden_root=self.config.test_gold.resolve(),
            )
        if scoring is not None:
            payload["scoring"] = {
                "status": scoring.status,
                "returncode": scoring.returncode,
                "duration_seconds": scoring.duration_seconds,
                "stderr": scoring.stderr[-4000:],
            }
        if error:
            payload["error"] = error
        _atomic_json(self.host_dir / "run_report.json", payload)
        self._host_line(
            f"status={status} phase={phase} score={score if score is not None else 'null'}",
        )
        return result

    def _host_line(self, message: str) -> None:
        print(f"[Test Harness] {message}", file=self.output, flush=True)


async def _run_frozen_replay(request: ReplayRequest) -> TestReplayExecution:
    request.runtime_root.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix=f"{request.phase}-", dir=request.runtime_root))
    remove_runtime = True
    started = time.monotonic()
    try:
        workdir = runtime / "frozen"
        snapshot_agent_workdir(request.frozen_snapshot, workdir)
        try:
            submission = load_submission(workdir)
            if submission.result_root.exists():
                _remove_path(submission.result_root)
            submission.result_root.parent.mkdir(parents=True, exist_ok=True)
            argv = render_replay_argv(
                submission,
                raw_root=request.raw_root,
                train_reference=request.train_reference,
                output_dir=submission.result_root,
                workdir=workdir,
            )
        except Exception as exc:
            return TestReplayExecution(
                "INVALID_SUBMISSION",
                None,
                "",
                f"{type(exc).__name__}: {exc}",
                time.monotonic() - started,
                None,
                0,
                0,
            )
        if argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        home = runtime / "home"
        scratch = runtime / "tmp"
        home.mkdir()
        scratch.mkdir()
        profile = runtime / "replay.sb"
        profile.write_text(
            build_macos_sandbox_profile(
                executable=Path(sys.executable),
                read_roots=sandbox_visible_recursive_roots(
                    workdir,
                    request.raw_root,
                    request.train_reference,
                    include_shell_state=True,
                ),
                write_roots=[workdir, home, scratch, Path("/dev/null")],
                allow_network=False,
            ),
            encoding="utf-8",
        )
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile),
            *argv,
            cwd=str(workdir),
            env=_restricted_python_environment(None, home, scratch),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr, timed_out = await _communicate_with_escalation(
            process,
            timeout=request.timeout_seconds,
        )
        stdout_text = stdout.decode(errors="replace")[-8000:]
        stderr_text = stderr.decode(errors="replace")[-8000:]
        if timed_out:
            return TestReplayExecution(
                "TIMEOUT",
                process.returncode,
                stdout_text,
                stderr_text,
                time.monotonic() - started,
                None,
                0,
                0,
            )
        if process.returncode != 0:
            return TestReplayExecution(
                "EXECUTION_FAILED",
                process.returncode,
                stdout_text,
                stderr_text,
                time.monotonic() - started,
                None,
                0,
                0,
            )
        try:
            validate_plain_directory_tree(submission.result_root)
            file_count, output_bytes = _structured_output_stats(submission.result_root)
        except ValueError as exc:
            return TestReplayExecution(
                "INVALID_OUTPUT",
                0,
                stdout_text,
                str(exc),
                time.monotonic() - started,
                None,
                0,
                0,
            )
        if file_count == 0:
            return TestReplayExecution(
                "INVALID_OUTPUT",
                0,
                stdout_text,
                "result package contains no structured files",
                time.monotonic() - started,
                None,
                0,
                output_bytes,
            )
        if request.retained_result_root.exists():
            _remove_path(request.retained_result_root)
        shutil.copytree(submission.result_root, request.retained_result_root)
        return TestReplayExecution(
            "SUCCESS",
            0,
            stdout_text,
            stderr_text,
            time.monotonic() - started,
            request.retained_result_root,
            file_count,
            output_bytes,
        )
    except ReplayProcessReapError:
        remove_runtime = False
        raise
    finally:
        if remove_runtime:
            shutil.rmtree(runtime, ignore_errors=True)


async def _communicate_with_escalation(
    process: asyncio.subprocess.Process,
    *,
    timeout: float,
    shutdown_timeout: float = 5.0,
) -> tuple[bytes, bytes, bool]:
    communication = asyncio.create_task(process.communicate())
    try:
        return (*await asyncio.wait_for(asyncio.shield(communication), timeout=timeout), False)
    except asyncio.TimeoutError:
        await _terminate_and_reap(process, communication, shutdown_timeout)
        stdout, stderr = communication.result()
        return stdout, stderr, True
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(
            _terminate_and_reap(process, communication, shutdown_timeout),
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
        raise


async def _terminate_and_reap(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes]],
    timeout: float,
) -> None:
    for requested_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if communication.done() or process.returncode is not None:
            break
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(communication), timeout=timeout)
            break
        except asyncio.TimeoutError:
            continue
    if not communication.done():
        raise ReplayProcessReapError("process group did not exit after SIGKILL")
    await communication


def _prepare_frozen_experiment(
    *,
    validation_experiment: Path,
    experiment: Path,
    frozen_snapshot: Path,
) -> tuple[Path, Path, str]:
    source, train_reference, source_hash = _load_validation_bundle(validation_experiment)
    _prepare_new_experiment(experiment)
    (experiment / "host").mkdir()
    snapshot_agent_workdir(source, frozen_snapshot)
    if directory_sha256(frozen_snapshot) != source_hash:
        raise RuntimeError("frozen preflight snapshot hash verification failed")
    return source, train_reference, source_hash


def _load_validation_bundle(validation_experiment: Path) -> tuple[Path, Path, str]:
    validation = validation_experiment.expanduser().resolve()
    report_path = validation / "host/run_report.json"
    manifest_path = validation / "host/run_manifest.json"
    _require_file(report_path, "Validation run report")
    _require_file(manifest_path, "Validation run manifest")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "SUCCESS_REPRODUCIBLE":
        raise ValueError("Validation status must be SUCCESS_REPRODUCIBLE")
    source = validation / "host/reproducible_snapshot"
    declared = str(report.get("reproducible_snapshot") or "").strip()
    if not declared or Path(declared).expanduser().resolve() != source.resolve():
        raise ValueError("Validation reproducible_snapshot does not match the frozen directory")
    validate_plain_directory_tree(source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_reference = Path(str(manifest.get("train_reference") or "")).expanduser().resolve()
    _require_directory(train_reference, "Train reference")
    return source.resolve(), train_reference, directory_sha256(source)


def _load_attestation(path: Path, expected_hash: str) -> dict[str, Any]:
    attestation_path = path.expanduser().resolve()
    _require_file(attestation_path, "preflight attestation")
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "SUCCESS":
        raise ValueError("preflight attestation must record SUCCESS")
    if payload.get("frozen_snapshot_sha256") != expected_hash:
        raise ValueError("preflight attestation frozen hash mismatch")
    return payload


def _verify_frozen_unchanged(source: Path, frozen: Path, expected_hash: str) -> None:
    source_hash = directory_sha256(source)
    frozen_hash = directory_sha256(frozen)
    if source_hash != expected_hash or frozen_hash != expected_hash:
        raise RuntimeError(
            "frozen pipeline changed during strict evaluation: "
            f"expected={expected_hash}, source={source_hash}, copy={frozen_hash}",
        )


def _validate_structured_execution(execution: TestReplayExecution) -> TestReplayExecution:
    if execution.status != "SUCCESS" or execution.result_root is None:
        return execution
    try:
        validate_plain_directory_tree(execution.result_root)
        file_count, output_bytes = _structured_output_stats(execution.result_root)
    except ValueError as exc:
        return replace(execution, status="INVALID_OUTPUT", stderr=str(exc))
    if file_count == 0:
        return replace(
            execution,
            status="INVALID_OUTPUT",
            stderr="result package contains no structured files",
            output_file_count=0,
            output_bytes=output_bytes,
        )
    return replace(
        execution,
        output_file_count=file_count,
        output_bytes=output_bytes,
    )


def _structured_output_stats(root: Path) -> tuple[int, int]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in IGNORED_PACKAGE_FILES
        and path.name.casefold().endswith(STRUCTURED_SUFFIXES)
    ]
    all_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return len(files), all_bytes


def _directory_identity(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    metadata = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        stat_result = path.stat()
        metadata.update(f"{relative}\0{stat_result.st_size}\0".encode())
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "total_size": sum(path.stat().st_size for path in files),
        "metadata_sha256": metadata.hexdigest(),
    }


def _sanitize_score_report(
    report: dict[str, Any],
    *,
    hidden_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in {"reference_file", "result_file"}
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            for root in hidden_roots:
                value = value.replace(str(root), "<hidden_path>")
        return value

    return clean(report)


def _restricted_python_environment(
    project_root: Path | None,
    home: Path,
    scratch: Path,
) -> dict[str, str]:
    environment = {
        "HOME": str(home),
        "TMPDIR": str(scratch),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if project_root is not None:
        environment["PYTHONPATH"] = str(project_root.resolve())
    return environment


def _prepare_new_experiment(experiment: Path) -> None:
    if experiment.exists() and any(experiment.iterdir()):
        raise ValueError("experiment directory must be new and empty")
    experiment.mkdir(parents=True, exist_ok=True)


def _require_directory(path: Path, label: str) -> None:
    if not path.expanduser().resolve().is_dir():
        raise ValueError(f"{label} directory does not exist: {path}")


def _require_file(path: Path, label: str) -> None:
    if not path.expanduser().resolve().is_file():
        raise ValueError(f"{label} does not exist: {path}")


def _require_positive_timeout(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")


def _remove_owned_path(path: Path | None, owner: Path) -> None:
    if path is None or not path.exists():
        return
    resolved = path.resolve()
    owner = owner.resolve()
    if resolved != owner and owner in resolved.parents:
        _remove_path(resolved)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_hidden_path(text: str, hidden_root: Path) -> str:
    return text.replace(str(hidden_root.resolve()), "<hidden_gold>")


def _replay_payload(
    replay: TestReplayExecution,
    *,
    hidden_root: Path | None = None,
) -> dict[str, Any]:
    stderr = replay.stderr[-4000:]
    if hidden_root is not None:
        stderr = _redact_hidden_path(stderr, hidden_root)
    return {
        "status": replay.status,
        "returncode": replay.returncode,
        "duration_seconds": replay.duration_seconds,
        "output_file_count": replay.output_file_count,
        "output_bytes": replay.output_bytes,
        "stderr": stderr,
    }


def _result_payload(result: PiTestHarnessResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": result.status,
        "phase": result.phase,
        "scored": result.scored,
        "score": result.score,
        "test_execution_count": result.test_execution_count,
        "replay_status": result.replay_status,
        "frozen_snapshot_sha256": result.frozen_snapshot_sha256,
        "frozen_snapshot": str(result.frozen_snapshot),
        "result_package": str(result.result_package) if result.result_package else "",
        "preflight_status": result.preflight_status,
        "attestation": str(result.attestation) if result.attestation else "",
        "replay_duration_seconds": result.replay_duration_seconds,
        "scoring_duration_seconds": result.scoring_duration_seconds,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
