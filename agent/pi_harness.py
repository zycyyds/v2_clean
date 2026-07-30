from __future__ import annotations

import json
import asyncio
import hashlib
import inspect
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

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
    public_feedback_from_equal_weight_report,
    score_equal_weight_reference_directory,
)


SCORE_TOLERANCE = 1e-6
ALLOWED_REPLAY_PLACEHOLDERS = {
    "raw_root",
    "train_raw",
    "train_reference",
    "output_dir",
    "workdir",
}
SNAPSHOT_EXCLUDES = {".agent_runs"}
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


class ReplayProcessReapError(RuntimeError):
    """Raised when a killed replay process cannot be reaped in time."""
_SHELL_LAUNCHERS = {"bash", "dash", "env", "fish", "ksh", "sh", "zsh"}


@dataclass(frozen=True)
class Submission:
    workdir: Path
    result_root: Path
    replay_argv: tuple[str, ...]


@dataclass(frozen=True)
class HarnessProgress:
    round_index: int = 0
    best_score: float = -1.0
    best_round: int = 0
    consecutive_non_improvements: int = 0
    stop_reason: str = ""


@dataclass(frozen=True)
class RoundDecision:
    promoted: bool
    restored_best: bool


@dataclass(frozen=True)
class PiValidationHarnessConfig:
    project_root: Path
    experiment_dir: Path
    train_raw: Path
    train_reference: Path
    validation_raw: Path
    validation_gold: Path
    evaluation_manifest: Path
    dataset_manifest: Path
    max_rounds: int = 20
    patience: int = 3
    target_score: float = 1.0
    max_iters: int = 10_000
    skill_dirs: tuple[Path, ...] = ()
    replay_timeout_seconds: float = 1_800.0


@dataclass(frozen=True)
class ReplayExecution:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    result_root: Path | None
    score_report: dict[str, Any] | None

    @property
    def score(self) -> float:
        if self.score_report is None:
            return 0.0
        return float((self.score_report.get("metrics") or {}).get("composite_score") or 0.0)


@dataclass(frozen=True)
class PiHarnessResult:
    status: str
    stop_reason: str
    rounds: int
    repair_rounds: int
    best_score: float
    reproducible_score: float
    best_snapshot: Path
    reproducible_snapshot: Path


Scorer = Callable[[str | Path, str | Path, dict[str, tuple[str, ...]]], dict[str, Any]]
ReplayRunner = Callable[[Path], Awaitable[ReplayExecution]]


def load_submission(workdir: str | Path) -> Submission:
    root = Path(workdir).expanduser().resolve()
    path = root / "submission.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("submission.json is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"submission.json is invalid JSON: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError("submission schema_version must be 1")

    result_text = str(payload.get("result_root") or "").strip()
    if not result_text:
        raise ValueError("submission result_root is required")
    if Path(result_text).is_absolute():
        raise ValueError("submission result_root must be relative")
    result_root = (root / result_text).resolve()
    if not _is_within(result_root, root):
        raise ValueError("submission result_root escapes the Agent workdir")
    if result_root == root:
        raise ValueError("submission result_root must be a dedicated subdirectory")

    replay = payload.get("replay")
    argv = replay.get("argv") if isinstance(replay, dict) else None
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("submission replay.argv must be a non-empty string array")
    if Path(argv[0]).name.lower() in _SHELL_LAUNCHERS:
        raise ValueError("submission replay.argv cannot execute shell command strings")
    placeholders = {name for item in argv for name in _PLACEHOLDER.findall(item)}
    unsupported = sorted(placeholders - ALLOWED_REPLAY_PLACEHOLDERS)
    if unsupported:
        raise ValueError("unsupported placeholder: " + ", ".join(unsupported))
    if "output_dir" not in placeholders:
        raise ValueError("submission replay.argv must contain {output_dir}")
    if "raw_root" not in placeholders:
        raise ValueError("submission replay.argv must contain {raw_root}")
    for item in argv:
        item_placeholders = set(_PLACEHOLDER.findall(item))
        static = _PLACEHOLDER.sub("placeholder", item)
        if Path(static).is_absolute() and not item_placeholders:
            raise ValueError("submission replay.argv cannot contain static absolute paths")

    return Submission(
        workdir=root,
        result_root=result_root,
        replay_argv=tuple(argv),
    )


def render_replay_argv(
    submission: Submission,
    *,
    raw_root: Path,
    train_reference: Path,
    output_dir: Path,
    workdir: Path,
    train_raw: Path | None = None,
) -> list[str]:
    values = {
        "raw_root": str(raw_root.resolve()),
        "train_reference": str(train_reference.resolve()),
        "output_dir": str(output_dir.resolve()),
        "workdir": str(workdir.resolve()),
    }
    if train_raw is not None:
        values["train_raw"] = str(train_raw.resolve())
    requested = {
        name
        for item in submission.replay_argv
        for name in _PLACEHOLDER.findall(item)
    }
    unavailable = sorted(requested - values.keys())
    if unavailable:
        raise ValueError(
            "placeholder unavailable in this replay context: "
            + ", ".join(unavailable),
        )
    return [item.format_map(values) for item in submission.replay_argv]


async def _kill_and_reap_process_group(
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 5.0,
) -> tuple[bytes, bytes]:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        return await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ReplayProcessReapError("Replay process did not exit after SIGKILL")
        return b"", b""


async def _close_worker_without_losing_cancellation(worker: Any) -> bool:
    """Finish bounded worker cleanup and report cancellation during cleanup."""
    close_task = asyncio.create_task(worker.close())
    cancelled = False
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            cancelled = True
    await close_task
    return cancelled


def record_round_score(
    state: HarnessProgress,
    score: float,
    *,
    max_rounds: int,
    patience: int,
    target_score: float,
    tolerance: float = SCORE_TOLERANCE,
    eligible_for_promotion: bool = True,
) -> tuple[HarnessProgress, RoundDecision]:
    if state.stop_reason:
        raise ValueError("cannot record a round after the Harness has stopped")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be finite and between 0 and 1")
    round_index = state.round_index + 1
    candidate = round(score, 6)
    promoted = eligible_for_promotion and (
        state.best_score < 0.0
        or candidate >= round(state.best_score + tolerance, 6)
    )
    updated = replace(
        state,
        round_index=round_index,
        best_score=candidate if promoted else state.best_score,
        best_round=round_index if promoted else state.best_round,
        consecutive_non_improvements=(
            0 if promoted else state.consecutive_non_improvements + 1
        ),
    )
    if updated.best_score >= target_score:
        updated = replace(updated, stop_reason="target_score")
    elif updated.consecutive_non_improvements >= patience:
        updated = replace(updated, stop_reason="patience")
    elif updated.round_index >= max_rounds:
        updated = replace(updated, stop_reason="max_rounds")
    return updated, RoundDecision(
        promoted=promoted,
        restored_best=not promoted and state.best_score >= 0.0,
    )


def snapshot_agent_workdir(source: str | Path, destination: str | Path) -> None:
    source_root = Path(source).expanduser().resolve()
    destination_root = Path(destination).expanduser().resolve()
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_root.with_name(f".{destination_root.name}.tmp-{uuid.uuid4().hex}")
    _copy_agent_entries(source_root, temporary)
    backup = destination_root.with_name(f".{destination_root.name}.old-{uuid.uuid4().hex}")
    try:
        if destination_root.exists():
            os.replace(destination_root, backup)
        os.replace(temporary, destination_root)
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup.exists() and not destination_root.exists():
            os.replace(backup, destination_root)


def restore_agent_workdir(destination: str | Path, snapshot: str | Path) -> None:
    destination_root = Path(destination).expanduser().resolve()
    snapshot_root = Path(snapshot).expanduser().resolve()
    if not snapshot_root.is_dir():
        raise ValueError(f"best snapshot is missing: {snapshot_root}")
    staging = destination_root.with_name(f".{destination_root.name}.restore-{uuid.uuid4().hex}")
    backup = destination_root.with_name(f".{destination_root.name}.backup-{uuid.uuid4().hex}")
    _copy_agent_entries(snapshot_root, staging)
    if directory_sha256(staging) != directory_sha256(snapshot_root):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("best restore staging verification failed")
    backup.mkdir()
    committed = False
    try:
        for item in list(destination_root.iterdir()):
            if item.name in SNAPSHOT_EXCLUDES:
                continue
            os.replace(item, backup / item.name)
        for item in list(staging.iterdir()):
            os.replace(item, destination_root / item.name)
        committed = True
    finally:
        if not committed:
            for item in list(destination_root.iterdir()):
                if item.name in SNAPSHOT_EXCLUDES:
                    continue
                _remove_path(item)
            for item in list(backup.iterdir()):
                os.replace(item, destination_root / item.name)
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def directory_sha256(root: str | Path) -> str:
    """Hash an Agent snapshot without following symlinks or .agent_runs."""
    base = Path(root).expanduser().resolve()
    digest = hashlib.sha256()
    if not base.is_dir():
        return digest.hexdigest()
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base)
        if relative.parts and relative.parts[0] in SNAPSHOT_EXCLUDES:
            continue
        name = relative.as_posix().encode()
        if path.is_symlink():
            digest.update(b"L\0" + name + b"\0" + os.readlink(path).encode())
        elif path.is_dir():
            digest.update(b"D\0" + name + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + name + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


class PiValidationHarness:
    def __init__(
        self,
        config: PiValidationHarnessConfig,
        *,
        worker: Any | None = None,
        scorer: Scorer = score_equal_weight_reference_directory,
        replay_runner: ReplayRunner | None = None,
        sandbox_probe: Callable[..., Any] | None = None,
        output: Any | None = None,
    ) -> None:
        self.config = config
        self.worker = worker
        self.scorer = scorer
        self.replay_runner = replay_runner or self._run_replay
        self.sandbox_probe = sandbox_probe or verify_agent_sandbox_access
        self.output = output or sys.stderr
        self.experiment_dir = config.experiment_dir.expanduser().resolve()
        self.agent_workdir = self.experiment_dir / "agent_workdir"
        self.host_dir = self.experiment_dir / "host"
        self.best_snapshot = self.host_dir / "best_snapshot"
        self._worker_launch: WorkerLaunch | None = None
        self._worker_log_handle: Any | None = None
        self._key_columns: dict[str, tuple[str, ...]] = {}
        self._model_environment: dict[str, str] | None = None
        self._round_executions: list[dict[str, Any]] = []

    async def run(self, prompt: str) -> PiHarnessResult:
        self._prepare(prompt)
        progress = HarnessProgress()
        repair_rounds = 0
        next_prompt = self._initial_prompt(prompt)
        last_replay: ReplayExecution | None = None
        try:
            if self.worker is None:
                self.worker, self._worker_launch = self._create_worker()
            probe_result = self.sandbox_probe(self.config, self._worker_launch)
            if inspect.isawaitable(probe_result):
                await probe_result
            await self.worker.start()
            while not progress.stop_reason:
                self._host_line(
                    f"round {progress.round_index + 1}/{self.config.max_rounds}: "
                    "Agent turn started",
                )
                agent_started = time.monotonic()
                turn_result = await self.worker.run_turn(next_prompt)
                agent_wall_seconds = round(time.monotonic() - agent_started, 6)
                self._host_line(
                    f"round {progress.round_index + 1}: hidden scoring started",
                )
                scoring_started = time.monotonic()
                report, submission_error = self._score_current()
                scoring_wall_seconds = round(time.monotonic() - scoring_started, 6)
                score = float((report.get("metrics") or {}).get("composite_score") or 0.0)
                progress, decision = record_round_score(
                    progress,
                    score,
                    max_rounds=self.config.max_rounds,
                    patience=self.config.patience,
                    target_score=self.config.target_score,
                    eligible_for_promotion=(
                        not submission_error
                        and all(
                            item.get("status") != "read_error"
                            for item in report.get("file_reports") or []
                        )
                    ),
                )
                self._host_line(
                    f"round {progress.round_index}: score={score:.6f} "
                    f"promoted={str(decision.promoted).lower()}",
                )
                best_hash = ""
                if decision.promoted:
                    snapshot_agent_workdir(self.agent_workdir, self.best_snapshot)
                    best_hash = directory_sha256(self.best_snapshot)
                elif decision.restored_best:
                    best_hash = directory_sha256(self.best_snapshot)
                    restore_agent_workdir(self.agent_workdir, self.best_snapshot)
                    if directory_sha256(self.agent_workdir) != best_hash:
                        raise RuntimeError("best rollback verification failed")
                    await self.worker.clear_file_cache()

                feedback = public_feedback_from_equal_weight_report(
                    report,
                    round_index=progress.round_index,
                    best_score=max(progress.best_score, 0.0),
                    promoted=decision.promoted,
                    restored_best=decision.restored_best,
                )
                if submission_error:
                    feedback["submission_error"] = self._sanitize(submission_error)
                if decision.restored_best:
                    feedback["best_snapshot_sha256"] = best_hash
                    feedback["required_action"] = (
                        "The host restored the formal best files. Re-read files before editing; "
                        "do not assume your rejected filesystem changes still exist."
                    )
                execution = {
                    "agent_wall_seconds": agent_wall_seconds,
                    "scoring_wall_seconds": scoring_wall_seconds,
                    "agent": _turn_metrics(turn_result),
                }
                self._round_executions.append(execution)
                self._record_round(progress.round_index, report, feedback, execution)
                next_prompt = self._feedback_prompt(feedback)

            if not self.best_snapshot.is_dir():
                snapshot_agent_workdir(self.agent_workdir, self.best_snapshot)
            self._host_line("independent replay started")
            last_replay = await self.replay_runner(self.best_snapshot)
            self._host_line(
                f"independent replay: status={last_replay.status} "
                f"score={last_replay.score:.6f}",
            )
            while not self._replay_is_acceptable(last_replay, progress.best_score):
                repair_rounds += 1
                replay_feedback = self._replay_feedback(last_replay, progress.best_score)
                self._atomic_json(
                    self.host_dir / "replay" / f"failure_{repair_rounds:04d}.json",
                    replay_feedback,
                )
                self._host_line(f"replay repair {repair_rounds}: Agent turn started")
                agent_started = time.monotonic()
                turn_result = await self.worker.run_turn(self._feedback_prompt(replay_feedback))
                agent_wall_seconds = round(time.monotonic() - agent_started, 6)
                scoring_started = time.monotonic()
                _, submission_error = self._score_current()
                scoring_wall_seconds = round(time.monotonic() - scoring_started, 6)
                if submission_error:
                    replay_feedback["submission_error"] = self._sanitize(submission_error)
                last_replay = await self.replay_runner(self.agent_workdir)
                execution = {
                    "phase": "replay_repair",
                    "agent_wall_seconds": agent_wall_seconds,
                    "scoring_wall_seconds": scoring_wall_seconds,
                    "agent": _turn_metrics(turn_result),
                    "replay": {
                        "status": last_replay.status,
                        "returncode": last_replay.returncode,
                        "duration_seconds": last_replay.duration_seconds,
                        "score": last_replay.score,
                    },
                }
                self._round_executions.append(execution)
                self._atomic_json(
                    self.host_dir / "replay" / f"repair_metrics_{repair_rounds:04d}.json",
                    execution,
                )
                self._host_line(
                    f"replay repair {repair_rounds}: status={last_replay.status} "
                    f"score={last_replay.score:.6f}",
                )

            final_snapshot = self.host_dir / "reproducible_snapshot"
            snapshot_agent_workdir(
                self.best_snapshot if repair_rounds == 0 else self.agent_workdir,
                final_snapshot,
            )
            result = PiHarnessResult(
                status="SUCCESS_REPRODUCIBLE",
                stop_reason=progress.stop_reason,
                rounds=progress.round_index,
                repair_rounds=repair_rounds,
                best_score=max(progress.best_score, 0.0),
                reproducible_score=last_replay.score,
                best_snapshot=self.best_snapshot,
                reproducible_snapshot=final_snapshot,
            )
            self._atomic_json(self.host_dir / "run_report.json", self._run_report_payload(result))
            return result
        except asyncio.CancelledError:
            if hasattr(self.worker, "interrupt"):
                try:
                    await self.worker.interrupt()
                except Exception:
                    pass
            result = PiHarnessResult(
                status="INTERRUPTED",
                stop_reason="keyboard_interrupt",
                rounds=progress.round_index,
                repair_rounds=repair_rounds,
                best_score=max(progress.best_score, 0.0),
                reproducible_score=0.0,
                best_snapshot=self.best_snapshot,
                reproducible_snapshot=self.best_snapshot,
            )
            self._atomic_json(self.host_dir / "run_report.json", self._run_report_payload(result))
            return result
        finally:
            cancelled_during_cleanup = False
            if self.worker is not None:
                cancelled_during_cleanup = await _close_worker_without_losing_cancellation(
                    self.worker,
                )
            if self._worker_log_handle is not None:
                self._worker_log_handle.close()
                self._worker_log_handle = None
            if cancelled_during_cleanup:
                result = PiHarnessResult(
                    status="INTERRUPTED",
                    stop_reason="keyboard_interrupt",
                    rounds=progress.round_index,
                    repair_rounds=repair_rounds,
                    best_score=max(progress.best_score, 0.0),
                    reproducible_score=0.0,
                    best_snapshot=self.best_snapshot,
                    reproducible_snapshot=self.best_snapshot,
                )
                self._atomic_json(
                    self.host_dir / "run_report.json",
                    self._run_report_payload(result),
                )
                return result

    def _prepare(self, prompt: str = "") -> None:
        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True)
        elif any(self.experiment_dir.iterdir()):
            raise ValueError("experiment directory must be new and empty")
        for path, label in (
            (self.config.train_raw, "train raw"),
            (self.config.train_reference, "train reference"),
            (self.config.validation_raw, "validation raw"),
            (self.config.validation_gold, "validation gold"),
        ):
            if not path.expanduser().resolve().is_dir():
                raise ValueError(f"{label} directory does not exist: {path}")
        if self.config.max_rounds < 1 or self.config.patience < 1 or self.config.max_iters < 1:
            raise ValueError("max_rounds, patience, and max_iters must be positive")
        if not math.isfinite(self.config.target_score) or not 0.0 <= self.config.target_score <= 1.0:
            raise ValueError("target_score must be finite and between 0 and 1")
        if not math.isfinite(self.config.replay_timeout_seconds) or self.config.replay_timeout_seconds <= 0:
            raise ValueError("replay_timeout_seconds must be positive and finite")
        for path, label in (
            (self.config.evaluation_manifest, "evaluation manifest"),
            (self.config.dataset_manifest, "dataset manifest"),
        ):
            if not path.expanduser().resolve().is_file():
                raise ValueError(f"{label} does not exist: {path}")
        self.agent_workdir.mkdir()
        (self.host_dir / "rounds").mkdir(parents=True)
        (self.host_dir / "public_feedback").mkdir(parents=True)
        (self.host_dir / "replay").mkdir(parents=True)
        self._key_columns = load_evaluation_manifest(
            self.config.evaluation_manifest,
            self.config.validation_gold,
        )
        self._model_environment = build_worker_model_environment("react_planner", "MiniMax-M3")
        model_identity = _public_model_identity(self._model_environment)
        git_identity = _git_identity(self.config.project_root)
        self._atomic_json(
            self.host_dir / "run_manifest.json",
            {
                "schema_version": 2,
                "project_root": str(self.config.project_root.resolve()),
                "train_raw": str(self.config.train_raw.resolve()),
                "train_reference": str(self.config.train_reference.resolve()),
                "validation_raw": str(self.config.validation_raw.resolve()),
                "validation_gold": str(self.config.validation_gold.resolve()),
                "evaluation_manifest": str(self.config.evaluation_manifest.resolve()),
                "dataset_manifest": str(self.config.dataset_manifest.resolve()),
                "dataset_manifest_sha256": _file_sha256(self.config.dataset_manifest),
                "evaluation_manifest_sha256": _file_sha256(self.config.evaluation_manifest),
                "prompt_sha256": _text_sha256(prompt),
                "keys_sha256": {
                    "train": _file_sha256(self.config.train_raw.resolve().parent / "keys.csv"),
                    "validation": _file_sha256(self.config.validation_raw.resolve().parent / "keys.csv"),
                },
                "git": git_identity,
                "model": model_identity,
                "skills": {
                    "enabled": bool(self.config.skill_dirs),
                    "directories": [str(path.resolve()) for path in self.config.skill_dirs],
                    "bundle_sha256": _directory_bundle_sha256(self.config.skill_dirs),
                },
                "scorer_version": SCORER_VERSION,
                "max_rounds": self.config.max_rounds,
                "patience": self.config.patience,
                "target_score": self.config.target_score,
                "max_iters": self.config.max_iters,
            },
        )

    def _create_worker(self) -> tuple[JsonlWorkerClient, WorkerLaunch]:
        model_environment = self._model_environment or build_worker_model_environment(
            "react_planner", "MiniMax-M3"
        )
        launch = build_sandboxed_worker_launch(
            SandboxedWorkerConfig(
                project_root=self.config.project_root,
                agent_workdir=self.agent_workdir,
                runtime_root=self.host_dir / "runtime",
                public_read_roots=(
                    self.config.train_raw,
                    self.config.train_reference,
                    self.config.validation_raw,
                ),
                skill_dirs=self.config.skill_dirs,
                max_iters=self.config.max_iters,
                model_environment=model_environment,
            ),
        )
        worker_log = (self.host_dir / "worker.log").open("a", encoding="utf-8")
        self._worker_log_handle = worker_log
        try:
            redactions = json.loads(model_environment.get("OPENAI_API_KEYS_JSON", "[]"))
        except json.JSONDecodeError:
            redactions = []
        output = _RedactingTee((self.output, worker_log), tuple(str(item) for item in redactions))
        return (
            JsonlWorkerClient(
                command=launch.command,
                cwd=launch.cwd,
                env=launch.env,
                output=output,
            ),
            launch,
        )

    def _score_current(self) -> tuple[dict[str, Any], str]:
        submission_error = ""
        try:
            submission = load_submission(self.agent_workdir)
            package = submission.result_root
        except Exception as exc:
            submission_error = f"{type(exc).__name__}: {exc}"
            package = self.agent_workdir / ".missing_submission_result"
        try:
            report = self.scorer(package, self.config.validation_gold, self._key_columns)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            submission_error = f"{submission_error}; {detail}".strip("; ")
            report = score_equal_weight_reference_directory(
                self.agent_workdir / ".missing_submission_result",
                self.config.validation_gold,
                self._key_columns,
            )
        if submission_error:
            report = {**report, "submission_status": "invalid", "submission_error": submission_error}
        return report, submission_error

    def _record_round(
        self,
        round_index: int,
        private_report: dict[str, Any],
        public_feedback: dict[str, Any],
        execution: dict[str, Any],
    ) -> None:
        name = f"round_{round_index:04d}.json"
        self._atomic_json(
            self.host_dir / "rounds" / name,
            {**private_report, "execution": execution},
        )
        self._atomic_json(self.host_dir / "public_feedback" / name, public_feedback)

    def _run_report_payload(self, result: PiHarnessResult) -> dict[str, Any]:
        usage_fields = (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "react_iterations",
            "tool_calls",
            "tool_errors",
            "api_failovers",
            "compactions",
        )
        totals = {
            field: sum(int(item.get("agent", {}).get(field) or 0) for item in self._round_executions)
            for field in usage_fields
        }
        totals["agent_wall_seconds"] = round(
            sum(float(item.get("agent_wall_seconds") or 0.0) for item in self._round_executions),
            6,
        )
        totals["scoring_wall_seconds"] = round(
            sum(float(item.get("scoring_wall_seconds") or 0.0) for item in self._round_executions),
            6,
        )
        return {**_dataclass_payload(result), "validation_usage": totals}

    def _initial_prompt(self, prompt: str) -> str:
        return (
            f"{prompt.strip()}\n\n"
            "[Validation Harness public contract]\n"
            f"Train raw: {self.config.train_raw.resolve()}\n"
            f"Train reference: {self.config.train_reference.resolve()}\n"
            f"Validation raw: {self.config.validation_raw.resolve()}\n"
            f"Agent workdir: {self.agent_workdir}\n"
            "Create reproducible scripts and current validation results. Before ending this turn, "
            "write submission.json with schema_version=1, a relative result_root, and replay.argv "
            "as a JSON string array. Use {raw_root}, {train_raw}, {train_reference}, "
            "{output_dir}, and optionally {workdir}; do not use a shell command string. "
            "Hidden gold and host reports are not "
            "available to you. The host will return aggregate and per-file scores after the turn."
        )

    @staticmethod
    def _feedback_prompt(feedback: dict[str, Any]) -> str:
        return (
            "[Validation Harness feedback]\n"
            + json.dumps(feedback, ensure_ascii=False, indent=2)
            + "\nContinue from the same task and context. Inspect current files, improve the "
            "reproducible pipeline and result package, run your own checks, then end the turn."
        )

    def _replay_feedback(self, replay: ReplayExecution, best_score: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "phase": "independent_replay",
            "status": replay.status,
            "best_score": max(best_score, 0.0),
            "returncode": replay.returncode,
            "duration_seconds": replay.duration_seconds,
            "stderr": self._sanitize(replay.stderr[-4000:]),
            "message": "Independent replay failed or scored below the formal best. Fix the replayable pipeline.",
        }
        if replay.score_report is not None:
            public = public_feedback_from_equal_weight_report(
                replay.score_report,
                round_index=0,
                best_score=max(best_score, 0.0),
                promoted=False,
                restored_best=False,
            )
            payload["replay_score"] = public["current_score"]
            payload["files"] = public["files"]
        return payload

    @staticmethod
    def _replay_is_acceptable(replay: ReplayExecution, best_score: float) -> bool:
        return (
            replay.status == "SUCCESS"
            and replay.score_report is not None
            and replay.score >= max(best_score, 0.0) - SCORE_TOLERANCE
        )

    async def _run_replay(self, source: Path) -> ReplayExecution:
        runtime_root = self.host_dir / "runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        replay_root = Path(
            tempfile.mkdtemp(prefix="replay-", dir=runtime_root),
        )
        remove_replay_root = True
        try:
            return await self._run_replay_in_directory(source, replay_root)
        except ReplayProcessReapError:
            remove_replay_root = False
            raise
        finally:
            if remove_replay_root:
                shutil.rmtree(replay_root, ignore_errors=True)

    async def _run_replay_in_directory(
        self,
        source: Path,
        replay_root: Path,
    ) -> ReplayExecution:
        replay_workdir = replay_root / "workdir"
        try:
            validate_plain_directory_tree(source)
        except ValueError as exc:
            return ReplayExecution(
                status="INVALID_SOURCE",
                returncode=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=0.0,
                result_root=None,
                score_report=None,
            )
        snapshot_agent_workdir(source, replay_workdir)
        started = time.monotonic()
        try:
            submission = load_submission(replay_workdir)
        except Exception as exc:
            return ReplayExecution(
                status="INVALID_SUBMISSION",
                returncode=None,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.monotonic() - started,
                result_root=None,
                score_report=None,
            )
        _remove_path(submission.result_root) if submission.result_root.exists() else None
        submission.result_root.parent.mkdir(parents=True, exist_ok=True)
        home = replay_root / "home"
        scratch = replay_root / "tmp"
        home.mkdir()
        scratch.mkdir()
        profile = replay_root / "replay.sb"
        profile.write_text(
            build_macos_sandbox_profile(
                executable=Path(sys.executable),
                read_roots=[
                    replay_workdir,
                    self.config.validation_raw,
                    self.config.train_raw,
                    self.config.train_reference,
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                    Path("/System"),
                    Path("/usr"),
                    Path("/bin"),
                    Path("/sbin"),
                    Path("/private/etc"),
                ],
                write_roots=[replay_workdir, home, scratch, Path("/dev/null")],
                allow_network=False,
            ),
            encoding="utf-8",
        )
        argv = render_replay_argv(
            submission,
            raw_root=self.config.validation_raw,
            train_raw=self.config.train_raw,
            train_reference=self.config.train_reference,
            output_dir=submission.result_root,
            workdir=replay_workdir,
        )
        if argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        command = ["/usr/bin/sandbox-exec", "-f", str(profile), *argv]
        environment = {
            "HOME": str(home),
            "TMPDIR": str(scratch),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "en_US.UTF-8"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(replay_workdir),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.replay_timeout_seconds,
            )
        except asyncio.TimeoutError:
            stdout, stderr = await _kill_and_reap_process_group(process)
            return ReplayExecution(
                status="TIMEOUT",
                returncode=process.returncode,
                stdout=stdout.decode(errors="replace")[-8000:],
                stderr=stderr.decode(errors="replace")[-8000:],
                duration_seconds=time.monotonic() - started,
                result_root=submission.result_root,
                score_report=None,
            )
        except asyncio.CancelledError:
            await _kill_and_reap_process_group(process)
            raise
        stdout_text = stdout.decode(errors="replace")[-8000:]
        stderr_text = stderr.decode(errors="replace")[-8000:]
        if process.returncode != 0:
            return ReplayExecution(
                status="EXECUTION_FAILED",
                returncode=process.returncode,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_seconds=time.monotonic() - started,
                result_root=submission.result_root,
                score_report=None,
            )
        try:
            validate_plain_directory_tree(submission.result_root)
        except ValueError as exc:
            return ReplayExecution(
                status="INVALID_OUTPUT",
                returncode=0,
                stdout=stdout_text,
                stderr=str(exc),
                duration_seconds=time.monotonic() - started,
                result_root=submission.result_root,
                score_report=None,
            )
        report = self.scorer(
            submission.result_root,
            self.config.validation_gold,
            self._key_columns,
        )
        return ReplayExecution(
            status="SUCCESS",
            returncode=0,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=time.monotonic() - started,
            result_root=submission.result_root,
            score_report=report,
        )

    def _sanitize(self, text: str) -> str:
        return text.replace(str(self.config.validation_gold.resolve()), "<hidden_gold>")

    def _host_line(self, message: str) -> None:
        print(f"[Harness] {message}", file=self.output, flush=True)

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


async def verify_agent_sandbox_access(
    config: PiValidationHarnessConfig,
    launch: WorkerLaunch | None,
) -> None:
    if launch is None:
        raise RuntimeError("sandbox launch is required for the access probe")
    public_file = _first_regular_file(config.validation_raw)
    gold_file = _first_regular_file(config.validation_gold)
    link = config.experiment_dir.resolve() / "agent_workdir" / ".gold_access_probe"
    link.symlink_to(gold_file)
    script = (
        "import agent, json, pathlib, subprocess, sys\n"
        "def readable(p):\n"
        "  try:\n"
        "    pathlib.Path(p).open('rb').read(1); return True\n"
        "  except OSError:\n"
        "    return False\n"
        "cat_ok = subprocess.run(['/bin/cat', sys.argv[2]], stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL).returncode == 0\n"
        "print(json.dumps([readable(sys.argv[1]), readable(sys.argv[2]), readable(sys.argv[3]), cat_ok]))\n"
    )
    env = {
        key: value
        for key, value in launch.env.items()
        if key not in {"OPENAI_API_KEY", "OPENAI_API_KEYS_JSON"}
    }
    try:
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/sandbox-exec",
            "-f",
            str(launch.profile_path),
            sys.executable,
            "-c",
            script,
            str(public_file),
            str(gold_file),
            str(link),
            cwd=str(config.experiment_dir / "agent_workdir"),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"sandbox access probe failed: {stderr.decode(errors='replace')[-1000:]}")
        access = json.loads(stdout)
        if access != [True, False, False, False]:
            raise RuntimeError(f"sandbox access boundary is unsafe: {access}")
    finally:
        link.unlink(missing_ok=True)


def _first_regular_file(root: Path) -> Path:
    for path in sorted(root.expanduser().resolve().rglob("*")):
        if path.is_file() and not path.is_symlink():
            return path
    raise ValueError(f"directory contains no regular files: {root}")


def validate_plain_directory_tree(root: str | Path) -> None:
    """Reject links and special files before host code reads a replay tree."""
    base = Path(root).expanduser().absolute()
    if not base.is_dir() or base.is_symlink():
        raise ValueError("replay tree must be a real directory")
    for path in base.rglob("*"):
        mode = path.lstat().st_mode
        relative = path.relative_to(base).as_posix()
        if stat.S_ISLNK(mode):
            raise ValueError(f"replay tree contains a symbolic link: {relative}")
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"replay tree contains a special file: {relative}")


def _dataclass_payload(result: PiHarnessResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "stop_reason": result.stop_reason,
        "rounds": result.rounds,
        "repair_rounds": result.repair_rounds,
        "best_score": result.best_score,
        "reproducible_score": result.reproducible_score,
        "best_snapshot": str(result.best_snapshot),
        "reproducible_snapshot": str(result.reproducible_snapshot),
    }


def _turn_metrics(payload: Any) -> dict[str, Any]:
    value = payload if isinstance(payload, dict) else {}
    scalar_fields = (
        "status",
        "finished_reason",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "react_iterations",
        "duration_seconds",
        "tool_calls",
        "tool_errors",
        "api_failovers",
        "compactions",
    )
    result = {field: value.get(field) for field in scalar_fields if field in value}
    result["stream_event_counts"] = dict(value.get("stream_event_counts") or {})
    result["text_block_ids"] = list(value.get("text_block_ids") or [])
    result["thinking_block_ids"] = list(value.get("thinking_block_ids") or [])
    return result


def _file_sha256(path: str | Path) -> str:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise ValueError(f"identity file does not exist: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _directory_bundle_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for root in sorted((path.expanduser().resolve() for path in paths), key=str):
        if not root.is_dir():
            raise ValueError(f"skill directory does not exist: {root}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name in {".DS_Store"} or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _public_model_identity(environment: dict[str, str]) -> dict[str, Any]:
    try:
        keys = json.loads(environment.get("OPENAI_API_KEYS_JSON", "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("worker API key bundle is invalid") from exc
    if not isinstance(keys, list):
        raise ValueError("worker API key bundle must be a list")
    return {
        "name": environment.get("MODEL_NAME", "MiniMax-M3"),
        "base_url": environment.get("OPENAI_API_BASE", ""),
        "temperature": float(environment.get("AGENT_TEMPERATURE", "0")),
        "seed": int(environment["AGENT_SEED"]) if environment.get("AGENT_SEED") else None,
        "api_key_count": len(keys),
    }


def _git_identity(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()

    def run(*args: str) -> bytes:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(completed.stderr.decode(errors="replace").strip())
        return completed.stdout

    try:
        commit = run("rev-parse", "HEAD").decode().strip()
        status = run("status", "--porcelain=v1")
        diff = run("diff", "--no-ext-diff", "--binary", "HEAD")
        untracked = run("ls-files", "--others", "--exclude-standard", "-z")
    except (OSError, ValueError) as exc:
        return {"commit": "", "dirty": None, "dirty_diff_sha256": "", "error": str(exc)}
    digest = hashlib.sha256(diff)
    included_untracked = 0
    ignored_roots = {".test_runs", "datasets", "experiments", "output", "outputs", "reports"}
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        relative = Path(raw_path.decode(errors="surrogateescape"))
        if relative.parts and relative.parts[0] in ignored_roots:
            continue
        path = root / relative
        if not path.is_file():
            continue
        digest.update(b"U\0" + raw_path + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        included_untracked += 1
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_diff_sha256": digest.hexdigest(),
        "untracked_files_included": included_untracked,
    }


class _RedactingTee:
    def __init__(self, streams: tuple[Any, ...], secrets: tuple[str, ...]) -> None:
        self.streams = streams
        self.secrets = tuple(secret for secret in secrets if secret)

    def write(self, value: str) -> int:
        redacted = value
        for secret in self.secrets:
            redacted = redacted.replace(secret, "<redacted-api-key>")
        for stream in self.streams:
            stream.write(redacted)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _copy_agent_entries(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.name in SNAPSHOT_EXCLUDES:
            continue
        _copy_path(item, destination / item.name)


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_symlink():
        destination.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents
