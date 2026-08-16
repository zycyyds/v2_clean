from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.pi_harness import directory_sha256
from agent.pi_harness_sandbox import require_path_isolated
from agent.pi_train_repair import (
    RepairReplayExecution,
    freeze_repair_source,
    run_frozen_repair_replay,
)
from workflow.pi_repair_audit import score_repair_output, write_sanitized_failure_cases


@dataclass(frozen=True)
class PiRepairTestConfig:
    project_root: Path
    train_experiment: Path
    test_experiment: Path
    test_raw: Path
    test_gold_log: Path
    replay_timeout_seconds: float = 1_800.0


@dataclass(frozen=True)
class PiRepairTestResult:
    status: str
    phase: str
    test_execution_count: int
    frozen_snapshot_sha256: str
    output_root: Path | None
    score_report: Path | None
    replay_status: str


TestReplayRunner = Callable[..., Awaitable[RepairReplayExecution]]


class PiRepairTestHarness:
    """Run one frozen repair pipeline replay and one host-private audit."""

    def __init__(
        self,
        config: PiRepairTestConfig,
        *,
        replay_runner: TestReplayRunner = run_frozen_repair_replay,
        output: Any | None = None,
    ) -> None:
        self.config = config
        self.replay_runner = replay_runner
        self.output = output or sys.stderr
        self.experiment = config.test_experiment.expanduser().resolve()
        self.host_dir = self.experiment / "host"
        self.frozen_snapshot = self.host_dir / "frozen_snapshot"
        self.test_output = self.host_dir / "test_output"

    async def run(self) -> PiRepairTestResult:
        source, source_hash, runtime_hash = self._prepare()
        _atomic_json(
            self.host_dir / "test_started.json",
            {
                "schema_version": 1,
                "started_at_unix": time.time(),
                "train_frozen_snapshot_sha256": source_hash,
                "frozen_snapshot_sha256": runtime_hash,
                "test_execution_count": 1,
            },
        )
        self._host_line("frozen Test replay started")
        try:
            replay = await self.replay_runner(
                source=self.frozen_snapshot,
                raw_root=self.config.test_raw,
                retained_output=self.test_output,
                runtime_root=self.host_dir / "replay_runtime",
                timeout_seconds=self.config.replay_timeout_seconds,
            )
        except asyncio.CancelledError:
            result = PiRepairTestResult(
                "INTERRUPTED",
                "test_replay",
                1,
                runtime_hash,
                None,
                None,
                "INTERRUPTED",
            )
            self._finish(result, None)
            return result
        if directory_sha256(source) != source_hash:
            raise RuntimeError("source Train snapshot changed during Test")
        if directory_sha256(self.frozen_snapshot) != runtime_hash:
            raise RuntimeError("frozen Test snapshot changed during replay")
        if replay.status != "SUCCESS" or replay.output_root is None:
            result = PiRepairTestResult(
                "REPLAY_FAILED",
                "test_replay",
                1,
                runtime_hash,
                replay.output_root,
                None,
                replay.status,
            )
            self._finish(result, replay)
            return result

        self._host_line("host-private raw repair scoring started")
        try:
            report = score_repair_output(
                output_root=replay.output_root,
                raw_root=self.config.test_raw,
                gold_log=self.config.test_gold_log,
            )
        except Exception as exc:
            result = PiRepairTestResult(
                "SCORING_FAILED",
                "host_private_scoring",
                1,
                runtime_hash,
                replay.output_root,
                None,
                replay.status,
            )
            self._finish(result, replay, error=f"{type(exc).__name__}: {exc}")
            return result

        score_path = self.host_dir / "score_report.json"
        _atomic_json(score_path, report)
        write_sanitized_failure_cases(
            report, self.host_dir / "sanitized_failure_cases.json"
        )
        result = PiRepairTestResult(
            "SUCCESS",
            "complete",
            1,
            runtime_hash,
            replay.output_root,
            score_path,
            replay.status,
        )
        self._finish(result, replay)
        return result

    def _prepare(self) -> tuple[Path, str, str]:
        train = _require_directory(self.config.train_experiment, "Train experiment")
        source = _require_directory(
            train / "host/reproducible_snapshot", "reproducible Train snapshot"
        )
        train_report_path = _require_file(train / "host/run_report.json", "Train run report")
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
        if train_report.get("status") != "SUCCESS_REPRODUCIBLE":
            raise ValueError("Train experiment is not SUCCESS_REPRODUCIBLE")
        source_hash = directory_sha256(source)
        expected_hash = str(train_report.get("frozen_snapshot_sha256") or "")
        if not expected_hash or expected_hash != source_hash:
            raise ValueError("Train frozen snapshot hash mismatch")
        if self.experiment.exists() and (
            not self.experiment.is_dir() or any(self.experiment.iterdir())
        ):
            raise ValueError("Test experiment directory must be new and empty")
        _require_directory(self.config.test_raw, "Test raw")
        _require_file(self.config.test_gold_log, "Test Gold log")
        if (
            not math.isfinite(self.config.replay_timeout_seconds)
            or self.config.replay_timeout_seconds <= 0
        ):
            raise ValueError("replay timeout must be positive and finite")
        for private, label in ((self.config.test_gold_log, "Test Gold"),):
            require_path_isolated(
                private,
                recursive_roots=(
                    self.config.test_raw,
                    source,
                    self.experiment,
                ),
                error_message=f"unsafe {label} path overlap",
            )

        self.host_dir.mkdir(parents=True)
        freeze_repair_source(source, self.frozen_snapshot)
        runtime_hash = directory_sha256(self.frozen_snapshot)
        _atomic_json(
            self.host_dir / "run_manifest.json",
            {
                "schema_version": 1,
                "mode": "one_shot_internal_test_raw_repair_audit",
                "train_experiment_sha256": _sha256(train_report_path),
                "test_raw_sha256": _directory_sha256(self.config.test_raw),
                "test_gold_log_sha256": _sha256(self.config.test_gold_log),
                "train_frozen_snapshot_sha256": source_hash,
                "frozen_snapshot_sha256": runtime_hash,
                "test_execution_limit": 1,
                "model_access": False,
                "api_credentials_required": False,
                "reporting_label": "internal_development_test_audit",
            },
        )
        return source, source_hash, runtime_hash

    def _finish(
        self,
        result: PiRepairTestResult,
        replay: RepairReplayExecution | None,
        *,
        error: str = "",
    ) -> None:
        payload: dict[str, Any] = {
            "status": result.status,
            "phase": result.phase,
            "test_execution_count": result.test_execution_count,
            "frozen_snapshot_sha256": result.frozen_snapshot_sha256,
            "output_root": str(result.output_root) if result.output_root else "",
            "score_report": str(result.score_report) if result.score_report else "",
            "replay_status": result.replay_status,
            "replay_duration_seconds": replay.duration_seconds if replay else 0.0,
        }
        if result.score_report and result.score_report.is_file():
            report = json.loads(result.score_report.read_text(encoding="utf-8"))
            payload["metrics"] = report.get("metrics") or {}
        if replay is not None:
            payload["replay"] = {
                "returncode": replay.returncode,
                "stderr": replay.stderr[-4000:],
            }
        if error:
            payload["error"] = error
        _atomic_json(self.host_dir / "run_report.json", payload)
        self._host_line(f"status={result.status} replay={result.replay_status}")

    def _host_line(self, message: str) -> None:
        print(f"[Pi Repair Test] {message}", file=self.output, flush=True)


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


def _sha256(path: str | Path) -> str:
    digest = __import__("hashlib").sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(root: str | Path) -> str:
    base = Path(root).expanduser().resolve()
    digest = __import__("hashlib").sha256()
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        digest.update(path.relative_to(base).as_posix().encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
