"""Host-only sequencing for the Validation and one-shot Test harnesses."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.pi_harness import (
    PiHarnessResult,
    PiValidationHarness,
    PiValidationHarnessConfig,
)
from agent.pi_test_harness import (
    PiTestHarness,
    PiTestHarnessConfig,
    PiTestHarnessResult,
)


@dataclass(frozen=True)
class PiFullExperimentConfig:
    validation: PiValidationHarnessConfig
    test_experiment: Path
    test_raw: Path
    test_gold: Path
    evaluation_manifest: Path
    test_replay_timeout_seconds: float = 1_800.0
    test_scoring_timeout_seconds: float = 3_600.0


@dataclass(frozen=True)
class PiFullExperimentResult:
    status: str
    phase: str
    validation_result: PiHarnessResult
    test_result: PiTestHarnessResult | None
    report_path: Path


ValidationRunner = Callable[[str], Awaitable[PiHarnessResult]]
TestRunner = Callable[[], Awaitable[PiTestHarnessResult]]

VALIDATION_STATUSES = frozenset(
    {"SUCCESS_REPRODUCIBLE", "INTERRUPTED", "FAILED", "REPLAY_FAILED"},
)
TEST_STATUSES = frozenset(
    {
        "SUCCESS",
        "PREFLIGHT_FAILED",
        "REPLAY_FAILED",
        "SCORING_FAILED",
        "INTERRUPTED",
        "CLEANUP_FAILED",
    },
)
FULL_STATUSES = TEST_STATUSES | {"VALIDATION_FAILED"}
PHASES = frozenset(
    {"validation", "preflight", "test_replay", "scoring", "complete"},
)
REPLAY_STATUSES = frozenset(
    {
        "SUCCESS",
        "INVALID_SOURCE",
        "INVALID_SUBMISSION",
        "TIMEOUT",
        "EXECUTION_FAILED",
        "INVALID_OUTPUT",
        "INTERRUPTED",
        "CLEANUP_FAILED",
    },
)
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
ERROR_CODES = frozenset(
    {
        "",
        "UNKNOWN",
        "VALIDATION_EXCEPTION",
        "VALIDATION_CANCELLED",
        "TEST_EXCEPTION",
        "TEST_CANCELLED",
    },
)


class PiFullExperiment:
    """Run Validation and gate exactly one Test harness invocation on success."""

    def __init__(
        self,
        config: PiFullExperimentConfig,
        *,
        validation_runner: ValidationRunner | None = None,
        test_runner: TestRunner | None = None,
    ) -> None:
        self.config = config
        self.validation_runner = validation_runner or self._run_validation
        self.test_runner = test_runner or self._run_test
        self.report_path = (
            config.validation.experiment_dir.expanduser().resolve()
            / "host/full_experiment_report.json"
        )

    async def run(self, prompt: str) -> PiFullExperimentResult:
        started = time.monotonic()
        validation_started = time.monotonic()
        try:
            validation = await self.validation_runner(prompt)
        except asyncio.CancelledError:
            validation = self._validation_terminal("INTERRUPTED", "validation_cancelled")
            return self._finish(
                status="INTERRUPTED",
                phase="validation",
                validation=validation,
                test=None,
                validation_duration=time.monotonic() - validation_started,
                test_duration=0.0,
                duration=time.monotonic() - started,
                error_code="VALIDATION_CANCELLED",
            )
        except Exception:
            validation = self._validation_terminal("FAILED", "validation_exception")
            return self._finish(
                status="VALIDATION_FAILED",
                phase="validation",
                validation=validation,
                test=None,
                validation_duration=time.monotonic() - validation_started,
                test_duration=0.0,
                duration=time.monotonic() - started,
                error_code="VALIDATION_EXCEPTION",
            )
        validation_duration = time.monotonic() - validation_started

        if validation.status != "SUCCESS_REPRODUCIBLE":
            return self._finish(
                status="VALIDATION_FAILED",
                phase="validation",
                validation=validation,
                test=None,
                validation_duration=validation_duration,
                test_duration=0.0,
                duration=time.monotonic() - started,
            )

        test_started = time.monotonic()
        try:
            test = await self.test_runner()
        except asyncio.CancelledError:
            test = self._test_terminal("INTERRUPTED", "INTERRUPTED")
            return self._finish(
                status="INTERRUPTED",
                phase="test_replay",
                validation=validation,
                test=test,
                validation_duration=validation_duration,
                test_duration=time.monotonic() - test_started,
                duration=time.monotonic() - started,
                error_code="TEST_CANCELLED",
            )
        except Exception:
            test = self._test_terminal("REPLAY_FAILED", "EXECUTION_FAILED")
            return self._finish(
                status="REPLAY_FAILED",
                phase="test_replay",
                validation=validation,
                test=test,
                validation_duration=validation_duration,
                test_duration=time.monotonic() - test_started,
                duration=time.monotonic() - started,
                error_code="TEST_EXCEPTION",
            )
        test_duration = time.monotonic() - test_started
        return self._finish(
            status="SUCCESS" if test.status == "SUCCESS" else test.status,
            phase="complete" if test.status == "SUCCESS" else test.phase,
            validation=validation,
            test=test,
            validation_duration=validation_duration,
            test_duration=test_duration,
            duration=time.monotonic() - started,
        )

    def _validation_terminal(self, status: str, stop_reason: str) -> PiHarnessResult:
        host_dir = self.config.validation.experiment_dir.expanduser().resolve() / "host"
        return PiHarnessResult(
            status=status,
            stop_reason=stop_reason,
            rounds=0,
            repair_rounds=0,
            best_score=0.0,
            reproducible_score=0.0,
            best_snapshot=host_dir / "best_snapshot",
            reproducible_snapshot=host_dir / "reproducible_snapshot",
        )

    def _test_terminal(self, status: str, replay_status: str) -> PiTestHarnessResult:
        host_dir = self.config.test_experiment.expanduser().resolve() / "host"
        execution_count = int((host_dir / "test_started.json").exists())
        return PiTestHarnessResult(
            status=status,
            phase="test_replay",
            scored=False,
            score=None,
            test_execution_count=execution_count,
            replay_status=replay_status,
            frozen_snapshot_sha256="",
            frozen_snapshot=host_dir / "frozen_snapshot",
            result_package=None,
            preflight_status="NOT_RUN",
            replay_duration_seconds=0.0,
            scoring_duration_seconds=0.0,
        )

    async def _run_validation(self, prompt: str) -> PiHarnessResult:
        return await PiValidationHarness(self.config.validation).run(prompt)

    async def _run_test(self) -> PiTestHarnessResult:
        test_config = PiTestHarnessConfig(
            project_root=self.config.validation.project_root,
            validation_experiment=self.config.validation.experiment_dir,
            preflight_attestation=None,
            test_experiment=self.config.test_experiment,
            test_raw=self.config.test_raw,
            test_gold=self.config.test_gold,
            evaluation_manifest=self.config.evaluation_manifest,
            replay_timeout_seconds=self.config.test_replay_timeout_seconds,
            scoring_timeout_seconds=self.config.test_scoring_timeout_seconds,
        )
        return await PiTestHarness(test_config).run()

    def _finish(
        self,
        *,
        status: str,
        phase: str,
        validation: PiHarnessResult,
        test: PiTestHarnessResult | None,
        validation_duration: float,
        test_duration: float,
        duration: float,
        error_code: str = "",
    ) -> PiFullExperimentResult:
        normalized_status = _known_text(status, FULL_STATUSES, "UNKNOWN")
        normalized_phase = _known_text(phase, PHASES, "unknown")
        scoring_execution_count = (
            1 if test is not None and test.phase in {"scoring", "complete"} else 0
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": normalized_status,
            "phase": normalized_phase,
            "error_code": _known_text(error_code, ERROR_CODES, "UNKNOWN"),
            "validation_experiment": self._redact_path_text(
                str(self.config.validation.experiment_dir.expanduser().resolve()),
            ),
            "test_experiment": self._redact_path_text(
                str(self.config.test_experiment.expanduser().resolve()),
            ),
            "validation": {
                "status": _known_text(
                    validation.status,
                    VALIDATION_STATUSES,
                    "UNKNOWN",
                ),
                "rounds": validation.rounds,
                "repair_rounds": validation.repair_rounds,
                "best_score": validation.best_score,
                "reproducible_score": validation.reproducible_score,
                "reproducible_snapshot": self._public_path(
                    validation.reproducible_snapshot,
                    self.config.validation.experiment_dir,
                ),
            },
            "test": self._test_payload(test),
            "test_execution_count": test.test_execution_count if test is not None else 0,
            "scoring_execution_count": scoring_execution_count,
            "validation_duration_seconds": round(validation_duration, 6),
            "test_duration_seconds": round(test_duration, 6),
            "duration_seconds": round(duration, 6),
        }
        _atomic_json(self.report_path, payload)
        return PiFullExperimentResult(
            status=normalized_status,
            phase=normalized_phase,
            validation_result=validation,
            test_result=test,
            report_path=self.report_path,
        )

    def _test_payload(self, test: PiTestHarnessResult | None) -> dict[str, Any] | None:
        if test is None:
            return None
        return {
            "status": _known_text(test.status, TEST_STATUSES, "UNKNOWN"),
            "phase": _known_text(test.phase, PHASES, "unknown"),
            "scored": test.scored,
            "score": test.score,
            "replay_status": _known_text(
                test.replay_status,
                REPLAY_STATUSES,
                "UNKNOWN",
            ),
            "frozen_snapshot_sha256": _sha256_text(test.frozen_snapshot_sha256),
            "frozen_snapshot": self._public_path(
                test.frozen_snapshot,
                self.config.test_experiment,
            ),
            "result_package": self._public_path(
                test.result_package,
                self.config.test_experiment,
            ),
            "replay_duration_seconds": test.replay_duration_seconds,
            "scoring_duration_seconds": test.scoring_duration_seconds,
        }

    def _public_path(self, path: Path | None, public_root: Path) -> str:
        if path is None:
            return ""
        try:
            resolved = path.expanduser().resolve()
            resolved.relative_to(public_root.expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            return ""
        return self._redact_path_text(str(resolved))

    def _redact_path_text(self, text: str) -> str:
        hidden_roots = (
            self.config.validation.validation_gold.expanduser().resolve(),
            self.config.test_gold.expanduser().resolve(),
        )
        redacted = text
        for root in sorted((str(path) for path in hidden_roots), key=len, reverse=True):
            redacted = redacted.replace(root, "<hidden>")
        return redacted


def _known_text(value: str, allowed: frozenset[str], fallback: str) -> str:
    return value if isinstance(value, str) and value in allowed else fallback


def _sha256_text(value: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        return ""
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
