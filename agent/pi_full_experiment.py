"""Host-only sequencing for the Validation and one-shot Test harnesses."""
from __future__ import annotations

import json
import os
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
        validation = await self.validation_runner(prompt)
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
        test = await self.test_runner()
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
    ) -> PiFullExperimentResult:
        scoring_execution_count = (
            1 if test is not None and test.phase in {"scoring", "complete"} else 0
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": self._redact_text(status),
            "phase": self._redact_text(phase),
            "validation_experiment": self._redact_text(
                str(self.config.validation.experiment_dir.expanduser().resolve()),
            ),
            "test_experiment": self._redact_text(
                str(self.config.test_experiment.expanduser().resolve()),
            ),
            "validation": {
                "status": self._redact_text(validation.status),
                "stop_reason": self._redact_text(validation.stop_reason),
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
            status=status,
            phase=phase,
            validation_result=validation,
            test_result=test,
            report_path=self.report_path,
        )

    def _test_payload(self, test: PiTestHarnessResult | None) -> dict[str, Any] | None:
        if test is None:
            return None
        return {
            "status": self._redact_text(test.status),
            "phase": self._redact_text(test.phase),
            "scored": test.scored,
            "score": test.score,
            "replay_status": self._redact_text(test.replay_status),
            "frozen_snapshot_sha256": self._redact_text(
                test.frozen_snapshot_sha256,
            ),
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
            path.expanduser().resolve().relative_to(public_root.expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            return ""
        return self._redact_text(str(path))

    def _redact_text(self, text: str) -> str:
        hidden_roots = (
            self.config.validation.validation_gold.expanduser().resolve(),
            self.config.test_gold.expanduser().resolve(),
        )
        redacted = text
        for root in sorted((str(path) for path in hidden_roots), key=len, reverse=True):
            redacted = redacted.replace(root, "<hidden>")
        return redacted


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
