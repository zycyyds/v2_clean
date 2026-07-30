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
            "status": status,
            "phase": phase,
            "validation_experiment": str(
                self.config.validation.experiment_dir.expanduser().resolve(),
            ),
            "test_experiment": str(self.config.test_experiment.expanduser().resolve()),
            "validation": {
                "status": validation.status,
                "stop_reason": validation.stop_reason,
                "rounds": validation.rounds,
                "repair_rounds": validation.repair_rounds,
                "best_score": validation.best_score,
                "reproducible_score": validation.reproducible_score,
                "reproducible_snapshot": str(validation.reproducible_snapshot),
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

    @staticmethod
    def _test_payload(test: PiTestHarnessResult | None) -> dict[str, Any] | None:
        if test is None:
            return None
        return {
            "status": test.status,
            "phase": test.phase,
            "scored": test.scored,
            "score": test.score,
            "replay_status": test.replay_status,
            "frozen_snapshot_sha256": test.frozen_snapshot_sha256,
            "frozen_snapshot": str(test.frozen_snapshot),
            "result_package": str(test.result_package) if test.result_package else "",
            "replay_duration_seconds": test.replay_duration_seconds,
            "scoring_duration_seconds": test.scoring_duration_seconds,
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
