"""Host-only sequencing for the Validation and one-shot Test harnesses."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
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
from agent.pi_harness_sandbox import (
    recursive_paths_overlap,
    require_path_isolated,
    require_test_gold_isolated,
    sandbox_visible_recursive_roots,
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
FULL_ROLE_TOPOLOGY_ERROR = "unsafe full experiment role topology"


def validate_full_experiment_role_topology(
    config: PiFullExperimentConfig,
    *,
    prompt_file: Path | None = None,
) -> None:
    """Reject path topologies that expose Test or Validation private inputs."""

    validation = config.validation
    recursive_roots = sandbox_visible_recursive_roots(
        validation.project_root / "agent",
        validation.project_root / "lib",
        validation.train_raw,
        validation.train_reference,
        validation.validation_raw,
        validation.experiment_dir,
        *validation.skill_dirs,
        include_shell_state=True,
    )
    literal_paths = (
        validation.project_root,
        validation.project_root / "config_loader.py",
        validation.project_root / "model_config.yaml",
        Path(sys.executable),
        Path("/dev/null"),
    )
    require_test_gold_isolated(
        config.test_gold,
        recursive_roots=(*recursive_roots, config.test_raw, config.test_experiment),
        literal_paths=literal_paths,
    )
    if recursive_paths_overlap(validation.validation_gold, config.test_gold):
        raise ValueError(FULL_ROLE_TOPOLOGY_ERROR)
    for validation_visible_data in (
        validation.train_raw,
        validation.train_reference,
        validation.validation_raw,
    ):
        if recursive_paths_overlap(config.test_raw, validation_visible_data):
            raise ValueError(FULL_ROLE_TOPOLOGY_ERROR)
    require_path_isolated(
        validation.validation_gold,
        recursive_roots,
        literal_paths=literal_paths,
        error_message=FULL_ROLE_TOPOLOGY_ERROR,
    )

    if prompt_file is not None:
        for gold_root in (validation.validation_gold, config.test_gold):
            if recursive_paths_overlap(prompt_file, gold_root):
                raise ValueError(FULL_ROLE_TOPOLOGY_ERROR)


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
        validate_full_experiment_role_topology(self.config)
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

        if validation.status == "INTERRUPTED":
            return self._finish(
                status="INTERRUPTED",
                phase="validation",
                validation=validation,
                test=None,
                validation_duration=validation_duration,
                test_duration=0.0,
                duration=time.monotonic() - started,
                error_code="VALIDATION_CANCELLED",
            )

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
            test = self._test_terminal(cancelled=True)
            return self._finish(
                status=test.status,
                phase=test.phase,
                validation=validation,
                test=test,
                validation_duration=validation_duration,
                test_duration=time.monotonic() - test_started,
                duration=time.monotonic() - started,
                error_code="TEST_CANCELLED",
            )
        except Exception:
            test = self._test_terminal(cancelled=False)
            return self._finish(
                status=test.status,
                phase=test.phase,
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

    def _test_terminal(self, *, cancelled: bool) -> PiTestHarnessResult:
        host_dir = self.config.test_experiment.expanduser().resolve() / "host"
        test_hash = _start_marker_hash(host_dir / "test_started.json")
        scoring_hash = _start_marker_hash(host_dir / "scoring_started.json")
        if scoring_hash:
            status = "INTERRUPTED" if cancelled else "SCORING_FAILED"
            phase = "scoring"
            replay_status = "SUCCESS"
            execution_count = 1
            frozen_hash = scoring_hash if scoring_hash == test_hash else ""
        else:
            status = "INTERRUPTED" if cancelled else "REPLAY_FAILED"
            phase = "test_replay"
            replay_status = "INTERRUPTED" if cancelled else "EXECUTION_FAILED"
            execution_count = int(bool(test_hash))
            frozen_hash = test_hash
        return PiTestHarnessResult(
            status=status,
            phase=phase,
            scored=False,
            score=None,
            test_execution_count=execution_count,
            replay_status=replay_status,
            frozen_snapshot_sha256=frozen_hash,
            frozen_snapshot=Path(),
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
        if not resolved.exists():
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


def _start_marker_hash(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        return ""
    return _sha256_text(payload.get("frozen_snapshot_sha256"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
