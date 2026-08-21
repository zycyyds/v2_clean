from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.pi_harness import (
    _close_worker_without_losing_cancellation,
    _directory_bundle_sha256,
    _public_model_identity,
)
from agent.pi_worker_client import (
    JsonlWorkerClient,
    SandboxedWorkerConfig,
    build_sandboxed_worker_launch,
)
from agent.pi_train_repair import default_repair_skill_dirs
from graph.cell_repair import (
    FCORR_MINIMUM_RECALL_AT_5,
    MAX_CANDIDATES,
    RULE_SCHEMA_VERSION,
    CellRepairError,
    _field_key,
    _normalized_model_name,
    _read_json,
    _read_pair_manifest,
    _sha256,
    _static_rule_issues,
    _validate_evidence,
    freeze_rule_registry,
    validate_fcorr,
)
from lib.agent_runtime import build_worker_model_environment


@dataclass(frozen=True)
class PiFcorrSynthesisConfig:
    project_root: Path
    evidence_dir: Path
    output_dir: Path
    agent_key: str = "react_planner"
    max_rounds: int = 8
    max_iters: int = 10_000
    fields: tuple[str, ...] = ()
    skill_dirs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PiFcorrSynthesisResult:
    status: str
    rounds: int
    output_dir: Path
    rule_count: int
    field_count: int
    registry: dict[str, Any]


@dataclass
class _FieldBest:
    source: str = ""
    validation: dict[str, Any] | None = None
    round_index: int = 0

    @property
    def recall_at_5(self) -> float:
        metrics = (self.validation or {}).get("metrics") or {}
        return float(metrics.get("candidate_recall_at_5") or 0.0)

    @property
    def accepted(self) -> bool:
        return bool(
            self.source
            and self.validation
            and self.validation.get("status") == "SUCCESS"
        )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _public_turn_metrics(turn: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "duration_seconds",
        "iterations",
        "model_calls",
        "tool_calls",
        "compression_count",
    }
    return {key: turn[key] for key in allowed if key in turn}


class PiFcorrSynthesisHarness:
    """Persistent Train-only Pi session that emits frozen field candidate rules."""

    def __init__(
        self,
        config: PiFcorrSynthesisConfig,
        *,
        worker: Any | None = None,
        worker_probe: Callable[..., Any] | None = None,
        output: Any | None = None,
    ) -> None:
        self.config = config
        self.worker = worker
        self.worker_probe = worker_probe
        self.output = output or sys.stderr
        self.project = config.project_root.expanduser().resolve()
        self.evidence_dir = config.evidence_dir.expanduser().resolve()
        self.output_dir = config.output_dir.expanduser().resolve()
        self.agent_workdir = self.output_dir / "agent_workdir"
        self.runtime_dir = self.output_dir / "agent_runtime"
        self.feedback_dir = self.agent_workdir / "host_feedback"
        self._model_environment: dict[str, str] | None = None
        self._turn_metrics: list[dict[str, Any]] = []

    async def run(self) -> PiFcorrSynthesisResult:
        pair_manifest, entries = self._prepare()
        best = {str(entry["field_id"]): _FieldBest() for entry in entries}
        histories: dict[str, list[dict[str, Any]]] = {
            str(entry["field_id"]): [] for entry in entries
        }
        rounds = 0
        cancelled = False
        try:
            if self.worker is None:
                self.worker = self._create_worker()
            if self.worker_probe is not None:
                probe = self.worker_probe(self.config, self.agent_workdir)
                if asyncio.iscoroutine(probe):
                    await probe
            await self.worker.start()
            prompt = self._initial_prompt(entries)
            for round_index in range(1, self.config.max_rounds + 1):
                rounds = round_index
                self._host_line(f"Pi F_corr Train round {round_index} started")
                started = time.monotonic()
                turn = await self.worker.run_turn(prompt)
                self._turn_metrics.append({
                    "round": round_index,
                    "duration_seconds": round(time.monotonic() - started, 6),
                    **_public_turn_metrics(turn),
                })
                report = self._validate_round(entries, best, histories, round_index)
                feedback_path = self.feedback_dir / f"round_{round_index:02d}.json"
                _atomic_json(feedback_path, report)
                accepted = sum(item.accepted for item in best.values())
                self._host_line(
                    f"Pi F_corr Train round {round_index}: accepted={accepted}/{len(entries)}"
                )
                if accepted == len(entries):
                    break
                if round_index < self.config.max_rounds:
                    await self.worker.clear_file_cache()
                    prompt = self._revision_prompt(feedback_path, entries, best)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if self.worker is not None:
                cleanup_cancelled = await _close_worker_without_losing_cancellation(self.worker)
                if cleanup_cancelled and not cancelled:
                    raise asyncio.CancelledError

        registry = self._materialize(pair_manifest, entries, best, histories, rounds)
        result = PiFcorrSynthesisResult(
            status="SUCCESS",
            rounds=rounds,
            output_dir=self.output_dir,
            rule_count=int(registry["rule_count"]),
            field_count=len(entries),
            registry=registry,
        )
        _atomic_json(
            self.output_dir / "pi_agent_run_report.json",
            {
                "schema_version": 1,
                "status": result.status,
                "workflow": "pi_agent_train_only_fcorr_synthesis",
                "rounds": rounds,
                "field_count": len(entries),
                "rule_count": result.rule_count,
                "turn_metrics": self._turn_metrics,
                "skills": {
                    "directories": [path.name for path in self._skill_dirs()],
                    "bundle_sha256": _directory_bundle_sha256(self._skill_dirs()),
                },
                "model": _public_model_identity(self._model_environment or {}),
                "validation_or_test_model_access": False,
            },
        )
        return result

    def _prepare(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not 1 <= self.config.max_rounds <= 12:
            raise CellRepairError("Pi F_corr max_rounds must be between 1 and 12")
        if self.config.max_iters < 1:
            raise CellRepairError("Pi F_corr max_iters must be positive")
        if self.output_dir.exists():
            raise CellRepairError(f"output directory already exists: {self.output_dir}")
        pair_manifest, entries = _read_pair_manifest(self.evidence_dir)
        selected = set(self.config.fields)
        known = {
            _field_key(str(entry["table"]), str(entry["column"])) for entry in entries
        }
        unknown = selected - known
        if unknown:
            raise CellRepairError(f"unknown requested fields: {sorted(unknown)}")
        entries = [
            entry for entry in entries
            if not selected
            or _field_key(str(entry["table"]), str(entry["column"])) in selected
        ]
        if not entries:
            raise CellRepairError("Pi F_corr selected no fields")
        for entry in entries:
            evidence = _read_json(
                self.evidence_dir / str(entry["path"]),
                f"field evidence {entry['field_id']}",
            )
            _validate_evidence(evidence)
        for skill in self._skill_dirs():
            if not (skill / "SKILL.md").is_file():
                raise CellRepairError(f"Pi correction skill is missing: {skill}")
        self.output_dir.mkdir(parents=True)
        self.agent_workdir.mkdir()
        (self.agent_workdir / "rules").mkdir()
        self.feedback_dir.mkdir()
        _atomic_text(
            self.agent_workdir / "validate_rule.py",
            """from __future__ import annotations

import json
import sys
from pathlib import Path

from graph.cell_repair import validate_fcorr


if len(sys.argv) != 3:
    raise SystemExit("usage: python validate_rule.py <evidence.json> <correction.py>")
evidence = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(sys.argv[2]).read_text(encoding="utf-8")
print(json.dumps(validate_fcorr(source, evidence), ensure_ascii=False, indent=2))
""",
        )
        shutil.copyfile(
            self.evidence_dir / "field_pairs_manifest.json",
            self.output_dir / "pair_manifest.json",
        )
        return pair_manifest, entries

    def _skill_dirs(self) -> tuple[Path, ...]:
        return self.config.skill_dirs or default_repair_skill_dirs(self.project)

    def _create_worker(self) -> JsonlWorkerClient:
        self._model_environment = build_worker_model_environment(
            self.config.agent_key,
            "MiniMax-M3",
        )
        model_name = self._model_environment.get("MODEL_NAME", "")
        if "minimaxm3" not in _normalized_model_name(model_name):
            raise CellRepairError(f"Pi F_corr requires MiniMax M3, got {model_name!r}")
        launch = build_sandboxed_worker_launch(
            SandboxedWorkerConfig(
                project_root=self.project,
                agent_workdir=self.agent_workdir,
                runtime_root=self.runtime_dir,
                public_read_roots=(self.evidence_dir, self.project / "graph"),
                skill_dirs=self._skill_dirs(),
                max_iters=self.config.max_iters,
                model_environment=self._model_environment,
            )
        )
        return JsonlWorkerClient(
            command=launch.command,
            cwd=launch.cwd,
            env=launch.env,
            output=self.output,
        )

    def _initial_prompt(self, entries: list[dict[str, Any]]) -> str:
        fields = [
            {
                "field_id": str(entry["field_id"]),
                "field": _field_key(str(entry["table"]), str(entry["column"])),
                "evidence_file": str((self.evidence_dir / str(entry["path"])).resolve()),
                "output_file": str(
                    (self.agent_workdir / "rules" / str(entry["field_id"]) / "correction.py")
                    .resolve()
                ),
            }
            for entry in entries
        ]
        contract = {
            "project_root": str(self.project),
            "validator_module": str((self.project / "graph" / "cell_repair.py").resolve()),
            "validator_import": "from graph.cell_repair import validate_fcorr",
            "validator_command": (
                f"python {self.agent_workdir / 'validate_rule.py'} "
                "<evidence_file> <output_file>"
            ),
            "evidence_root": str(self.evidence_dir),
            "workdir": str(self.agent_workdir),
            "fields": fields,
        }
        return """\
[Pi Agent Train-only F_corr contract]

Generate deterministic Top-5 candidate functions from the de-identified Train-only field evidence.
There is no Validation or Test access. Use the four loaded correction skills as analysis lenses:
intra-table, entity-alignment, cross-table, and task-oriented. Read their SKILL.md files before
designing rules. Use file tools and local Python statistics; do not paste whole evidence files into
the model conversation and do not inspect paths outside the declared evidence root and workdir.

For every listed field, write exactly one output correction.py. It must define exactly:
def GenerateCandidates(input_string, row_context):
and return zero to five unique ordered dicts with exactly string keys value, rule_id, evidence.
The final source may not import modules, access files or networks, use randomness, memorize row or
patient identifiers, or use private paths. The runtime supplies re; allowed regex calls are
re.fullmatch, re.match, re.search and re.sub. Return [] when Train evidence does not establish a
rule. Analyze complete evidence with scripts and verify every source locally with
graph.cell_repair.validate_fcorr before finishing. Do not generate corrected data and do not select
or apply repairs. The worker PYTHONPATH already contains project_root. Do not search the filesystem for the
validator, read the full validator module, or list project_root. Prefer the declared
validator_command, which prints the complete Train validation JSON; the host will independently
validate every source again after the turn.

Declared contract:\n""" + json.dumps(contract, ensure_ascii=False, indent=2)

    def _revision_prompt(
        self,
        feedback_path: Path,
        entries: list[dict[str, Any]],
        best: dict[str, _FieldBest],
    ) -> str:
        unresolved = [
            _field_key(str(entry["table"]), str(entry["column"]))
            for entry in entries
            if not best[str(entry["field_id"])].accepted
        ]
        return (
            "Continue the same Train-only F_corr task. Host validation is in "
            f"{feedback_path}. Read that JSON with a file tool and revise only unresolved fields: "
            + json.dumps(unresolved, ensure_ascii=False)
            + ". Keep accepted field sources unchanged. Diagnose broad missing patterns with local "
            "scripts instead of adding row-specific mappings. Re-run validate_fcorr for changed "
            "sources and finish only after writing the complete correction.py files."
        )

    def _validate_round(
        self,
        entries: list[dict[str, Any]],
        best: dict[str, _FieldBest],
        histories: dict[str, list[dict[str, Any]]],
        round_index: int,
    ) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        for entry in entries:
            field_id = str(entry["field_id"])
            field = _field_key(str(entry["table"]), str(entry["column"]))
            source_path = self.agent_workdir / "rules" / field_id / "correction.py"
            evidence = _read_json(
                self.evidence_dir / str(entry["path"]),
                f"field evidence {field}",
            )
            source = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
            if not source:
                validation = {
                    "status": "FAILED",
                    "issues": ["correction.py is missing"],
                    "metrics": {},
                    "missing_candidate_pairs": [],
                }
            else:
                validation = validate_fcorr(
                    source,
                    evidence,
                    minimum_recall_at_5=FCORR_MINIMUM_RECALL_AT_5,
                )
            current = {
                "round": round_index,
                "field_id": field_id,
                "field": field,
                "source_sha256": _text_sha256(source) if source else "",
                "validation": validation,
            }
            histories[field_id].append(current)
            candidate_recall = float(
                ((validation.get("metrics") or {}).get("candidate_recall_at_5") or 0.0)
            )
            existing = best[field_id]
            candidate_key = (
                validation.get("status") == "SUCCESS",
                candidate_recall,
                float((validation.get("metrics") or {}).get("candidate_recall_at_1") or 0.0),
            )
            existing_key = (
                existing.accepted,
                existing.recall_at_5,
                float(((existing.validation or {}).get("metrics") or {}).get(
                    "candidate_recall_at_1"
                ) or 0.0),
            )
            if source and candidate_key > existing_key:
                best[field_id] = _FieldBest(source, validation, round_index)
            reports.append({
                **current,
                "best_round": best[field_id].round_index,
                "best_accepted": best[field_id].accepted,
                "best_candidate_recall_at_5": best[field_id].recall_at_5,
            })
        return {
            "schema_version": 1,
            "status": "SUCCESS",
            "workflow": "pi_agent_train_only_fcorr_feedback",
            "round": round_index,
            "minimum_candidate_recall_at_5_exclusive": FCORR_MINIMUM_RECALL_AT_5,
            "accepted_count": sum(item.accepted for item in best.values()),
            "field_count": len(entries),
            "fields": reports,
        }

    def _materialize(
        self,
        pair_manifest: dict[str, Any],
        entries: list[dict[str, Any]],
        best: dict[str, _FieldBest],
        histories: dict[str, list[dict[str, Any]]],
        rounds: int,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        model_name = (
            (self._model_environment or {}).get("MODEL_NAME")
            or "MiniMax-M3"
        )
        for entry in entries:
            field_id = str(entry["field_id"])
            field_output = self.output_dir / "fields" / field_id
            field_output.mkdir(parents=True)
            evidence_path = field_output / "evidence.json"
            shutil.copyfile(self.evidence_dir / str(entry["path"]), evidence_path)
            conversation_path = field_output / "conversation.json"
            _atomic_json(conversation_path, {
                "runtime": "persistent_pi_agent",
                "round_count": rounds,
                "history": histories[field_id],
            })
            selected = best[field_id]
            status = "FROZEN" if selected.accepted else "FCORR_REJECTED"
            validation = selected.validation or {
                "status": "FAILED",
                "issues": ["Agent produced no valid source"],
                "metrics": {},
                "missing_candidate_pairs": [],
            }
            field_manifest: dict[str, Any] = {
                "schema_version": RULE_SCHEMA_VERSION,
                "status": status,
                "workflow": "gidcl_multicandidate_fcorr",
                "synthesis_runtime": "persistent_pi_agent",
                "field_id": field_id,
                "table": entry["table"],
                "column": entry["column"],
                "model": model_name,
                "agent_key": self.config.agent_key,
                "attempt_count": rounds,
                "maximum_attempts": self.config.max_rounds,
                "selected_round": selected.round_index,
                "minimum_candidate_recall_at_5_exclusive": FCORR_MINIMUM_RECALL_AT_5,
                "maximum_candidates": MAX_CANDIDATES,
                "generation_config": {"temperature": 0.0, "seed": 666},
                "evidence_path": f"fields/{field_id}/evidence.json",
                "evidence_sha256": _sha256(evidence_path),
                "conversation_path": f"fields/{field_id}/conversation.json",
                "conversation_sha256": _sha256(conversation_path),
                "validation": validation,
                "test_time_llm_access": False,
            }
            if selected.accepted:
                source_path = field_output / "correction.py"
                static_issues = _static_rule_issues(selected.source)
                if static_issues:
                    raise CellRepairError(
                        f"selected Pi source became invalid for {field_id}: {static_issues}"
                    )
                _atomic_text(source_path, selected.source.rstrip() + "\n")
                field_manifest["source_sha256"] = _sha256(source_path)
            manifest_path = field_output / "field_manifest.json"
            _atomic_json(manifest_path, field_manifest)
            results.append({
                "field_id": field_id,
                "table": entry["table"],
                "column": entry["column"],
                "status": status,
                "field_manifest": f"fields/{field_id}/field_manifest.json",
            })
        synthesis_manifest = {
            "schema_version": RULE_SCHEMA_VERSION,
            "status": "SUCCESS",
            "workflow": "gidcl_multicandidate_fcorr_synthesis",
            "synthesis_runtime": "persistent_pi_agent",
            "field_result_count": len(results),
            "status_counts": dict(Counter(item["status"] for item in results)),
            "models": [model_name],
            "agent_key": self.config.agent_key,
            "maximum_attempts": self.config.max_rounds,
            "completed_rounds": rounds,
            "minimum_candidate_recall_at_5_exclusive": FCORR_MINIMUM_RECALL_AT_5,
            "maximum_candidates": MAX_CANDIDATES,
            "pair_manifest_path": "pair_manifest.json",
            "pair_manifest_sha256": _sha256(self.output_dir / "pair_manifest.json"),
            "source_pair_manifest_sha256": pair_manifest.get("manifest_sha256", ""),
            "fields": results,
        }
        _atomic_json(self.output_dir / "synthesis_manifest.json", synthesis_manifest)
        return freeze_rule_registry(synthesis_dir=self.output_dir)

    def _host_line(self, value: str) -> None:
        self.output.write(value + "\n")
        self.output.flush()


async def synthesize_fcorr_with_pi(
    *,
    project_root: str | Path,
    evidence_dir: str | Path,
    output_dir: str | Path,
    agent_key: str = "react_planner",
    max_rounds: int = 8,
    max_iters: int = 10_000,
    fields: tuple[str, ...] = (),
    skill_dirs: tuple[str | Path, ...] = (),
) -> PiFcorrSynthesisResult:
    return await PiFcorrSynthesisHarness(
        PiFcorrSynthesisConfig(
            project_root=Path(project_root),
            evidence_dir=Path(evidence_dir),
            output_dir=Path(output_dir),
            agent_key=agent_key,
            max_rounds=max_rounds,
            max_iters=max_iters,
            fields=tuple(fields),
            skill_dirs=tuple(Path(path) for path in skill_dirs),
        )
    ).run()
