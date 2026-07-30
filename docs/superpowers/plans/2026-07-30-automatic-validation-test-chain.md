# Automatic Validation-to-Test Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one command that runs the existing Validation Agent loop and, only after `SUCCESS_REPRODUCIBLE`, automatically executes and scores the frozen Pipeline on Test exactly once.

**Architecture:** Keep `PiValidationHarness` and `PiTestHarness` as independent phase owners. Add a small host-only orchestrator that sequences them, and make the Test preflight attestation optional so the automatic path can validate the frozen Validation snapshot directly. Preserve the existing standalone preflight path for optional diagnostics.

**Tech Stack:** Python 3.10, asyncio, dataclasses, argparse, pytest, AgentScope 2.0.4.post1, macOS `sandbox-exec`.

---

## File Structure

- Modify `agent/pi_test_harness.py`: support direct frozen-snapshot Test execution and write the one-shot start marker.
- Modify `agent/pi_test_harness_cli.py`: make preflight attestation optional for standalone Test use.
- Create `agent/pi_full_experiment.py`: host-only sequencing and combined-report ownership.
- Create `agent/pi_full_experiment_cli.py`: one-command public interface and terminal signal handling.
- Modify `tests/test_pi_test_harness.py`: direct-mode and one-shot marker regression coverage.
- Modify `tests/test_pi_test_harness_cli.py`: optional-attestation CLI coverage.
- Create `tests/test_pi_full_experiment.py`: orchestration behavior and combined report coverage.
- Create `tests/test_pi_full_experiment_cli.py`: full CLI parsing and exit-code coverage.

### Task 1: Direct One-Shot Test Mode

**Files:**
- Modify: `agent/pi_test_harness.py:28-48,274-340`
- Modify: `tests/test_pi_test_harness.py`

- [ ] **Step 1: Write failing tests for direct mode and the marker**

Add a helper that creates `PiTestHarnessConfig` without an attestation and tests that Test starts once:

```python
def _direct_test_config(tmp_path: Path) -> PiTestHarnessConfig:
    validation, _ = _validation_fixture(tmp_path)
    test_raw = tmp_path / "dataset/test/raw"
    test_gold = tmp_path / "dataset/test/reference_private"
    test_raw.mkdir(parents=True)
    test_gold.mkdir(parents=True)
    (test_raw / "data.csv").write_text("id,value\n2,test\n", encoding="utf-8")
    (test_gold / "data.csv").write_text("id,value\n2,gold\n", encoding="utf-8")
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "files": {"data.csv": {"key_columns": ["id"]}}}),
        encoding="utf-8",
    )
    return PiTestHarnessConfig(
        project_root=Path(__file__).parents[1],
        validation_experiment=validation,
        test_experiment=tmp_path / "test_experiment",
        test_raw=test_raw,
        test_gold=test_gold,
        evaluation_manifest=manifest,
        preflight_attestation=None,
        replay_timeout_seconds=10.0,
        scoring_timeout_seconds=10.0,
    )


def test_direct_test_mode_runs_without_preflight_and_writes_one_shot_marker(tmp_path: Path) -> None:
    config = _direct_test_config(tmp_path)
    replay_calls: list[ReplayRequest] = []

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        return ScoreExecution("SUCCESS", 0, "", 0.1, _score_report(0.75))

    result = asyncio.run(
        PiTestHarness(
            config,
            replay_runner=_successful_replay(replay_calls),
            score_runner=score_runner,
        ).run(),
    )

    marker = config.test_experiment / "host/test_started.json"
    assert result.status == "SUCCESS"
    assert len(replay_calls) == 1
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["frozen_snapshot_sha256"]


def test_direct_test_mode_rejects_second_invocation_before_replay(tmp_path: Path) -> None:
    config = _direct_test_config(tmp_path)
    replay_calls: list[ReplayRequest] = []

    async def score_runner(*_args, **_kwargs) -> ScoreExecution:
        return ScoreExecution("SUCCESS", 0, "", 0.1, _score_report(0.75))

    asyncio.run(
        PiTestHarness(
            config,
            replay_runner=_successful_replay(replay_calls),
            score_runner=score_runner,
        ).run(),
    )
    with pytest.raises(ValueError, match="new and empty|already started"):
        asyncio.run(PiTestHarness(config, replay_runner=_successful_replay(replay_calls)).run())
    assert len(replay_calls) == 1
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q \
  tests/test_pi_test_harness.py::test_direct_test_mode_runs_without_preflight_and_writes_one_shot_marker \
  tests/test_pi_test_harness.py::test_direct_test_mode_rejects_second_invocation_before_replay
```

Expected: FAIL because `preflight_attestation` is required and no marker is written.

- [ ] **Step 3: Make attestation optional and write the marker before replay**

Move the optional field after required dataclass fields:

```python
@dataclass(frozen=True)
class PiTestHarnessConfig:
    project_root: Path
    validation_experiment: Path
    test_experiment: Path
    test_raw: Path
    test_gold: Path
    evaluation_manifest: Path
    preflight_attestation: Path | None = None
    replay_timeout_seconds: float = 1_800.0
    scoring_timeout_seconds: float = 3_600.0
```

In `PiTestHarness.run()`, validate an attestation only when supplied, record the optional metadata in `run_manifest.json`, and create the marker immediately before replay:

```python
attestation: dict[str, Any] = {}
attestation_sha256 = ""
if self.config.preflight_attestation is not None:
    attestation = _load_attestation(self.config.preflight_attestation, source_hash)
    attestation_sha256 = _file_sha256(self.config.preflight_attestation)

_atomic_json(
    self.host_dir / "test_started.json",
    {
        "schema_version": 1,
        "frozen_snapshot_sha256": frozen_hash,
        "started_at_unix": time.time(),
    },
)
```

The manifest stores `preflight_attestation_sha256` as an empty string and `preflight_raw_identity` as `{}` in direct mode. Do not include Test Gold paths in either file.

- [ ] **Step 4: Run the focused Test harness tests**

Run:

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q tests/test_pi_test_harness.py
```

Expected: all tests pass, including existing attested mode and new direct mode.

- [ ] **Step 5: Commit Task 1**

```bash
git add agent/pi_test_harness.py tests/test_pi_test_harness.py
git commit -m "feat: allow direct one-shot Test execution"
```

### Task 2: Host-Only Full Experiment Orchestrator

**Files:**
- Create: `agent/pi_full_experiment.py`
- Create: `tests/test_pi_full_experiment.py`

- [ ] **Step 1: Write failing orchestration tests**

Define small fake phase runners and assert both branches:

```python
def test_validation_success_automatically_runs_test_once(tmp_path: Path) -> None:
    calls: list[str] = []
    validation_result = _validation_result(tmp_path, status="SUCCESS_REPRODUCIBLE")
    test_result = _test_result(tmp_path, status="SUCCESS", score=0.88)

    async def run_validation(_prompt: str):
        calls.append("validation")
        return validation_result

    async def run_test():
        calls.append("test")
        return test_result

    result = asyncio.run(
        PiFullExperiment(
            _full_config(tmp_path),
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    assert calls == ["validation", "test"]
    assert result.status == "SUCCESS"
    assert result.test_result is test_result


@pytest.mark.parametrize("status", ["INTERRUPTED", "FAILED", "REPLAY_FAILED"])
def test_validation_non_success_never_starts_test(tmp_path: Path, status: str) -> None:
    test_calls = 0

    async def run_validation(_prompt: str):
        return _validation_result(tmp_path, status=status)

    async def run_test():
        nonlocal test_calls
        test_calls += 1
        raise AssertionError("Test must not start")

    result = asyncio.run(
        PiFullExperiment(
            _full_config(tmp_path),
            validation_runner=run_validation,
            test_runner=run_test,
        ).run("prompt"),
    )

    assert result.status == "VALIDATION_FAILED"
    assert result.test_result is None
    assert test_calls == 0
```

Also assert the combined report records `test_execution_count=1` and derives `scoring_execution_count=1` only for Test phases `scoring` or `complete`.

- [ ] **Step 2: Run the orchestrator tests and verify RED**

Run:

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q tests/test_pi_full_experiment.py
```

Expected: FAIL with `ModuleNotFoundError: agent.pi_full_experiment`.

- [ ] **Step 3: Implement the minimal orchestrator**

Create these public types:

```python
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
```

`PiFullExperiment.run(prompt)` must:

```python
validation = await self.validation_runner(prompt)
if validation.status != "SUCCESS_REPRODUCIBLE":
    return self._finish("VALIDATION_FAILED", "validation", validation, None)

test = await self.test_runner()
return self._finish(
    "SUCCESS" if test.status == "SUCCESS" else test.status,
    "complete" if test.status == "SUCCESS" else test.phase,
    validation,
    test,
)
```

Default runners construct the existing `PiValidationHarness` and a direct-mode `PiTestHarness` with `preflight_attestation=None`. Save `full_experiment_report.json` under the Validation experiment's `host/` directory, and sanitize it so it contains scores, hashes, counts, durations, and public result paths but no Gold paths or values.

- [ ] **Step 4: Run orchestrator and existing phase tests**

Run:

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q \
  tests/test_pi_full_experiment.py \
  tests/test_pi_validation_harness.py \
  tests/test_pi_test_harness.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add agent/pi_full_experiment.py tests/test_pi_full_experiment.py
git commit -m "feat: chain Validation and Test harnesses"
```

### Task 3: Unified CLI

**Files:**
- Create: `agent/pi_full_experiment_cli.py`
- Create: `tests/test_pi_full_experiment_cli.py`
- Modify: `agent/pi_test_harness_cli.py`
- Modify: `tests/test_pi_test_harness_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Test parsing of all existing Validation arguments plus the Test arguments:

```python
def test_full_cli_parses_validation_and_automatic_test_inputs() -> None:
    args = parse_args(
        [
            "--experiment-dir", "/runs/validation",
            "--train-raw", "/data/train/raw",
            "--train-reference", "/data/train/reference",
            "--validation-raw", "/data/validation/raw",
            "--validation-gold", "/private/validation/gold",
            "--evaluation-manifest", "/manifests/evaluation.json",
            "--dataset-manifest", "/data/split_manifest.json",
            "--prompt-file", "/prompts/task.md",
            "--test-experiment", "/runs/test",
            "--test-raw", "/data/test/raw",
            "--test-gold", "/private/test/gold",
        ],
    )
    assert args.max_rounds == 20
    assert args.patience == 3
    assert args.test_replay_timeout == 1800.0
    assert args.test_scoring_timeout == 3600.0
```

Update the standalone Test CLI test to prove `--preflight-attestation` is optional while still accepted when provided.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q \
  tests/test_pi_full_experiment_cli.py \
  tests/test_pi_test_harness_cli.py
```

Expected: FAIL because the full CLI does not exist and standalone Test still requires attestation.

- [ ] **Step 3: Implement CLI parsing and execution**

`agent.pi_full_experiment_cli` builds `PiValidationHarnessConfig` and `PiFullExperimentConfig`, reads the prompt file, and executes the orchestrator through existing `_run_with_terminal_signals()`.

Exit codes:

```python
if received_signal is not None or result.status == "INTERRUPTED":
    return 130
return 0 if result.status == "SUCCESS" else 1
```

Print only the combined public fields: status, phase, Validation rounds/best/reproducible score, Test status/score, execution count, frozen hash, and report path.

In `agent.pi_test_harness_cli`, change:

```python
parser.add_argument("--preflight-attestation")
```

and construct `Path(args.preflight_attestation)` only when the value is present.

- [ ] **Step 4: Run all CLI tests**

Run:

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q \
  tests/test_pi_full_experiment_cli.py \
  tests/test_pi_test_harness_cli.py \
  tests/test_pi_harness_cli.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add \
  agent/pi_full_experiment_cli.py \
  agent/pi_test_harness_cli.py \
  tests/test_pi_full_experiment_cli.py \
  tests/test_pi_test_harness_cli.py
git commit -m "feat: add automatic full experiment CLI"
```

### Task 4: Isolation and Failure-Path Integration Tests

**Files:**
- Modify: `tests/test_pi_full_experiment.py`
- Modify: `tests/test_pi_test_harness.py`

- [ ] **Step 1: Add failing tests for failure-path counts and redaction**

Cover these concrete assertions:

```python
assert replay_calls == 1
assert score_calls == 0  # replay failure
assert score_calls == 1  # scorer failure, without replay retry
assert test_result.test_execution_count == 1
assert "reference_private" not in combined_report_text
assert str(config.test_gold.resolve()) not in combined_report_text
assert not (validation_experiment / "agent_workdir" / "test_started.json").exists()
```

Add a cancellation case where Validation returns `INTERRUPTED` and prove the Test experiment directory was never created. Add a direct-mode sandbox case alongside the existing attested sandbox test to prove Test Gold, Validation Gold, Train raw, `workflow/`, host directories, network, and API keys remain unavailable to the Pipeline.

- [ ] **Step 2: Run the new integration tests and verify RED where behavior is missing**

Run:

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q \
  tests/test_pi_full_experiment.py \
  tests/test_pi_test_harness.py
```

Expected: any missing report redaction or phase-count behavior fails with the corresponding assertion.

- [ ] **Step 3: Make the minimal report and isolation corrections**

Adjust only the orchestrator report serializer and direct-mode Test setup. Do not modify the scoring formula, replay sandbox roots, Agent sandbox, or existing process-group cleanup.

- [ ] **Step 4: Re-run the integration tests**

Run the same command from Step 2.

Expected: all selected tests pass. On a nested Codex sandbox, real `sandbox-exec` tests may be blocked by the outer environment; record this separately and run them in a normal terminal if necessary.

- [ ] **Step 5: Commit Task 4**

```bash
git add tests/test_pi_full_experiment.py tests/test_pi_test_harness.py agent/pi_full_experiment.py agent/pi_test_harness.py
git commit -m "test: verify automatic Test isolation and one-shot behavior"
```

### Task 5: Full Verification

**Files:**
- Verify only; no formal Test data execution.

- [ ] **Step 1: Run static patch checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; the existing local absolute-path evaluation manifest remains untracked and unstaged.

- [ ] **Step 2: Run the complete automated suite**

```bash
conda run -n py3102 env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider -q
```

Expected: the existing 279 tests plus the new connector tests pass in a normal host terminal.

- [ ] **Step 3: Run a synthetic end-to-end CLI test**

Use pytest fixtures only; do not point the command at the completed 5000-case Test directory or Test raw. Verify the command reaches automatic Test after synthetic Validation success and blocks Test after synthetic Validation failure.

- [ ] **Step 4: Audit sensitive output**

```bash
rg -n "reference_private|API_KEY|sk-[A-Za-z0-9_-]+" \
  tests/.tmp_full_experiment 2>/dev/null
```

Expected: no Gold paths, Gold values, or credentials in public combined reports, Agent transcript, or worker logs.

- [ ] **Step 5: Record final status**

Report the new commit IDs, focused test counts, full-suite count, and any outer-sandbox-only test limitation. Do not run or alter the already completed formal 5000-case Test.
