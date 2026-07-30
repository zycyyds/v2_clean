# Automatic Validation-to-Test Chain Design

## Objective

Connect the already verified Validation and strict one-shot Test harnesses into
one host-controlled experiment command:

```text
Validation Agent loop
-> reproducible frozen snapshot
-> one automatic Test execution
-> one hidden Test scoring pass
-> final combined report
```

The Test phase starts immediately after Validation returns
`SUCCESS_REPRODUCIBLE`. It does not require the 4000-case preflight or manual
confirmation.

## Architecture

Add a top-level orchestration module and CLI. The orchestrator owns sequencing
only and reuses the existing `PiValidationHarness` and `PiTestHarness` behavior.
It must not move Test execution into the Validation harness or expose Test
inputs to the Agent worker.

The Test harness gains a direct mode that validates the Validation report and
frozen snapshot itself. Existing preflight support remains available as an
optional standalone diagnostic path but is not part of the automatic chain.

## Execution Contract

1. Run Validation using the existing persistent Agent and hidden Validation
   scoring loop.
2. If Validation does not return `SUCCESS_REPRODUCIBLE`, stop without reading
   Test raw data or invoking Test scoring.
3. Verify the reported reproducible snapshot exists and calculate its content
   hash.
4. Atomically record that Test is starting before launching the Pipeline.
5. Execute the frozen Pipeline on Test raw exactly once.
6. If Test execution fails, times out, or produces invalid output, stop without
   retrying, repairing, or scoring.
7. If execution succeeds, invoke the isolated host scorer exactly once.
8. Write a combined final report containing Validation status and score,
   frozen snapshot hash, Test execution count, scoring count, Test status and
   score, timings, and terminal phase.

## Isolation

- The Agent can access only the existing public Train and Validation inputs and
  its own workspace.
- Test raw and Test Gold are not passed to the Agent process or its sandbox.
- The frozen Pipeline can read Test raw and the already authorized Train
  reference, but not Test Gold, Validation Gold, host reports, API keys,
  network resources, or `workflow/`.
- Only the isolated host scorer can read Test Gold.
- Test scores and file metrics are never returned to the Agent.

## One-Shot Semantics

The Test experiment directory is immutable for one formal execution. Before
launch, the orchestrator creates an atomic `test_started.json` marker containing
the frozen hash and start time. A second invocation against a directory with
that marker must fail before starting the Pipeline, regardless of whether the
first execution succeeded, failed, timed out, or was interrupted.

Test failures are not retried. A low Test score is still a successful completed
Test when execution and scoring both finish.

## CLI

Add `python -m agent.pi_full_experiment_cli`. It accepts the existing Validation
arguments plus:

```text
--test-experiment
--test-raw
--test-gold
--test-replay-timeout
--test-scoring-timeout
```

No preflight attestation or confirmation flag is required.

## Cancellation

SIGINT, SIGTERM, and SIGHUP use the existing bounded process-group cleanup.
Cancellation during Validation prevents Test from starting. Cancellation after
`test_started.json` is written leaves the marker in place and records
`INTERRUPTED`; it must not allow an automatic Test retry.

## Verification

Automated tests must prove:

- Validation success starts Test automatically without preflight.
- Every non-success Validation status prevents Test access and execution.
- Test Pipeline execution and hidden scoring are each called exactly once.
- Test execution failure does not score or retry.
- Test scoring failure does not rerun the Pipeline.
- The one-shot marker blocks every second invocation.
- Test inputs and scores never enter Agent IPC, transcript, or worker logs.
- Cancellation in each phase leaves no child process group running.
- Existing standalone Validation, optional preflight, and strict Test tests
  remain passing.

The already completed formal 5000-case Test must not be rerun. Connector
integration tests use synthetic data and mocked phase boundaries; the next new
formal experiment is the first end-to-end use of the automatic chain.
