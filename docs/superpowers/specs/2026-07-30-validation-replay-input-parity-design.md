# Validation Replay Input Parity Design

## Objective

Keep the Validation Harness replay model simple: after the Agent loop ends, run the formal best Pipeline in a clean directory with the same public dataset inputs available during normal rounds. If execution fails or the replay score is below the formal best, return sanitized feedback to the same persistent Agent and replay its repaired Pipeline until reproducibility succeeds.

## Current Defect

Normal Validation Agent turns can read:

- `train/raw`
- `train/reference`
- `validation/raw`

Independent replay can currently read only `train/reference` and `validation/raw`. A best Pipeline that legitimately derives rules from paired training raw/reference data therefore fails only during replay. The Agent then changes its derivation algorithm to avoid `train/raw`, which can change the output and lower the score even though the original best Pipeline was valid under the normal-round input contract.

## Replay Contract

Add `{train_raw}` to the allowed replay placeholders. The complete placeholder set remains generic:

- `{train_raw}`: public training raw directory
- `{train_reference}`: public training reference directory
- `{raw_root}`: validation raw directory for Validation replay
- `{output_dir}`: result directory declared by `result_root`
- `{workdir}`: clean replay workspace root

The Agent continues to declare an argv array in `submission.json`; the Harness does not prescribe script layout or command argument order.

## Replay Flow

1. Copy the complete formal best Agent workspace bundle into a new temporary replay directory.
2. Remove only the declared result directory so stale outputs cannot satisfy replay.
3. Render the declared argv using the five placeholders above.
4. Run the command without a shell in the macOS sandbox.
5. Permit read-only access to `train/raw`, `train/reference`, `validation/raw`, the copied Pipeline bundle, and required system runtime paths.
6. Permit writes only inside the replay workspace, isolated home, and temporary directory.
7. Keep network access and Validation/Test Gold access denied.
8. Score the newly generated result package in the host process.

The Harness treats scripts, learned rules, and helper files as an opaque Pipeline bundle. It does not know about or special-case ICD mappings, feature lists, or any dataset-specific artifact.

## Repair Flow

Replay is acceptable only when the command succeeds, produces a valid result package, and scores no lower than the formal best within the existing tolerance.

On replay failure or score regression:

1. Preserve the formal best snapshot unchanged.
2. Send only sanitized execution details and public per-file score summaries to the same persistent Agent context.
3. Let the Agent modify its Pipeline bundle in the active workspace.
4. Replay that repaired workspace in a fresh temporary directory.
5. Repeat until replay is acceptable or the user interrupts the run.

This design does not add a separate training mode, frozen-artifact mode, dataset-specific gate, or scoring tool visible to the Agent.

## Interface Changes

- Add `train_raw: Path` to replay argv rendering.
- Add `{train_raw}` to `ALLOWED_REPLAY_PLACEHOLDERS`.
- Pass `PiValidationHarnessConfig.train_raw` when rendering replay argv.
- Add `train_raw` to replay sandbox read roots.
- Update the Agent submission prompt to document `{train_raw}`.

## Verification

Tests must verify:

- `{train_raw}` renders to the configured absolute training raw path.
- Unknown placeholders remain rejected.
- Replay can read `train/raw`, `train/reference`, and `validation/raw`.
- Replay still cannot read Validation Gold, including through symlinks or subprocesses.
- Existing best snapshot copying and result-root clearing behavior remains unchanged.
- A synthetic Pipeline that derives an artifact from `train/raw` and `train/reference` reproduces its best score in clean replay.
- Replay failures still return to the same Agent context for repair.

## Scope

This change affects only Validation independent replay input parity. It does not change scoring, normal Agent rounds, Test Gold isolation, Pipeline structure, model configuration, or the existing process-backend fixes.
