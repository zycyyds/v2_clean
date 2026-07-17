# Reference-Guided Correction Design

## Goal

Add an isolated 10-shot correction benchmark without changing the existing
`reference-guided-train-validate` workflow. The Agent learns only from ten
paired dirty and clean stay bundles, then corrects all remaining stays in one
run without seeing error labels, modification logs, repair hints, or private
clean answers.

## Dataset

The source archive contains `clean/`, `dirty/`, and host-only ground truth. A
deterministic builder creates `10 train : remaining correction` splits. Every
cell update is assigned by the clean row's canonical `stay_id`; inserted rows
are assigned by their recorded or current `stay_id`. The ten train stays are
selected greedily to cover all four error classes and eighteen subtypes while
the Agent-visible files contain no class or subtype metadata.

The materialized layout is:

```text
train/raw
train/reference
train/keys.csv
correction/raw
correction/reference_private
correction/keys.csv
host_private/train_modification_log.csv
host_private/correction_modification_log.csv
split_manifest.json
```

Business tables retain their original relative names and gzip formats.
Unmodified summary files are copied into both splits. Ground-truth paths and
error taxonomy are never included in the sanitized Agent contract.

## Workflow

Add `reference-guided-correct` as a new `main.py` workflow. It creates one
fresh `ReferenceCodeAgent` context with read access only to `train/raw`,
`train/reference`, and `correction/raw`. It does not use validation rounds,
patience, promotion, checkpoint, or test orchestration.

The Agent must publish a complete corrected package under its workspace. The
prompt describes a generic dirty-to-clean task and conservative modification
rules, but does not enumerate injected error classes or subtypes. The Agent
may write and run local correction code in its isolated workspace.

Only after the Agent session and context are closed may the host read
`correction/reference_private` and `host_private` for evaluation.

## Evaluation

The primary evaluator operates at injected-error locations and compares dirty,
result, and clean data. It reports exact repairs, expected inserted-row
deletions, unresolved errors, incorrect repairs, and collateral changes outside
the ground-truth set. Precision, recall, F1, exact repair rate, row deletion
rate, and clean preservation are reported overall and by error class/subtype.

Whole-package similarity is secondary because millions of unchanged rows would
otherwise hide correction failures. The first version treats exact recovery as
the objective metric and lists non-exact neutralizations separately for later
semantic review.

## Isolation And Failure Behavior

- The existing validation workflow and scorer are unchanged.
- Agent read roots never include the dataset root, `host_private`, correction
  private reference, or evaluator output.
- Missing/incomplete result packages fail before private evaluation.
- Agent failure records `correction_failed` and preserves all inputs.
- Dataset construction is deterministic and refuses to overwrite a mismatched
  existing manifest.

## Future Integration

If the isolated benchmark succeeds, dirty examples can later be injected into
the train, validation, and test raw sides of the main workflow while all
references remain clean. That integration is explicitly outside this change.
