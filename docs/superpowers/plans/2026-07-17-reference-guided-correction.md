# Reference-Guided Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 10-shot dirty-to-clean dataset and a single-run correction workflow with host-private targeted scoring.

**Architecture:** `workflow/correction_dataset.py` materializes the split from the supplied archive. `workflow/reference_correction.py` owns the fresh Agent session, sanitized contract, package gate, and private evaluation. `main.py` exposes preparation and execution as isolated workflow choices.

**Tech Stack:** Python 3.10, standard-library `csv`, `gzip`, `zipfile`, AgentScope, pytest.

## Global Constraints

- Do not change `reference-guided-train-validate` behavior.
- Do not expose ground truth, error taxonomy, private clean outputs, or keys to the Agent.
- Preserve original table paths and gzip serialization.
- Select exactly ten canonical train stays and place all remaining canonical stays in correction.
- Use targeted correction metrics as primary and package similarity only as secondary.

---

### Task 1: Deterministic correction dataset builder

**Files:**
- Create: `workflow/correction_dataset.py`
- Create: `tests/test_reference_correction.py`

**Interfaces:**
- Produces: `build_correction_dataset(archive_path, output_dir, train_count=10) -> dict`
- Produces: the split layout and manifest defined in the design.

- [ ] Write a synthetic-archive test covering canonical stay assignment, inserted rows, gzip preservation, ten-train selection, and hidden logs.
- [ ] Run `PYTHONPATH=. conda run -n py310 pytest -q tests/test_reference_correction.py -k dataset` and verify it fails because the module is absent.
- [ ] Implement archive profiling, canonical modification mapping, greedy stay selection, streaming table partitioning, and atomic manifest writing.
- [ ] Re-run the targeted tests and verify they pass.

### Task 2: Targeted correction evaluator

**Files:**
- Create: `workflow/correction_evaluation.py`
- Modify: `tests/test_reference_correction.py`

**Interfaces:**
- Produces: `evaluate_correction_package(dirty_root, result_root, clean_root, modification_log, output_dir) -> dict`

- [ ] Add failing tests for exact cell repair, inserted-row deletion, missed repair, wrong repair, and collateral modification.
- [ ] Run the evaluator tests and verify expected failures.
- [ ] Implement stable row loading, ground-truth matching, outside-ground-truth diffing, aggregate metrics, and per-class/subtype reports.
- [ ] Re-run evaluator tests and verify they pass.

### Task 3: Single-run ReferenceCodeAgent workflow

**Files:**
- Create: `workflow/reference_correction.py`
- Modify: `main.py`
- Modify: `tests/test_reference_correction.py`

**Interfaces:**
- Produces: `ReferenceCorrectionConfig` and `ReferenceCorrectionWorkflow.run_sync()`.
- Adds workflows: `prepare-correction-split` and `reference-guided-correct`.

- [ ] Add failing CLI, prompt privacy, read-root isolation, package-gate, and mocked Agent/evaluator tests.
- [ ] Run those tests and verify they fail for the missing workflow.
- [ ] Implement CLI wiring, sanitized contract, fresh Agent session, full-package gate, and post-session private evaluation.
- [ ] Re-run targeted tests and verify they pass.

### Task 4: Real dataset build and end-to-end verification

**Files:**
- Generate: `datasets/icu_dirty_correction_10train_990correction_seed_20260702/`

**Interfaces:**
- Consumes the supplied `错误类型构造.zip`.
- Produces the user-facing launch command and verified manifest.

- [ ] Run the new preparation command against the real archive.
- [ ] Verify 10/990 key counts, zero overlap, 594 ground-truth records partitioned exactly once, all four classes and eighteen subtypes represented in train, and no private paths in the sanitized contract.
- [ ] Run `PYTHONPATH=. conda run -n py310 pytest -q` and `git diff --check`.
- [ ] Provide the exact `reference-guided-correct` command with a fresh experiment directory.
