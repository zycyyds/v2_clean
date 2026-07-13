# MIMIC-IV ICU Mortality Full Data Reproduction Implementation Plan

> **For Codex:** Execute this plan task by task. Keep the original MIMIC pipeline source and its existing `data` directory read-only.

**Goal:** Build and run a deterministic, resumable wrapper around the original MIMIC-IV-Data-Pipeline ICU mortality data-generation path through notebook Block 7, with all five feature families and no machine learning.

**Architecture:** Add a small `reproduction` package to `v2_clean`. A parent runner owns configuration, checkpoints, manifests, and validation. Each heavy original-pipeline stage runs in a fresh child process with the isolated run directory as its working directory so the original relative paths continue to work and memory is released between stages. The original repository and MIMIC raw files are exposed read-only through symlinks; all generated files remain under `v2_clean/reproductions`.

**Tech Stack:** Python 3.10, stdlib (`argparse`, `dataclasses`, `hashlib`, `json`, `pathlib`, `subprocess`), pandas, pytest, original MIMIC-IV-Data-Pipeline modules.

---

### Task 1: Define the fixed configuration and isolated workspace

**Files:**
- Create: `reproduction/__init__.py`
- Create: `reproduction/mimic_icu_mortality.py`
- Test: `tests/test_mimic_reproduction.py`

**Step 1: Write failing tests**

Add tests asserting that the default configuration contains exactly:

- ICU / Mortality / MIMIC-IV 3.1.
- No Disease Filter.
- five feature flags enabled.
- grouped ICD-10 root diagnosis mode.
- include time 72, prediction window 2, bucket 1, no imputation.
- original project `data` is never selected as output.
- isolated paths resolve under the configured run root.

**Step 2: Run the focused tests and confirm failure**

Run: `conda run -n py310 pytest -q tests/test_mimic_reproduction.py`

Expected: import or missing API failure.

**Step 3: Implement minimal configuration and path model**

Add immutable `ReproductionConfig` and `RunLayout` dataclasses. Include an explicit canonical JSON representation used for checkpoint fingerprints. Validate that source, raw data, mapping, and run root are distinct and that run root is not inside the original project `data` directory.

**Step 4: Re-run focused tests**

Expected: configuration/path tests pass.

### Task 2: Create isolated links and stage/checkpoint primitives

**Files:**
- Modify: `reproduction/mimic_icu_mortality.py`
- Test: `tests/test_mimic_reproduction.py`

**Step 1: Write failing tests**

Cover:

- creation of `data/cohort`, `data/features`, `data/summary`, `data/csv`, `data/dict`, `logs`, and `checkpoints`.
- raw `mimiciv/3.1` and ICD mapping links point to configured sources.
- an existing wrong link is rejected instead of silently replaced.
- file fingerprints contain path, size, and mtime without hashing the full 9.9 GB raw dataset.
- a checkpoint is reusable only when stage name, fixed configuration fingerprint, relevant input fingerprints, and expected outputs all match.

**Step 2: Run focused tests and confirm failure**

**Step 3: Implement workspace initialization and atomic JSON writes**

Use symlinks only for read-only inputs. Write checkpoints through a temporary sibling and `Path.replace`. Never delete or overwrite an unrelated existing run directory. A non-empty run root without this runner's metadata must fail closed.

**Step 4: Re-run focused tests**

### Task 3: Implement original-pipeline stage workers

**Files:**
- Create: `reproduction/mimic_stage_worker.py`
- Modify: `reproduction/mimic_icu_mortality.py`
- Test: `tests/test_mimic_reproduction.py`

**Step 1: Write failing tests with fake source modules**

Test generated worker commands and stage dispatch without loading real MIMIC data. Assert exact original calls:

1. `day_intervals_cohort_v3.extract_data("ICU", "Mortality", 0, "No Disease Filter", run_root, "")`.
2. `feature_icu(cohort_output, "mimiciv/3.1", True, True, True, True, True)`.
3. `preprocess_features_icu(cohort_output, True, "Convert ICD-9 to ICD-10 and group ICD-10 codes", False, False, False, 0, 0)`.
4. `generate_summary_icu(True, True, True, True, True)`.
5. `data_generation_icu.Generator(cohort_output, True, False, False, True, True, True, True, True, False, 72, 1, 2)`.

**Step 2: Run focused tests and confirm failure**

**Step 3: Implement stage workers**

- Prepend the original project, cohort module, feature-selection module, and model module to `sys.path`.
- Run with the isolated run root as `cwd`.
- Before original imports, create the directories required by import-time side effects.
- After cohort extraction, move the Version 3 root outputs into `data/cohort` and verify content size is unchanged.
- Keep extraction and diagnosis grouping as separate checkpoints so the six-column intermediate cannot be mistaken for final output.
- Redirect child stdout/stderr to a per-stage log while also recording return code and elapsed time.

**Step 4: Re-run focused tests**

### Task 4: Implement orchestration, resume, and force-stage behavior

**Files:**
- Modify: `reproduction/mimic_icu_mortality.py`
- Create: `scripts/reproduce_mimic_icu_mortality.py`
- Test: `tests/test_mimic_reproduction.py`

**Step 1: Write failing tests**

Cover:

- stages execute in order: setup, cohort, features, diagnosis, summaries, stay generation, validation.
- a valid checkpoint skips a completed stage.
- a changed config or input invalidates that stage and all downstream stages.
- `--force-stage diagnosis` re-runs diagnosis and all downstream stages but not cohort/features.
- a failed child process stops the chain and records `failed` with its log path.
- no command refers to original project `data` as output.

**Step 2: Run focused tests and confirm failure**

**Step 3: Implement the CLI and parent runner**

CLI arguments:

- `--source-root`, default original project path.
- `--run-root`, default `reproductions/mimic_icu_mortality_v3_1_full_v1`.
- `--force-stage`, optional stage name.
- `--validate-only`, optional.
- `--dry-run`, optional, printing exact stages and paths without creating data products.

The parent writes `run_state.json` after every transition and retains failed logs for resume.

**Step 4: Re-run focused tests**

### Task 5: Implement machine-readable validation and manifest generation

**Files:**
- Create: `reproduction/mimic_validation.py`
- Modify: `reproduction/mimic_icu_mortality.py`
- Test: `tests/test_mimic_reproduction.py`

**Step 1: Write failing fixture-based tests**

Create small CSV fixtures covering pass/fail cases for:

- adult cohort and binary labels.
- cohort key uniqueness reporting.
- exact four-column diagnosis schema.
- feature stay IDs being a subset of cohort IDs.
- labels requiring original integer-hour `los >= 74` semantics.
- one `demo.csv`, `static.csv`, and `dynamic.csv` per label stay.
- missing, empty, and extra stay directories.
- all expected dictionary files.
- exactly 10 summary files and 17 gold candidates.
- no model artifacts.

**Step 2: Run focused tests and confirm failure**

**Step 3: Implement scalable validators**

- Read only key/schema columns for large feature checks, using chunks for chart.
- Count CSV rows without loading the full chart file into memory.
- Record every check as `{name, passed, observed, expected, details}`.
- Produce `validation_report.json` with top-level `passed` and `failed_validation` status.
- Produce `reproduction_manifest.json` containing fixed config, source fingerprints, stage timings, output path, size, row count, schema, and the explicit 17-file gold-candidate list.

Expected dictionary names from the enabled original Generator:

- `dataDic`, `hadmDic`, `ethVocab`, `ageVocab`, `insVocab`
- `medVocab`, `outVocab`, `chartVocab`, `condVocab`, `procVocab`, `metaDic`

**Step 4: Re-run focused tests**

### Task 6: Verify the implementation before touching full data

**Files:**
- Modify as needed based on failures.

**Step 1: Run focused suite**

Run: `conda run -n py310 pytest -q tests/test_mimic_reproduction.py`

Expected: all pass.

**Step 2: Run existing main-chain suite**

Run: `conda run -n py310 pytest -q tests/test_main_chain.py`

Expected: no regression.

**Step 3: Run dry-run against real paths**

Run:

```bash
conda run -n py310 python scripts/reproduce_mimic_icu_mortality.py \
  --source-root /Users/mac/PycharmProjects/MIMIC-IV-Data-Pipeline-main \
  --run-root /Users/mac/PycharmProjects/v2_clean/reproductions/mimic_icu_mortality_v3_1_full_v1 \
  --dry-run
```

Expected: all paths and exact fixed parameters are printed; original project `data` is not writable output.

### Task 7: Run the real full reproduction

**Files:**
- Generate only under: `reproductions/mimic_icu_mortality_v3_1_full_v1/`

**Step 1: Check disk capacity and environment**

Require enough free disk for feature intermediates and approximately 30,000 per-stay directories. Fail before execution if the configured safety threshold is not met.

**Step 2: Start the full command in the foreground**

Run the same command without `--dry-run`. Do not run two copies against the same run root.

**Step 3: Inspect every stage checkpoint**

After each stage, confirm status, expected outputs, row counts, schemas, elapsed time, and log tail. If a stage fails, fix the wrapper only when the issue is adaptation/environmental; do not alter original business rules.

**Step 4: Complete Block 7 generation**

Confirm labels and all per-stay CSVs are present before validation.

### Task 8: Final validation and handoff

**Files:**
- Verify: `reproductions/mimic_icu_mortality_v3_1_full_v1/validation_report.json`
- Verify: `reproductions/mimic_icu_mortality_v3_1_full_v1/reproduction_manifest.json`

**Step 1: Run validation-only mode**

Run the CLI with `--validate-only` to independently re-check the completed outputs.

**Step 2: Compare with the untouched original project**

Verify no files under `/Users/mac/PycharmProjects/MIMIC-IV-Data-Pipeline-main/data` changed during the run, using the captured pre-run fingerprint snapshot.

**Step 3: Report the final inventory**

Report:

- cohort and label counts, including positive/negative cases.
- row counts and schemas for all five final feature files.
- the 10 summary files.
- number of stay directories and three-file completeness.
- dictionary inventory.
- exact 17 gold-candidate paths.
- per-stage and total elapsed time.
- any original warnings that did not fail a gate.

**Step 4: Commit implementation only after verification**

Stage only the new reproduction code, tests, CLI, and plan. Do not commit generated MIMIC outputs or unrelated existing worktree changes.
