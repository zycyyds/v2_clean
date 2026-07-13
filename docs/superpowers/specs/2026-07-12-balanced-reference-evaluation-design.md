# Balanced Reference Evaluation Design

## Purpose

Redesign the reference-directory score so that:

- very large tables such as `preproc_chart_icu.csv` cannot dominate the total score by row count;
- small files remain visible without allowing the ten summary files to dominate merely because there are more of them;
- partially correct rows receive credit for their correct cells;
- missing and extra rows are both penalized;
- physical CSV row order does not affect correctness;
- every expected column has equal weight inside its file.

This document defines the scoring contract only. It does not change the evaluator implementation.

## Overall Score

The score is hierarchical rather than globally micro-averaged over all rows and cells.

```text
overall_score =
    0.20 * core_score
  + 0.60 * feature_score
  + 0.20 * summary_score
```

### Core group: 20 percent

The core group is split into:

```text
core_score = 0.75 * cohort_file_score + 0.25 * labels_file_score
```

This gives the cohort file 15 percent of the overall score and the labels file 5 percent. The lower labels weight avoids over-counting the label already present in the cohort file.

### Feature group: 60 percent

The five feature files are equally weighted:

- `features/preproc_chart_icu.csv`
- `features/preproc_diag_icu.csv`
- `features/preproc_med_icu.csv`
- `features/preproc_out_icu.csv`
- `features/preproc_proc_icu.csv`

Each feature file contributes 12 percent of the overall score, regardless of its row count.

### Summary group: 20 percent

The ten files under `summary/` are equally weighted. Each contributes 2 percent of the overall score. A summary file with 71 rows and one with 1,570 rows therefore have equal importance inside the summary contract.

## Per-File Score

Each expected file receives an independent score between zero and one:

```text
file_score =
    0.10 * schema_score
  + 0.20 * key_structure_f1
  + 0.40 * row_aligned_cell_f1
  + 0.30 * exact_row_f1
```

If an expected file is missing, its file score is zero.

### Schema score: 10 percent

Schema score checks whether the result exposes the expected table contract.

```text
column_recall    = matched_expected_columns / expected_columns
column_precision = matched_expected_columns / result_columns
schema_score     = F1(column_precision, column_recall)
```

Missing columns reduce recall. Unexpected extra columns reduce precision. Column order is not part of this metric; serialization and ordering requirements may be enforced separately by a quality gate.

### Key and row structure F1: 20 percent

A key identifies which reference and result records represent the same business event. Physical row number is never the identity of a record.

Keys must be configured per file and should use stable identity fields rather than target values. Conceptual examples include:

| File family | Candidate identity fields |
| --- | --- |
| cohort and labels | `stay_id` |
| chart | `stay_id`, `itemid`, event time |
| diagnosis | stable admission/stay identity plus a diagnosis occurrence identity |
| medication | stay, item, order and stable time fields |
| output event | stay, item and chart time |
| procedure | stay, item and start time |
| summary | `itemid` or `new_icd_code` |

The exact key contract must be derived and fixed per file before implementation. A value being evaluated, such as `Age`, `valuenum`, or a derived statistic, must not be used merely to make the key unique.

Key matching is multiset-based, so legitimate repeated events are retained. For every key group:

```text
matched_key_occurrences = min(reference_occurrences, result_occurrences)
key_recall               = matched_key_occurrences / reference_rows
key_precision            = matched_key_occurrences / result_rows
key_structure_f1         = F1(key_precision, key_recall)
```

This penalizes missing records, extra records, under-produced duplicates, and over-produced duplicates. It does not penalize duplicates that also exist in the reference.

### Row-aligned cell F1: 40 percent

Cell values must never be compared as independent column bags. Rows are aligned by their business key first, and cells are compared only within aligned records.

This avoids a false match such as:

```text
reference: stay 1001 -> Age 65; stay 1002 -> Age 72
result:    stay 1001 -> Age 72; stay 1002 -> Age 65
```

Although both Age columns contain the same value set, both row-aligned Age cells are wrong.

For key groups containing multiple rows, pairing must be one-to-one and independent of file order. The pairing objective is to maximize matched expected cells within the group. Unpaired reference rows are missing; unpaired result rows are extra.

After pairing, each expected column receives its own precision, recall, and F1:

```text
column_cell_recall    = matched_cells_in_column / reference_cells_in_column
column_cell_precision = matched_cells_in_column / result_cells_in_column
column_cell_f1        = F1(column_cell_precision, column_cell_recall)

row_aligned_cell_f1 = arithmetic mean of all expected column_cell_f1 values
```

All expected columns are equally weighted in this final mean. A column does not gain weight because it contains more rows, more non-empty values, or a particular data type.

Empty values, sentinels, numeric normalization, timestamps, and other serialization forms must use one explicit normalization contract. Extra rows contribute result cells to the precision denominator and therefore reduce the score.

### Exact row F1: 30 percent

An exact row match requires every expected cell in the aligned row to match after normalization.

```text
exact_row_recall    = exact_rows / reference_rows
exact_row_precision = exact_rows / result_rows
exact_row_f1        = F1(exact_row_precision, exact_row_recall)
```

A row with nine correct cells and one incorrect cell receives no exact-row credit, but its nine correct cells still contribute to row-aligned cell F1. Exact-row F1 therefore rewards complete reconstruction without erasing partial correctness.

## Duplicate-Key Pairing

Pairing duplicate-key groups by physical order is forbidden because sorting differences would change the score. The required semantics are:

1. Partition reference and result rows by the configured business key.
2. Remove exact row matches first.
3. Pair remaining rows one-to-one to maximize the number of matching expected cells.
4. Do not allow one row to match more than once.
5. Count unpaired reference rows as missing and unpaired result rows as extra.

The implementation may use an exact or optimized equivalent algorithm, but it must reproduce these semantics deterministically at validation scale.

## Reporting Contract

The evaluator should report enough detail to explain every score:

- overall score and the three group scores;
- every file's final score and overall contribution;
- schema score, key precision/recall/F1, row-aligned cell F1, and exact-row precision/recall/F1;
- per-column cell precision, recall, F1, matched count, reference count, and result count;
- missing rows, extra rows, duplicate-count mismatches, and the configured key columns;
- comparison mode and whether an optimized large-table path was used.

Large-table optimization must not silently change scoring semantics. Chart, medication, and output-event tables must retain partial cell credit under the same contract as smaller files.

## Acceptance Examples

1. Reordering a correct file does not change its score.
2. Swapping values between two different keys reduces the affected column score.
3. A ten-cell row with nine correct cells earns nine cells of partial credit but no exact-row credit.
4. A legitimate repeated key matching the reference count is not penalized.
5. Extra duplicate rows reduce key precision, cell precision, and exact-row precision.
6. Missing rows reduce key recall, cell recall, and exact-row recall.
7. A file with all rows and cells correct receives 1.0.
8. A missing expected file receives 0.0.
9. Each feature file contributes equally even when chart has orders of magnitude more rows.
10. The ten summary files together contribute exactly 20 percent, not ten times the influence of a single business layer.

## Current Baseline Interpretation

Under the accepted weighting, the ten currently byte-identical summary files would earn the full summary contribution of 0.20. Remaining improvements would be visible independently in cohort, labels, chart, diagnosis, medication, output-event, and procedure scores rather than being submerged in a single global row count.
