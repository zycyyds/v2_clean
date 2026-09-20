# Raha Hospital curated 10-shot Pi Agent benchmark

This experiment keeps the existing AgentScope 2.0 Pi harness unchanged. It uses ten
dirty-clean row pairs selected by deterministic changed-column coverage, freezes the
generated pipeline, and executes it once on the remaining 990 rows.

The protocol is a curated capability audit, not an unbiased leaderboard submission.
Test clean data and cell differences remain host-private.

## Build

```bash
python -m benchmark_experiments.hospital_curated10_v1.build_dataset \
  --output-dir datasets/raha_hospital_curated10_v1
```

## Train and freeze

Run `agent.pi_harness_cli` with the dataset's `train` directories and use
`train_only_replay` for the required validation arguments. Set `--max-rounds 1` and
`--patience 1`. Load the four existing correction skills.

## One-shot test

Run `agent.pi_test_harness_cli` once with `correction/raw` and
`correction/reference_private`. The model credentials must be absent during this step.

## Private cell scoring

```bash
python -m benchmark_experiments.hospital_curated10_v1.score_cells \
  --dirty-root datasets/raha_hospital_curated10_v1/correction/raw \
  --clean-root datasets/raha_hospital_curated10_v1/correction/reference_private \
  --result-root experiments/hospital_curated10_test/host/test_result_package \
  --output experiments/hospital_curated10_test/host/cell_score_report.json
```
