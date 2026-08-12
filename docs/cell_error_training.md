# Cell error detection training

This pipeline freezes Qwen Embedding v1 and compares `triple_mlp`,
`fullrow_rgcn`, and `strict_rgcn` under one supervised Cell split.

## Protocol

- folds 0/1/2: train
- fold 3: validation
- fold 4: internal test, withheld unless `--evaluate-internal-test` is set
- checkpoint selection: validation Average Precision
- threshold selection: validation Macro-F1 after restoring the best checkpoint
- score output: uncalibrated `dirty_score`
- graph protocol: transductive full graph with preassigned folds; the upstream
  supervision builder groups folds by subject, but training cannot rederive and
  independently audit subjects from the current private mask artifact
- every target Cell's exact forward and reverse edge IDs are removed before sampling

The first data preparation run scans `cell_observations.jsonl` once and creates
private caches under `supervision-dir`. The R-GCN additionally creates a private
incoming-edge CSR cache there. These derived files are data products and must not
be committed. Concurrent seeds share a preparation lock, so only one process
builds or verifies a missing cache while the others wait and then reuse it.

## Windows commands

Run all commands from `D:\mimic_graph\v2_clean-graph` in the
`graph-embedding` Conda environment.

Development run, validation only:

```bat
python -m graph.cell_training_cli ^
  --graph-dir "D:\mimic_graph\data\graph_dirty_cell_supervised_v2" ^
  --supervision-dir "D:\mimic_graph\private\graph_cell_supervision_v2" ^
  --embedding-dir "D:\mimic_graph\embeddings\qwen3_0.6b_full_train_v1" ^
  --output-dir "D:\mimic_graph\experiments\triple_mlp_dev_666" ^
  --model-type triple_mlp ^
  --seed 666 ^
  --device cuda
```

For either R-GCN, change `--model-type` to `fullrow_rgcn` or `strict_rgcn`.
Their default batch size is 64 and fanout is `16,8`.
The shared default hidden dimension is 128; CUDA uses vectorized Fast-R-GCN-style
messages while CPU tests retain a low-memory relation-grouped implementation.

After model and hyperparameters are frozen, run seeds 666, 667, and 668 into
three new output directories and add `--evaluate-internal-test`. Do not reuse
fold 4 results to change the model.

Aggregate the three formal reports:

```bat
python -m graph.cell_training_aggregate_cli ^
  --reports ^
    "D:\mimic_graph\experiments\strict_rgcn_666" ^
    "D:\mimic_graph\experiments\strict_rgcn_667" ^
    "D:\mimic_graph\experiments\strict_rgcn_668" ^
  --output "D:\mimic_graph\experiments\strict_rgcn_aggregate.json"
```

Each run writes `config.json`, `progress.json`, `history.json`, `best_checkpoint.pt`,
`latest_checkpoint.pt`, `predictions.csv`, and `report.json`. Resume an
interrupted run with the exact same arguments plus `--resume`; optimizer and
Python/NumPy/PyTorch CPU/CUDA RNG states are restored. The report records
SHA-256 values for the best checkpoint and predictions.

Monitor a running experiment from another Windows terminal:

```bat
powershell -NoProfile -Command "while ($true) { Clear-Host; Get-Date; Get-Content 'D:\mimic_graph\experiments\strict_rgcn_666\progress.json'; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits; Start-Sleep 5 }"
```
