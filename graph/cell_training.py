from __future__ import annotations

import json
import hashlib
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from .cell_metrics import binary_metrics, select_macro_f1_threshold
from .cell_models import CellRGCN, TripleMLP
from .cell_sampling import TargetMaskedNeighborSampler
from .cell_training_data import (
    CellRecords,
    FrozenEmbeddingStore,
    PreparedCellTrainingData,
    prepare_cell_training_data,
    prepare_graph_cache,
    write_predictions_csv,
)


TRAIN_FOLDS = (0, 1, 2)
VALIDATION_FOLDS = (3,)
TEST_FOLDS = (4,)
EVALUATION_SAMPLING_EPOCH = 0


class CellTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    graph_dir: str
    supervision_dir: str
    embedding_dir: str
    output_dir: str
    model_type: str
    seed: int
    device: str = "auto"
    epochs: int = 40
    batch_size: int = 256
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    patience: int = 7
    rgcn_layers: int = 2
    rgcn_bases: int = 16
    fanouts: tuple[int, ...] = (16, 8)
    evaluate_internal_test: bool = False
    resume: bool = False

    def validate(self) -> None:
        if self.model_type not in {"triple_mlp", "fullrow_rgcn", "strict_rgcn"}:
            raise CellTrainingError(f"unsupported model type: {self.model_type}")
        if self.seed < 0 or self.epochs < 1 or self.batch_size < 1:
            raise CellTrainingError("seed, epochs and batch_size must be valid positive values")
        if self.hidden_dim < 8 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise CellTrainingError("invalid optimizer or hidden dimension configuration")
        if not 0 <= self.dropout < 1 or self.patience < 1:
            raise CellTrainingError("dropout and patience are invalid")
        if self.model_type != "triple_mlp" and len(self.fanouts) != self.rgcn_layers:
            raise CellTrainingError("R-GCN fanout count must match layer count")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 5:
                raise
            time.sleep(0.025 * (2 ** attempt))


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 5:
                raise
            time.sleep(0.025 * (2 ** attempt))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise CellTrainingError("CUDA was requested but is not available")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _batches(indices: np.ndarray, batch_size: int, *, rng: np.random.Generator | None) -> Iterator[np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).copy()
    if rng is not None:
        rng.shuffle(values)
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def _build_model(config: TrainingConfig, data: PreparedCellTrainingData, table_count: int) -> nn.Module:
    if config.model_type == "triple_mlp":
        return TripleMLP(data.identity.embedding_dimension, config.hidden_dim, config.dropout)
    return CellRGCN(
        data.identity.embedding_dimension,
        config.hidden_dim,
        data.identity.relation_count,
        table_count,
        strict_rows=config.model_type == "strict_rgcn",
        layer_count=config.rgcn_layers,
        basis_count=min(config.rgcn_bases, data.identity.relation_count),
        dropout=config.dropout,
    )


class _BatchForward:
    def __init__(
        self,
        config: TrainingConfig,
        data: PreparedCellTrainingData,
        device: torch.device,
    ) -> None:
        self.config = config
        self.data = data
        self.device = device
        self.store = FrozenEmbeddingStore(data)
        self.relation_tensor = torch.from_numpy(
            self.store.relation_rows(np.arange(data.identity.relation_count))
        ).to(device)
        self.sampler: TargetMaskedNeighborSampler | None = None
        self.metadata = None
        if config.model_type != "triple_mlp":
            graph, metadata = prepare_graph_cache(data)
            self.metadata = metadata
            self.sampler = TargetMaskedNeighborSampler(graph, metadata, fanouts=config.fanouts)

    @property
    def table_count(self) -> int:
        return len(self.metadata.tables) if self.metadata is not None else 1

    def __call__(
        self,
        model: nn.Module,
        record_indices: np.ndarray,
        *,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        records = self.data.records
        if self.config.model_type == "triple_mlp":
            row = torch.from_numpy(self.store.node_rows(records.row_id[record_indices])).to(self.device)
            relation = torch.from_numpy(
                self.store.relation_rows(records.relation_id[record_indices])
            ).to(self.device)
            value = torch.from_numpy(
                self.store.node_rows(records.value_node_id[record_indices])
            ).to(self.device)
            logits = model(row, relation, value)
            labels = torch.from_numpy(records.label[record_indices].astype(np.float32)).to(self.device)
            return logits, labels

        assert self.sampler is not None
        sampled = self.sampler.sample(
            records, record_indices, seed=self.config.seed, epoch=epoch
        )
        if self.config.model_type == "strict_rgcn":
            node_values = np.zeros(
                (len(sampled.node_id), self.data.identity.embedding_dimension),
                dtype=np.float32,
            )
            value_mask = sampled.node_type != 0
            node_values[value_mask] = self.store.node_rows(sampled.node_id[value_mask])
        else:
            node_values = self.store.node_rows(sampled.node_id)
        node_embedding = torch.from_numpy(node_values).to(self.device)
        logits = model(
            node_embedding,
            self.relation_tensor,
            torch.from_numpy(sampled.node_type.astype(np.int64)).to(self.device),
            torch.from_numpy(sampled.table_id.astype(np.int64)).to(self.device),
            torch.from_numpy(sampled.edge_source).to(self.device),
            torch.from_numpy(sampled.edge_target).to(self.device),
            torch.from_numpy(sampled.edge_relation).to(self.device),
            torch.from_numpy(sampled.edge_hop).to(self.device),
            torch.from_numpy(sampled.target_row).to(self.device),
            torch.from_numpy(sampled.target_value).to(self.device),
            torch.from_numpy(sampled.target_relation).to(self.device),
        )
        labels = torch.from_numpy(sampled.label).to(self.device)
        return logits, labels


def _predict(
    model: nn.Module,
    forward: _BatchForward,
    indices: np.ndarray,
    batch_size: int,
    *,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loss_sum = 0.0
    example_count = 0
    with torch.no_grad():
        for batch in _batches(indices, batch_size, rng=None):
            logits, target = forward(model, batch, epoch=epoch)
            loss_sum += float(
                nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="sum"
                )
            )
            example_count += len(target)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(target.cpu().numpy().astype(np.int8))
    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
        loss_sum / example_count,
    )


def _source_metrics(
    records: CellRecords,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    sources = records.source[indices]
    for source in sorted(np.unique(sources)):
        mask = sources == source
        source_name = {0: "sampled_clean", 1: "original_dirty", 2: "synthetic_dirty"}.get(
            int(source), f"source_{int(source)}"
        )
        source_scores = probabilities[mask]
        source_predictions = source_scores >= threshold
        if int(source) == 0:
            false_positives = int(source_predictions.sum())
            result[source_name] = {
                "count": int(mask.sum()),
                "tn": int(mask.sum()) - false_positives,
                "fp": false_positives,
                "specificity": 1.0 - false_positives / int(mask.sum()),
                "false_positive_rate": false_positives / int(mask.sum()),
                "dirty_score_mean": float(source_scores.mean()),
                "dirty_score_median": float(np.median(source_scores)),
                "dirty_score_p05": float(np.quantile(source_scores, 0.05)),
                "dirty_score_p95": float(np.quantile(source_scores, 0.95)),
            }
        else:
            true_positives = int(source_predictions.sum())
            result[source_name] = {
                "count": int(mask.sum()),
                "tp": true_positives,
                "fn": int(mask.sum()) - true_positives,
                "recall": true_positives / int(mask.sum()),
                "dirty_score_mean": float(source_scores.mean()),
                "dirty_score_median": float(np.median(source_scores)),
                "dirty_score_p05": float(np.quantile(source_scores, 0.05)),
                "dirty_score_p95": float(np.quantile(source_scores, 0.95)),
            }
    return result


def run_training(config: TrainingConfig) -> dict[str, Any]:
    config.validate()
    _seed_everything(config.seed)
    device = _device(config.device)
    output = Path(config.output_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise CellTrainingError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise CellTrainingError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = prepare_cell_training_data(
        config.graph_dir, config.supervision_dir, config.embedding_dir
    )
    forward = _BatchForward(config, data, device)
    model = _build_model(config, data, forward.table_count).to(device)
    train_indices = data.records.indices_for_folds(TRAIN_FOLDS)
    validation_indices = data.records.indices_for_folds(VALIDATION_FOLDS)
    test_indices = data.records.indices_for_folds(TEST_FOLDS)
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise CellTrainingError("train, validation and test folds must all be non-empty")

    for split, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("internal_test", test_indices),
    ):
        if set(np.unique(data.records.label[indices]).tolist()) != {0, 1}:
            raise CellTrainingError(f"{split} split must contain both clean and dirty Cells")

    train_labels = data.records.label[train_indices]
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    if not positives or not negatives:
        raise CellTrainingError("training data must contain both labels")
    positive_weight = negatives / positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint_path = output / "best_checkpoint.pt"
    latest_path = output / "latest_checkpoint.pt"
    start_epoch = 0
    best_auprc = -math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    identity = data.identity.to_dict()

    if config.resume:
        if not latest_path.is_file():
            raise CellTrainingError("resume requested but latest checkpoint is missing")
        state = torch.load(latest_path, map_location=device, weights_only=False)
        comparable = dict(asdict(config))
        comparable["resume"] = False
        if state.get("config") != comparable:
            raise CellTrainingError("resume configuration does not match checkpoint")
        if state.get("training_identity") != identity:
            raise CellTrainingError("resume data identity does not match checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_auprc = float(state["best_auprc"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state["history"])
        if "rng_state" not in state:
            raise CellTrainingError("resume checkpoint is missing RNG state")
        _restore_rng_state(state["rng_state"])

    frozen_config = dict(asdict(config))
    frozen_config["resume"] = False
    experiment_config = {
        key: value
        for key, value in frozen_config.items()
        if key not in {"graph_dir", "supervision_dir", "embedding_dir", "output_dir", "seed"}
    }
    _atomic_json(output / "config.json", {
        **frozen_config,
        "resolved_device": str(device),
        "train_folds": list(TRAIN_FOLDS),
        "validation_folds": list(VALIDATION_FOLDS),
        "test_folds": list(TEST_FOLDS),
        "loss": {
            "type": "frequency_weighted_binary_cross_entropy_with_logits",
            "positive_weight": positive_weight,
            "positive_label": "dirty",
        },
        "training_identity": identity,
    })
    started = time.perf_counter()
    _atomic_json(output / "progress.json", {
        "schema_version": 1,
        "status": "RUNNING",
        "model_type": config.model_type,
        "seed": config.seed,
        "epochs_requested": config.epochs,
        "epochs_completed": start_epoch,
        "best_epoch": best_epoch,
        "best_validation_average_precision": (
            best_auprc if math.isfinite(best_auprc) else None
        ),
        "stale_epochs": stale_epochs,
    })
    for epoch in range(start_epoch, config.epochs):
        if stale_epochs >= config.patience:
            break
        model.train()
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, epoch]))
        train_loss_sum = 0.0
        train_example_count = 0
        for batch in _batches(train_indices, config.batch_size, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            logits, labels = forward(model, batch, epoch=epoch)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(labels)
            train_example_count += len(labels)

        validation_labels, validation_probabilities, validation_loss = _predict(
            model,
            forward,
            validation_indices,
            config.batch_size,
            epoch=EVALUATION_SAMPLING_EPOCH,
        )
        validation_metrics = binary_metrics(
            validation_labels, validation_probabilities, threshold=0.5
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_example_count,
            "validation_loss": validation_loss,
            "validation_metrics_at_0_5": validation_metrics,
        }
        history.append(record)
        improved = float(validation_metrics["average_precision"]) > best_auprc + 1e-12
        if improved:
            best_auprc = float(validation_metrics["average_precision"])
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(checkpoint_path, {
                "model": model.state_dict(),
                "epoch": epoch,
                "validation_average_precision": best_auprc,
                "config": frozen_config,
                "training_identity": identity,
            })
        else:
            stale_epochs += 1
        _atomic_torch_save(latest_path, {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_auprc": best_auprc,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "rng_state": _rng_state(),
            "config": frozen_config,
            "training_identity": identity,
        })
        _atomic_json(output / "history.json", history)
        _atomic_json(output / "progress.json", {
            "schema_version": 1,
            "status": "RUNNING",
            "model_type": config.model_type,
            "seed": config.seed,
            "epochs_requested": config.epochs,
            "epochs_completed": epoch + 1,
            "latest_epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_average_precision": best_auprc,
            "stale_epochs": stale_epochs,
            "runtime_seconds": time.perf_counter() - started,
        })
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"val_average_precision={validation_metrics['average_precision']:.6f} "
            f"val_macro_f1_at_0.5={validation_metrics['macro_f1']:.6f}",
            flush=True,
        )
        if stale_epochs >= config.patience:
            break

    if not checkpoint_path.is_file():
        raise CellTrainingError("training did not create a best checkpoint")
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    best_epoch = int(best["epoch"])
    validation_labels, validation_probabilities, _ = _predict(
        model,
        forward,
        validation_indices,
        config.batch_size,
        epoch=EVALUATION_SAMPLING_EPOCH,
    )
    threshold = select_macro_f1_threshold(validation_labels, validation_probabilities)
    best["validation_threshold"] = threshold
    best["threshold_selection"] = "maximum_validation_macro_f1_after_epoch_selection"
    _atomic_torch_save(checkpoint_path, best)
    predictions: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    evaluation_splits = [
        ("train", train_indices),
        ("validation", validation_indices),
    ]
    if config.evaluate_internal_test:
        evaluation_splits.append(("internal_test", test_indices))
    fold_reports: dict[str, Any] = {}
    for split, indices in evaluation_splits:
        labels, probabilities, loss = _predict(
            model,
            forward,
            indices,
            config.batch_size,
            epoch=EVALUATION_SAMPLING_EPOCH,
        )
        metrics = binary_metrics(labels, probabilities, threshold)
        split_reports[split] = {
            "loss": loss,
            "metrics": metrics,
            "metrics_at_0_5": binary_metrics(labels, probabilities, 0.5),
            "by_source": _source_metrics(
                data.records, indices, probabilities, threshold
            ),
        }
        for fold in sorted(np.unique(data.records.fold[indices])):
            fold_mask = data.records.fold[indices] == fold
            fold_reports[str(int(fold))] = {
                "split": split,
                "metrics": binary_metrics(
                    labels[fold_mask], probabilities[fold_mask], threshold
                ),
                "metrics_at_0_5": binary_metrics(
                    labels[fold_mask], probabilities[fold_mask], 0.5
                ),
            }
        for position, record_index in enumerate(indices):
            predictions.append({
                "split": split,
                "observation_index": int(data.records.observation_index[record_index]),
                "row_id": int(data.records.row_id[record_index]),
                "relation_id": int(data.records.relation_id[record_index]),
                "value_node_id": int(data.records.value_node_id[record_index]),
                "fold": int(data.records.fold[record_index]),
                "source": int(data.records.source[record_index]),
                "label": int(labels[position]),
                "dirty_score": float(probabilities[position]),
                "threshold": threshold,
                "prediction": int(probabilities[position] >= threshold),
                "model_type": config.model_type,
                "seed": config.seed,
            })
    predictions.sort(key=lambda item: item["observation_index"])
    write_predictions_csv(output / "predictions.csv", iter(predictions))
    checkpoint_sha256 = _sha256(checkpoint_path)
    predictions_sha256 = _sha256(output / "predictions.csv")
    report = {
        "schema_version": 1,
        "status": "SUCCESS",
        "model_type": config.model_type,
        "seed": config.seed,
        "experiment_config": experiment_config,
        "training_identity": identity,
        "best_epoch": best_epoch,
        "selection_metric": "validation_average_precision",
        "selection_metric_alias": "validation_auprc",
        "threshold_selection": "maximum_validation_macro_f1_after_epoch_selection",
        "evaluation_protocol": "transductive_full_graph_preassigned_folds",
        "fold_provenance": {
            "assignment": "upstream_supervision_subject_grouped_folds",
            "subject_disjoint_reverified_at_training": False,
        },
        "score_semantics": "dirty_score_not_calibrated_probability",
        "sampling": {
            "training": "deterministic_per_seed_epoch_and_observation",
            "evaluation": "deterministic_per_seed_and_observation",
            "evaluation_sampling_epoch": EVALUATION_SAMPLING_EPOCH,
            "target_edge_masking": "exact_forward_and_reverse_edge_ids_before_sampling",
        },
        "validation_threshold": threshold,
        "loss": {
            "type": "frequency_weighted_binary_cross_entropy_with_logits",
            "positive_weight": positive_weight,
            "positive_label": "dirty",
        },
        "split_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "internal_test": len(test_indices),
        },
        "internal_test_evaluated": config.evaluate_internal_test,
        "splits": split_reports,
        "folds": fold_reports,
        "runtime_seconds": time.perf_counter() - started,
        "outputs": {
            "best_checkpoint": {
                "path": checkpoint_path.name,
                "sha256": checkpoint_sha256,
            },
            "latest_checkpoint": latest_path.name,
            "history": "history.json",
            "predictions": {
                "path": "predictions.csv",
                "sha256": predictions_sha256,
                "row_count": len(predictions),
            },
            "config": "config.json",
            "progress": "progress.json",
        },
    }
    _atomic_json(output / "report.json", report)
    _atomic_json(output / "progress.json", {
        "schema_version": 1,
        "status": "SUCCESS",
        "model_type": config.model_type,
        "seed": config.seed,
        "epochs_requested": config.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_auprc,
        "validation_threshold": threshold,
        "internal_test_evaluated": config.evaluate_internal_test,
        "runtime_seconds": report["runtime_seconds"],
    })
    return report
