from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .cell_sampling import TargetMaskedNeighborSampler
from .cell_training_data import (
    FrozenEmbeddingStore,
    PreparedCellTrainingData,
    prepare_cell_training_data,
    prepare_graph_cache,
)
from .embedding import Qwen3EmbeddingBackend
from .features import canonical_scalar
from .schema import DEFAULT_SCHEMA


RANKING_SCHEMA_VERSION = 1
TRAIN_FOLDS = (0, 1, 2)
VALIDATION_FOLD = 3
EVALUATION_SAMPLING_EPOCH = 0
MAX_CANDIDATES = 5
PREPARED_OUTPUTS = frozenset({
    "row_hidden.npy",
    "relation_hidden.npy",
    "value_hidden.npy",
    "ranking_examples.npz",
    "values.jsonl",
    "validation_candidates.jsonl",
})
EXAMPLE_ARRAYS = frozenset({
    "context_observation",
    "context_relation",
    "context_fold",
    "pair_context",
    "pair_positive",
    "pair_negative",
    "pair_weight",
    "validation_group_observation",
    "validation_group_context",
    "validation_group_gold",
    "validation_group_gold_position",
    "validation_group_label",
    "validation_offsets",
    "validation_candidate_ids",
})


class CandidateRankingError(ValueError):
    pass


@dataclass(frozen=True)
class PrepareRankingConfig:
    graph_dir: str
    supervision_dir: str
    embedding_dir: str
    detector_checkpoint: str
    predictions: str
    train_candidates: str
    train_candidate_manifest: str
    validation_candidates: str
    validation_candidate_manifest: str
    paired_log: str
    paired_log_manifest: str
    qwen_model: str
    output_dir: str
    device: str = "auto"
    qwen_revision: str | None = None
    qwen_batch_size: int = 32
    rgcn_batch_size: int = 64

    def validate(self) -> None:
        if self.qwen_batch_size < 1 or self.rgcn_batch_size < 1:
            raise CandidateRankingError("batch sizes must be positive")


@dataclass(frozen=True)
class RankTrainingConfig:
    prepared_dir: str
    output_dir: str
    seed: int
    device: str = "auto"
    epochs: int = 80
    batch_size: int = 512
    rank_dim: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    patience: int = 10

    def validate(self) -> None:
        if self.seed < 0 or self.epochs < 1 or self.batch_size < 1:
            raise CandidateRankingError("seed, epochs and batch size must be valid")
        if self.rank_dim < 4 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise CandidateRankingError("rank dimension or optimizer configuration is invalid")
        if not 0 <= self.dropout < 1 or self.patience < 1:
            raise CandidateRankingError("dropout or patience is invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _atomic_torch_save(torch: Any, path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    _replace_with_retry(temporary, path)


def _replace_with_retry(temporary: Path, path: Path) -> None:
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 5:
                raise
            time.sleep(0.025 * (2 ** attempt))


def _new_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise CandidateRankingError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _required_file(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise CandidateRankingError(f"missing {label}: {value}")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateRankingError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateRankingError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CandidateRankingError(
                        f"{label} record is not an object at line {line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        if isinstance(exc, CandidateRankingError):
            raise
        raise CandidateRankingError(f"invalid {label}: {path}") from exc
    if not rows:
        raise CandidateRankingError(f"{label} is empty: {path}")
    return rows


def _candidate_strings(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_CANDIDATES:
        raise CandidateRankingError("candidates must be a list of at most five records")
    result: list[str] = []
    for candidate in value:
        if not isinstance(candidate, dict) or set(candidate) != {"value", "rule_id", "evidence"}:
            raise CandidateRankingError("candidate record has an invalid schema")
        candidate_value = candidate["value"]
        if (
            not isinstance(candidate_value, str)
            or not isinstance(candidate["rule_id"], str)
            or not candidate["rule_id"]
            or not isinstance(candidate["evidence"], str)
            or not candidate["evidence"]
        ):
            raise CandidateRankingError("candidate fields must be non-empty strings")
        if candidate_value not in result:
            result.append(candidate_value)
    if len(result) != len(value):
        raise CandidateRankingError("candidate values must be unique per target")
    return result


def load_candidate_artifact(path: str | Path, expected_split: str) -> dict[int, dict[str, Any]]:
    candidate_path = _required_file(path, f"{expected_split} candidates")
    lowered_parts = {part.casefold() for part in candidate_path.parts}
    if "internal_test" in lowered_parts or expected_split == "internal_test":
        raise CandidateRankingError("Internal Test candidates are forbidden during ranker training")
    rows = _read_jsonl(candidate_path, f"{expected_split} candidates")
    result: dict[int, dict[str, Any]] = {}
    exact_schema = {
        "split", "observation_index", "table", "raw_row_index", "column",
        "current_value", "dirty_score", "field_rule", "reason", "candidates",
    }
    for row in rows:
        if set(row) != exact_schema:
            raise CandidateRankingError(
                "candidate artifact has an invalid or private top-level schema"
            )
        if row["split"] != expected_split:
            raise CandidateRankingError(
                f"expected only {expected_split} candidates, got {row['split']!r}"
            )
        index = int(row["observation_index"])
        if index < 0 or index in result:
            raise CandidateRankingError("candidate observation indices must be unique and non-negative")
        normalized = dict(row)
        normalized["observation_index"] = index
        normalized["raw_row_index"] = int(row["raw_row_index"])
        normalized["candidate_values"] = _candidate_strings(row["candidates"])
        result[index] = normalized
    return result


def _validate_candidate_manifest(
    manifest_path: Path,
    candidate_path: Path,
    expected_split: str,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, f"{expected_split} candidate manifest")
    if (
        manifest.get("workflow") != "offline_frozen_multicandidate_fcorr_execution"
        or manifest.get("split") != expected_split
        or manifest.get("selection_performed") is not False
        or manifest.get("llm_called") is not False
        or int(manifest.get("maximum_candidates_per_target", -1)) != MAX_CANDIDATES
        or manifest.get("candidate_values_sha256") != _sha256(candidate_path)
        or not isinstance(manifest.get("rule_registry_sha256"), str)
    ):
        raise CandidateRankingError(
            f"{expected_split} candidate manifest is not a valid frozen execution"
        )
    return manifest


def _normalize_table(raw_file: str) -> str:
    value = str(raw_file).replace("\\", "/")
    if value.endswith(".csv.gz"):
        return value[:-7]
    if value.endswith(".csv"):
        return value[:-4]
    return value


def _read_selected_gold(
    path: Path,
    coordinates: set[tuple[str, int, str]],
) -> dict[tuple[str, int, str], dict[str, str]]:
    result: dict[tuple[str, int, str], dict[str, str]] = {}
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise CandidateRankingError(f"cannot read paired log: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"raw_file", "raw_row_index", "column", "clean_value", "dirty_value"}
        if not required.issubset(reader.fieldnames or ()):
            raise CandidateRankingError("paired log has an invalid schema")
        for row in reader:
            key = (
                _normalize_table(row["raw_file"]),
                int(row["raw_row_index"]) + 1,
                str(row["column"]),
            )
            if key not in coordinates:
                continue
            if key in result:
                raise CandidateRankingError(f"duplicate paired coordinate: {key}")
            result[key] = {str(name): str(value or "") for name, value in row.items()}
    return result


_NUMBER = __import__("re").compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_MISSING = {"", "nan", "none", "null", "nat"}


def value_embedding_text(
    table: str,
    column: str,
    value: str,
    *,
    icd_version: str = "",
) -> str:
    field_spec = DEFAULT_SCHEMA.field_for(table, column)
    canonical = (
        canonical_scalar(value, field_spec.canonicalizer, icd_version=icd_version)
        if field_spec
        else None
    )
    if field_spec and canonical is not None:
        domain = field_spec.domain
        normalized = canonical
        shared = True
    else:
        stripped = value.strip()
        normalized = "<MISSING>" if stripped.casefold() in _MISSING else stripped
        domain = f"{table}.{column}"
        shared = False
    if normalized == "<MISSING>":
        value_type = "missing"
    elif shared:
        value_type = "identifier"
    elif _NUMBER.fullmatch(normalized):
        value_type = "numeric"
    else:
        from datetime import datetime

        try:
            datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError:
            value_type = "text"
        else:
            value_type = "datetime"
    return json.dumps(
        {
            "domain": domain,
            "kind": "value",
            "shared": shared,
            "value": normalized,
            "value_type": value_type,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def exact_candidate_position(candidates: Sequence[str], target: str) -> int:
    try:
        return list(candidates).index(target)
    except ValueError:
        return -1


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise CandidateRankingError("ranking supports only auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise CandidateRankingError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _load_strict_rgcn(
    torch: Any,
    state: dict[str, Any],
    data: PreparedCellTrainingData,
    table_count: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    from .cell_models import CellRGCN

    config = state.get("config")
    if not isinstance(config, dict) or config.get("model_type") != "strict_rgcn":
        raise CandidateRankingError("detector checkpoint must be a Strict R-GCN checkpoint")
    if state.get("training_identity") != data.identity.to_dict():
        raise CandidateRankingError("detector checkpoint identity does not match graph artifacts")
    fanouts = tuple(int(value) for value in config.get("fanouts", ()))
    layer_count = int(config.get("rgcn_layers", 0))
    if layer_count < 1 or len(fanouts) != layer_count:
        raise CandidateRankingError("detector checkpoint has an invalid R-GCN sampling contract")
    model = CellRGCN(
        data.identity.embedding_dimension,
        int(config["hidden_dim"]),
        data.identity.relation_count,
        table_count,
        strict_rows=True,
        layer_count=layer_count,
        basis_count=min(int(config["rgcn_bases"]), data.identity.relation_count),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, config


def _encode_context_batch(
    torch: Any,
    model: Any,
    sampler: TargetMaskedNeighborSampler,
    store: FrozenEmbeddingStore,
    relation_input: Any,
    records: Any,
    record_indices: np.ndarray,
    *,
    seed: int,
    device: Any,
) -> np.ndarray:
    sampled = sampler.sample(
        records,
        record_indices,
        seed=seed,
        epoch=EVALUATION_SAMPLING_EPOCH,
    )
    node_values = np.zeros((len(sampled.node_id), store.nodes.shape[1]), dtype=np.float32)
    value_mask = sampled.node_type != 0
    node_values[value_mask] = store.node_rows(sampled.node_id[value_mask])
    captured: list[Any] = []

    def capture_head_inputs(_module: Any, args: tuple[Any, ...]) -> None:
        captured.append(args[0].detach())

    hook = model.head.register_forward_pre_hook(capture_head_inputs)
    try:
        with torch.inference_mode():
            model(
                torch.from_numpy(node_values).to(device),
                relation_input,
                torch.from_numpy(sampled.node_type.astype(np.int64)).to(device),
                torch.from_numpy(sampled.table_id.astype(np.int64)).to(device),
                torch.from_numpy(sampled.edge_source).to(device),
                torch.from_numpy(sampled.edge_target).to(device),
                torch.from_numpy(sampled.edge_relation).to(device),
                torch.from_numpy(sampled.edge_hop).to(device),
                torch.from_numpy(sampled.target_row).to(device),
                torch.from_numpy(sampled.target_value).to(device),
                torch.from_numpy(sampled.target_relation).to(device),
            )
    finally:
        hook.remove()
    if len(captured) != 1:
        raise CandidateRankingError("failed to capture Strict R-GCN Row representations")
    return captured[0].float().cpu().numpy()


def _encode_values(
    backend: Qwen3EmbeddingBackend,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunks.append(backend.encode(texts[start:start + batch_size]).values)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _verify_qwen_identity(
    backend_metadata: dict[str, Any],
    embedding_manifest: dict[str, Any],
) -> None:
    identity = embedding_manifest.get("identity")
    expected = identity.get("backend") if isinstance(identity, dict) else None
    if not isinstance(expected, dict):
        raise CandidateRankingError("embedding manifest is missing backend identity")
    keys = (
        "type", "model_artifact_sha256", "tokenizer_sha256", "pooling",
        "normalization", "max_length",
    )
    mismatched = [key for key in keys if expected.get(key) != backend_metadata.get(key)]
    if mismatched:
        raise CandidateRankingError(
            f"Qwen backend does not match graph embeddings: {', '.join(mismatched)}"
        )


def export_train_validation_gold(
    *,
    graph_dir: str | Path,
    supervision_dir: str | Path,
    paired_log: str | Path,
    predictions: str | Path,
    detector_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    graph = Path(graph_dir).expanduser().resolve()
    supervision = Path(supervision_dir).expanduser().resolve()
    source_path = _required_file(paired_log, "source paired log")
    source_predictions_path = _required_file(predictions, "source detector predictions")
    detector_report_path = _required_file(detector_report, "detector report")
    masks_path = _required_file(supervision / "supervision_masks.npz", "supervision masks")
    observations_path = _required_file(graph / "cell_observations.jsonl", "Cell observations")
    graph_manifest_path = _required_file(graph / "graph_manifest.json", "graph manifest")
    supervision_manifest_path = _required_file(
        supervision / "supervision_manifest.json", "supervision manifest"
    )
    output = _new_output(output_dir)
    detector = _read_json(detector_report_path, "detector report")
    detector_outputs = detector.get("outputs")
    checkpoint_output = (
        detector_outputs.get("best_checkpoint") if isinstance(detector_outputs, dict) else None
    )
    predictions_output = (
        detector_outputs.get("predictions") if isinstance(detector_outputs, dict) else None
    )
    if (
        detector.get("status") != "SUCCESS"
        or detector.get("model_type") != "strict_rgcn"
        or not isinstance(detector.get("training_identity"), dict)
        or not isinstance(checkpoint_output, dict)
        or not isinstance(checkpoint_output.get("sha256"), str)
        or not isinstance(predictions_output, dict)
        or predictions_output.get("sha256") != _sha256(source_predictions_path)
    ):
        raise CandidateRankingError("detector report does not bind the supplied Strict R-GCN outputs")
    try:
        with np.load(masks_path) as masks:
            indices = np.asarray(masks["cell_indices"], dtype=np.int64)
            labels = np.asarray(masks["cell_labels"], dtype=np.int8)
            folds = np.asarray(masks["cell_folds"], dtype=np.int8)
    except (OSError, ValueError, KeyError) as exc:
        raise CandidateRankingError("invalid supervision masks") from exc
    if len({len(indices), len(labels), len(folds)}) != 1:
        raise CandidateRankingError("supervision arrays have inconsistent lengths")
    selected = set(int(value) for value in indices[(labels == 1) & (folds <= VALIDATION_FOLD)])
    if np.any((folds < 0) | (folds > 4)) or not selected:
        raise CandidateRankingError("supervision folds are invalid or Train/Validation Gold is empty")
    coordinates: set[tuple[str, int, str]] = set()
    try:
        with observations_path.open(encoding="utf-8") as handle:
            for observation_index, line in enumerate(handle):
                if observation_index not in selected:
                    continue
                observation = json.loads(line)
                coordinates.add((
                    str(observation["table"]),
                    int(observation["row_number"]),
                    str(observation["column"]),
                ))
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise CandidateRankingError("cannot map scoped Gold observations") from exc
    if len(coordinates) != len(selected):
        raise CandidateRankingError("scoped Gold observations are missing or duplicate")

    scoped_predictions_path = output / "predictions_train_validation.csv"
    prediction_count = 0
    seen_prediction_indices: set[int] = set()
    with source_predictions_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required_predictions = {
            "split", "observation_index", "fold", "label", "prediction",
            "model_type", "seed",
        }
        if not required_predictions.issubset(reader.fieldnames or ()):
            raise CandidateRankingError("source predictions have an invalid schema")
        with scoped_predictions_path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                split = str(row["split"])
                if split not in {"train", "validation"}:
                    continue
                fold = int(row["fold"])
                if (
                    (split == "train" and fold not in TRAIN_FOLDS)
                    or (split == "validation" and fold != VALIDATION_FOLD)
                    or row["model_type"] != "strict_rgcn"
                    or int(row["seed"]) != int(detector["seed"])
                    or row["label"] not in {"0", "1"}
                    or row["prediction"] not in {"0", "1"}
                ):
                    raise CandidateRankingError("source predictions violate the fold/model contract")
                observation_index = int(row["observation_index"])
                if observation_index in seen_prediction_indices:
                    raise CandidateRankingError("source predictions contain duplicate observations")
                seen_prediction_indices.add(observation_index)
                prediction_count += 1
                writer.writerow(row)
    if prediction_count < 1:
        raise CandidateRankingError("scoped Train/Validation predictions are empty")

    target_path = output / "paired_gold_train_validation.csv"
    seen: set[tuple[str, int, str]] = set()
    with source_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"raw_file", "raw_row_index", "column", "clean_value", "dirty_value"}
        if not required.issubset(reader.fieldnames or ()):
            raise CandidateRankingError("source paired log has an invalid schema")
        with target_path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                coordinate = (
                    _normalize_table(row["raw_file"]),
                    int(row["raw_row_index"]) + 1,
                    str(row["column"]),
                )
                if coordinate not in coordinates:
                    continue
                if coordinate in seen:
                    raise CandidateRankingError(f"duplicate scoped Gold coordinate: {coordinate}")
                seen.add(coordinate)
                writer.writerow(row)
    if seen != coordinates:
        missing = sorted(coordinates - seen)
        raise CandidateRankingError(f"source paired log is missing scoped Gold: {missing[:5]}")
    report = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "host_private_train_validation_gold_scope",
        "folds": [0, 1, 2, 3],
        "internal_test_gold_exported": False,
        "source_contains_unscoped_gold": True,
        "record_count": len(seen),
        "prediction_count": prediction_count,
        "detector": {
            "model_type": "strict_rgcn",
            "seed": int(detector["seed"]),
            "training_identity": detector["training_identity"],
            "best_checkpoint_sha256": checkpoint_output["sha256"],
            "source_predictions_sha256": predictions_output["sha256"],
            "report_sha256": _sha256(detector_report_path),
        },
        "inputs": {
            "source_paired_log_sha256": _sha256(source_path),
            "source_predictions_sha256": _sha256(source_predictions_path),
            "graph_manifest_sha256": _sha256(graph_manifest_path),
            "supervision_manifest_sha256": _sha256(supervision_manifest_path),
            "supervision_masks_sha256": _sha256(masks_path),
            "cell_observations_sha256": _sha256(observations_path),
        },
        "outputs": {
            "paired_gold": {
                "path": target_path.name,
                "sha256": _sha256(target_path),
            },
            "predictions": {
                "path": scoped_predictions_path.name,
                "sha256": _sha256(scoped_predictions_path),
            },
        },
    }
    _atomic_json(output / "scoped_gold_manifest.json", report)
    return report


def _batches(values: np.ndarray, batch_size: int, rng: np.random.Generator | None) -> Iterator[np.ndarray]:
    indices = np.asarray(values, dtype=np.int64).copy()
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def _read_scoped_predictions(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    result: dict[str, dict[int, dict[str, str]]] = {"train": {}, "validation": {}}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split", "observation_index", "fold", "label", "prediction",
            "model_type", "seed",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise CandidateRankingError("scoped predictions have an invalid schema")
        for row in reader:
            split = str(row["split"])
            if split not in result:
                raise CandidateRankingError("scoped predictions contain a forbidden split")
            index = int(row["observation_index"])
            if index in result[split]:
                raise CandidateRankingError("scoped predictions contain duplicate observations")
            result[split][index] = dict(row)
    if not result["train"] or not result["validation"]:
        raise CandidateRankingError("scoped predictions must contain Train and Validation")
    return result


def prepare_ranking_data(config: PrepareRankingConfig) -> dict[str, Any]:
    config.validate()
    output = _new_output(config.output_dir)
    train_path = _required_file(config.train_candidates, "Train candidate artifact")
    validation_path = _required_file(config.validation_candidates, "Validation candidate artifact")
    train_manifest_path = _required_file(
        config.train_candidate_manifest, "Train candidate execution manifest"
    )
    validation_manifest_path = _required_file(
        config.validation_candidate_manifest, "Validation candidate execution manifest"
    )
    checkpoint_path = _required_file(config.detector_checkpoint, "detector checkpoint")
    predictions_path = _required_file(config.predictions, "scoped detector predictions")
    paired_log_path = _required_file(config.paired_log, "paired log")
    paired_log_manifest_path = _required_file(
        config.paired_log_manifest, "scoped paired-log manifest"
    )
    train_candidates = load_candidate_artifact(train_path, "train")
    validation_candidates = load_candidate_artifact(validation_path, "validation")
    train_candidate_manifest = _validate_candidate_manifest(
        train_manifest_path, train_path, "train"
    )
    validation_candidate_manifest = _validate_candidate_manifest(
        validation_manifest_path, validation_path, "validation"
    )
    if (
        train_candidate_manifest["rule_registry_sha256"]
        != validation_candidate_manifest["rule_registry_sha256"]
        or int(train_candidate_manifest.get("target_count", -1)) != len(train_candidates)
        or int(validation_candidate_manifest.get("target_count", -1))
        != len(validation_candidates)
    ):
        raise CandidateRankingError(
            "Train and Validation candidates must use one frozen registry and complete targets"
        )
    if set(train_candidates) & set(validation_candidates):
        raise CandidateRankingError("Train and Validation candidate observations overlap")

    data = prepare_cell_training_data(
        config.graph_dir, config.supervision_dir, config.embedding_dir
    )
    gold_manifest = _read_json(paired_log_manifest_path, "scoped paired-log manifest")
    gold_outputs = gold_manifest.get("outputs")
    gold_output = gold_outputs.get("paired_gold") if isinstance(gold_outputs, dict) else None
    scoped_predictions_output = (
        gold_outputs.get("predictions") if isinstance(gold_outputs, dict) else None
    )
    gold_inputs = gold_manifest.get("inputs")
    scoped_detector = gold_manifest.get("detector")
    if (
        gold_manifest.get("workflow") != "host_private_train_validation_gold_scope"
        or gold_manifest.get("folds") != [0, 1, 2, 3]
        or gold_manifest.get("internal_test_gold_exported") is not False
        or not isinstance(gold_output, dict)
        or gold_output.get("sha256") != _sha256(paired_log_path)
        or not isinstance(scoped_predictions_output, dict)
        or scoped_predictions_output.get("sha256") != _sha256(predictions_path)
        or not isinstance(scoped_detector, dict)
        or scoped_detector.get("model_type") != "strict_rgcn"
        or scoped_detector.get("best_checkpoint_sha256") != _sha256(checkpoint_path)
        or scoped_detector.get("training_identity") != data.identity.to_dict()
        or not isinstance(gold_inputs, dict)
        or gold_inputs.get("graph_manifest_sha256")
        != data.identity.graph_manifest_sha256
        or gold_inputs.get("supervision_manifest_sha256")
        != data.identity.supervision_manifest_sha256
        or gold_inputs.get("supervision_masks_sha256")
        != data.identity.supervision_masks_sha256
    ):
        raise CandidateRankingError(
            "paired log is not a verified Train/Validation-only Gold artifact"
        )
    prediction_rows = _read_scoped_predictions(predictions_path)
    records = data.records
    observation_to_record = {
        int(observation): index for index, observation in enumerate(records.observation_index)
    }
    for split, expected_fold, rows in (
        ("train", set(TRAIN_FOLDS), train_candidates),
        ("validation", {VALIDATION_FOLD}, validation_candidates),
    ):
        for observation in rows:
            record_index = observation_to_record.get(observation)
            if record_index is None or int(records.fold[record_index]) not in expected_fold:
                raise CandidateRankingError(
                    f"{split} candidates reference an observation outside the allowed folds"
                )

    for split, expected_folds in (("train", set(TRAIN_FOLDS)), ("validation", {VALIDATION_FOLD})):
        expected_prediction_observations = {
            int(records.observation_index[index])
            for index in np.flatnonzero(np.isin(records.fold, np.asarray(sorted(expected_folds))))
        }
        if set(prediction_rows[split]) != expected_prediction_observations:
            raise CandidateRankingError(
                f"scoped {split} predictions must cover every supervised Cell"
            )
        expected_positive: set[int] = set()
        for observation, prediction in prediction_rows[split].items():
            record_index = observation_to_record.get(observation)
            if record_index is None:
                raise CandidateRankingError("predictions reference an unknown supervised observation")
            if (
                int(records.fold[record_index]) not in expected_folds
                or int(prediction["fold"]) != int(records.fold[record_index])
                or int(prediction["label"]) != int(records.label[record_index])
                or prediction["model_type"] != "strict_rgcn"
                or int(prediction["seed"]) != int(scoped_detector["seed"])
            ):
                raise CandidateRankingError("predictions disagree with supervision or detector identity")
            if prediction["prediction"] == "1":
                expected_positive.add(observation)
        actual = set(train_candidates if split == "train" else validation_candidates)
        if actual != expected_positive:
            raise CandidateRankingError(
                f"{split} candidates must cover every and only detector-positive Cell"
            )

    train_dirty_records = np.flatnonzero(
        np.isin(records.fold, np.asarray(TRAIN_FOLDS)) & (records.label == 1)
    )
    validation_dirty_total = int(
        np.sum((records.fold == VALIDATION_FOLD) & (records.label == 1))
    )
    context_observations = set(int(records.observation_index[index]) for index in train_dirty_records)
    context_observations.update(train_candidates)
    context_observations.update(validation_candidates)
    context_record_indices = np.asarray(
        sorted(observation_to_record[index] for index in context_observations), dtype=np.int64
    )
    ordered_observations = records.observation_index[context_record_indices].astype(np.int64)
    context_index = {int(value): index for index, value in enumerate(ordered_observations)}
    needed_rows = set(int(records.row_id[index]) for index in context_record_indices)

    observations: dict[int, dict[str, Any]] = {}
    icd_versions: dict[int, str] = {}
    observations_path = Path(config.graph_dir).expanduser().resolve() / "cell_observations.jsonl"
    try:
        with observations_path.open(encoding="utf-8") as handle:
            for observation_index, line in enumerate(handle):
                value = json.loads(line)
                row_id = int(value["row_id"])
                if observation_index in context_observations:
                    observations[observation_index] = value
                if row_id in needed_rows and str(value["column"]) == "icd_version":
                    icd_versions[row_id] = str(value.get("raw_value") or "")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise CandidateRankingError("cannot read required graph Cell observations") from exc
    missing = sorted(context_observations - set(observations))
    if missing:
        raise CandidateRankingError(f"missing graph observations: {missing[:5]}")

    for candidate_rows in (train_candidates, validation_candidates):
        for observation_index, row in candidate_rows.items():
            observation = observations[observation_index]
            actual = (
                str(row["table"]), int(row["raw_row_index"]), str(row["column"]),
                str(row["current_value"]),
            )
            expected = (
                str(observation["table"]), int(observation["row_number"]) - 1,
                str(observation["column"]), str(observation.get("raw_value") or ""),
            )
            if actual != expected:
                raise CandidateRankingError(
                    f"candidate coordinate mismatch at observation {observation_index}"
                )

    dirty_observations = {
        int(records.observation_index[index])
        for index in np.flatnonzero(
            np.isin(records.fold, np.asarray((*TRAIN_FOLDS, VALIDATION_FOLD)))
            & (records.label == 1)
        )
    }
    dirty_coordinates = {
        (
            str(observations[index]["table"]), int(observations[index]["row_number"]),
            str(observations[index]["column"]),
        )
        for index in dirty_observations
        if index in observations
    }
    gold = _read_selected_gold(paired_log_path, dirty_coordinates)

    value_texts: list[str] = []
    value_records: list[dict[str, str]] = []
    value_id_by_text: dict[str, int] = {}

    def intern_value(observation_index: int, raw_value: str) -> int:
        observation = observations[observation_index]
        row_id = int(observation["row_id"])
        text = value_embedding_text(
            str(observation["table"]),
            str(observation["column"]),
            str(raw_value),
            icd_version=icd_versions.get(row_id, ""),
        )
        existing = value_id_by_text.get(text)
        if existing is not None:
            return existing
        value_id = len(value_texts)
        value_id_by_text[text] = value_id
        value_texts.append(text)
        value_records.append({"embedding_text": text, "raw_value": str(raw_value)})
        return value_id

    pair_context: list[int] = []
    pair_positive: list[int] = []
    pair_negative: list[int] = []
    pair_weight: list[float] = []
    train_positive_observations = 0
    for record_index in train_dirty_records:
        observation_index = int(records.observation_index[record_index])
        observation = observations[observation_index]
        coordinate = (
            str(observation["table"]), int(observation["row_number"]),
            str(observation["column"]),
        )
        truth = gold.get(coordinate)
        current = str(observation.get("raw_value") or "")
        if truth is None or current != truth["dirty_value"]:
            raise CandidateRankingError(f"missing or mismatched Train Gold at {coordinate}")
        clean = truth["clean_value"]
        negatives = [current]
        candidate_row = train_candidates.get(observation_index)
        if candidate_row is not None:
            negatives.extend(candidate_row["candidate_values"])
        negatives = list(dict.fromkeys(value for value in negatives if value != clean))
        if not negatives:
            continue
        positive_id = intern_value(observation_index, clean)
        negative_ids = list(dict.fromkeys(
            intern_value(observation_index, negative) for negative in negatives
        ))
        negative_ids = [value_id for value_id in negative_ids if value_id != positive_id]
        if not negative_ids:
            continue
        weight = 1.0 / len(negative_ids)
        for negative_id in negative_ids:
            pair_context.append(context_index[observation_index])
            pair_positive.append(positive_id)
            pair_negative.append(negative_id)
            pair_weight.append(weight)
        train_positive_observations += 1

    train_clean_detector_examples = 0
    for observation_index, candidate_row in train_candidates.items():
        record_index = observation_to_record[observation_index]
        if int(records.label[record_index]) != 0:
            continue
        current = str(observations[observation_index].get("raw_value") or "")
        negatives = list(dict.fromkeys(
            value for value in candidate_row["candidate_values"] if value != current
        ))
        if not negatives:
            continue
        positive_id = intern_value(observation_index, current)
        negative_ids = list(dict.fromkeys(
            intern_value(observation_index, negative) for negative in negatives
        ))
        negative_ids = [value_id for value_id in negative_ids if value_id != positive_id]
        if not negative_ids:
            continue
        weight = 1.0 / len(negative_ids)
        for negative_id in negative_ids:
            pair_context.append(context_index[observation_index])
            pair_positive.append(positive_id)
            pair_negative.append(negative_id)
            pair_weight.append(weight)
        train_clean_detector_examples += 1
    if not pair_context:
        raise CandidateRankingError("no Train pairwise ranking examples were constructed")

    validation_group_observation: list[int] = []
    validation_group_context: list[int] = []
    validation_group_gold: list[int] = []
    validation_group_gold_position: list[int] = []
    validation_group_label: list[int] = []
    validation_offsets = [0]
    validation_candidate_ids: list[int] = []
    validation_candidate_raw: list[list[str]] = []
    for observation_index, candidate_row in sorted(validation_candidates.items()):
        record_index = observation_to_record[observation_index]
        label = int(records.label[record_index])
        observation = observations[observation_index]
        current = str(observation.get("raw_value") or "")
        if label == 1:
            coordinate = (
                str(observation["table"]), int(observation["row_number"]),
                str(observation["column"]),
            )
            truth = gold.get(coordinate)
            if truth is None or current != truth["dirty_value"]:
                raise CandidateRankingError(f"missing or mismatched Validation Gold at {coordinate}")
            target = truth["clean_value"]
        else:
            target = current
        candidates = candidate_row["candidate_values"]
        validation_group_observation.append(observation_index)
        validation_group_context.append(context_index[observation_index])
        validation_group_gold.append(intern_value(observation_index, target))
        validation_group_gold_position.append(exact_candidate_position(candidates, target))
        validation_group_label.append(label)
        validation_candidate_ids.extend(intern_value(observation_index, value) for value in candidates)
        validation_candidate_raw.append(list(candidates))
        validation_offsets.append(len(validation_candidate_ids))

    backend = Qwen3EmbeddingBackend(
        config.qwen_model,
        device=config.device,
        max_length=int(data.embedding_manifest["identity"]["max_length"]),
        revision=config.qwen_revision,
    )
    _verify_qwen_identity(backend.metadata, data.embedding_manifest)
    value_input = _encode_values(backend, value_texts, config.qwen_batch_size)
    qwen_metadata = backend.metadata
    del backend
    gc.collect()

    try:
        import torch
    except ImportError as exc:
        raise CandidateRankingError("ranking preparation requires torch") from exc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = _resolve_device(torch, config.device)
    graph, metadata = prepare_graph_cache(data)
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, detector_config = _load_strict_rgcn(
        torch, checkpoint_state, data, len(metadata.tables), device
    )
    del checkpoint_state
    fanouts = tuple(int(value) for value in detector_config["fanouts"])
    sampler = TargetMaskedNeighborSampler(graph, metadata, fanouts=fanouts)
    store = FrozenEmbeddingStore(data)
    relation_input = torch.from_numpy(
        store.relation_rows(np.arange(data.identity.relation_count))
    ).to(device)
    row_hidden_chunks: list[np.ndarray] = []
    for batch in _batches(
        np.arange(len(context_record_indices)), config.rgcn_batch_size, rng=None
    ):
        row_hidden_chunks.append(_encode_context_batch(
            torch,
            model,
            sampler,
            store,
            relation_input,
            records,
            context_record_indices[batch],
            seed=int(detector_config["seed"]),
            device=device,
        ))
    row_hidden = np.concatenate(row_hidden_chunks, axis=0).astype(np.float32, copy=False)
    with torch.inference_mode():
        relation_hidden = model.relation_projection(relation_input).float().cpu().numpy()
        value_hidden_chunks: list[np.ndarray] = []
        for start in range(0, len(value_input), config.qwen_batch_size * 4):
            value_hidden_chunks.append(
                model.node_projection(
                    torch.from_numpy(value_input[start:start + config.qwen_batch_size * 4]).to(device)
                ).float().cpu().numpy()
            )
        value_hidden = np.concatenate(value_hidden_chunks, axis=0).astype(np.float32, copy=False)

    context_relations = records.relation_id[context_record_indices].astype(np.int64)
    context_folds = records.fold[context_record_indices].astype(np.int8)
    np.save(output / "row_hidden.npy", row_hidden)
    np.save(output / "relation_hidden.npy", relation_hidden)
    np.save(output / "value_hidden.npy", value_hidden)
    np.savez_compressed(
        output / "ranking_examples.npz",
        context_observation=ordered_observations,
        context_relation=context_relations,
        context_fold=context_folds,
        pair_context=np.asarray(pair_context, dtype=np.int64),
        pair_positive=np.asarray(pair_positive, dtype=np.int64),
        pair_negative=np.asarray(pair_negative, dtype=np.int64),
        pair_weight=np.asarray(pair_weight, dtype=np.float32),
        validation_group_observation=np.asarray(validation_group_observation, dtype=np.int64),
        validation_group_context=np.asarray(validation_group_context, dtype=np.int64),
        validation_group_gold=np.asarray(validation_group_gold, dtype=np.int64),
        validation_group_gold_position=np.asarray(
            validation_group_gold_position, dtype=np.int8
        ),
        validation_group_label=np.asarray(validation_group_label, dtype=np.int8),
        validation_offsets=np.asarray(validation_offsets, dtype=np.int64),
        validation_candidate_ids=np.asarray(validation_candidate_ids, dtype=np.int64),
    )
    with (output / "values.jsonl").open("w", encoding="utf-8") as handle:
        for value_id, record in enumerate(value_records):
            handle.write(json.dumps({"value_id": value_id, **record}, ensure_ascii=True, sort_keys=True) + "\n")
    with (output / "validation_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for observation, candidates in zip(validation_group_observation, validation_candidate_raw):
            handle.write(json.dumps({
                "observation_index": observation,
                "candidates": candidates,
            }, ensure_ascii=True, sort_keys=True) + "\n")

    report = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "strict_rgcn_complex_candidate_ranking_prepare",
        "privacy": "HOST_PRIVATE_TRAIN_VALIDATION_ONLY",
        "internal_test_gold_accessed": False,
        "internal_test_candidates_accessed": False,
        "folds": {"train": list(TRAIN_FOLDS), "validation": [VALIDATION_FOLD]},
        "training_identity": data.identity.to_dict(),
        "detector": {
            "model_type": "strict_rgcn",
            "checkpoint_sha256": _sha256(checkpoint_path),
            "scoped_predictions_sha256": _sha256(predictions_path),
            "config": detector_config,
            "frozen": True,
        },
        "candidate_protocol": {
            "rule_registry_sha256": train_candidate_manifest["rule_registry_sha256"],
            "maximum_candidates": MAX_CANDIDATES,
            "selection_performed": False,
            "llm_called": False,
        },
        "qwen_backend": qwen_metadata,
        "counts": {
            "context_count": len(ordered_observations),
            "unique_value_count": len(value_texts),
            "train_pair_count": len(pair_context),
            "train_dirty_positive_count": train_positive_observations,
            "train_clean_detector_example_count": train_clean_detector_examples,
            "validation_detector_positive_count": len(validation_group_observation),
            "validation_dirty_total": validation_dirty_total,
            "validation_candidate_count": len(validation_candidate_ids),
        },
        "inputs": {
            "train_candidates_sha256": _sha256(train_path),
            "train_candidate_manifest_sha256": _sha256(train_manifest_path),
            "validation_candidates_sha256": _sha256(validation_path),
            "validation_candidate_manifest_sha256": _sha256(validation_manifest_path),
            "scoped_predictions_sha256": _sha256(predictions_path),
            "paired_log_sha256": _sha256(paired_log_path),
            "paired_log_manifest_sha256": _sha256(paired_log_manifest_path),
            "cell_observations_sha256": _sha256(observations_path),
        },
        "outputs": {},
    }
    for name in sorted(PREPARED_OUTPUTS):
        report["outputs"][name] = {"sha256": _sha256(output / name)}
    _atomic_json(output / "prepared_manifest.json", report)
    return report


def _complex_class(torch: Any) -> type:
    class ComplexCandidateRanker(torch.nn.Module):
        def __init__(self, input_dim: int, rank_dim: int, dropout: float) -> None:
            super().__init__()
            output_dim = rank_dim * 2
            self.row_projection = torch.nn.Linear(input_dim, output_dim)
            self.relation_projection = torch.nn.Linear(input_dim, output_dim)
            self.value_projection = torch.nn.Linear(input_dim, output_dim)
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, row: Any, relation: Any, value: Any) -> Any:
            row = self.dropout(self.row_projection(row))
            relation = self.dropout(self.relation_projection(relation))
            value = self.dropout(self.value_projection(value))
            row_re, row_im = row.chunk(2, dim=-1)
            rel_re, rel_im = relation.chunk(2, dim=-1)
            value_re, value_im = value.chunk(2, dim=-1)
            return (
                row_re * rel_re * value_re
                + row_re * rel_im * value_im
                + row_im * rel_re * value_im
                - row_im * rel_im * value_re
            ).sum(dim=-1) / math.sqrt(row_re.shape[-1])

    return ComplexCandidateRanker


def _validation_metrics(
    torch: Any,
    model: Any,
    row_hidden: Any,
    relation_hidden: Any,
    value_hidden: Any,
    examples: Any,
    validation_dirty_total: int,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    offsets = examples["validation_offsets"]
    observations = examples["validation_group_observation"]
    labels = examples["validation_group_label"]
    contexts = examples["validation_group_context"]
    gold_ids = examples["validation_group_gold"]
    gold_positions = examples["validation_group_gold_position"]
    candidate_ids = examples["validation_candidate_ids"]
    covered = 0
    top1 = 0
    top3 = 0
    reciprocal_rank_sum = 0.0
    clean_count = 0
    clean_kept = 0
    score_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for group_index, observation in enumerate(observations):
            start, end = int(offsets[group_index]), int(offsets[group_index + 1])
            ids = candidate_ids[start:end]
            gold_id = int(gold_ids[group_index])
            context = int(contexts[group_index])
            relation_id = int(examples["context_relation"][context])
            if len(ids):
                id_tensor = torch.from_numpy(ids.astype(np.int64)).to(device)
                scores = model(
                    row_hidden[context].unsqueeze(0).expand(len(ids), -1),
                    relation_hidden[relation_id].unsqueeze(0).expand(len(ids), -1),
                    value_hidden[id_tensor],
                )
                candidate_scores = [
                    {"value_id": int(ids[index]), "score": float(scores[index].cpu())}
                    for index in range(len(ids))
                ]
            else:
                scores = torch.empty(0, device=device)
                candidate_scores = []
            if int(labels[group_index]) == 0:
                clean_count += 1
                keep_score = model(
                    row_hidden[context].unsqueeze(0),
                    relation_hidden[relation_id].unsqueeze(0),
                    value_hidden[gold_id].unsqueeze(0),
                )[0]
                keep_selected = not len(scores) or bool((keep_score >= scores.max()).item())
                clean_kept += keep_selected
                score_rows.append({
                    "observation_index": int(observation),
                    "is_dirty": False,
                    "keep_original_selected": keep_selected,
                    "keep_original_score": float(keep_score.cpu()),
                    "scores": candidate_scores,
                })
                continue
            gold_position = int(gold_positions[group_index])
            if gold_position < 0:
                score_rows.append({
                    "observation_index": int(observation),
                    "is_dirty": True,
                    "gold_in_candidates": False,
                    "rank": 0,
                    "scores": candidate_scores,
                })
                continue
            covered += 1
            order = torch.argsort(scores, descending=True, stable=True).cpu().numpy()
            rank = int(np.flatnonzero(order == gold_position)[0]) + 1
            top1 += rank <= 1
            top3 += rank <= 3
            reciprocal_rank_sum += 1.0 / rank
            score_rows.append({
                "observation_index": int(observation),
                "is_dirty": True,
                "gold_in_candidates": True,
                "rank": rank,
                "scores": candidate_scores,
            })
    conditional_denominator = covered or 1
    total_denominator = validation_dirty_total or 1
    detected_dirty = int(np.sum(labels == 1))
    detected_denominator = detected_dirty or 1
    return {
        "validation_dirty_total": validation_dirty_total,
        "validation_detected_dirty": detected_dirty,
        "candidate_gold_coverage_count": covered,
        "candidate_recall_at_5": covered / detected_denominator,
        "rule_candidate_recall_at_5": covered / detected_denominator,
        "joint_candidate_recall_at_5": covered / total_denominator,
        "conditional_hits_at_1": top1 / conditional_denominator,
        "conditional_hits_at_3": top3 / conditional_denominator,
        "conditional_mrr": reciprocal_rank_sum / conditional_denominator,
        "joint_recall_at_1": top1 / total_denominator,
        "joint_recall_at_3": top3 / total_denominator,
        "clean_detector_positive_count": clean_count,
        "keep_original_accuracy": clean_kept / clean_count if clean_count else None,
    }, score_rows


def _validate_prepared_arrays(
    manifest: dict[str, Any],
    row_array: np.ndarray,
    relation_array: np.ndarray,
    value_array: np.ndarray,
    examples: dict[str, np.ndarray],
) -> None:
    if set(examples) != EXAMPLE_ARRAYS:
        raise CandidateRankingError("prepared ranking example arrays have an invalid schema")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise CandidateRankingError("prepared manifest is missing counts")
    context_count = len(examples["context_observation"])
    pair_count = len(examples["pair_context"])
    group_count = len(examples["validation_group_observation"])
    candidate_count = len(examples["validation_candidate_ids"])
    if (
        row_array.ndim != 2
        or relation_array.ndim != 2
        or value_array.ndim != 2
        or len({row_array.shape[1], relation_array.shape[1], value_array.shape[1]}) != 1
        or row_array.shape[0] != context_count
        or int(counts.get("context_count", -1)) != context_count
        or int(counts.get("train_pair_count", -1)) != pair_count
        or int(counts.get("validation_detector_positive_count", -1)) != group_count
        or int(counts.get("validation_candidate_count", -1)) != candidate_count
        or int(counts.get("unique_value_count", -1)) != value_array.shape[0]
    ):
        raise CandidateRankingError("prepared array shapes disagree with the manifest")
    if len(np.unique(examples["context_observation"])) != context_count:
        raise CandidateRankingError("prepared context observations must be unique")
    if (
        examples["context_relation"].shape != (context_count,)
        or examples["context_fold"].shape != (context_count,)
        or not set(np.unique(examples["context_fold"]).tolist()).issubset({0, 1, 2, 3})
        or examples["context_relation"].min(initial=0) < 0
        or examples["context_relation"].max(initial=-1) >= relation_array.shape[0]
    ):
        raise CandidateRankingError("prepared context arrays violate fold/relation bounds")
    for name in ("pair_positive", "pair_negative", "pair_weight"):
        if examples[name].shape != (pair_count,):
            raise CandidateRankingError("prepared pair arrays have inconsistent lengths")
    if (
        examples["pair_context"].min(initial=0) < 0
        or examples["pair_context"].max(initial=-1) >= context_count
        or not set(np.unique(examples["context_fold"][examples["pair_context"]]).tolist()).issubset(
            set(TRAIN_FOLDS)
        )
        or examples["pair_positive"].min(initial=0) < 0
        or examples["pair_positive"].max(initial=-1) >= value_array.shape[0]
        or examples["pair_negative"].min(initial=0) < 0
        or examples["pair_negative"].max(initial=-1) >= value_array.shape[0]
        or np.any(examples["pair_positive"] == examples["pair_negative"])
        or np.any(~np.isfinite(examples["pair_weight"]))
        or np.any(examples["pair_weight"] <= 0)
    ):
        raise CandidateRankingError("prepared Train pair arrays violate index/value bounds")
    for name in (
        "validation_group_context", "validation_group_gold",
        "validation_group_gold_position", "validation_group_label",
    ):
        if examples[name].shape != (group_count,):
            raise CandidateRankingError("prepared Validation group arrays have inconsistent lengths")
    offsets = examples["validation_offsets"]
    if (
        offsets.shape != (group_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != candidate_count
        or np.any(offsets[1:] < offsets[:-1])
        or examples["validation_group_context"].min(initial=0) < 0
        or examples["validation_group_context"].max(initial=-1) >= context_count
        or np.any(
            examples["context_fold"][examples["validation_group_context"]]
            != VALIDATION_FOLD
        )
        or examples["validation_group_gold"].min(initial=0) < 0
        or examples["validation_group_gold"].max(initial=-1) >= value_array.shape[0]
        or not set(np.unique(examples["validation_group_label"]).tolist()).issubset({0, 1})
        or examples["validation_candidate_ids"].min(initial=0) < 0
        or examples["validation_candidate_ids"].max(initial=-1) >= value_array.shape[0]
    ):
        raise CandidateRankingError("prepared Validation arrays violate index/fold bounds")
    validation_context = examples["validation_group_context"]
    if (
        len(np.unique(examples["validation_group_observation"])) != group_count
        or len(np.unique(validation_context)) != group_count
        or np.any(
            examples["validation_group_observation"]
            != examples["context_observation"][validation_context]
        )
    ):
        raise CandidateRankingError("prepared Validation observation/context mapping is invalid")
    for group_index, position in enumerate(examples["validation_group_gold_position"]):
        group_size = int(offsets[group_index + 1] - offsets[group_index])
        if int(position) < -1 or int(position) >= group_size:
            raise CandidateRankingError("prepared exact Gold position is out of range")
        if int(position) >= 0:
            candidate_position = int(offsets[group_index]) + int(position)
            if (
                int(examples["validation_candidate_ids"][candidate_position])
                != int(examples["validation_group_gold"][group_index])
            ):
                raise CandidateRankingError(
                    "prepared exact Gold position disagrees with candidate embedding ID"
                )


def train_complex_ranker(config: RankTrainingConfig) -> dict[str, Any]:
    config.validate()
    prepared = Path(config.prepared_dir).expanduser().resolve()
    manifest_path = _required_file(prepared / "prepared_manifest.json", "prepared manifest")
    manifest = _read_json(manifest_path, "prepared manifest")
    if (
        manifest.get("workflow") != "strict_rgcn_complex_candidate_ranking_prepare"
        or manifest.get("internal_test_gold_accessed") is not False
        or manifest.get("internal_test_candidates_accessed") is not False
        or manifest.get("folds") != {"train": [0, 1, 2], "validation": [3]}
    ):
        raise CandidateRankingError("prepared artifacts do not satisfy the Train/Validation-only contract")
    manifest_outputs = manifest.get("outputs")
    if not isinstance(manifest_outputs, dict) or set(manifest_outputs) != PREPARED_OUTPUTS:
        raise CandidateRankingError("prepared manifest does not declare the fixed artifact set")
    for name, metadata in manifest_outputs.items():
        path = _required_file(prepared / name, f"prepared artifact {name}")
        if not isinstance(metadata, dict) or metadata.get("sha256") != _sha256(path):
            raise CandidateRankingError(f"prepared artifact hash mismatch: {name}")
    output = _new_output(config.output_dir)
    try:
        import torch
    except ImportError as exc:
        raise CandidateRankingError("ComplEx training requires torch") from exc
    device = _resolve_device(torch, config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    row_array = np.load(prepared / "row_hidden.npy", mmap_mode="r")
    relation_array = np.load(prepared / "relation_hidden.npy", mmap_mode="r")
    value_array = np.load(prepared / "value_hidden.npy", mmap_mode="r")
    with np.load(prepared / "ranking_examples.npz") as loaded:
        examples = {name: np.asarray(loaded[name]) for name in loaded.files}
    _validate_prepared_arrays(manifest, row_array, relation_array, value_array, examples)
    row_hidden = torch.from_numpy(np.asarray(row_array, dtype=np.float32)).to(device)
    relation_hidden = torch.from_numpy(np.asarray(relation_array, dtype=np.float32)).to(device)
    value_hidden = torch.from_numpy(np.asarray(value_array, dtype=np.float32)).to(device)
    ranker_class = _complex_class(torch)
    model = ranker_class(row_hidden.shape[1], config.rank_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pair_count = len(examples["pair_context"])
    if pair_count < 1:
        raise CandidateRankingError("prepared Train pairs are empty")
    train_indices = np.arange(pair_count, dtype=np.int64)
    history: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    best_epoch = -1
    stale_epochs = 0
    checkpoint_path = output / "best_checkpoint.pt"
    started = time.perf_counter()
    for epoch in range(config.epochs):
        if stale_epochs >= config.patience:
            break
        model.train()
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, epoch]))
        loss_sum = 0.0
        weight_sum = 0.0
        for batch in _batches(train_indices, config.batch_size, rng):
            context_np = examples["pair_context"][batch].astype(np.int64)
            relation_np = examples["context_relation"][context_np].astype(np.int64)
            positive_np = examples["pair_positive"][batch].astype(np.int64)
            negative_np = examples["pair_negative"][batch].astype(np.int64)
            weights = torch.from_numpy(examples["pair_weight"][batch]).to(device)
            context = torch.from_numpy(context_np).to(device)
            relation = torch.from_numpy(relation_np).to(device)
            positive = torch.from_numpy(positive_np).to(device)
            negative = torch.from_numpy(negative_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            positive_score = model(
                row_hidden[context], relation_hidden[relation], value_hidden[positive]
            )
            negative_score = model(
                row_hidden[context], relation_hidden[relation], value_hidden[negative]
            )
            losses = torch.nn.functional.softplus(-(positive_score - negative_score))
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float((losses * weights).sum().detach().cpu())
            weight_sum += float(weights.sum().detach().cpu())
        metrics, _ = _validation_metrics(
            torch, model, row_hidden, relation_hidden, value_hidden, examples,
            int(manifest["counts"]["validation_dirty_total"]), device,
        )
        record = {
            "epoch": epoch,
            "train_pairwise_loss": loss_sum / weight_sum,
            "validation": metrics,
        }
        history.append(record)
        key = (float(metrics["conditional_hits_at_1"]), float(metrics["conditional_mrr"]))
        if key > best_key:
            best_key = key
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(torch, checkpoint_path, {
                "model": model.state_dict(),
                "epoch": epoch,
                "config": asdict(config),
                "prepared_manifest_sha256": _sha256(manifest_path),
                "validation": metrics,
            })
        else:
            stale_epochs += 1
        _atomic_json(output / "history.json", history)
        _atomic_json(output / "progress.json", {
            "schema_version": RANKING_SCHEMA_VERSION,
            "status": "RUNNING",
            "epochs_completed": epoch + 1,
            "best_epoch": best_epoch,
            "best_conditional_hits_at_1": best_key[0],
            "stale_epochs": stale_epochs,
            "runtime_seconds": time.perf_counter() - started,
            "internal_test_accessed": False,
        })
        print(
            f"epoch={epoch} pairwise_loss={record['train_pairwise_loss']:.6f} "
            f"val_conditional_hits_at_1={metrics['conditional_hits_at_1']:.6f} "
            f"val_joint_recall_at_1={metrics['joint_recall_at_1']:.6f}",
            flush=True,
        )
    if not checkpoint_path.is_file():
        raise CandidateRankingError("training did not create a best checkpoint")
    best_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    metrics, score_rows = _validation_metrics(
        torch, model, row_hidden, relation_hidden, value_hidden, examples,
        int(manifest["counts"]["validation_dirty_total"]), device,
    )
    scores_path = output / "validation_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as handle:
        for row in score_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    report = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "status": "SUCCESS",
        "workflow": "strict_rgcn_frozen_complex_candidate_ranker",
        "evaluation_role": "validation_model_selection",
        "internal_test_evaluated": False,
        "selection_applied": False,
        "best_epoch": best_epoch,
        "config": asdict(config),
        "validation": metrics,
        "candidate_ceiling_note": (
            "Candidate Recall@5 is an upstream pool ceiling; ComplEx only ranks provided candidates."
        ),
        "inputs": {"prepared_manifest_sha256": _sha256(manifest_path)},
        "outputs": {
            "best_checkpoint_sha256": _sha256(checkpoint_path),
            "validation_scores_sha256": _sha256(scores_path),
        },
    }
    _atomic_json(output / "ranking_report.json", report)
    _atomic_json(output / "progress.json", {
        "schema_version": RANKING_SCHEMA_VERSION,
        "status": "SUCCESS",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_conditional_hits_at_1": metrics["conditional_hits_at_1"],
        "runtime_seconds": time.perf_counter() - started,
        "internal_test_accessed": False,
    })
    return report
