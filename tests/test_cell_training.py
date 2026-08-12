from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from graph.cell_metrics import binary_metrics, select_macro_f1_threshold
from graph.cell_models import CellRGCN, StrictMLP
from graph.cell_sampling import TargetMaskedNeighborSampler
from graph.cell_training import TrainingConfig, run_training
from graph.cell_training_aggregate import (
    CellTrainingAggregateError,
    aggregate_training_reports,
)
from graph.cell_training_data import (
    CellRecords,
    CellTrainingDataError,
    GraphCSR,
    NodeMetadata,
    prepare_cell_training_data,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, bad_embedding_identity: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    graph = tmp_path / "graph"
    supervision = tmp_path / "private"
    embeddings = tmp_path / "embeddings"
    graph.mkdir()
    supervision.mkdir()
    embeddings.mkdir()
    relation_ids = {"table.value": 0, "table.value__rev": 1}
    (graph / "relation_ids.json").write_text(json.dumps(relation_ids), encoding="utf-8")
    nodes = []
    observations = []
    edge_index = np.empty((2, 20), dtype=np.int64)
    edge_type = np.empty(20, dtype=np.int16)
    for index in range(10):
        row_id = index * 2
        value_id = row_id + 1
        nodes.extend((
            {"node_id": row_id, "node_type": "Row", "table": "table", "embedding_text": f"row {index}"},
            {"node_id": value_id, "node_type": "Value", "embedding_text": f"value {index}"},
        ))
        edge_index[:, 2 * index] = (row_id, value_id)
        edge_index[:, 2 * index + 1] = (value_id, row_id)
        edge_type[2 * index:2 * index + 2] = (0, 1)
        observations.append({
            "observation_index": index,
            "row_id": row_id,
            "value_node_id": value_id,
            "relation": "table.value",
            "forward_edge_id": 2 * index,
            "reverse_edge_id": 2 * index + 1,
        })
    nodes_path = graph / "nodes.jsonl"
    nodes_path.write_text("".join(json.dumps(value) + "\n" for value in nodes), encoding="utf-8")
    (graph / "cell_observations.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in observations), encoding="utf-8"
    )
    (graph / "relation_texts.jsonl").write_text(
        "".join(json.dumps({"relation_id": index, "embedding_text": name}) + "\n" for name, index in relation_ids.items()),
        encoding="utf-8",
    )
    np.save(graph / "edge_index.npy", edge_index)
    np.save(graph / "edge_type.npy", edge_type)
    graph_manifest = {
        "node_count": 20,
        "observation_count": 10,
        "triple_count": 10,
        "edge_count": 20,
        "relation_count": 2,
        "table_counts": {"table": 10},
        "storage": {
            "nodes": "nodes.jsonl",
            "cell_observations": "cell_observations.jsonl",
            "relation_texts": "relation_texts.jsonl",
            "edge_index": "edge_index.npy",
            "edge_type": "edge_type.npy",
        },
    }
    graph_manifest_path = graph / "graph_manifest.json"
    graph_manifest_path.write_text(json.dumps(graph_manifest), encoding="utf-8")
    labels = np.asarray([0, 1] * 5, dtype=np.int8)
    folds = np.repeat(np.arange(5, dtype=np.int8), 2)
    sources = np.where(labels == 0, 0, 2).astype(np.int8)
    np.savez_compressed(
        supervision / "supervision_masks.npz",
        cell_indices=np.arange(10, dtype=np.int64),
        cell_labels=labels,
        cell_folds=folds,
        cell_source=sources,
    )
    (supervision / "supervision_manifest.json").write_text(json.dumps({
        "status": "SUCCESS",
        "training_observation_count": 10,
        "dirty_training_count": 5,
        "clean_training_count": 5,
        "relation_ids_sha256": _sha256(graph / "relation_ids.json"),
    }), encoding="utf-8")
    rng = np.random.default_rng(7)
    node_values = rng.normal(size=(20, 8)).astype(np.float32)
    node_values /= np.linalg.norm(node_values, axis=1, keepdims=True)
    relation_values = rng.normal(size=(2, 8)).astype(np.float32)
    relation_values /= np.linalg.norm(relation_values, axis=1, keepdims=True)
    np.save(embeddings / "node_embeddings.f16.npy", node_values.astype(np.float16))
    np.save(embeddings / "relation_embeddings.f16.npy", relation_values.astype(np.float16))
    graph_hash = "wrong" if bad_embedding_identity else _sha256(graph_manifest_path)
    embedding_manifest = {
        "status": "SUCCESS",
        "embedding_dimension": 8,
        "completed": {"nodes": 20, "relations": 2},
        "identity": {
            "graph_manifest_sha256": graph_hash,
            "nodes_sha256": _sha256(nodes_path),
            "relations_sha256": _sha256(graph / "relation_texts.jsonl"),
            "graph_node_count": 20,
            "relation_count": 2,
            "identity_sha256": "fixture-v1",
        },
        "outputs": {
            "node_embeddings": {
                "path": "node_embeddings.f16.npy",
                "shape": [20, 8],
                "sha256": _sha256(embeddings / "node_embeddings.f16.npy"),
            },
            "relation_embeddings": {
                "path": "relation_embeddings.f16.npy",
                "shape": [2, 8],
                "sha256": _sha256(embeddings / "relation_embeddings.f16.npy"),
            },
        },
    }
    (embeddings / "embedding_manifest.json").write_text(
        json.dumps(embedding_manifest), encoding="utf-8"
    )
    return graph, supervision, embeddings


def test_prepares_supervised_cell_records_and_reuses_private_cache(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    prepared = prepare_cell_training_data(graph, supervision, embeddings)

    assert prepared.records.observation_index.tolist() == list(range(10))
    assert prepared.records.row_id.tolist() == list(range(0, 20, 2))
    assert prepared.records.value_node_id.tolist() == list(range(1, 20, 2))
    assert prepared.records.relation_id.tolist() == [0] * 10
    assert prepared.records.forward_edge_id.tolist() == list(range(0, 20, 2))
    assert prepared.records.reverse_edge_id.tolist() == list(range(1, 20, 2))
    assert prepared.records.indices_for_folds((0, 1, 2)).tolist() == list(range(6))
    assert (supervision / "cell_training_records.npz").is_file()
    assert len(prepare_cell_training_data(graph, supervision, embeddings).records) == 10


def test_rejects_embedding_from_another_graph(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path, bad_embedding_identity=True)
    with pytest.raises(CellTrainingDataError, match="identity"):
        prepare_cell_training_data(graph, supervision, embeddings)


def test_rejects_non_success_manifest_and_embedding_hash_mismatch(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    manifest_path = supervision / "supervision_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "FAILED"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CellTrainingDataError, match="status"):
        prepare_cell_training_data(graph, supervision, embeddings)

    graph, supervision, embeddings = _fixture(tmp_path / "hash")
    values = np.load(embeddings / "node_embeddings.f16.npy")
    values[0, 0] += np.float16(0.5)
    np.save(embeddings / "node_embeddings.f16.npy", values)
    with pytest.raises(CellTrainingDataError, match="SHA-256"):
        prepare_cell_training_data(graph, supervision, embeddings)


def test_rejects_out_of_range_supervision_and_wrong_edge_mapping(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    masks_path = supervision / "supervision_masks.npz"
    with np.load(masks_path) as arrays:
        values = {name: np.asarray(arrays[name]) for name in arrays.files}
    values["cell_indices"] = values["cell_indices"].copy()
    values["cell_indices"][-1] = 10
    np.savez_compressed(masks_path, **values)
    with pytest.raises(CellTrainingDataError, match="outside graph observation range"):
        prepare_cell_training_data(graph, supervision, embeddings)

    graph, supervision, embeddings = _fixture(tmp_path / "edge")
    observations_path = graph / "cell_observations.jsonl"
    observations = [json.loads(line) for line in observations_path.read_text().splitlines()]
    observations[0]["forward_edge_id"] = 2
    observations_path.write_text(
        "".join(json.dumps(value) + "\n" for value in observations), encoding="utf-8"
    )
    with pytest.raises(CellTrainingDataError, match="edge mapping"):
        prepare_cell_training_data(graph, supervision, embeddings)


def test_sampler_masks_only_exact_target_edge_pair_and_is_batch_invariant() -> None:
    # e0/e1 are targets; e2/e3 are exact parallel edges and must remain.
    sources = np.asarray([0, 1, 0, 1, 0, 2, 3, 1], dtype=np.int64)
    targets = np.asarray([1, 0, 1, 0, 2, 0, 1, 3], dtype=np.int64)
    relations = np.asarray([0, 1, 0, 1, 2, 3, 0, 1], dtype=np.int16)
    order = np.argsort(targets, kind="stable")
    offsets = np.zeros(5, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(targets, minlength=4))
    edge_position = np.empty(len(order), dtype=np.uint32)
    edge_position[order] = np.arange(len(order), dtype=np.uint32)
    graph = GraphCSR(
        offsets,
        sources[order].astype(np.int32),
        relations[order],
        order.astype(np.uint32),
        edge_position,
    )
    metadata = NodeMetadata(
        node_type=np.asarray([0, 1, 1, 0], dtype=np.uint8),
        table_id=np.asarray([0, -1, -1, 0], dtype=np.int16),
        tables=("table",),
    )
    records = CellRecords(
        observation_index=np.asarray([100, 101]),
        row_id=np.asarray([0, 0]),
        relation_id=np.asarray([0, 2], dtype=np.int32),
        value_node_id=np.asarray([1, 2]),
        forward_edge_id=np.asarray([0, 4]),
        reverse_edge_id=np.asarray([1, 5]),
        label=np.asarray([1, 0], dtype=np.int8),
        fold=np.asarray([0, 0], dtype=np.int8),
        source=np.asarray([2, 0], dtype=np.int8),
    )
    sampler = TargetMaskedNeighborSampler(graph, metadata, fanouts=(8,))
    alone = sampler.sample(records, np.asarray([0]), seed=7, epoch=0)
    together = sampler.sample(records, np.asarray([0, 1]), seed=7, epoch=0)

    assert 0 not in alone.edge_id and 1 not in alone.edge_id
    assert {2, 3}.issubset(set(alone.edge_id.tolist()))
    first_component = together.edge_id[together.edge_sample == 0]
    assert sorted(first_component.tolist()) == sorted(alone.edge_id.tolist())
    second_component = together.edge_id[together.edge_sample == 1]
    assert 4 not in second_component and 5 not in second_component
    assert 0 in second_component or 1 in second_component


def test_metrics_select_threshold_only_from_given_labels() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    threshold = select_macro_f1_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    assert threshold == pytest.approx(0.6)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_average_precision_is_invariant_to_order_with_tied_scores() -> None:
    first = binary_metrics(
        np.asarray([1, 0, 1, 0]), np.asarray([0.8, 0.8, 0.2, 0.2]), 0.5
    )
    second = binary_metrics(
        np.asarray([0, 1, 0, 1]), np.asarray([0.8, 0.8, 0.2, 0.2]), 0.5
    )
    assert first["average_precision"] == pytest.approx(second["average_precision"])
    assert first["average_precision"] == pytest.approx(0.5)


def test_rgcn_cpu_forward_backward_is_finite() -> None:
    model = CellRGCN(
        input_dim=8, hidden_dim=8, relation_count=4, table_count=1,
        strict_rows=True, layer_count=2, basis_count=2, dropout=0.0,
    )
    node_embedding = torch.randn(4, 8)
    relation_embedding = torch.randn(4, 8)
    logits = model(
        node_embedding, relation_embedding,
        torch.tensor([0, 1, 1, 0]), torch.tensor([0, -1, -1, 0]),
        torch.tensor([1, 2, 3]), torch.tensor([0, 0, 1]), torch.tensor([1, 3, 0]),
        torch.tensor([1, 0, 0]),
        torch.tensor([0]), torch.tensor([1]), torch.tensor([0]),
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_strict_mlp_forward_backward_uses_table_relation_and_value() -> None:
    model = StrictMLP(
        input_dim=8, hidden_dim=8, table_count=2, layer_count=2, dropout=0.0
    )
    logits = model(
        torch.tensor([0, 1]),
        torch.randn(2, 8),
        torch.randn(2, 8),
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0])
    )
    loss.backward()
    assert logits.shape == (2,)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rgcn_cuda_forward_backward_is_finite() -> None:
    model = CellRGCN(
        input_dim=8, hidden_dim=8, relation_count=4, table_count=1,
        strict_rows=False, layer_count=2, basis_count=2, dropout=0.0,
    ).cuda()
    logits = model(
        torch.randn(4, 8, device="cuda"), torch.randn(4, 8, device="cuda"),
        torch.tensor([0, 1, 1, 0], device="cuda"),
        torch.tensor([0, -1, -1, 0], device="cuda"),
        torch.tensor([1, 2, 3], device="cuda"),
        torch.tensor([0, 0, 1], device="cuda"),
        torch.tensor([1, 3, 0], device="cuda"),
        torch.tensor([1, 0, 0], device="cuda"),
        torch.tensor([0], device="cuda"),
        torch.tensor([1], device="cuda"),
        torch.tensor([0], device="cuda"),
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.ones(1, device="cuda")
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_triple_mlp_training_smoke_writes_frozen_test_report(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    output = tmp_path / "run"
    report = run_training(TrainingConfig(
        graph_dir=str(graph), supervision_dir=str(supervision), embedding_dir=str(embeddings),
        output_dir=str(output), model_type="triple_mlp", seed=666, device="cpu",
        epochs=5, batch_size=4, hidden_dim=8, learning_rate=1e-2,
        dropout=0.0, patience=5, evaluate_internal_test=True,
    ))
    assert report["status"] == "SUCCESS"
    assert report["split_counts"] == {"train": 6, "validation": 2, "internal_test": 2}
    assert report["selection_metric"] == "validation_average_precision"
    assert report["threshold_selection"] == "maximum_validation_macro_f1_after_epoch_selection"
    assert report["internal_test_evaluated"] is True
    assert set(report["folds"]) == {"0", "1", "2", "3", "4"}
    assert report["splits"]["internal_test"]["metrics_at_0_5"]["threshold"] == 0.5
    assert (output / "best_checkpoint.pt").is_file()
    assert (output / "predictions.csv").is_file()
    assert json.loads((output / "progress.json").read_text())["status"] == "SUCCESS"
    history = json.loads((output / "history.json").read_text())
    assert min(item["train_loss"] for item in history[1:]) < history[0]["train_loss"]
    header = (output / "predictions.csv").read_text().splitlines()[0]
    assert "row_id" in header and "dirty_score" in header and "model_type" in header
    assert json.loads((output / "report.json").read_text())["status"] == "SUCCESS"
    assert len(report["outputs"]["best_checkpoint"]["sha256"]) == 64
    assert report["outputs"]["predictions"]["row_count"] == 10


def test_internal_test_is_withheld_by_default(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    report = run_training(TrainingConfig(
        graph_dir=str(graph), supervision_dir=str(supervision), embedding_dir=str(embeddings),
        output_dir=str(tmp_path / "run"), model_type="triple_mlp", seed=666,
        device="cpu", epochs=1, batch_size=4, hidden_dim=8, patience=1,
    ))
    assert report["internal_test_evaluated"] is False
    assert "internal_test" not in report["splits"]
    assert set(report["folds"]) == {"0", "1", "2", "3"}


def test_strict_mlp_training_is_a_no_graph_control(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    report = run_training(TrainingConfig(
        graph_dir=str(graph), supervision_dir=str(supervision), embedding_dir=str(embeddings),
        output_dir=str(tmp_path / "run"), model_type="strict_mlp", seed=666,
        device="cpu", epochs=1, batch_size=4, hidden_dim=8, patience=1,
    ))
    assert report["status"] == "SUCCESS"
    assert report["sampling"]["graph_message_passing"] is False
    assert report["sampling"]["target_edge_masking"].startswith("not_applicable")
    assert not (supervision / "cell_graph_cache").exists()


def test_completed_checkpoint_can_be_resumed_with_same_identity(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    config = TrainingConfig(
        graph_dir=str(graph), supervision_dir=str(supervision), embedding_dir=str(embeddings),
        output_dir=str(tmp_path / "run"), model_type="triple_mlp", seed=666,
        device="cpu", epochs=1, batch_size=4, hidden_dim=8, patience=1,
    )
    first = run_training(config)
    resumed = run_training(TrainingConfig(**{**config.__dict__, "resume": True}))
    assert resumed["best_epoch"] == first["best_epoch"]
    assert resumed["validation_threshold"] == pytest.approx(first["validation_threshold"])


def test_early_stopped_checkpoint_resume_is_idempotent(tmp_path: Path) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    config = TrainingConfig(
        graph_dir=str(graph), supervision_dir=str(supervision), embedding_dir=str(embeddings),
        output_dir=str(tmp_path / "run"), model_type="triple_mlp", seed=666,
        device="cpu", epochs=5, batch_size=4, hidden_dim=8, patience=1,
        learning_rate=1e-12, dropout=0.0,
    )
    first = run_training(config)
    first_history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert len(first_history) < config.epochs
    resumed = run_training(TrainingConfig(**{**config.__dict__, "resume": True}))
    resumed_history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert resumed_history == first_history
    assert resumed["best_epoch"] == first["best_epoch"]


@pytest.mark.parametrize("model_type", ("fullrow_rgcn", "strict_rgcn"))
def test_rgcn_training_smoke_uses_target_masked_sampler(
    tmp_path: Path, model_type: str
) -> None:
    graph, supervision, embeddings = _fixture(tmp_path)
    report = run_training(TrainingConfig(
        graph_dir=str(graph), supervision_dir=str(supervision), embedding_dir=str(embeddings),
        output_dir=str(tmp_path / "run"), model_type=model_type, seed=666,
        device="cpu", epochs=1, batch_size=2, hidden_dim=8, patience=1,
        rgcn_layers=2, rgcn_bases=2, fanouts=(4, 2),
    ))
    assert report["status"] == "SUCCESS"
    assert report["sampling"]["graph_message_passing"] is True
    assert report["sampling"]["target_edge_masking"].startswith("exact_forward")
    assert (supervision / "cell_graph_cache" / "manifest.json").is_file()


def test_aggregate_requires_formal_seeds_and_reports_sample_standard_deviation(
    tmp_path: Path,
) -> None:
    identity = {"embedding_identity_sha256": "fixture"}
    paths = []
    for offset, seed in enumerate((666, 667, 668)):
        report = {
            "status": "SUCCESS",
            "model_type": "triple_mlp",
            "seed": seed,
            "training_identity": identity,
            "experiment_config": {"hidden_dim": 8, "model_type": "triple_mlp"},
            "internal_test_evaluated": True,
            "validation_threshold": 0.4 + offset * 0.1,
            "best_epoch": offset + 1,
            "splits": {},
        }
        for split in ("validation", "internal_test"):
            report["splits"][split] = {"metrics": {
                "average_precision": 0.6 + offset * 0.1,
                "auroc": 0.7,
                "macro_f1": 0.5,
                "dirty_precision": 0.5,
                "dirty_recall": 0.5,
                "dirty_f1": 0.5,
                "confusion_matrix": {"tn": 1, "fp": 2, "fn": 3, "tp": 4},
            }}
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    aggregate = aggregate_training_reports(paths, tmp_path / "aggregate.json")
    assert aggregate["seeds"] == [666, 667, 668]
    assert aggregate["splits"]["internal_test"]["average_precision"]["mean"] == pytest.approx(0.7)
    assert aggregate["splits"]["internal_test"]["average_precision"]["standard_deviation"] == pytest.approx(0.1)
    assert aggregate["splits"]["internal_test"]["confusion_matrix_total"]["tp"] == 12
    with pytest.raises(CellTrainingAggregateError, match="formal aggregate"):
        aggregate_training_reports(paths[:2])

    changed = json.loads(paths[-1].read_text())
    changed["experiment_config"]["hidden_dim"] = 16
    paths[-1].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(CellTrainingAggregateError, match="experiment configuration"):
        aggregate_training_reports(paths)
