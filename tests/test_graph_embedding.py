from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from graph.embedding import (
    EmbeddingBatch,
    EmbeddingBuildError,
    EmbeddingOutOfMemoryError,
    build_embeddings,
    Qwen3EmbeddingBackend,
)


class FakeBackend:
    dimension = 1024

    def __init__(
        self,
        *,
        identity: str = "fake-v1",
        oom_above: int | None = None,
        fail_after: int | None = None,
        invalid: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self.identity = identity
        self.oom_above = oom_above
        self.fail_after = fail_after
        self.invalid = invalid
        self.max_length = max_length
        self.calls = 0
        self.seen: list[str] = []

    @property
    def metadata(self) -> dict:
        metadata = {
            "type": "fake",
            "model": self.identity,
            "resolved_revision": self.identity,
            "device": "cpu",
            "compute_dtype": "float32",
        }
        if self.max_length is not None:
            metadata["max_length"] = self.max_length
        return metadata

    def encode(self, texts: list[str]) -> EmbeddingBatch:
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise EmbeddingOutOfMemoryError("synthetic OOM")
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("synthetic interruption")
        self.calls += 1
        self.seen.extend(texts)
        values = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.resize(np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1.0, self.dimension)
            vector /= np.linalg.norm(vector)
            values.append(vector)
        array = np.stack(values)
        if self.invalid == "nan":
            array[0, 0] = np.nan
        elif self.invalid == "shape":
            array = array[:, :-1]
        return EmbeddingBatch(array, sum(len(text) for text in texts))


def _write_graph(root: Path, *, node_count: int = 5, relation_count: int = 3) -> Path:
    root.mkdir()
    nodes = [
        {"node_id": index, "node_type": "Row", "embedding_text": f"node text {index}"}
        for index in range(node_count)
    ]
    relations = [
        {"relation_id": index, "relation": f"relation_{index}", "embedding_text": f"relation text {index}"}
        for index in range(relation_count)
    ]
    (root / "nodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in nodes), encoding="utf-8"
    )
    (root / "relation_texts.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in relations), encoding="utf-8"
    )
    (root / "graph_manifest.json").write_text(json.dumps({
        "node_count": node_count,
        "relation_count": relation_count,
        "storage": {"nodes": "nodes.jsonl", "relation_texts": "relation_texts.jsonl"},
    }), encoding="utf-8")
    return root


def test_builds_ordered_float16_memmaps_and_complete_manifest(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph")
    output = tmp_path / "embeddings"
    backend = FakeBackend()

    manifest = build_embeddings(graph, output, backend, batch_size=2)

    nodes = np.load(output / "node_embeddings.f16.npy")
    relations = np.load(output / "relation_embeddings.f16.npy")
    assert manifest["status"] == "SUCCESS"
    assert nodes.shape == (5, 1024)
    assert relations.shape == (3, 1024)
    assert nodes.dtype == np.float16
    assert backend.seen == [
        "node text 0", "node text 1", "node text 2", "node text 3", "node text 4",
        "relation text 0", "relation text 1", "relation text 2",
    ]
    assert json.loads((output / "progress.json").read_text())["status"] == "SUCCESS"
    report = json.loads((output / "benchmark_report.json").read_text())
    assert report["nodes_per_second"] > 0
    assert "model_load_seconds" in report


def test_max_nodes_creates_small_partial_benchmark_not_full_sized_array(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=20)
    output = tmp_path / "partial"

    manifest = build_embeddings(graph, output, FakeBackend(), max_nodes=3)

    assert manifest["status"] == "PARTIAL_BENCHMARK"
    assert np.load(output / "node_embeddings.f16.npy").shape == (3, 1024)
    assert manifest["identity"]["graph_node_count"] == 20
    assert manifest["identity"]["requested_node_count"] == 3


def test_oom_halves_batch_size_and_retries_without_skipping(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=7)
    output = tmp_path / "embeddings"
    backend = FakeBackend(oom_above=2)

    build_embeddings(graph, output, backend, batch_size=5)

    progress = json.loads((output / "progress.json").read_text())
    assert progress["effective_batch_size"] == 2
    assert progress["oom_reductions"] == 1
    assert backend.seen[:7] == [f"node text {index}" for index in range(7)]


def test_resume_continues_after_last_flushed_batch(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=6)
    output = tmp_path / "embeddings"
    first = FakeBackend(fail_after=1)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        build_embeddings(graph, output, first, batch_size=2)
    assert json.loads((output / "progress.json").read_text())["nodes_completed"] == 2

    second = FakeBackend()
    manifest = build_embeddings(graph, output, second, batch_size=2, resume=True)

    assert manifest["status"] == "SUCCESS"
    assert second.seen[:4] == ["node text 2", "node text 3", "node text 4", "node text 5"]


def test_resume_rejects_changed_source_or_backend_identity(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=4)
    output = tmp_path / "embeddings"
    with pytest.raises(RuntimeError):
        build_embeddings(graph, output, FakeBackend(fail_after=1), batch_size=2)

    node_path = graph / "nodes.jsonl"
    node_path.write_text(node_path.read_text().replace("node text 3", "changed text"))
    with pytest.raises(EmbeddingBuildError, match="identity"):
        build_embeddings(graph, output, FakeBackend(), batch_size=2, resume=True)

    node_path.write_text(node_path.read_text().replace("changed text", "node text 3"))
    with pytest.raises(EmbeddingBuildError, match="identity"):
        build_embeddings(graph, output, FakeBackend(identity="fake-v2"), batch_size=2, resume=True)


def test_resume_rejects_corrupt_progress_and_array_contract(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=4)
    output = tmp_path / "embeddings"
    with pytest.raises(RuntimeError):
        build_embeddings(graph, output, FakeBackend(fail_after=1), batch_size=2)

    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress["nodes_completed"] = 99
    progress_path.write_text(json.dumps(progress))
    with pytest.raises(EmbeddingBuildError, match="nodes_completed"):
        build_embeddings(graph, output, FakeBackend(), batch_size=2, resume=True)

    progress["nodes_completed"] = 2
    progress_path.write_text(json.dumps(progress))
    (output / "node_embeddings.f16.npy").write_bytes(b"broken")
    with pytest.raises(EmbeddingBuildError, match="resume array"):
        build_embeddings(graph, output, FakeBackend(), batch_size=2, resume=True)


def test_resume_rejects_same_shape_committed_chunk_corruption(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=4)
    output = tmp_path / "embeddings"
    with pytest.raises(RuntimeError):
        build_embeddings(graph, output, FakeBackend(fail_after=1), batch_size=2)
    array = np.load(output / "node_embeddings.f16.npy", mmap_mode="r+")
    array[0, 0] = np.float16(0)
    array.flush()
    del array

    with pytest.raises(EmbeddingBuildError, match="checksum mismatch"):
        build_embeddings(graph, output, FakeBackend(), batch_size=2, resume=True)


def test_relation_phase_interruption_resumes_without_reencoding_nodes(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=2, relation_count=3)
    output = tmp_path / "embeddings"
    with pytest.raises(RuntimeError):
        build_embeddings(graph, output, FakeBackend(fail_after=1), batch_size=4)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["nodes_completed"] == 2
    assert progress["relations_completed"] == 0

    resumed = FakeBackend()
    build_embeddings(graph, output, resumed, batch_size=4, resume=True)
    assert resumed.seen == ["relation text 0", "relation text 1", "relation text 2"]


def test_existing_lock_rejects_concurrent_resume(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=4)
    output = tmp_path / "embeddings"
    with pytest.raises(RuntimeError):
        build_embeddings(graph, output, FakeBackend(fail_after=1), batch_size=2)
    (output / "embedding.lock").write_text('{"pid": 999, "hostname": "other"}\n')

    with pytest.raises(EmbeddingBuildError, match="locked"):
        build_embeddings(graph, output, FakeBackend(), batch_size=2, resume=True)


def test_backend_and_requested_max_length_must_match(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph")
    with pytest.raises(EmbeddingBuildError, match="max_length"):
        build_embeddings(
            graph,
            tmp_path / "output",
            FakeBackend(max_length=512),
            max_length=1024,
        )


def test_manifest_record_counts_must_match_jsonl(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=5, relation_count=3)
    manifest_path = graph / "graph_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["node_count"] = 4
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EmbeddingBuildError, match="nodes.jsonl count"):
        build_embeddings(graph, tmp_path / "short-output", FakeBackend())

    manifest["node_count"] = 5
    manifest["relation_count"] = 4
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EmbeddingBuildError, match="relation_texts.jsonl count"):
        build_embeddings(graph, tmp_path / "long-output", FakeBackend())


def test_oom_at_batch_size_one_fails_without_skipping(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph", node_count=2)
    output = tmp_path / "embeddings"
    with pytest.raises(EmbeddingBuildError, match="batch size 1"):
        build_embeddings(graph, output, FakeBackend(oom_above=0), batch_size=2)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["nodes_completed"] == 0
    assert progress["effective_batch_size"] == 1


def test_keyboard_interrupt_records_status_and_releases_lock(tmp_path: Path) -> None:
    class InterruptBackend(FakeBackend):
        def encode(self, texts: list[str]) -> EmbeddingBatch:
            raise KeyboardInterrupt

    graph = _write_graph(tmp_path / "graph")
    output = tmp_path / "embeddings"
    with pytest.raises(KeyboardInterrupt):
        build_embeddings(graph, output, InterruptBackend())
    assert json.loads((output / "progress.json").read_text())["status"] == "INTERRUPTED"
    assert not (output / "embedding.lock").exists()


@pytest.mark.skipif(
    not os.environ.get("QWEN3_EMBEDDING_MODEL"),
    reason="set QWEN3_EMBEDDING_MODEL to run the real Qwen3 smoke test",
)
def test_real_qwen_backend_is_normalized_and_repeatable() -> None:
    backend = Qwen3EmbeddingBackend(
        os.environ["QWEN3_EMBEDDING_MODEL"],
        device=os.environ.get("QWEN3_EMBEDDING_DEVICE", "auto"),
        max_length=1024,
    )
    texts = [
        '{"kind":"value","domain":"entity.stay_id","value":"30526645"}',
        '{"kind":"relation","table":"icu/chartevents","column":"valuenum"}',
    ]
    first = backend.encode(texts).values
    second = backend.encode(texts).values
    cosine = np.sum(first * second, axis=1) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    )
    assert first.shape == (2, 1024)
    assert np.isfinite(first).all()
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=2e-3)
    assert float(cosine.min()) >= 0.9999


@pytest.mark.parametrize("invalid", ["nan", "shape"])
def test_invalid_backend_output_is_rejected(tmp_path: Path, invalid: str) -> None:
    graph = _write_graph(tmp_path / "graph")
    with pytest.raises(EmbeddingBuildError):
        build_embeddings(graph, tmp_path / "output", FakeBackend(invalid=invalid))


def test_non_contiguous_ids_and_existing_output_are_rejected(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path / "graph")
    nodes_path = graph / "nodes.jsonl"
    nodes_path.write_text(nodes_path.read_text().replace('"node_id": 2', '"node_id": 7'))
    with pytest.raises(EmbeddingBuildError, match="non-contiguous"):
        build_embeddings(graph, tmp_path / "bad-output", FakeBackend())

    good_graph = _write_graph(tmp_path / "good-graph")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("do not replace")
    with pytest.raises(EmbeddingBuildError, match="new or empty"):
        build_embeddings(good_graph, output, FakeBackend())
