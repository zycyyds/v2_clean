from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import numpy as np
from numpy.lib.format import open_memmap

try:
    import resource
except ImportError:  # Windows
    resource = None


EMBEDDING_SCHEMA_VERSION = 1
NODE_EMBEDDINGS_FILE = "node_embeddings.f16.npy"
RELATION_EMBEDDINGS_FILE = "relation_embeddings.f16.npy"
MANIFEST_FILE = "embedding_manifest.json"
PROGRESS_FILE = "progress.json"
BENCHMARK_FILE = "benchmark_report.json"
CHUNKS_FILE = "chunks.jsonl"
LOCK_FILE = "embedding.lock"
BACKEND_IMPLEMENTATION_VERSION = 3
_WINDOWS_REPLACE_RETRYABLE_ERRORS = {5, 32}
_ATOMIC_REPLACE_RETRY_DELAYS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0)


class EmbeddingBuildError(ValueError):
    pass


class EmbeddingOutOfMemoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingBatch:
    values: np.ndarray
    token_count: int = 0


class EmbeddingBackend(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    def encode(self, texts: Sequence[str]) -> EmbeddingBatch: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_sha256_and_count(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                count += 1
    return digest.hexdigest(), count


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _directory_artifact_identity(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(_file_sha256(path).encode("ascii"))
        count += 1
    if count == 0:
        raise EmbeddingBuildError(f"model snapshot contains no artifacts: {root}")
    return digest.hexdigest(), count


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for delay in (*_ATOMIC_REPLACE_RETRY_DELAYS, None):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in _WINDOWS_REPLACE_RETRYABLE_ERRORS or delay is None:
                raise
            time.sleep(delay)


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


class _OutputLock:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / LOCK_FILE
        self.acquired = False

    def __enter__(self) -> "_OutputLock":
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_unix": time.time(),
        }
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            detail = ""
            try:
                detail = f": {self.path.read_text(encoding='utf-8').strip()}"
            except OSError:
                pass
            raise EmbeddingBuildError(f"embedding output is locked by another process{detail}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingBuildError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EmbeddingBuildError(f"{label} must contain a JSON object: {path}")
    return value


def _storage_dtype(name: str) -> np.dtype[Any]:
    if name != "float16":
        raise EmbeddingBuildError(f"unsupported storage dtype: {name}")
    return np.dtype(np.float16)


def _source_paths(graph_dir: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    storage = manifest.get("storage")
    if not isinstance(storage, dict):
        raise EmbeddingBuildError("graph manifest is missing storage metadata")
    nodes = graph_dir / str(storage.get("nodes", "nodes.jsonl"))
    relations = graph_dir / str(storage.get("relation_texts", "relation_texts.jsonl"))
    if not nodes.is_file() or not relations.is_file():
        raise EmbeddingBuildError("graph is missing nodes.jsonl or relation_texts.jsonl")
    return nodes, relations


def _records(
    path: Path,
    *,
    id_field: str,
    expected_count: int,
    start: int,
) -> Iterator[tuple[int, str]]:
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if seen >= expected_count:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmbeddingBuildError(f"invalid JSONL record at {path}:{line_number}") from exc
            record_id = record.get(id_field)
            if record_id != seen:
                raise EmbeddingBuildError(
                    f"non-contiguous {id_field} at {path}:{line_number}: expected {seen}, got {record_id}"
                )
            text = record.get("embedding_text")
            if not isinstance(text, str) or not text:
                raise EmbeddingBuildError(f"missing embedding_text at {path}:{line_number}")
            if seen >= start:
                yield seen, text
            seen += 1
    if seen != expected_count:
        raise EmbeddingBuildError(f"record count mismatch for {path}: expected {expected_count}, got {seen}")


def _validate_encoded(batch: EmbeddingBatch, expected_rows: int, dimension: int) -> np.ndarray:
    values = np.asarray(batch.values)
    if values.shape != (expected_rows, dimension):
        raise EmbeddingBuildError(
            f"backend returned shape {values.shape}, expected {(expected_rows, dimension)}"
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise EmbeddingBuildError(f"backend returned non-floating dtype: {values.dtype}")
    if not np.isfinite(values).all():
        raise EmbeddingBuildError("backend returned NaN or infinite embeddings")
    norms = np.linalg.norm(values.astype(np.float32, copy=False), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3, rtol=2e-3):
        raise EmbeddingBuildError("backend embeddings are not L2-normalized")
    return values


def _identity(
    *,
    graph_manifest_sha256: str,
    nodes_sha256: str,
    relations_sha256: str,
    graph_node_count: int,
    requested_node_count: int,
    relation_count: int,
    backend: EmbeddingBackend,
    max_length: int,
    storage_dtype: str,
) -> dict[str, Any]:
    value = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "graph_manifest_sha256": graph_manifest_sha256,
        "nodes_sha256": nodes_sha256,
        "relations_sha256": relations_sha256,
        "graph_node_count": graph_node_count,
        "requested_node_count": requested_node_count,
        "relation_count": relation_count,
        "embedding_dimension": backend.dimension,
        "backend": {
            key: value
            for key, value in backend.metadata.items()
            if key not in {"model_load_seconds"}
        },
        "max_length": max_length,
        "pooling": "last_token",
        "normalization": "l2_fp32",
        "storage_dtype": storage_dtype,
    }
    return {**value, "identity_sha256": _json_sha256(value)}


def _initial_progress(identity_sha256: str, batch_size: int) -> dict[str, Any]:
    return {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "status": "RUNNING",
        "nodes_completed": 0,
        "relations_completed": 0,
        "effective_batch_size": batch_size,
        "nodes_batches_completed": 0,
        "relations_batches_completed": 0,
        "nodes_tokens": 0,
        "relations_tokens": 0,
        "nodes_encoding_seconds": 0.0,
        "relations_encoding_seconds": 0.0,
        "oom_reductions": 0,
    }


def _validate_progress(
    progress: dict[str, Any],
    *,
    identity_sha256: str,
    requested_node_count: int,
    relation_count: int,
) -> None:
    if progress.get("identity_sha256") != identity_sha256:
        raise EmbeddingBuildError("resume identity does not match the current graph, model, or parameters")
    integer_bounds = {
        "nodes_completed": requested_node_count,
        "relations_completed": relation_count,
    }
    for field, upper in integer_bounds.items():
        value = progress.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= upper:
            raise EmbeddingBuildError(f"invalid resume progress field: {field}")
    batch_size = progress.get("effective_batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise EmbeddingBuildError("invalid resume progress field: effective_batch_size")


def _open_output_array(
    path: Path,
    *,
    shape: tuple[int, int],
    dtype: np.dtype[Any],
    resume: bool,
) -> np.memmap:
    if resume:
        try:
            array = np.load(path, mmap_mode="r+")
        except (OSError, ValueError) as exc:
            raise EmbeddingBuildError(f"invalid resume array: {path}") from exc
        if array.shape != shape or array.dtype != dtype:
            raise EmbeddingBuildError(
                f"resume array contract mismatch for {path}: shape={array.shape}, dtype={array.dtype}"
            )
        return array
    return open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _array_chunk_sha256(array: np.memmap, start: int, end: int) -> str:
    values = np.asarray(array[start:end])
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _append_chunk(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_committed_chunks(
    path: Path,
    *,
    arrays: dict[str, np.memmap],
    completed: dict[str, int],
) -> None:
    if not path.is_file():
        if any(completed.values()):
            raise EmbeddingBuildError("resume chunk log is missing")
        return
    kept: list[dict[str, Any]] = []
    cursors = {kind: 0 for kind in arrays}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmbeddingBuildError(f"invalid chunk log at {path}:{line_number}") from exc
            kind = record.get("kind")
            if kind not in arrays:
                raise EmbeddingBuildError(f"invalid chunk kind at {path}:{line_number}")
            start = record.get("start")
            end = record.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
                raise EmbeddingBuildError(f"invalid chunk range at {path}:{line_number}")
            if end > completed[kind]:
                continue
            if start != cursors[kind]:
                raise EmbeddingBuildError(f"non-contiguous committed chunk at {path}:{line_number}")
            actual = _array_chunk_sha256(arrays[kind], start, end)
            if record.get("sha256") != actual:
                raise EmbeddingBuildError(
                    f"committed embedding chunk checksum mismatch: {kind}[{start}:{end}]"
                )
            cursors[kind] = end
            kept.append(record)
    for kind, expected in completed.items():
        if cursors[kind] != expected:
            raise EmbeddingBuildError(
                f"chunk log does not cover completed {kind}: expected {expected}, got {cursors[kind]}"
            )
    with path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _encode_records(
    *,
    source: Path,
    id_field: str,
    count: int,
    start: int,
    destination: np.memmap,
    backend: EmbeddingBackend,
    progress: dict[str, Any],
    progress_path: Path,
    progress_prefix: str,
    chunk_path: Path,
) -> None:
    records = _records(
        source,
        id_field=id_field,
        expected_count=count,
        start=start,
    )
    pending: list[tuple[int, str]] = []
    exhausted = False
    while start < count:
        effective_batch_size = int(progress["effective_batch_size"])
        while len(pending) < effective_batch_size and not exhausted:
            try:
                pending.append(next(records))
            except StopIteration:
                exhausted = True
        if not pending:
            break
        current = pending[:effective_batch_size]
        texts = [text for _, text in current]
        began = time.perf_counter()
        try:
            encoded = backend.encode(texts)
        except EmbeddingOutOfMemoryError:
            if effective_batch_size == 1:
                raise EmbeddingBuildError("embedding backend ran out of memory at batch size 1") from None
            progress["effective_batch_size"] = max(1, effective_batch_size // 2)
            progress["oom_reductions"] = int(progress["oom_reductions"]) + 1
            _atomic_json(progress_path, progress)
            continue
        elapsed = time.perf_counter() - began
        values = _validate_encoded(encoded, len(current), backend.dimension)
        first = current[0][0]
        last = current[-1][0] + 1
        destination[first:last] = values.astype(destination.dtype, copy=False)
        destination.flush()
        _append_chunk(chunk_path, {
            "kind": progress_prefix,
            "start": first,
            "end": last,
            "sha256": _array_chunk_sha256(destination, first, last),
        })
        del pending[:len(current)]
        start = last
        progress[f"{progress_prefix}_completed"] = start
        progress[f"{progress_prefix}_batches_completed"] = int(
            progress[f"{progress_prefix}_batches_completed"]
        ) + 1
        progress[f"{progress_prefix}_tokens"] = int(progress[f"{progress_prefix}_tokens"]) + int(
            encoded.token_count
        )
        progress[f"{progress_prefix}_encoding_seconds"] = float(
            progress[f"{progress_prefix}_encoding_seconds"]
        ) + elapsed
        _atomic_json(progress_path, progress)


def _build_embeddings_locked(
    graph_dir: str | Path,
    output_dir: str | Path,
    backend: EmbeddingBackend,
    *,
    batch_size: int = 16,
    max_length: int = 1024,
    storage_dtype: str = "float16",
    max_nodes: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    graph_dir = Path(graph_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if batch_size <= 0:
        raise EmbeddingBuildError("batch_size must be positive")
    if max_length <= 0:
        raise EmbeddingBuildError("max_length must be positive")
    backend_max_length = backend.metadata.get("max_length")
    if backend_max_length is not None and backend_max_length != max_length:
        raise EmbeddingBuildError(
            f"backend max_length {backend_max_length} does not match requested max_length {max_length}"
        )
    if backend.dimension != 1024:
        raise EmbeddingBuildError(f"Qwen3 embedding dimension must be 1024, got {backend.dimension}")
    dtype = _storage_dtype(storage_dtype)

    manifest_path = graph_dir / "graph_manifest.json"
    if not manifest_path.is_file():
        raise EmbeddingBuildError(f"missing graph manifest: {manifest_path}")
    graph_manifest = _load_json_object(manifest_path, "graph manifest")
    try:
        graph_node_count = int(graph_manifest["node_count"])
        relation_count = int(graph_manifest["relation_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingBuildError("graph manifest has invalid node_count or relation_count") from exc
    if graph_node_count < 0 or relation_count < 0:
        raise EmbeddingBuildError("graph counts must be non-negative")
    if max_nodes is not None and (max_nodes <= 0 or max_nodes > graph_node_count):
        raise EmbeddingBuildError("max_nodes must be between 1 and graph node_count")
    requested_node_count = graph_node_count if max_nodes is None else max_nodes
    nodes_path, relations_path = _source_paths(graph_dir, graph_manifest)

    source_hash_started = time.perf_counter()
    graph_manifest_sha256 = _file_sha256(manifest_path)
    nodes_sha256, actual_node_count = _jsonl_sha256_and_count(nodes_path)
    relations_sha256, actual_relation_count = _jsonl_sha256_and_count(relations_path)
    if actual_node_count != graph_node_count:
        raise EmbeddingBuildError(
            f"nodes.jsonl count does not match graph manifest: {actual_node_count} != {graph_node_count}"
        )
    if actual_relation_count != relation_count:
        raise EmbeddingBuildError(
            f"relation_texts.jsonl count does not match graph manifest: "
            f"{actual_relation_count} != {relation_count}"
        )
    source_hash_seconds = time.perf_counter() - source_hash_started
    identity = _identity(
        graph_manifest_sha256=graph_manifest_sha256,
        nodes_sha256=nodes_sha256,
        relations_sha256=relations_sha256,
        graph_node_count=graph_node_count,
        requested_node_count=requested_node_count,
        relation_count=relation_count,
        backend=backend,
        max_length=max_length,
        storage_dtype=storage_dtype,
    )

    if resume:
        if not output_dir.is_dir():
            raise EmbeddingBuildError(f"resume output directory does not exist: {output_dir}")
        existing_manifest = _load_json_object(output_dir / MANIFEST_FILE, "embedding manifest")
        progress = _load_json_object(output_dir / PROGRESS_FILE, "embedding progress")
        if existing_manifest.get("identity") != identity:
            raise EmbeddingBuildError("resume manifest identity does not match the current request")
        _validate_progress(
            progress,
            identity_sha256=identity["identity_sha256"],
            requested_node_count=requested_node_count,
            relation_count=relation_count,
        )
    else:
        existing = [path for path in output_dir.iterdir() if path.name != LOCK_FILE]
        if not output_dir.is_dir() or existing:
            raise EmbeddingBuildError(f"output_dir must be new or empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        progress = _initial_progress(identity["identity_sha256"], batch_size)
        initial_manifest = {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "status": "RUNNING",
            "identity": identity,
            "outputs": {
                "node_embeddings": NODE_EMBEDDINGS_FILE,
                "relation_embeddings": RELATION_EMBEDDINGS_FILE,
                "progress": PROGRESS_FILE,
                "benchmark_report": BENCHMARK_FILE,
                "chunks": CHUNKS_FILE,
            },
        }
        _atomic_json(output_dir / MANIFEST_FILE, initial_manifest)
        _atomic_json(output_dir / PROGRESS_FILE, progress)

    node_array = _open_output_array(
        output_dir / NODE_EMBEDDINGS_FILE,
        shape=(requested_node_count, backend.dimension),
        dtype=dtype,
        resume=resume,
    )
    relation_array = _open_output_array(
        output_dir / RELATION_EMBEDDINGS_FILE,
        shape=(relation_count, backend.dimension),
        dtype=dtype,
        resume=resume,
    )
    if resume:
        _verify_committed_chunks(
            output_dir / CHUNKS_FILE,
            arrays={"nodes": node_array, "relations": relation_array},
            completed={
                "nodes": int(progress["nodes_completed"]),
                "relations": int(progress["relations_completed"]),
            },
        )
    run_started = time.perf_counter()
    try:
        _encode_records(
            source=nodes_path,
            id_field="node_id",
            count=requested_node_count,
            start=int(progress["nodes_completed"]),
            destination=node_array,
            backend=backend,
            progress=progress,
            progress_path=output_dir / PROGRESS_FILE,
            progress_prefix="nodes",
            chunk_path=output_dir / CHUNKS_FILE,
        )
        _encode_records(
            source=relations_path,
            id_field="relation_id",
            count=relation_count,
            start=int(progress["relations_completed"]),
            destination=relation_array,
            backend=backend,
            progress=progress,
            progress_path=output_dir / PROGRESS_FILE,
            progress_prefix="relations",
            chunk_path=output_dir / CHUNKS_FILE,
        )
    except BaseException as exc:
        progress["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        progress["last_error_type"] = type(exc).__name__
        _atomic_json(output_dir / PROGRESS_FILE, progress)
        raise
    finally:
        node_array.flush()
        relation_array.flush()
        del node_array, relation_array

    total_run_seconds = time.perf_counter() - run_started
    status = "SUCCESS" if requested_node_count == graph_node_count else "PARTIAL_BENCHMARK"
    progress["status"] = "FINALIZING"
    _atomic_json(output_dir / PROGRESS_FILE, progress)
    try:
        node_sha256 = _file_sha256(output_dir / NODE_EMBEDDINGS_FILE)
        relation_sha256 = _file_sha256(output_dir / RELATION_EMBEDDINGS_FILE)
        node_seconds = float(progress["nodes_encoding_seconds"])
        node_rate = requested_node_count / node_seconds if node_seconds > 0 else None
        runtime_stats = getattr(backend, "runtime_stats", {})
        benchmark = {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "status": status,
            "requested_node_count": requested_node_count,
            "graph_node_count": graph_node_count,
            "relation_count": relation_count,
            "source_hash_seconds": source_hash_seconds,
            "model_load_seconds": backend.metadata.get("model_load_seconds"),
            "run_wall_seconds": total_run_seconds,
            "node_encoding_seconds": node_seconds,
            "relation_encoding_seconds": float(progress["relations_encoding_seconds"]),
            "node_tokens": int(progress["nodes_tokens"]),
            "relation_tokens": int(progress["relations_tokens"]),
            "nodes_per_second": node_rate,
            "node_tokens_per_second": (
                int(progress["nodes_tokens"]) / node_seconds if node_seconds > 0 else None
            ),
            "estimated_full_graph_encoding_seconds": (
                graph_node_count / node_rate if node_rate is not None and node_rate > 0 else None
            ),
            "host_peak_rss_bytes": _peak_rss_bytes(),
            "device_peak_allocated_bytes": runtime_stats.get("device_peak_allocated_bytes"),
            "device_peak_reserved_bytes": runtime_stats.get("device_peak_reserved_bytes"),
            "effective_batch_size": int(progress["effective_batch_size"]),
            "oom_reductions": int(progress["oom_reductions"]),
        }
        _atomic_json(output_dir / BENCHMARK_FILE, benchmark)
        final_manifest = {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "status": status,
            "identity": identity,
            "completed": {
                "nodes": requested_node_count,
                "relations": relation_count,
            },
            "device": backend.metadata.get("device"),
            "compute_dtype": backend.metadata.get("compute_dtype"),
            "storage_dtype": storage_dtype,
            "embedding_dimension": backend.dimension,
            "outputs": {
                "node_embeddings": {
                    "path": NODE_EMBEDDINGS_FILE,
                    "shape": [requested_node_count, backend.dimension],
                    "sha256": node_sha256,
                },
                "relation_embeddings": {
                    "path": RELATION_EMBEDDINGS_FILE,
                    "shape": [relation_count, backend.dimension],
                    "sha256": relation_sha256,
                },
                "progress": PROGRESS_FILE,
                "benchmark_report": BENCHMARK_FILE,
                "chunks": CHUNKS_FILE,
            },
        }
        progress["status"] = status
        _atomic_json(output_dir / PROGRESS_FILE, progress)
        _atomic_json(output_dir / MANIFEST_FILE, final_manifest)
        return final_manifest
    except BaseException as exc:
        progress["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        progress["last_error_type"] = type(exc).__name__
        _atomic_json(output_dir / PROGRESS_FILE, progress)
        raise


def build_embeddings(
    graph_dir: str | Path,
    output_dir: str | Path,
    backend: EmbeddingBackend,
    *,
    batch_size: int = 16,
    max_length: int = 1024,
    storage_dtype: str = "float16",
    max_nodes: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    resolved_output = Path(output_dir).expanduser().resolve()
    if resume:
        if not resolved_output.is_dir():
            raise EmbeddingBuildError(f"resume output directory does not exist: {resolved_output}")
    else:
        if resolved_output.exists() and (
            not resolved_output.is_dir() or any(resolved_output.iterdir())
        ):
            raise EmbeddingBuildError(f"output_dir must be new or empty: {resolved_output}")
        resolved_output.mkdir(parents=True, exist_ok=True)
    with _OutputLock(resolved_output):
        return _build_embeddings_locked(
            graph_dir,
            resolved_output,
            backend,
            batch_size=batch_size,
            max_length=max_length,
            storage_dtype=storage_dtype,
            max_nodes=max_nodes,
            resume=resume,
        )


class Qwen3EmbeddingBackend:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        max_length: int = 1024,
        revision: str | None = None,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as functional
            import transformers
            from huggingface_hub import snapshot_download
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EmbeddingBuildError(
                "Qwen3 embedding requires torch and transformers; install the graph embedding dependencies"
            ) from exc
        if max_length <= 0:
            raise EmbeddingBuildError("max_length must be positive")
        self._torch = torch
        self._functional = functional
        self._device = self._resolve_device(torch, device)
        self._compute_dtype = self._resolve_dtype(torch, self._device)
        load_started = time.perf_counter()
        model_path = Path(model_name).expanduser()
        if model_path.is_dir():
            snapshot = model_path.resolve()
            resolved_revision = revision or "local-snapshot"
            public_model_name = snapshot.name
        else:
            snapshot = Path(snapshot_download(repo_id=model_name, revision=revision)).resolve()
            resolved_revision = snapshot.name
            public_model_name = model_name
        artifact_sha256, artifact_count = _directory_artifact_identity(snapshot)
        self._tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            padding_side="left",
        )
        self._model = AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=self._compute_dtype,
        )
        self._model.to(self._device)
        self._model.eval()
        self._model.requires_grad_(False)
        self._max_length = max_length
        self._dimension = int(self._model.config.hidden_size)
        if self._dimension != 1024:
            raise EmbeddingBuildError(
                f"Qwen/Qwen3-Embedding-0.6B must provide 1024 dimensions, got {self._dimension}"
            )
        commit_hash = getattr(self._model.config, "_commit_hash", None)
        tokenizer_identity = {
            "class": type(self._tokenizer).__name__,
            "vocab_size": int(len(self._tokenizer)),
            "padding_side": self._tokenizer.padding_side,
            "model_max_length": int(self._tokenizer.model_max_length),
        }
        self._metadata = {
            "type": "qwen3_transformers",
            "backend_implementation_version": BACKEND_IMPLEMENTATION_VERSION,
            "model": public_model_name,
            "requested_revision": revision,
            "resolved_revision": commit_hash or resolved_revision,
            "model_artifact_sha256": artifact_sha256,
            "model_artifact_count": artifact_count,
            "device": self._device,
            "compute_dtype": str(self._compute_dtype).removeprefix("torch."),
            "torch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
            "max_length": max_length,
            "pooling": "last_token",
            "normalization": "l2_fp32",
            "tokenizer_sha256": _json_sha256(tokenizer_identity),
            "tokenizer": tokenizer_identity,
            "model_load_seconds": time.perf_counter() - load_started,
        }
        self._device_peak_allocated = 0
        self._device_peak_reserved = 0
        self._sample_device_memory()

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested not in {"cpu", "mps", "cuda"}:
            raise EmbeddingBuildError(f"unsupported device: {requested}")
        if requested == "cuda" and not torch.cuda.is_available():
            raise EmbeddingBuildError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise EmbeddingBuildError("MPS was requested but is not available")
        return requested

    @staticmethod
    def _resolve_dtype(torch: Any, device: str) -> Any:
        if device == "cpu":
            return torch.float32
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def runtime_stats(self) -> dict[str, Any]:
        return {
            "device_peak_allocated_bytes": self._device_peak_allocated or None,
            "device_peak_reserved_bytes": self._device_peak_reserved or None,
        }

    def _sample_device_memory(self) -> None:
        if self._device == "cuda":
            self._device_peak_allocated = max(
                self._device_peak_allocated,
                int(self._torch.cuda.max_memory_allocated()),
            )
            self._device_peak_reserved = max(
                self._device_peak_reserved,
                int(self._torch.cuda.max_memory_reserved()),
            )
        elif self._device == "mps":
            self._device_peak_allocated = max(
                self._device_peak_allocated,
                int(self._torch.mps.current_allocated_memory()),
            )
            driver_memory = getattr(self._torch.mps, "driver_allocated_memory", None)
            if driver_memory is not None:
                self._device_peak_reserved = max(
                    self._device_peak_reserved,
                    int(driver_memory()),
                )

    def _clear_device_cache(self) -> None:
        if self._device == "cuda":
            self._torch.cuda.empty_cache()
        elif self._device == "mps":
            self._torch.mps.empty_cache()

    def encode(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(np.empty((0, self.dimension), dtype=np.float32), 0)
        try:
            encoded = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            token_count = int(encoded["attention_mask"].sum().item())
            inputs = {name: tensor.to(self._device) for name, tensor in encoded.items()}
            with self._torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
                attention_mask = inputs["attention_mask"]
                if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
                    pooled = hidden[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    pooled = hidden[
                        self._torch.arange(hidden.shape[0], device=hidden.device),
                        sequence_lengths,
                    ]
                normalized = self._functional.normalize(pooled.float(), p=2, dim=1)
                self._sample_device_memory()
            values = normalized.cpu().numpy()
            return EmbeddingBatch(values, token_count)
        except BaseException as exc:
            is_oom = isinstance(exc, getattr(self._torch, "OutOfMemoryError", RuntimeError)) or (
                isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
            )
            if not is_oom:
                raise
            self._clear_device_cache()
            raise EmbeddingOutOfMemoryError(str(exc)) from exc
