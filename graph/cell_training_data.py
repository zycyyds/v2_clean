from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


CELL_CACHE_SCHEMA_VERSION = 1
CELL_CACHE_FILE = "cell_training_records.npz"
CELL_CACHE_MANIFEST = "cell_training_records.json"
GRAPH_CACHE_DIR = "cell_graph_cache"
HASH_CACHE_FILE = "cell_training_hashes.json"
PREPARATION_LOCK_FILE = ".cell_training_prepare.lock"


class CellTrainingDataError(ValueError):
    pass


@contextmanager
def _preparation_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / PREPARATION_LOCK_FILE
    deadline = time.monotonic() + 3600.0
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}))
            break
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 6 * 3600
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise CellTrainingDataError(
                    f"timed out waiting for private cache preparation lock: {lock_path}"
                )
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_sha256(path: Path, cache: dict[str, Any]) -> str:
    stat = path.stat()
    key = str(path.resolve())
    cached = cache.get(key)
    if (
        isinstance(cached, dict)
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
    ):
        return str(cached["sha256"])
    value = _sha256(path)
    cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": value}
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CellTrainingDataError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CellTrainingDataError(f"{label} must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _required_file(root: Path, name: str, label: str) -> Path:
    path = root / name
    if not path.is_file():
        raise CellTrainingDataError(f"missing {label}: {path}")
    return path


@dataclass(frozen=True)
class TrainingIdentity:
    graph_manifest_sha256: str
    supervision_manifest_sha256: str
    supervision_masks_sha256: str
    nodes_sha256: str
    relations_sha256: str
    relation_ids_sha256: str
    edge_index_sha256: str
    edge_type_sha256: str
    node_embeddings_sha256: str
    relation_embeddings_sha256: str
    embedding_identity_sha256: str
    node_count: int
    observation_count: int
    edge_count: int
    relation_count: int
    embedding_dimension: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CellRecords:
    observation_index: np.ndarray
    row_id: np.ndarray
    relation_id: np.ndarray
    value_node_id: np.ndarray
    forward_edge_id: np.ndarray
    reverse_edge_id: np.ndarray
    label: np.ndarray
    fold: np.ndarray
    source: np.ndarray

    def __len__(self) -> int:
        return int(self.observation_index.shape[0])

    def indices_for_folds(self, folds: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.fold, np.asarray(tuple(folds), dtype=np.int8)))

    def take(self, indices: np.ndarray) -> "CellRecords":
        return CellRecords(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> "CellRecords":
        try:
            with np.load(path) as arrays:
                missing = set(cls.__dataclass_fields__) - set(arrays.files)
                if missing:
                    raise CellTrainingDataError(f"cell cache is missing arrays: {sorted(missing)}")
                return cls(**{name: np.asarray(arrays[name]) for name in cls.__dataclass_fields__})
        except (OSError, ValueError) as exc:
            if isinstance(exc, CellTrainingDataError):
                raise
            raise CellTrainingDataError(f"invalid cell cache: {path}") from exc


@dataclass(frozen=True)
class NodeMetadata:
    node_type: np.ndarray
    table_id: np.ndarray
    tables: tuple[str, ...]


@dataclass(frozen=True)
class GraphCSR:
    offsets: np.ndarray
    neighbors: np.ndarray
    relation_id: np.ndarray
    edge_id: np.ndarray
    edge_position: np.ndarray


@dataclass(frozen=True)
class PreparedCellTrainingData:
    identity: TrainingIdentity
    records: CellRecords
    graph_manifest: dict[str, Any]
    embedding_manifest: dict[str, Any]
    graph_dir: Path
    supervision_dir: Path
    embedding_dir: Path

    @property
    def node_embeddings_path(self) -> Path:
        return self.embedding_dir / str(
            self.embedding_manifest["outputs"]["node_embeddings"]["path"]
        )

    @property
    def relation_embeddings_path(self) -> Path:
        return self.embedding_dir / str(
            self.embedding_manifest["outputs"]["relation_embeddings"]["path"]
        )


def _validate_supervision_arrays(
    path: Path,
    observation_count: int,
    expected_count: int,
    expected_dirty: int,
) -> dict[str, np.ndarray]:
    try:
        with np.load(path) as values:
            required = {"cell_indices", "cell_labels", "cell_folds", "cell_source"}
            missing = required - set(values.files)
            if missing:
                raise CellTrainingDataError(f"supervision masks are missing arrays: {sorted(missing)}")
            arrays = {name: np.asarray(values[name]) for name in required}
    except (OSError, ValueError) as exc:
        if isinstance(exc, CellTrainingDataError):
            raise
        raise CellTrainingDataError(f"invalid supervision masks: {path}") from exc

    lengths = {len(value) for value in arrays.values()}
    if lengths != {expected_count}:
        raise CellTrainingDataError(
            f"expected {expected_count} supervised Cells, got lengths={sorted(lengths)}"
        )
    indices = arrays["cell_indices"]
    if len(np.unique(indices)) != len(indices):
        raise CellTrainingDataError("supervised Cell indices must be unique")
    if indices.min(initial=0) < 0 or indices.max(initial=-1) >= observation_count:
        raise CellTrainingDataError("supervised Cell index is outside graph observation range")
    if set(np.unique(arrays["cell_labels"]).tolist()) != {0, 1}:
        raise CellTrainingDataError("Cell labels must contain both 0 and 1")
    if set(np.unique(arrays["cell_folds"]).tolist()) != {0, 1, 2, 3, 4}:
        raise CellTrainingDataError("Cell folds must be exactly 0..4")
    if not set(np.unique(arrays["cell_source"]).tolist()).issubset({0, 1, 2}):
        raise CellTrainingDataError("Cell source must use only 0, 1 or 2")
    labels = arrays["cell_labels"]
    sources = arrays["cell_source"]
    if np.any((sources == 0) != (labels == 0)):
        raise CellTrainingDataError(
            "sampled-clean source must be clean and dirty sources must be dirty"
        )
    if int(arrays["cell_labels"].sum()) != expected_dirty:
        raise CellTrainingDataError(f"expected {expected_dirty} dirty Cell labels")
    return arrays


def _validate_identity(
    graph_dir: Path,
    supervision_dir: Path,
    embedding_dir: Path,
) -> tuple[TrainingIdentity, dict[str, Any], dict[str, Any], dict[str, Any]]:
    hash_cache_path = supervision_dir / HASH_CACHE_FILE
    hash_cache: dict[str, Any] = {}
    if hash_cache_path.is_file():
        try:
            loaded_cache = json.loads(hash_cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded_cache, dict):
                hash_cache = loaded_cache
        except (OSError, json.JSONDecodeError):
            hash_cache = {}
    graph_manifest_path = _required_file(graph_dir, "graph_manifest.json", "graph manifest")
    graph_manifest = _read_json(graph_manifest_path, "graph manifest")
    supervision_manifest = _read_json(
        _required_file(supervision_dir, "supervision_manifest.json", "supervision manifest"),
        "supervision manifest",
    )
    embedding_manifest = _read_json(
        _required_file(embedding_dir, "embedding_manifest.json", "embedding manifest"),
        "embedding manifest",
    )
    if supervision_manifest.get("status") != "SUCCESS":
        raise CellTrainingDataError("supervision manifest status must be SUCCESS")
    if embedding_manifest.get("status") != "SUCCESS":
        raise CellTrainingDataError("embedding manifest status must be SUCCESS")

    try:
        node_count = int(graph_manifest["node_count"])
        observation_count = int(graph_manifest.get("observation_count", graph_manifest["triple_count"]))
        edge_count = int(graph_manifest["edge_count"])
        relation_count = int(graph_manifest["relation_count"])
        embedding_dimension = int(embedding_manifest["embedding_dimension"])
        embedding_identity = embedding_manifest["identity"]
        outputs = embedding_manifest["outputs"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CellTrainingDataError("graph or embedding manifest has an invalid shape contract") from exc
    if edge_count != observation_count * 2:
        raise CellTrainingDataError("typed-value graph must contain two directed edges per Cell")
    if int(embedding_manifest["completed"]["nodes"]) != node_count:
        raise CellTrainingDataError("node embedding count does not match graph")
    if int(embedding_manifest["completed"]["relations"]) != relation_count:
        raise CellTrainingDataError("relation embedding count does not match graph")

    graph_hash = _cached_sha256(graph_manifest_path, hash_cache)
    relation_ids_path = _required_file(graph_dir, "relation_ids.json", "relation IDs")
    relation_ids_hash = _cached_sha256(relation_ids_path, hash_cache)
    if embedding_identity.get("graph_manifest_sha256") != graph_hash:
        raise CellTrainingDataError("embedding graph identity does not match current graph")
    if supervision_manifest.get("relation_ids_sha256") != relation_ids_hash:
        raise CellTrainingDataError("supervision relation identity does not match current graph")
    if int(embedding_identity.get("graph_node_count", -1)) != node_count:
        raise CellTrainingDataError("embedding identity node count does not match graph")
    if int(embedding_identity.get("relation_count", -1)) != relation_count:
        raise CellTrainingDataError("embedding identity relation count does not match graph")

    node_output = outputs["node_embeddings"]
    relation_output = outputs["relation_embeddings"]
    if list(node_output.get("shape", [])) != [node_count, embedding_dimension]:
        raise CellTrainingDataError("node embedding shape in manifest does not match graph")
    if list(relation_output.get("shape", [])) != [relation_count, embedding_dimension]:
        raise CellTrainingDataError("relation embedding shape in manifest does not match graph")
    node_embeddings_path = _required_file(
        embedding_dir, str(node_output["path"]), "node embeddings"
    )
    relation_embeddings_path = _required_file(
        embedding_dir, str(relation_output["path"]), "relation embeddings"
    )
    for path, shape in (
        (node_embeddings_path, (node_count, embedding_dimension)),
        (relation_embeddings_path, (relation_count, embedding_dimension)),
    ):
        try:
            array = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise CellTrainingDataError(f"invalid embedding array: {path}") from exc
        if array.shape != shape or array.dtype != np.float16:
            raise CellTrainingDataError(
                f"embedding array contract mismatch at {path}: {array.shape}/{array.dtype}"
            )

    nodes_path = _required_file(
        graph_dir, str(graph_manifest["storage"]["nodes"]), "graph nodes"
    )
    relations_path = _required_file(
        graph_dir, str(graph_manifest["storage"]["relation_texts"]), "relation texts"
    )
    expected_hashes = (
        (nodes_path, str(embedding_identity.get("nodes_sha256", "")), "graph nodes"),
        (
            relations_path,
            str(embedding_identity.get("relations_sha256", "")),
            "relation texts",
        ),
        (
            node_embeddings_path,
            str(node_output.get("sha256", "")),
            "node embeddings",
        ),
        (
            relation_embeddings_path,
            str(relation_output.get("sha256", "")),
            "relation embeddings",
        ),
    )
    for path, expected_hash, label in expected_hashes:
        if len(expected_hash) != 64 or _cached_sha256(path, hash_cache) != expected_hash:
            raise CellTrainingDataError(f"{label} SHA-256 does not match embedding identity")

    edge_index_path = _required_file(
        graph_dir, str(graph_manifest["storage"]["edge_index"]), "edge index"
    )
    edge_type_path = _required_file(
        graph_dir, str(graph_manifest["storage"]["edge_type"]), "edge types"
    )
    supervision_manifest_path = supervision_dir / "supervision_manifest.json"
    supervision_masks_path = _required_file(
        supervision_dir, "supervision_masks.npz", "supervision masks"
    )
    identity = TrainingIdentity(
        graph_manifest_sha256=graph_hash,
        supervision_manifest_sha256=_cached_sha256(supervision_manifest_path, hash_cache),
        supervision_masks_sha256=_cached_sha256(supervision_masks_path, hash_cache),
        nodes_sha256=str(embedding_identity.get("nodes_sha256", "")),
        relations_sha256=str(embedding_identity.get("relations_sha256", "")),
        relation_ids_sha256=relation_ids_hash,
        edge_index_sha256=_cached_sha256(edge_index_path, hash_cache),
        edge_type_sha256=_cached_sha256(edge_type_path, hash_cache),
        node_embeddings_sha256=str(node_output["sha256"]),
        relation_embeddings_sha256=str(relation_output["sha256"]),
        embedding_identity_sha256=str(embedding_identity.get("identity_sha256", "")),
        node_count=node_count,
        observation_count=observation_count,
        edge_count=edge_count,
        relation_count=relation_count,
        embedding_dimension=embedding_dimension,
    )
    _atomic_json(hash_cache_path, hash_cache)
    return identity, graph_manifest, supervision_manifest, embedding_manifest


def _build_cell_records(
    graph_dir: Path,
    masks: dict[str, np.ndarray],
    relation_ids: dict[str, int],
) -> CellRecords:
    targets = np.asarray(masks["cell_indices"], dtype=np.int64)
    order = np.argsort(targets)
    sorted_targets = targets[order]
    positions = {int(value): int(order[index]) for index, value in enumerate(sorted_targets)}
    count = len(targets)
    result = {
        "observation_index": targets.copy(),
        "row_id": np.full(count, -1, dtype=np.int64),
        "relation_id": np.full(count, -1, dtype=np.int32),
        "value_node_id": np.full(count, -1, dtype=np.int64),
        "forward_edge_id": np.full(count, -1, dtype=np.int64),
        "reverse_edge_id": np.full(count, -1, dtype=np.int64),
        "label": np.asarray(masks["cell_labels"], dtype=np.int8),
        "fold": np.asarray(masks["cell_folds"], dtype=np.int8),
        "source": np.asarray(masks["cell_source"], dtype=np.int8),
    }
    next_target = 0
    observations_path = _required_file(graph_dir, "cell_observations.jsonl", "Cell observations")
    with observations_path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if next_target >= count:
                break
            target = int(sorted_targets[next_target])
            if line_index < target:
                continue
            if line_index != target:
                raise CellTrainingDataError(f"missing Cell observation index {target}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CellTrainingDataError(
                    f"invalid Cell observation JSON at line {line_index + 1}"
                ) from exc
            position = positions[target]
            if int(record.get("observation_index", line_index)) != target:
                raise CellTrainingDataError(f"Cell observation index mismatch at {target}")
            relation = str(record.get("relation") or "")
            if relation not in relation_ids:
                raise CellTrainingDataError(f"unknown forward relation at Cell {target}: {relation}")
            try:
                result["row_id"][position] = int(record["row_id"])
                result["relation_id"][position] = int(relation_ids[relation])
                result["value_node_id"][position] = int(record["value_node_id"])
                result["forward_edge_id"][position] = int(record["forward_edge_id"])
                result["reverse_edge_id"][position] = int(record["reverse_edge_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CellTrainingDataError(f"incomplete Cell observation at {target}") from exc
            next_target += 1
    if next_target != count or any((result[name] < 0).any() for name in (
        "row_id", "relation_id", "value_node_id", "forward_edge_id", "reverse_edge_id"
    )):
        raise CellTrainingDataError("not all supervised Cells mapped to graph triples")
    return CellRecords(**result)


def _validate_cell_records(
    records: CellRecords,
    masks: dict[str, np.ndarray],
    identity: TrainingIdentity,
    graph_dir: Path,
    graph_manifest: dict[str, Any],
    relation_ids: dict[str, int],
) -> None:
    if len(records) != len(masks["cell_indices"]):
        raise CellTrainingDataError("Cell record cache count does not match supervision masks")
    for record_name, mask_name in (
        ("observation_index", "cell_indices"),
        ("label", "cell_labels"),
        ("fold", "cell_folds"),
        ("source", "cell_source"),
    ):
        if not np.array_equal(getattr(records, record_name), masks[mask_name]):
            raise CellTrainingDataError(
                f"Cell record cache {record_name} does not match supervision masks"
            )
    if (
        records.row_id.min(initial=0) < 0
        or records.value_node_id.min(initial=0) < 0
        or records.relation_id.min(initial=0) < 0
        or records.forward_edge_id.min(initial=0) < 0
        or records.reverse_edge_id.min(initial=0) < 0
        or records.row_id.max(initial=-1) >= identity.node_count
        or records.value_node_id.max(initial=-1) >= identity.node_count
        or records.relation_id.max(initial=-1) >= identity.relation_count
        or records.forward_edge_id.max(initial=-1) >= identity.edge_count
        or records.reverse_edge_id.max(initial=-1) >= identity.edge_count
    ):
        raise CellTrainingDataError("supervised Cell mapping contains an out-of-range graph ID")

    try:
        edge_index_path = graph_dir / str(graph_manifest["storage"]["edge_index"])
        edge_type_path = graph_dir / str(graph_manifest["storage"]["edge_type"])
        edge_index = np.load(edge_index_path, mmap_mode="r")
        edge_type = np.load(edge_type_path, mmap_mode="r")
    except (KeyError, OSError, ValueError) as exc:
        raise CellTrainingDataError("cannot validate supervised Cells against graph edges") from exc
    if (
        edge_index.shape != (2, identity.edge_count)
        or edge_type.shape != (identity.edge_count,)
    ):
        raise CellTrainingDataError("graph edge arrays do not match the graph manifest")
    forward = records.forward_edge_id
    reverse = records.reverse_edge_id
    reverse_relation = np.full(identity.relation_count, -1, dtype=np.int64)
    for name, relation_id in relation_ids.items():
        if not name.endswith("__rev"):
            reverse_name = f"{name}__rev"
            if reverse_name not in relation_ids:
                raise CellTrainingDataError(f"missing reverse relation ID for {name}")
            reverse_relation[relation_id] = relation_ids[reverse_name]
    expected_reverse = reverse_relation[records.relation_id]
    valid = (
        (edge_index[0, forward] == records.row_id)
        & (edge_index[1, forward] == records.value_node_id)
        & (edge_type[forward] == records.relation_id)
        & (edge_index[0, reverse] == records.value_node_id)
        & (edge_index[1, reverse] == records.row_id)
        & (edge_type[reverse] == expected_reverse)
    )
    if not bool(np.all(valid)):
        bad = int(np.flatnonzero(~valid)[0])
        raise CellTrainingDataError(
            "supervised Cell edge mapping does not match edge_index/edge_type "
            f"at observation {int(records.observation_index[bad])}"
        )


def prepare_cell_training_data(
    graph_dir: str | Path,
    supervision_dir: str | Path,
    embedding_dir: str | Path,
) -> PreparedCellTrainingData:
    graph_dir = Path(graph_dir).expanduser().resolve()
    supervision_dir = Path(supervision_dir).expanduser().resolve()
    embedding_dir = Path(embedding_dir).expanduser().resolve()
    with _preparation_lock(supervision_dir):
        return _prepare_cell_training_data_locked(
            graph_dir, supervision_dir, embedding_dir
        )


def _prepare_cell_training_data_locked(
    graph_dir: Path,
    supervision_dir: Path,
    embedding_dir: Path,
) -> PreparedCellTrainingData:
    identity, graph_manifest, supervision_manifest, embedding_manifest = _validate_identity(
        graph_dir, supervision_dir, embedding_dir
    )
    try:
        expected_count = int(supervision_manifest["training_observation_count"])
        expected_dirty = int(supervision_manifest["dirty_training_count"])
        expected_clean = int(supervision_manifest["clean_training_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CellTrainingDataError("supervision manifest has invalid training counts") from exc
    if expected_count != expected_dirty + expected_clean or expected_dirty < 1 or expected_clean < 1:
        raise CellTrainingDataError("supervision training counts are inconsistent")
    masks = _validate_supervision_arrays(
        _required_file(supervision_dir, "supervision_masks.npz", "supervision masks"),
        identity.observation_count,
        expected_count,
        expected_dirty,
    )
    cache_path = supervision_dir / CELL_CACHE_FILE
    cache_manifest_path = supervision_dir / CELL_CACHE_MANIFEST
    expected_cache_identity = {
        "schema_version": CELL_CACHE_SCHEMA_VERSION,
        "training_identity": identity.to_dict(),
    }
    use_cache = False
    if cache_path.is_file() and cache_manifest_path.is_file():
        cache_manifest = _read_json(cache_manifest_path, "Cell cache manifest")
        use_cache = all(cache_manifest.get(key) == value for key, value in expected_cache_identity.items())
    relation_ids_raw = _read_json(graph_dir / "relation_ids.json", "relation IDs")
    relation_ids = {str(key): int(value) for key, value in relation_ids_raw.items()}
    if sorted(relation_ids.values()) != list(range(identity.relation_count)):
        raise CellTrainingDataError("relation IDs must be unique, contiguous and match manifest")
    if use_cache:
        records = CellRecords.load(cache_path)
    else:
        records = _build_cell_records(graph_dir, masks, relation_ids)
        temporary = cache_path.with_name(f".{cache_path.name}.tmp.npz")
        records.save(temporary)
        _replace_with_retry(temporary, cache_path)
        _atomic_json(cache_manifest_path, {
            **expected_cache_identity,
            "record_count": len(records),
            "fold_counts": {
                str(fold): int((records.fold == fold).sum()) for fold in range(5)
            },
        })
    _validate_cell_records(
        records, masks, identity, graph_dir, graph_manifest, relation_ids
    )
    return PreparedCellTrainingData(
        identity=identity,
        records=records,
        graph_manifest=graph_manifest,
        embedding_manifest=embedding_manifest,
        graph_dir=graph_dir,
        supervision_dir=supervision_dir,
        embedding_dir=embedding_dir,
    )


def _graph_cache_identity(data: PreparedCellTrainingData) -> dict[str, Any]:
    edge_index_name = str(data.graph_manifest["storage"]["edge_index"])
    edge_type_name = str(data.graph_manifest["storage"]["edge_type"])
    nodes_name = str(data.graph_manifest["storage"]["nodes"])
    return {
        "schema_version": CELL_CACHE_SCHEMA_VERSION,
        "training_identity": data.identity.to_dict(),
        "edge_index": edge_index_name,
        "edge_type": edge_type_name,
        "nodes": nodes_name,
    }


def prepare_graph_cache(data: PreparedCellTrainingData) -> tuple[GraphCSR, NodeMetadata]:
    with _preparation_lock(data.supervision_dir):
        return _prepare_graph_cache_locked(data)


def _prepare_graph_cache_locked(
    data: PreparedCellTrainingData,
) -> tuple[GraphCSR, NodeMetadata]:
    cache = data.supervision_dir / GRAPH_CACHE_DIR
    manifest_path = cache / "manifest.json"
    expected = _graph_cache_identity(data)
    files = {
        "offsets": cache / "offsets.npy",
        "neighbors": cache / "neighbors.npy",
        "relation_id": cache / "relation_id.npy",
        "edge_id": cache / "edge_id.npy",
        "edge_position": cache / "edge_position.npy",
        "node_type": cache / "node_type.npy",
        "table_id": cache / "table_id.npy",
        "tables": cache / "tables.json",
    }
    valid = manifest_path.is_file() and all(path.is_file() for path in files.values())
    if valid:
        manifest = _read_json(manifest_path, "graph cache manifest")
        valid = all(manifest.get(key) == value for key, value in expected.items())
    if not valid:
        temporary = cache.with_name(f".{cache.name}.building")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        edge_index = np.load(
            data.graph_dir / str(data.graph_manifest["storage"]["edge_index"]), mmap_mode="r"
        )
        edge_type = np.load(
            data.graph_dir / str(data.graph_manifest["storage"]["edge_type"]), mmap_mode="r"
        )
        if edge_index.shape != (2, data.identity.edge_count) or edge_type.shape != (data.identity.edge_count,):
            raise CellTrainingDataError("graph edge arrays do not match manifest")
        sources = np.asarray(edge_index[0])
        targets = np.asarray(edge_index[1])
        offsets = np.zeros(data.identity.node_count + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(np.bincount(targets, minlength=data.identity.node_count))
        order = np.argsort(targets, kind="stable")
        np.save(temporary / "offsets.npy", offsets)
        edge_position = np.empty(data.identity.edge_count, dtype=np.uint32)
        edge_position[order] = np.arange(data.identity.edge_count, dtype=np.uint32)
        np.save(temporary / "neighbors.npy", np.asarray(sources[order], dtype=np.int32))
        np.save(temporary / "relation_id.npy", np.asarray(edge_type[order], dtype=np.int16))
        np.save(temporary / "edge_id.npy", np.asarray(order, dtype=np.uint32))
        np.save(temporary / "edge_position.npy", edge_position)

        table_names = sorted(str(name) for name in data.graph_manifest.get("table_counts", {}))
        table_map = {name: index for index, name in enumerate(table_names)}
        node_type = np.full(data.identity.node_count, 1, dtype=np.uint8)
        table_id = np.full(data.identity.node_count, -1, dtype=np.int16)
        nodes_path = data.graph_dir / str(data.graph_manifest["storage"]["nodes"])
        digest = hashlib.sha256()
        seen_nodes = 0
        with nodes_path.open("rb") as handle:
            for expected_node_id, raw_line in enumerate(handle):
                seen_nodes += 1
                digest.update(raw_line)
                try:
                    node = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CellTrainingDataError(
                        f"invalid node JSON at line {expected_node_id + 1}"
                    ) from exc
                node_id = int(node.get("node_id", -1))
                if node_id != expected_node_id or node_id >= data.identity.node_count:
                    raise CellTrainingDataError("node IDs must be contiguous and ordered")
                if node.get("node_type") == "Row":
                    node_type[node_id] = 0
                    table = str(node.get("table") or "")
                    if table not in table_map:
                        raise CellTrainingDataError(f"unknown Row table: {table}")
                    table_id[node_id] = table_map[table]
        if seen_nodes != data.identity.node_count:
            raise CellTrainingDataError("node file count does not match graph manifest")
        if data.identity.nodes_sha256 and digest.hexdigest() != data.identity.nodes_sha256:
            raise CellTrainingDataError("current nodes.jsonl does not match embedding identity")
        np.save(temporary / "node_type.npy", node_type)
        np.save(temporary / "table_id.npy", table_id)
        (temporary / "tables.json").write_text(
            json.dumps(table_names, indent=2) + "\n", encoding="utf-8"
        )
        _atomic_json(temporary / "manifest.json", expected)
        if cache.exists():
            shutil.rmtree(cache)
        _replace_with_retry(temporary, cache)

    try:
        arrays = {
            name: np.load(files[name], mmap_mode="r")
            for name in (
                "offsets", "neighbors", "relation_id", "edge_id", "edge_position",
                "node_type", "table_id",
            )
        }
    except (OSError, ValueError) as exc:
        raise CellTrainingDataError("invalid private graph cache array") from exc
    expected_contracts = {
        "offsets": ((data.identity.node_count + 1,), np.dtype(np.int64)),
        "neighbors": ((data.identity.edge_count,), np.dtype(np.int32)),
        "relation_id": ((data.identity.edge_count,), np.dtype(np.int16)),
        "edge_id": ((data.identity.edge_count,), np.dtype(np.uint32)),
        "edge_position": ((data.identity.edge_count,), np.dtype(np.uint32)),
        "node_type": ((data.identity.node_count,), np.dtype(np.uint8)),
        "table_id": ((data.identity.node_count,), np.dtype(np.int16)),
    }
    for name, (shape, dtype) in expected_contracts.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise CellTrainingDataError(
                f"private graph cache contract mismatch for {name}"
            )
    offsets = arrays["offsets"]
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != data.identity.edge_count
        or np.any(offsets[1:] < offsets[:-1])
        or arrays["neighbors"].min(initial=0) < 0
        or arrays["neighbors"].max(initial=-1) >= data.identity.node_count
        or arrays["relation_id"].min(initial=0) < 0
        or arrays["relation_id"].max(initial=-1) >= data.identity.relation_count
        or arrays["edge_id"].max(initial=0) >= data.identity.edge_count
        or arrays["edge_position"].max(initial=0) >= data.identity.edge_count
    ):
        raise CellTrainingDataError("private graph cache contains out-of-range values")
    graph = GraphCSR(
        offsets=offsets,
        neighbors=arrays["neighbors"],
        relation_id=arrays["relation_id"],
        edge_id=arrays["edge_id"],
        edge_position=arrays["edge_position"],
    )
    tables_raw = json.loads(files["tables"].read_text(encoding="utf-8"))
    metadata = NodeMetadata(
        node_type=arrays["node_type"],
        table_id=arrays["table_id"],
        tables=tuple(str(value) for value in tables_raw),
    )
    return graph, metadata


class FrozenEmbeddingStore:
    def __init__(self, data: PreparedCellTrainingData) -> None:
        self.nodes = np.load(data.node_embeddings_path, mmap_mode="r")
        self.relations = np.load(data.relation_embeddings_path, mmap_mode="r")

    def node_rows(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.nodes[indices], dtype=np.float32)

    def relation_rows(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.relations[indices], dtype=np.float32)


def write_predictions_csv(path: Path, rows: Iterator[dict[str, Any]]) -> None:
    fields = [
        "split",
        "observation_index",
        "row_id",
        "relation_id",
        "value_node_id",
        "fold",
        "source",
        "label",
        "dirty_score",
        "threshold",
        "prediction",
        "model_type",
        "seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
