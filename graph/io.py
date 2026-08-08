from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_graph_arrays(output_dir: Path, edge_index: list[tuple[int, int]], edge_type: list[int], features: np.ndarray) -> None:
    np.savez_compressed(
        output_dir / "edges.npz",
        edge_index=np.asarray(edge_index, dtype=np.int64).T if edge_index else np.empty((2, 0), dtype=np.int64),
        edge_type=np.asarray(edge_type, dtype=np.int16),
    )
    np.savez_compressed(output_dir / "node_features.npz", node_features=features.astype(np.float32, copy=False))
