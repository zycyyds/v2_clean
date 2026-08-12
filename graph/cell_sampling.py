from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cell_training_data import CellRecords, GraphCSR, NodeMetadata


@dataclass(frozen=True)
class SampledCellBatch:
    node_id: np.ndarray
    node_type: np.ndarray
    table_id: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    edge_relation: np.ndarray
    edge_id: np.ndarray
    edge_sample: np.ndarray
    edge_hop: np.ndarray
    target_row: np.ndarray
    target_value: np.ndarray
    target_relation: np.ndarray
    label: np.ndarray
    record_index: np.ndarray


class TargetMaskedNeighborSampler:
    """Build disjoint target-specific ego graphs with both target directions removed."""

    def __init__(
        self,
        graph: GraphCSR,
        metadata: NodeMetadata,
        *,
        fanouts: Sequence[int] = (16, 8),
    ) -> None:
        if not fanouts or any(value <= 0 for value in fanouts):
            raise ValueError("fanouts must contain positive integers")
        self.graph = graph
        self.metadata = metadata
        self.fanouts = tuple(int(value) for value in fanouts)

    def _sample_edges(
        self,
        node_id: int,
        blocked: frozenset[int],
        fanout: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        start = int(self.graph.offsets[node_id])
        end = int(self.graph.offsets[node_id + 1])
        blocked_positions = sorted(
            int(self.graph.edge_position[edge_id])
            for edge_id in blocked
            if start <= int(self.graph.edge_position[edge_id]) < end
        )
        available = end - start - len(blocked_positions)
        if available <= 0:
            return np.empty(0, dtype=np.int64)
        count = min(fanout, available)
        compact = np.arange(available, dtype=np.int64) if count == available else np.sort(
            rng.choice(available, size=count, replace=False)
        )
        positions = compact + start
        for blocked_position in blocked_positions:
            positions += positions >= blocked_position
        return positions

    def sample(
        self,
        records: CellRecords,
        record_indices: np.ndarray,
        *,
        seed: int,
        epoch: int,
    ) -> SampledCellBatch:
        nodes: list[int] = []
        node_types: list[int] = []
        table_ids: list[int] = []
        edge_sources: list[int] = []
        edge_targets: list[int] = []
        edge_relations: list[int] = []
        edge_ids: list[int] = []
        edge_samples: list[int] = []
        edge_hops: list[int] = []
        target_rows: list[int] = []
        target_values: list[int] = []
        target_relations: list[int] = []
        labels: list[int] = []

        for sample_id, record_index in enumerate(np.asarray(record_indices, dtype=np.int64)):
            observation = int(records.observation_index[record_index])
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, observation]))
            blocked = frozenset((
                int(records.forward_edge_id[record_index]),
                int(records.reverse_edge_id[record_index]),
            ))
            local: dict[int, int] = {}

            def intern(global_node: int) -> int:
                value = local.get(global_node)
                if value is not None:
                    return value
                value = len(nodes)
                local[global_node] = value
                nodes.append(global_node)
                node_types.append(int(self.metadata.node_type[global_node]))
                table_ids.append(int(self.metadata.table_id[global_node]))
                return value

            row = int(records.row_id[record_index])
            value = int(records.value_node_id[record_index])
            target_rows.append(intern(row))
            target_values.append(intern(value))
            target_relations.append(int(records.relation_id[record_index]))
            labels.append(int(records.label[record_index]))

            frontier = [row, value]
            visited = {row, value}
            for hop, fanout in enumerate(self.fanouts):
                next_frontier: list[int] = []
                for target in frontier:
                    for position in self._sample_edges(target, blocked, fanout, rng):
                        source = int(self.graph.neighbors[position])
                        edge_sources.append(intern(source))
                        edge_targets.append(intern(target))
                        edge_relations.append(int(self.graph.relation_id[position]))
                        edge_ids.append(int(self.graph.edge_id[position]))
                        edge_samples.append(sample_id)
                        edge_hops.append(hop)
                        if source not in visited:
                            visited.add(source)
                            next_frontier.append(source)
                frontier = next_frontier
                if not frontier:
                    break

        return SampledCellBatch(
            node_id=np.asarray(nodes, dtype=np.int64),
            node_type=np.asarray(node_types, dtype=np.uint8),
            table_id=np.asarray(table_ids, dtype=np.int16),
            edge_source=np.asarray(edge_sources, dtype=np.int64),
            edge_target=np.asarray(edge_targets, dtype=np.int64),
            edge_relation=np.asarray(edge_relations, dtype=np.int64),
            edge_id=np.asarray(edge_ids, dtype=np.int64),
            edge_sample=np.asarray(edge_samples, dtype=np.int64),
            edge_hop=np.asarray(edge_hops, dtype=np.int64),
            target_row=np.asarray(target_rows, dtype=np.int64),
            target_value=np.asarray(target_values, dtype=np.int64),
            target_relation=np.asarray(target_relations, dtype=np.int64),
            label=np.asarray(labels, dtype=np.float32),
            record_index=np.asarray(record_indices, dtype=np.int64),
        )
