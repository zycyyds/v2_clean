from __future__ import annotations

import torch
from torch import nn


class CellBinaryHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        row: torch.Tensor,
        relation: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (row, relation, value, row * value, torch.abs(row - value)), dim=-1
        )
        return self.network(features).squeeze(-1)


class TripleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.row_projection = nn.Linear(input_dim, hidden_dim)
        self.relation_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.head = CellBinaryHead(hidden_dim, dropout)

    def forward(
        self,
        row: torch.Tensor,
        relation: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.row_projection(row),
            self.relation_projection(relation),
            self.value_projection(value),
        )


class StrictMLP(nn.Module):
    """Strict R-GCN control that retains self-updates but removes graph messages."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        table_count: int,
        layer_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        self.node_projection = nn.Linear(input_dim, hidden_dim)
        self.relation_projection = nn.Linear(input_dim, hidden_dim)
        self.row_type = nn.Parameter(torch.empty(hidden_dim))
        self.table_embedding = nn.Embedding(table_count, hidden_dim)
        self.self_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(layer_count)
        ])
        self.head = CellBinaryHead(hidden_dim, dropout)
        nn.init.normal_(self.row_type, std=0.02)

    def forward(
        self,
        table_id: torch.Tensor,
        relation: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if (table_id < 0).any():
            raise ValueError("strict Row inputs require a table ID")
        row = self.row_type.unsqueeze(0) + self.table_embedding(table_id)
        value = self.node_projection(value)
        for self_update in self.self_updates:
            row = self_update(row)
            value = self_update(value)
        return self.head(
            row,
            self.relation_projection(relation),
            value,
        )


class BasisRGCNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        relation_count: int,
        basis_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if basis_count < 1 or basis_count > relation_count:
            raise ValueError("basis_count must be between 1 and relation_count")
        self.relation_count = relation_count
        self.bases = nn.Parameter(torch.empty(basis_count, hidden_dim, hidden_dim))
        self.coefficients = nn.Parameter(torch.empty(relation_count, basis_count))
        self.self_loop = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        edge_relation: torch.Tensor,
        relation_state: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(node_state)
        if edge_source.numel():
            if node_state.is_cuda:
                relation_weights = torch.einsum(
                    "rb,bij->rij", self.coefficients, self.bases
                )
                messages = torch.bmm(
                    node_state[edge_source].unsqueeze(1),
                    relation_weights[edge_relation],
                ).squeeze(1)
                messages = messages + self.relation_message(
                    relation_state[edge_relation]
                )
                pair = edge_target * self.relation_count + edge_relation
                _, inverse, counts = torch.unique(
                    pair, sorted=False, return_inverse=True, return_counts=True
                )
                messages = messages / counts[inverse].to(messages.dtype).unsqueeze(-1)
                aggregate.index_add_(0, edge_target, messages)
                updated = self.self_loop(node_state) + aggregate
                return self.dropout(torch.nn.functional.gelu(self.normalization(updated)))

            order = torch.argsort(edge_relation, stable=True)
            sorted_relation = edge_relation[order]
            unique_relation, counts = torch.unique_consecutive(
                sorted_relation, return_counts=True
            )
            relation_weights = torch.einsum(
                "rb,bij->rij", self.coefficients[unique_relation], self.bases
            )
            start = 0
            for group_index, count in enumerate(counts.tolist()):
                positions = order[start:start + count]
                source = edge_source[positions]
                target = edge_target[positions]
                relation_id = unique_relation[group_index]
                messages = node_state[source] @ relation_weights[group_index]
                messages = messages + self.relation_message(
                    relation_state[relation_id]
                ).unsqueeze(0)
                relation_degree = torch.bincount(
                    target, minlength=node_state.shape[0]
                ).to(messages.dtype)
                messages = messages / relation_degree[target].unsqueeze(-1)
                aggregate.index_add_(0, target, messages)
                start += count
        updated = self.self_loop(node_state) + aggregate
        return self.dropout(torch.nn.functional.gelu(self.normalization(updated)))


class CellRGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        relation_count: int,
        table_count: int,
        *,
        strict_rows: bool,
        layer_count: int,
        basis_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        self.strict_rows = strict_rows
        self.node_projection = nn.Linear(input_dim, hidden_dim)
        self.relation_projection = nn.Linear(input_dim, hidden_dim)
        self.row_type = nn.Parameter(torch.empty(hidden_dim))
        self.table_embedding = nn.Embedding(table_count, hidden_dim)
        self.layers = nn.ModuleList([
            BasisRGCNLayer(hidden_dim, relation_count, basis_count, dropout)
            for _ in range(layer_count)
        ])
        self.head = CellBinaryHead(hidden_dim, dropout)
        nn.init.normal_(self.row_type, std=0.02)

    def forward(
        self,
        node_embedding: torch.Tensor,
        relation_embedding: torch.Tensor,
        node_type: torch.Tensor,
        table_id: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        edge_relation: torch.Tensor,
        edge_hop: torch.Tensor,
        target_row: torch.Tensor,
        target_value: torch.Tensor,
        target_relation: torch.Tensor,
    ) -> torch.Tensor:
        node_state = self.node_projection(node_embedding)
        relation_state = self.relation_projection(relation_embedding)
        if self.strict_rows:
            row_mask = node_type == 0
            if row_mask.any():
                row_tables = table_id[row_mask]
                if (row_tables < 0).any():
                    raise ValueError("strict Row nodes require a table ID")
                node_state = node_state.clone()
                node_state[row_mask] = self.row_type + self.table_embedding(row_tables)
        layer_count = len(self.layers)
        for layer_index, layer in enumerate(self.layers):
            # Sampling starts at the target nodes, while propagation starts at
            # the outermost frontier and moves back toward those targets.
            hop = layer_count - layer_index - 1
            mask = edge_hop == hop
            node_state = layer(
                node_state,
                edge_source[mask],
                edge_target[mask],
                edge_relation[mask],
                relation_state,
            )
        return self.head(
            node_state[target_row],
            relation_state[target_relation],
            node_state[target_value],
        )
