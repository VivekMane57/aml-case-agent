"""
GraphSAGE encoder + edge classifier.

Node identity handling: each account has a learnable ID embedding, but during training a
random fraction (id_drop) of nodes get a shared "unknown" vector instead. At inference,
accounts never seen in training also get the "unknown" vector. This forces the model to
predict from structure + edge features rather than memorising account IDs, and gives
brand-new accounts a sensible (trained) representation instead of an untrained one.

id_drop >= 1.0 is the control setting: IDs are never used (train or inference), so the
model sees only edge features and graph structure.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int = 64, hidden_dim: int = 64,
                 num_layers: int = 2, id_drop: float = 0.5):
        super().__init__()
        self.node_embed = nn.Embedding(num_nodes, embed_dim)
        nn.init.normal_(self.node_embed.weight, std=0.1)

        self.unk = nn.Parameter(torch.zeros(1, embed_dim))            # shared "unknown account" vector
        self.register_buffer("seen", torch.zeros(num_nodes, dtype=torch.bool))
        self.id_drop = id_drop

        self.convs = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim

    def forward(self, n_id: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_embed(n_id)
        if self.training:
            drop = torch.rand(x.size(0), device=x.device) < self.id_drop
        elif self.id_drop >= 1.0:
            drop = torch.ones(x.size(0), dtype=torch.bool, device=x.device)   # control run: never use IDs
        else:
            drop = ~self.seen[n_id]            # accounts never seen in train -> unknown
        x = torch.where(drop.unsqueeze(1), self.unk.expand_as(x), x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x


class EdgeClassifier(nn.Module):
    def __init__(self, num_nodes: int, edge_feat_dim: int, embed_dim: int = 64,
                 hidden_dim: int = 64, id_drop: float = 0.5):
        super().__init__()
        self.encoder = GraphSAGEEncoder(num_nodes, embed_dim=embed_dim,
                                        hidden_dim=hidden_dim, id_drop=id_drop)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, n_id, edge_index, src_local, dst_local, edge_attr):
        h = self.encoder(n_id, edge_index)
        h_src, h_dst = h[src_local], h[dst_local]
        return self.classifier(torch.cat([h_src, h_dst, edge_attr], dim=-1)).squeeze(-1)  