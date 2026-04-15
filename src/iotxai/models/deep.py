from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DeepBatch:
    x: torch.Tensor
    y: torch.Tensor


class CNN1DModel(nn.Module):
    def __init__(self, seq_len: int = 10, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [B, T, 1] -> [B, 1, T]
        x = x.transpose(1, 2)
        return self.features(x).squeeze(-1)


class RecurrentClassifier(nn.Module):
    def __init__(self, kind: str, input_size: int = 1, hidden_dim: int = 32, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.kind = kind
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, hidden = self.rnn(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_cols = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(position * div_term[:cos_cols])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, seq_len: int = 10, d_model: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model=d_model, max_len=seq_len + 2)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.fc(x).squeeze(-1)


class GNNClassifier(nn.Module):
    """
    Optional simple GNN over a chain graph of 10 encoded features.
    Requires torch_geometric.
    """
    def __init__(self, hidden_dim: int = 32, num_node_features: int = 1):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, 1)
        self.global_mean_pool = global_mean_pool

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


def build_deep_model(name: str, hidden_dim: int = 32, num_layers: int = 3, dropout: float = 0.2):
    if name == "CNN":
        return CNN1DModel(hidden_dim=hidden_dim, dropout=dropout)
    if name in {"RNN", "LSTM", "GRU"}:
        return RecurrentClassifier(kind=name, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if name == "Transformer":
        return TransformerClassifier(d_model=hidden_dim, num_layers=max(1, min(num_layers, 4)), dropout=dropout)
    if name == "GNN":
        return GNNClassifier(hidden_dim=hidden_dim)
    raise ValueError(f"Unsupported deep model: {name}")
