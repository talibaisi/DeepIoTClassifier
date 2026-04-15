from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler


@dataclass
class FeatureSelectionResult:
    chi2_features: List[str]
    cnn_guided_features: List[str]


def chi2_filter(df: pd.DataFrame, y: pd.Series, top_k: int = 40) -> List[str]:
    top_k = min(top_k, df.shape[1])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df)
    selector = SelectKBest(score_func=chi2, k=top_k)
    selector.fit(X_scaled, y)
    mask = selector.get_support()
    return list(df.columns[mask])


def cnn_guided_ranking(
    df: pd.DataFrame,
    y: pd.Series,
    top_k: int = 20,
    epochs: int = 10,
    batch_size: int = 256,
    random_state: int = 42,
) -> List[str]:
    """
    Lightweight approximation of a CNN-guided feature selector.
    A 1D CNN is trained over the raw scaled features, and mean absolute
    input gradients are used to rank features.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(random_state)
    X = df.to_numpy(dtype=np.float32)
    y_np = y.to_numpy(dtype=np.float32)

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    X_tensor = torch.tensor(X[:, None, :], dtype=torch.float32)
    y_tensor = torch.tensor(y_np[:, None], dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    class SmallCNN(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = SmallCNN(df.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    sample_idx = rng.choice(len(X), size=min(2048, len(X)), replace=False)
    xb = torch.tensor(X[sample_idx][:, None, :], dtype=torch.float32, requires_grad=True)
    logits = model(xb)
    score = logits.sigmoid().mean()
    score.backward()
    grads = xb.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    ranked_idx = np.argsort(-grads)[: min(top_k, len(grads))]
    return [df.columns[i] for i in ranked_idx]


def hybrid_feature_selector(
    df: pd.DataFrame,
    y: pd.Series,
    chi2_top_k: int = 40,
    cnn_top_k: int = 20,
    epochs: int = 10,
    batch_size: int = 256,
    random_state: int = 42,
) -> FeatureSelectionResult:
    chi2_cols = chi2_filter(df, y, top_k=chi2_top_k)
    guided_cols = cnn_guided_ranking(
        df[chi2_cols],
        y,
        top_k=min(cnn_top_k, len(chi2_cols)),
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    )
    return FeatureSelectionResult(chi2_features=chi2_cols, cnn_guided_features=guided_cols)
