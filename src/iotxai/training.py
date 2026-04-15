from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .metrics import compute_metrics


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).reshape(len(X), -1, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class TrainResult:
    metrics: Dict[str, object]
    scaler: StandardScaler
    model: object


def fit_classical_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    model.fit(X_train, y_train)
    return model


def predict_classical(model, X_test: pd.DataFrame):
    y_pred = model.predict(X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_score = 1.0 / (1.0 + np.exp(-scores))
    return y_pred, y_score


def prepare_deep_arrays(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train)
    X_test_np = scaler.transform(X_test)
    return X_train_np.astype(np.float32), X_test_np.astype(np.float32), scaler


def train_torch_model(
    model: torch.nn.Module,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    batch_size: int = 256,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
) -> TrainResult:
    X_train_np, X_test_np, scaler = prepare_deep_arrays(X_train, X_test)
    ds_train = SequenceDataset(X_train_np, y_train.to_numpy(dtype=np.float32))
    ds_test = SequenceDataset(X_test_np, y_test.to_numpy(dtype=np.float32))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_f1 = -1.0

    for _ in range(epochs):
        model.train()
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        metrics = evaluate_torch_model(model, dl_test, device=device)
        if metrics["F1Score"] > best_f1:
            best_f1 = metrics["F1Score"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate_torch_model(model, dl_test, device=device)
    return TrainResult(metrics=metrics, scaler=scaler, model=model)


def evaluate_torch_model(model: torch.nn.Module, data_loader: DataLoader, device: str = "cpu") -> Dict[str, object]:
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            pred = (probs >= 0.5).astype(int)
            y_true.extend(yb.numpy().astype(int).tolist())
            y_pred.extend(pred.tolist())
            y_score.extend(probs.tolist())
    return compute_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)


def get_device(config_device: str = "auto") -> str:
    if config_device != "auto":
        return config_device
    return "cuda" if torch.cuda.is_available() else "cpu"
