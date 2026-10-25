#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:25:55 2026

@author: thaer

Deep models (PyTorch). The ten properties are fed as a sequence of 10 steps with one
value per step: a step is a property position, not a point in time.
"""
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(4)

class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                            nn.AdaptiveMaxPool1d(1))
        s.h = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x):                       # x: (batch, 10, 1)
        return s.h(s.f(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)

class Recurrent(nn.Module):
    def __init__(s, cell):
        super().__init__()
        s.r = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell](1, 64, num_layers=2, batch_first=True)
        s.h = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x):
        out, _ = s.r(x)
        return s.h(out[:, -1, :]).squeeze(-1)

class Transformer(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Linear(1, 32); s.pos = nn.Parameter(torch.zeros(1, 10, 32))
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)
        s.enc = nn.TransformerEncoder(layer, num_layers=2)
        s.h = nn.Linear(32, 1)
    def forward(s, x):
        return s.h(s.enc(s.emb(x) + s.pos).mean(1)).squeeze(-1)

def build(name):
    return CNN() if name == "CNN" else Transformer() if name == "Transformer" else Recurrent(name)

DEEP = ["CNN", "RNN", "LSTM", "GRU", "Transformer"]

class DeepClassifier:
    """sklearn-like wrapper: Adam (lr 1e-3), batch 256, up to 50 epochs, early stopping
    (patience 5) on a stratified 10% validation part of the training data."""
    def __init__(s, name, seed=42, max_epochs=50, patience=5):
        s.name, s.seed, s.max_epochs, s.patience = name, seed, max_epochs, patience
    def fit(s, X, y):
        from sklearn.model_selection import train_test_split
        torch.manual_seed(s.seed); np.random.seed(s.seed)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, stratify=y, random_state=s.seed)
        s.model = build(s.name)
        opt = torch.optim.Adam(s.model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(-1); yt = torch.tensor(ytr, dtype=torch.float32)
        Xv = torch.tensor(Xva, dtype=torch.float32).unsqueeze(-1); yv = torch.tensor(yva, dtype=torch.float32)
        best, best_state, bad = 1e9, None, 0
        g = torch.Generator().manual_seed(s.seed)
        for ep in range(s.max_epochs):
            s.model.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), 256):
                b = perm[i:i + 256]
                opt.zero_grad(); l = lossf(s.model(Xt[b]), yt[b]); l.backward(); opt.step()
            s.model.eval()
            with torch.no_grad():
                vl = lossf(s.model(Xv), yv).item()
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in s.model.state_dict().items()}
            else:
                bad += 1
                if bad >= s.patience:
                    break
        s.model.load_state_dict(best_state); s.epochs_run = ep + 1
        return s
    def predict_proba(s, X):
        s.model.eval()
        with torch.no_grad():
            p = torch.sigmoid(s.model(torch.tensor(X, dtype=torch.float32).unsqueeze(-1))).numpy()
        return np.c_[1 - p, p]
    def n_params(s):
        return int(sum(p.numel() for p in s.model.parameters()))
