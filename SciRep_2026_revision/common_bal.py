#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:04:38 2026

@author: thaer

Data split and metrics for the balanced 50,000-flow subset.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from common import metrics, classical_models, SEED
N_FOLDS = 10
def load_split():
    raw = pd.read_csv("data/ciciot2023_balanced_50k.csv")
    feats = [c for c in raw.columns if c != "label"]
    X = raw[feats].values.astype(float)
    lab = raw["label"].values
    y = (lab != "BenignTraffic").astype(int)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=lab, random_state=SEED)
    return X[tr], X[te], y[tr], y[te], lab[tr], lab[te], feats
def folds(lab):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(lab)), lab))
