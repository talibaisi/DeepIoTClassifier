#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:45:11 2026

@author: thaer
"""
import pickle, json, numpy as np, matplotlib
matplotlib.use("Agg")
from common_bal import load_split
from pipeline import PROPS
Xtr, Xte, ytr, yte, ltr, lte, feats = load_split()
c = np.load("cache/final.npz")
def model(n): return pickle.load(open(f"models/{n}.pkl", "rb"))
def scaled(n):
    sc = model(n)["scaler"]; return sc.transform(c["A"]), sc.transform(c["B"])
CLASSICAL = ["DT", "RF", "NB", "LR", "GB", "XGB", "Bagging", "KNN", "CatBoost", "Stacking-LR", "Stacking-DT", "SVM"]
FILE = {"DT": "DT", "RF": "RF", "NB": "NB", "LR": "LR", "GB": "GB", "XGB": "XGB", "Bagging": "Bagging", "KNN": "KNN",
        "CatBoost": "CatBoost", "Stacking-LR": "StackingLR", "Stacking-DT": "StackingDT", "SVM": "SVM"}
def instance_index():
    """First test flow that is an attack and is detected by every model (used for all local explanations)."""
    ok = (yte == 1)
    for n in CLASSICAL + ["RNN"]:
        m = model(n); _, B = scaled(n)
        ok &= (m["model"].predict_proba(B)[:, 1] >= 0.5)
    return int(np.where(ok)[0][0])
