#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:32:41 2026

@author: thaer

Ablation (R1.5, R2.5): same split, same 10 folds, same models (XGBoost, RNN).
Arm A: raw CICIoT2023 features (Yeo-Johnson + standardisation), 39 dims.
Arm B: all 39 features encoded (Yeo-Johnson -> encoding -> 10 properties), 10 dims.
Arm C: proposed (Yeo-Johnson -> top-k selection -> encoding -> 10 properties), 10 dims.
"""
import sys, os, json, time, numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from common_bal import load_split, folds
from common import metrics
from pipeline import AminoAcidEncoder
from deep import DeepClassifier
from xgboost import XGBClassifier
Xtr, Xte, ytr, yte, ltr, lte, feats = load_split()
F = folds(ltr) + [(None, None)]
os.makedirs("results/ablation", exist_ok=True); os.makedirs("cache_abl", exist_ok=True)
arm, model = sys.argv[1], sys.argv[2]
for i, (a, b) in enumerate(F):
    tag = "test" if a is None else f"fold{i}"
    out = f"results/ablation/{arm}_{model}_{tag}.json"
    if os.path.exists(out): continue
    XA, yA = (Xtr, ytr) if a is None else (Xtr[a], ytr[a])
    XB, yB = (Xte, yte) if a is None else (Xtr[b], ytr[b])
    t0 = time.perf_counter()
    if arm == "A":
        pt = PowerTransformer().fit(XA); A, t1 = pt.transform(XA), time.perf_counter(); B = pt.transform(XB)
        pre_ms = 1000 * (time.perf_counter() - t1) / len(XB)
    elif arm == "C":
        c = np.load(f"cache/{'final' if a is None else tag}.npz"); A, B = c["A"], c["B"]; pre_ms = float(c["ms_per_flow"])
    else:
        cf = f"cache_abl/B_{tag}.npz"
        if not os.path.exists(cf):
            enc = AminoAcidEncoder(len(feats)).fit(XA, yA)
            t1 = time.perf_counter(); B_ = enc.transform(XB); ms = 1000 * (time.perf_counter() - t1) / len(XB)
            np.savez(cf, A=enc.transform(XA), B=B_, ms=ms)
        c = np.load(cf); A, B, pre_ms = c["A"], c["B"], float(c["ms"])
    sc = StandardScaler().fit(A)
    m = XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, n_jobs=4) if model == "XGB" else DeepClassifier("RNN")
    m.fit(sc.transform(A), yA)
    r = metrics(yB, m.predict_proba(sc.transform(B))[:, 1]); r["dims"] = int(A.shape[1]); r["preprocess_ms_per_flow"] = pre_ms
    json.dump(r, open(out, "w")); print(arm, model, tag, "acc %.4f mcc %.4f" % (r["accuracy"], r["mcc"]), flush=True)
