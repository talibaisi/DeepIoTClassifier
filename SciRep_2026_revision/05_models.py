#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:07:42 2026

@author: thaer

10-fold CV on the training split (properties computed per fold) + final test.
Usage: python 05_models.py <model> [<model> ...]   (resumable)
"""
import sys, os, json, time, pickle, numpy as np
from sklearn.preprocessing import StandardScaler
from common_bal import load_split, folds
from common import metrics, classical_models
from deep import DeepClassifier, DEEP
Xtr, Xte, ytr, yte, ltr, lte, feats = load_split()
F = folds(ltr); CL = classical_models()
os.makedirs("results/cv", exist_ok=True); os.makedirs("results/test", exist_ok=True)
def make(n): return DeepClassifier(n) if n in DEEP else CL[n]()
def fit_eval(n, A, ya, B):
    sc = StandardScaler().fit(A); m = make(n)
    t0 = time.perf_counter(); m.fit(sc.transform(A), ya); tf = time.perf_counter() - t0
    return sc, m, m.predict_proba(sc.transform(B))[:, 1], tf
for n in sys.argv[1:]:
    for i, (a, b) in enumerate(F):
        f = f"results/cv/{n}_fold{i}.json"
        if os.path.exists(f): continue
        c = np.load(f"cache/fold{i}.npz")
        _, m, p, tf = fit_eval(n, c["A"], ytr[a], c["B"])
        r = metrics(ytr[b], p); r["fit_seconds"] = tf; json.dump(r, open(f, "w"))
        print(n, i, "acc %.4f mcc %.4f" % (r["accuracy"], r["mcc"]), flush=True)
    f = f"results/test/{n}.json"
    if not os.path.exists(f):
        c = np.load("cache/final.npz")
        sc, m, p, tf = fit_eval(n, c["A"], ytr, c["B"])
        r = metrics(yte, p); r["fit_seconds"] = tf
        pred = (p >= 0.5).astype(int)
        r["per_label"] = {l: {"n": int((lte == l).sum()), "pred_attack": int(pred[lte == l].sum())} for l in sorted(set(lte))}
        if n in DEEP: r["n_params"] = m.n_params(); r["epochs_run"] = m.epochs_run
        json.dump(r, open(f, "w"), indent=1)
        pickle.dump({"scaler": sc, "model": m}, open(f"models/{n}.pkl", "wb"))
        print(n, "TEST acc %.4f f1 %.4f mcc %.4f fpr %.4f" % (r["accuracy"], r["f1"], r["mcc"], r["fpr"]), r["confusion_matrix"], flush=True)
