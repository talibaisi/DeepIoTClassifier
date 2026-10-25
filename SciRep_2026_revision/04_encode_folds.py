#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:06:31 2026

@author: thaer

Fit the encoder (Yeo-Johnson + top-k selection) on each training part and cache
the ten properties for the CV folds and for the final train/test split.
"""
import json, os, time, numpy as np
from common_bal import load_split, folds
from pipeline import AminoAcidEncoder
K = json.load(open("results/03_select_k.json"))["chosen_k"]
Xtr, Xte, ytr, yte, ltr, lte, feats = load_split()
jobs = [(f"fold{i}", a, b) for i, (a, b) in enumerate(folds(ltr))] + [("final", None, None)]
for name, a, b in jobs:
    f = f"cache/{name}.npz"
    if os.path.exists(f):
        continue
    A_X, A_y = (Xtr, ytr) if a is None else (Xtr[a], ytr[a])
    B_X = Xte if a is None else Xtr[b]
    enc = AminoAcidEncoder(K).fit(A_X, A_y)
    t0 = time.perf_counter(); B = enc.transform(B_X); t = time.perf_counter() - t0
    A = enc.transform(A_X)
    np.savez(f, A=A, B=B, sel=enc.selected, mi=enc.mi, ms_per_flow=1000 * t / len(B_X))
    print(name, "selected:", [feats[i] for i in enc.selected], "%.3f ms/flow" % (1000 * t / len(B_X)), flush=True)
