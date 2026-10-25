#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:04:38 2026

@author: thaer

Choose k (number of selected features before encoding) by 5-fold CV on the
training split only, using XGBoost as the reference model.
"""
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef
from common_bal import load_split, SEED
from pipeline import AminoAcidEncoder
Xtr, Xte, ytr, yte, ltr, lte, feats = load_split()
res = {}
for k in [3, 5, 8, 10]:
    scores = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ltr):
        enc = AminoAcidEncoder(k).fit(Xtr[a], ytr[a])
        A, B = enc.transform(Xtr[a]), enc.transform(Xtr[b])
        m = XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=SEED, n_jobs=4).fit(A, ytr[a])
        scores.append(matthews_corrcoef(ytr[b], m.predict(B)))
    res[k] = [float(np.mean(scores)), float(np.std(scores))]
    print(k, res[k], flush=True)
best = max(res, key=lambda k: res[k][0])
json.dump({"cv_mcc_by_k": res, "chosen_k": best}, open("results/03_select_k.json", "w"), indent=2)
print("chosen k =", best)
