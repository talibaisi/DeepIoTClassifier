#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:25:55 2026

@author: thaer

Shared settings, data loading, metrics and model definitions.
"""
import json, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)

SEED = 42
N_FOLDS = 10
PROPS = ["MolecularWeight", "Aromaticity", "InstabilityIndex", "IsoelectricPoint",
         "AlphaHelix", "ReducedCysteines", "DisulfideBridges", "Gravy", "BetaTurn", "BetaStrand"]

def load_split(kind="encoded"):
    """Returns X_train, X_test, y_train, y_test, label_train, label_test.
    The same 80/20 split (stratified on the 34 original labels, seed 42) is used for
    every experiment, for both the encoded and the raw features."""
    enc = pd.read_csv("data/encoded_50k.csv")
    raw = pd.read_csv("data/ciciot2023_subset_50k.csv")
    idx = np.arange(len(enc))
    tr, te = train_test_split(idx, test_size=0.2, stratify=enc["label"], random_state=SEED)
    if kind == "encoded":
        X = enc[PROPS].values
    else:  # raw CICIoT2023 features
        X = raw.drop(columns=["label"]).values.astype(float)
    y, lab = enc["y"].values, enc["label"].values
    return X[tr], X[te], y[tr], y[te], lab[tr], lab[te]

def folds(y_label):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y_label)), y_label))

def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "roc_auc": roc_auc_score(y, prob), "mcc": matthews_corrcoef(y, pred),
            "g_mean": float(np.sqrt(tpr * tnr)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}

#==================================================
#classical models
def classical_models():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                  BaggingClassifier, StackingClassifier)
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    base = lambda: [("rf", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=4)),
                    ("xgb", XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED, n_jobs=4)),
                    ("cat", CatBoostClassifier(iterations=200, verbose=0, random_seed=SEED, thread_count=4))]
    return {
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=4),
        "NB": lambda: GaussianNB(),
        "DT": lambda: DecisionTreeClassifier(random_state=SEED),
        "LR": lambda: LogisticRegression(max_iter=1000),
        "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=SEED),
        "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=4),
        "GB": lambda: GradientBoostingClassifier(random_state=SEED),
        "XGB": lambda: XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED, n_jobs=4),
        "CatBoost": lambda: CatBoostClassifier(iterations=200, verbose=0, random_seed=SEED, thread_count=4),
        "Bagging": lambda: BaggingClassifier(DecisionTreeClassifier(random_state=SEED), n_estimators=10, random_state=SEED, n_jobs=4),
        "Stacking-LR": lambda: StackingClassifier(base(), final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=1),
        "Stacking-DT": lambda: StackingClassifier(base(), final_estimator=DecisionTreeClassifier(random_state=SEED), cv=5, n_jobs=1),
    }

def preprocessor():
    """Yeo-Johnson transformation followed by standardisation (fitted on training data only)."""
    from sklearn.preprocessing import PowerTransformer
    return PowerTransformer(method="yeo-johnson", standardize=True)
