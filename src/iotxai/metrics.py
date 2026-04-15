from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import gmean
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_score=None) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    g_mean = float(np.sqrt(tpr * tnr))
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score) if y_score is not None else None

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1Score": float(f1),
        "FPR": float(fpr),
        "ROCAUC": float(roc_auc) if roc_auc is not None else None,
        "MCC": float(mcc),
        "GMean": float(g_mean),
        "ConfusionMatrix": cm.tolist(),
    }


def aggregate_metric_dicts(metric_list):
    keys = ["Accuracy", "F1Score", "FPR", "Recall"]
    out = {}
    for key in keys:
        values = np.array([m[key] for m in metric_list], dtype=float)
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return out
