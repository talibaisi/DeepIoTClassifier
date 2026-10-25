#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:49:13 2026

@author: thaer
"""
import json, glob, numpy as np
CL = ["KNN","NB","DT","LR","SVM","RF","GB","XGB","CatBoost","Bagging","Stacking-LR","Stacking-DT"]
DL = ["CNN","RNN","LSTM","GRU","Transformer"]
def row(n):
    r = json.load(open(f"results/test/{n}.json")); cm = r["confusion_matrix"]
    return f"{n} & {r['accuracy']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} & {r['fpr']:.4f} & {r['roc_auc']:.4f} & {r['mcc']:.4f} & {r['g_mean']:.4f} & [[{cm[0][0]:,}, {cm[0][1]:,}], [{cm[1][0]:,}, {cm[1][1]:,}]] \\\\".replace(",", "{,}").replace("{,}}", ",}").replace("]{,} [", "], [")
def cv(n):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"results/cv/{n}_fold*.json"))]
    s = lambda k: (100*np.mean([r[k] for r in R]), 100*np.std([r[k] for r in R]))
    m = (np.mean([r['mcc'] for r in R]), np.std([r['mcc'] for r in R]))
    return f"{n} & {s('accuracy')[0]:.2f} $\\pm$ {s('accuracy')[1]:.2f} & {s('f1')[0]:.2f} $\\pm$ {s('f1')[1]:.2f} & {s('fpr')[0]:.2f} $\\pm$ {s('fpr')[1]:.2f} & {s('recall')[0]:.2f} $\\pm$ {s('recall')[1]:.2f} & {m[0]:.3f} $\\pm$ {m[1]:.3f} \\\\"
print("%%TEST"); [print(row(n)) for n in CL]; print("%%DL"); [print(row(n)) for n in DL]
print("%%CV"); [print(cv(n)) for n in CL+DL]
def abl(arm, m):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"results/ablation/{arm}_{m}_fold*.json"))]; t = json.load(open(f"results/ablation/{arm}_{m}_test.json"))
    return (np.mean([r['mcc'] for r in R]), np.std([r['mcc'] for r in R]), t)
print("%%ABL")
for arm, name in [("A","Raw CICIoT2023 features"),("B","Encoding of all 39 features"),("C","Proposed: selection, then encoding")]:
    for m in ["XGB","RNN"]:
        a, s, t = abl(arm, m)
        print(f"{name} & {m} & {t['dims']} & {a:.3f} $\\pm$ {s:.3f} & {100*t['accuracy']:.2f} & {100*t['f1']:.2f} & {t['mcc']:.3f} & {100*t['fpr']:.2f} & {t['preprocess_ms_per_flow']:.3f} \\\\")
