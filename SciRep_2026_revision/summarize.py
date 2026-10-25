#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:32:24 2026

@author: thaer
"""
import json,glob,numpy as np
order=["KNN","NB","DT","LR","SVM","RF","GB","XGB","CatBoost","Bagging","Stacking-LR","Stacking-DT","CNN","RNN","LSTM","GRU","Transformer"]
print("TEST"); 
for n in order:
    try: r=json.load(open(f"results/test/{n}.json"))
    except: continue
    print(f"{n:12s} acc {r['accuracy']:.4f} prec {r['precision']:.4f} rec {r['recall']:.4f} f1 {r['f1']:.4f} fpr {r['fpr']:.4f} auc {r['roc_auc']:.4f} mcc {r['mcc']:.4f} gm {r['g_mean']:.4f} {r['confusion_matrix']} {r.get('n_params','')}")
print("CV")
for n in order:
    fs=sorted(glob.glob(f"results/cv/{n}_fold*.json"))
    if len(fs)<10: print(n,"incomplete",len(fs)); continue
    R=[json.load(open(f)) for f in fs]
    s=lambda k: (100*np.mean([r[k] for r in R]),100*np.std([r[k] for r in R]))
    print(f"{n:12s} acc %.2f+-%.2f f1 %.2f+-%.2f fpr %.2f+-%.2f rec %.2f+-%.2f mcc %.4f+-%.4f"%(*s('accuracy'),*s('f1'),*s('fpr'),*s('recall'),np.mean([r['mcc'] for r in R]),np.std([r['mcc'] for r in R])))
