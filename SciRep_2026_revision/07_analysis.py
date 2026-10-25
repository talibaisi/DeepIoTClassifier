#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:43:38 2026

@author: thaer

R1.4 correlations, R2.6.1 per-category detection, R2.4 RNN architecture (Table 1), compute of the deep models (Table 9).
"""
import json, pickle, io, time, platform, os, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from common_bal import load_split
from pipeline import PROPS
from deep import DEEP
Xtr, Xte, ytr, yte, ltr, lte, feats = load_split()
c = np.load("cache/final.npz"); A, B, sel = c["A"], c["B"], c["sel"]
out = {"selected_features": [feats[i] for i in sel], "mi_selected": [float(c["mi"][i]) for i in sel]}
#==================================================
#correlations between each property and each raw feature (training split)
rho = np.zeros((10, len(feats)))
for i in range(10):
    for j in range(len(feats)):
        r = spearmanr(A[:, i], Xtr[:, j]).correlation
        rho[i, j] = 0.0 if np.isnan(r) else r
pd.DataFrame(rho, index=PROPS, columns=feats).to_csv("results/07_spearman_props_vs_raw.csv")
out["corr_with_selected"] = {p: {feats[j]: round(float(rho[i, j]), 3) for j in sel} for i, p in enumerate(PROPS)}
out["top3_raw_per_property"] = {p: [(feats[j], round(float(rho[i, j]), 3)) for j in np.argsort(-np.abs(rho[i]))[:3]] for i, p in enumerate(PROPS)}
# separation of each property between classes (Cliff's delta-like AUC on test)
from sklearn.metrics import roc_auc_score
out["univariate_auc_test"] = {p: round(float(max(roc_auc_score(yte, B[:, i]), 1 - roc_auc_score(yte, B[:, i]))), 3) for i, p in enumerate(PROPS)}
#==================================================
#per-category detection on the test split (RNN and XGB)
for n in ["RNN", "XGB"]:
    r = json.load(open(f"results/test/{n}.json"))
    out[f"per_label_{n}"] = {l: {"n": v["n"], "detected": v["pred_attack"], "rate": round(v["pred_attack"] / v["n"], 4)} for l, v in r["per_label"].items()}
#==================================================
#RNN architecture and compute for deep models
comp = {}
for n in DEEP:
    m = pickle.load(open(f"models/{n}.pkl", "rb"))["model"]; net = m.model.eval()
    buf = io.BytesIO(); torch.save(net.state_dict(), buf)
    x1 = torch.tensor(B[:1], dtype=torch.float32).unsqueeze(-1)
    xb = torch.tensor(B[:1000], dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        for _ in range(100): net(x1)
        t0 = time.perf_counter()
        for i in range(1000): net(torch.tensor(B[i:i+1], dtype=torch.float32).unsqueeze(-1))
        lat = (time.perf_counter() - t0) / 1000 * 1000
        t0 = time.perf_counter(); net(xb); thr = 1000 / (time.perf_counter() - t0)
    comp[n] = {"params": int(sum(p.numel() for p in net.parameters())), "size_kb": round(len(buf.getvalue()) / 1024, 1),
               "latency_ms_batch1": round(lat, 3), "throughput_flows_per_s_batch1000": int(thr)}
    if n == "RNN":
        out["rnn_architecture"] = str(net)
        out["rnn_layer_params"] = {k: int(v.numel()) for k, v in net.state_dict().items()}
out["compute"] = comp
out["encoding_ms_per_flow"] = float(c["ms_per_flow"])
out["machine"] = {"platform": platform.platform(), "processor": platform.machine(), "cpus": os.cpu_count(), "torch": torch.__version__, "threads": torch.get_num_threads()}
json.dump(out, open("results/07_analysis.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["selected_features", "corr_with_selected", "top3_raw_per_property", "univariate_auc_test", "compute", "encoding_ms_per_flow", "machine"]}, indent=1))
print(out["rnn_architecture"]); print(out["rnn_layer_params"])
