#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:47:43 2026

@author: thaer
"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pipeline import to_sequence, properties, PROPS, AminoAcidEncoder
from common_bal import load_split
Xtr, Xte, ytr, yte, ltr, lte, feats = load_split()
out = {}
# worked example on three transformed values
vals = [0.62, -1.35, 2.08]
s = "".join(format(v, ".2f") for v in vals); seq = to_sequence(vals)
out["worked_example"] = {"values": vals, "string": s, "sequence": seq, "L": len(seq),
                         "props": {p: round(float(v), 3) for p, v in zip(PROPS, properties(seq))}}
# sequence lengths on the test split (final encoder)
c = np.load("cache/final.npz")
enc = AminoAcidEncoder(3).fit(Xtr, ytr)
assert list(enc.selected) == list(c["sel"])
seqs = enc.sequences(Xte); L = np.array([len(q) for q in seqs])
out["seq_len_test"] = {"min": int(L.min()), "mean": round(float(L.mean()), 2), "max": int(L.max())}
out["example_test_sequences"] = seqs[:3]
out["lambda_yj_selected"] = {feats[i]: round(float(enc.yj.lambdas_[i]), 4) for i in enc.selected}
# grouped per-category detection (test)
fam = lambda l: ("Benign" if l == "BenignTraffic" else "DDoS" if l.startswith("DDoS") else "DoS" if l.startswith("DoS") else
                 "Mirai" if l.startswith("Mirai") else "Reconnaissance" if l.startswith("Recon") or l == "VulnerabilityScan" else
                 "Spoofing" if l in ("DNS_Spoofing", "MITM-ArpSpoofing") else "Brute force" if l == "DictionaryBruteForce" else "Web-based")
an = json.load(open("results/07_analysis.json"))
g = {}
for n in ["RNN", "XGB"]:
    for l, v in an[f"per_label_{n}"].items():
        d = g.setdefault(fam(l), {"classes": set(), "n": 0, "RNN": 0, "XGB": 0}); d["classes"].add(l)
        if n == "RNN": d["n"] += v["n"]
        d[n] += v["detected"]
out["family_detection"] = {k: {"n_classes": len(v["classes"]), "n_test": v["n"], "RNN_rate": round(v["RNN"] / v["n"], 4), "XGB_rate": round(v["XGB"] / v["n"], 4)} for k, v in g.items()}
# subset composition by family (whole 50k and split)
raw = pd.read_csv("data/ciciot2023_balanced_50k.csv"); fm = raw.label.map(fam)
out["family_counts_total"] = fm.value_counts().to_dict()
out["train_counts"] = {"benign": int((ytr == 0).sum()), "attack": int((ytr == 1).sum())}
out["test_counts"] = {"benign": int((yte == 0).sum()), "attack": int((yte == 1).sum())}
json.dump(out, open("results/09_extras.json", "w"), indent=1)
print(json.dumps(out, indent=1))
# correlation heatmap
rho = pd.read_csv("results/07_spearman_props_vs_raw.csv", index_col=0)
fig, ax = plt.subplots(figsize=(13, 4.2))
im = ax.imshow(rho.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax.set_yticks(range(10)); ax.set_yticklabels(rho.index); ax.set_xticks(range(rho.shape[1])); ax.set_xticklabels(rho.columns, rotation=75, ha="right", fontsize=8)
plt.colorbar(im, ax=ax, label="Spearman $\\rho$"); ax.set_title("Correlation between the ten structural properties and the raw CICIoT2023 features (training split)")
plt.tight_layout(); plt.savefig("figs/corr_props_raw.png", dpi=170); plt.close()
