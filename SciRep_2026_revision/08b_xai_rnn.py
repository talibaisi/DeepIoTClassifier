#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:45:57 2026

@author: thaer

RNN explanations: SHAP (GradientExplainer), LIME, Captum Integrated Gradients, Alibi anchors,
weights, hidden states, and representative flows (R2.8).
"""
import sys, json, os, numpy as np, torch, matplotlib.pyplot as plt
from xai_common import *
part = sys.argv[1]
idx = json.load(open("results/xai_instance.json"))["idx"]
m = model("RNN"); sc, clf = m["scaler"], m["model"]; net = clf.model.eval()
A, B = sc.transform(c["A"]), sc.transform(c["B"])
T = lambda X: torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
class Prob(torch.nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, x): return torch.sigmoid(s.n(x)).unsqueeze(-1)
pnet = Prob(net).eval()
outf = "results/08_xai_rnn.json"
out = json.load(open(outf)) if os.path.exists(outf) else {}
rng = np.random.default_rng(42)
if part == "shap":
    import shap
    bg = T(A[rng.choice(len(A), 200, replace=False)]); s = rng.choice(len(B), 500, replace=False)
    ex = shap.GradientExplainer(pnet, bg); sv = np.array(ex.shap_values(T(B[s])))
    sv = sv.reshape(len(s), 10)
    plt.figure(); shap.summary_plot(sv, B[s], feature_names=PROPS, show=False); plt.tight_layout(); plt.savefig("figs/shap_summary_RNN.png", dpi=150); plt.close()
    plt.figure(); shap.summary_plot(sv, B[s], feature_names=PROPS, plot_type="bar", show=False); plt.tight_layout(); plt.savefig("figs/shap_bar_RNN.png", dpi=150); plt.close()
    one = np.array(ex.shap_values(T(B[idx:idx + 1]))).reshape(10)
    base = float(pnet(bg).mean())
    shap.force_plot(base, np.round(one, 4), np.round(B[idx], 3), feature_names=[str(i + 1) for i in range(10)], matplotlib=True, show=False, figsize=(14, 3))
    plt.savefig("figs/shap_force_RNN.png", dpi=150, bbox_inches="tight"); plt.close()
    out["shap_mean_abs"] = {p: round(float(v), 4) for p, v in zip(PROPS, np.abs(sv).mean(0))}
    out["shap_instance"] = {p: round(float(v), 4) for p, v in zip(PROPS, one)}
    print(sorted(out["shap_mean_abs"].items(), key=lambda kv: -kv[1]))
if part == "lime":
    from lime.lime_tabular import LimeTabularExplainer
    ex = LimeTabularExplainer(A, feature_names=PROPS, class_names=["Benign", "Attack"], discretize_continuous=True, random_state=42)
    e = ex.explain_instance(B[idx], clf.predict_proba, num_features=10, num_samples=3000)
    fig = e.as_pyplot_figure(label=1); fig.set_size_inches(6, 4); plt.title("LIME - RNN (attack class)"); plt.tight_layout()
    fig.savefig("figs/lime_explaination_RNN.png", dpi=150); plt.close(fig)
    out["lime"] = [(r, round(float(w), 5)) for r, w in e.as_list(label=1)]
    print(out["lime"])
if part == "captum":
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(lambda x: torch.sigmoid(net(x)))
    base = T(A.mean(0, keepdims=True))
    s = rng.choice(len(B), 1000, replace=False)
    att = ig.attribute(T(B[s]), baselines=base, n_steps=50).squeeze(-1).detach().numpy()
    one = ig.attribute(T(B[idx:idx + 1]), baselines=base, n_steps=50).squeeze(-1).detach().numpy()[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    o = np.argsort(np.abs(att).mean(0))
    ax[0].barh(np.array(PROPS)[o], np.abs(att).mean(0)[o], color="#4C72B0"); ax[0].set_title("Mean |IG attribution| (1,000 test flows)")
    ax[1].barh(PROPS, one, color=["#55A868" if v > 0 else "#C44E52" for v in one]); ax[1].set_title("IG attribution, explained attack flow"); ax[1].axvline(0, color="k", lw=0.6)
    plt.tight_layout(); plt.savefig("figs/captum_IG_RNN.png", dpi=150); plt.close()
    out["ig_mean_abs"] = {p: round(float(v), 4) for p, v in zip(PROPS, np.abs(att).mean(0))}
    out["ig_instance"] = {p: round(float(v), 4) for p, v in zip(PROPS, one)}
    # representative flows (R2.8): a detected flood, a detected Mirai flow, and a missed reconnaissance flow
    prob = clf.predict_proba(B)[:, 1]
    picks = {}
    for name, cond in [("DDoS-UDP_Flood (detected)", (lte == "DDoS-UDP_Flood") & (prob >= .5)),
                       ("Mirai-greeth_flood (detected)", (lte == "Mirai-greeth_flood") & (prob >= .5)),
                       ("Recon-PortScan (missed)", (lte == "Recon-PortScan") & (prob < .5)),
                       ("BenignTraffic (correct)", (lte == "BenignTraffic") & (prob < .5))]:
        j = int(np.where(cond)[0][0]); a = ig.attribute(T(B[j:j + 1]), baselines=base, n_steps=50).squeeze(-1).detach().numpy()[0]
        top = np.argsort(-np.abs(a))[:3]
        picks[name] = {"prob_attack": round(float(prob[j]), 4),
                       "raw_selected": {feats[i]: float(Xte[j, i]) for i in c["sel"]},
                       "top_ig": [(PROPS[t], round(float(a[t]), 4), round(float(c["B"][j, t]), 4)) for t in top]}
    out["representative_flows"] = picks
    print(json.dumps(picks, indent=1))
if part == "anchors":
    from alibi.explainers import AnchorTabular
    ex = AnchorTabular(lambda X: (clf.predict_proba(X)[:, 1] >= .5).astype(int), feature_names=PROPS, seed=42)
    ex.fit(A, disc_perc=(25, 50, 75))
    att_idx = np.where(yte == 1)[0][:15].tolist() + np.where(yte == 0)[0][:15].tolist()
    res = []
    for j in att_idx:
        e = ex.explain(B[j], threshold=0.95)
        res.append({"label": str(lte[j]), "anchor": e.anchor, "precision": float(e.precision), "coverage": float(e.coverage)})
    P = np.array([r["precision"] for r in res]); C = np.array([r["coverage"] for r in res])
    plt.figure(figsize=(5.5, 4)); plt.scatter(C, P, c=["#C44E52" if r["label"] != "BenignTraffic" else "#4C72B0" for r in res])
    plt.xlabel("Coverage"); plt.ylabel("Precision"); plt.title("Anchors for the RNN (red: attack, blue: benign)"); plt.ylim(0.8, 1.01); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("figs/alibi_anchor_tabular_RNN.png", dpi=150); plt.close()
    out["anchors"] = res; out["anchor_precision_min"] = float(P.min()); out["anchor_precision_mean"] = float(P.mean())
    print(P.min(), P.mean(), C.mean(), res[0], res[-1])
if part == "internals":
    sd = net.state_dict()
    for l in [0, 1]:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        im = ax[0].imshow(sd[f"r.weight_ih_l{l}"].numpy(), aspect="auto", cmap="coolwarm"); ax[0].set_title(f"Layer {l}: input-hidden"); plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(sd[f"r.weight_hh_l{l}"].numpy(), aspect="auto", cmap="coolwarm"); ax[1].set_title(f"Layer {l}: hidden-hidden"); plt.colorbar(im, ax=ax[1])
        ax[2].hist((sd[f"r.bias_ih_l{l}"] + sd[f"r.bias_hh_l{l}"]).numpy(), bins=20, color="#4C72B0"); ax[2].set_title(f"Layer {l}: bias distribution")
        plt.tight_layout(); plt.savefig(f"figs/weights_visual_RNN_layer{l}.png", dpi=150); plt.close()
    with torch.no_grad():
        hs, _ = net.r(T(B[idx:idx + 1]))
    plt.figure(figsize=(9, 3.6)); plt.imshow(hs[0].numpy().T, aspect="auto", cmap="viridis")
    plt.xticks(range(10), PROPS, rotation=45, ha="right"); plt.ylabel("Hidden unit"); plt.title("RNN hidden state (last layer) at each property position"); plt.colorbar()
    plt.tight_layout(); plt.savefig("figs/hidden_states_RNN.png", dpi=150); plt.close(); print("ok")
json.dump(out, open(outf, "w"), indent=1)
