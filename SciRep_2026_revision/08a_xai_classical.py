#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:45:25 2026

@author: thaer

LIME (12 classical models), SHAP (XGBoost, GB, CatBoost), ELI5 (DT, RF, LR, GB, XGB).
"""
import sys, json, os, numpy as np, matplotlib.pyplot as plt
from xai_common import *
from lime.lime_tabular import LimeTabularExplainer
part = sys.argv[1]
idx = json.load(open("results/xai_instance.json"))["idx"] if os.path.exists("results/xai_instance.json") else None
if idx is None:
    idx = instance_index(); json.dump({"idx": idx, "label": str(lte[idx])}, open("results/xai_instance.json", "w"))
out = json.load(open("results/08_xai_classical.json")) if os.path.exists("results/08_xai_classical.json") else {}
if part == "lime":
    for n in CLASSICAL:
        A, B = scaled(n); m = model(n)["model"]
        ex = LimeTabularExplainer(A, feature_names=PROPS, class_names=["Benign", "Attack"], discretize_continuous=True, random_state=42)
        e = ex.explain_instance(B[idx], m.predict_proba, num_features=10, num_samples=3000)
        fig = e.as_pyplot_figure(label=1); fig.set_size_inches(6, 4); plt.title(f"LIME - {n} (attack class)"); plt.tight_layout()
        fig.savefig(f"figs/{FILE[n]}_LIME_output_for_bi_classification.png", dpi=150); plt.close(fig)
        out.setdefault("lime", {})[n] = e.as_list(label=1)
        print("lime", n, e.as_list(label=1)[:3], flush=True)
if part == "shap":
    import shap
    names = {"XGB": "XGBoost", "GB": "GradientBoost", "CatBoost": "CatBoost"}
    force = {"XGB": "XGBoost_SHAP_BI_class_4DecRounding.png", "GB": "GradientBoost_SHAP_BI_class_5DecRounding.png", "CatBoost": "Catboost_SHAP_BI_class_4DecRounding.png"}
    for n in ["XGB", "GB", "CatBoost"]:
        A, B = scaled(n); m = model(n)["model"]
        rng = np.random.default_rng(42); s = rng.choice(len(B), 1000, replace=False)
        ex = shap.TreeExplainer(m); sv = ex.shap_values(B[s])
        if isinstance(sv, list): sv = sv[1]
        plt.figure(); shap.summary_plot(sv, B[s], feature_names=PROPS, plot_type="bar", show=False); plt.tight_layout()
        plt.savefig(f"figs/{names[n]}_SHAP_BAR_Plot.png", dpi=150); plt.close()
        plt.figure(); shap.summary_plot(sv, B[s], feature_names=PROPS, show=False); plt.tight_layout()
        plt.savefig(f"figs/{names[n]}_SHAP_Impact_on_model_output.png", dpi=150); plt.close()
        one = ex.shap_values(B[idx:idx + 1]); one = one[1] if isinstance(one, list) else one
        ev = np.ravel(ex.expected_value)[-1]
        shap.force_plot(ev, np.round(one[0], 4), np.round(B[idx], 4), feature_names=PROPS, matplotlib=True, show=False, figsize=(14, 3))
        plt.savefig(force[n], dpi=150, bbox_inches="tight") if False else plt.savefig(f"figs/{force[n]}", dpi=150, bbox_inches="tight"); plt.close()
        out.setdefault("shap_mean_abs", {})[n] = {p: round(float(v), 4) for p, v in zip(PROPS, np.abs(sv).mean(0))}
        print("shap", n, sorted(out["shap_mean_abs"][n].items(), key=lambda kv: -kv[1])[:3], flush=True)
if part == "eli5":
    import eli5
    fname = {"DT": "DT_ELI5_Bi.png", "RF": "RF_ELI5_Bi.png", "LR": "LR_ELI5_Bi.png", "GB": "GB_ELI5_Bi.png", "XGB": "XGBoost_ELI5_Bi.png"}
    for n in ["DT", "RF", "LR", "GB", "XGB"]:
        A, B = scaled(n); m = model(n)["model"]
        df = eli5.explain_prediction_df(m, B[idx], feature_names=PROPS)
        df = df[df["target"].astype(str).isin(["1", "True"])] if "target" in df and df["target"].nunique() > 1 else df
        df = df[["feature", "weight", "value"]].reset_index(drop=True)
        out.setdefault("eli5", {})[n] = [(r.feature, round(float(r.weight), 4)) for r in df.itertuples()]
        fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(df) + 0.8)); ax.axis("off")
        cells = [[f"{w:+.4f}", f, f"{v:.3f}" if f != "<BIAS>" else ""] for f, w, v in zip(df.feature, df.weight, df.value)]
        t = ax.table(cellText=cells, colLabels=["Contribution", "Feature", "Value"], loc="center", cellLoc="left")
        for i, (f, w) in enumerate(zip(df.feature, df.weight), start=1):
            t[i, 0].set_facecolor("#c8f0c8" if w > 0 else "#f6c6c6")
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.25)
        plt.title(f"ELI5 - {n} (y = attack)", fontsize=10); plt.tight_layout()
        plt.savefig(f"figs/{fname[n]}", dpi=150, bbox_inches="tight"); plt.close()
        print("eli5", n, out["eli5"][n][:3], flush=True)
json.dump(out, open("results/08_xai_classical.json", "w"), indent=1)
