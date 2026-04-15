from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_bar_plot(names: List[str], values: List[float], title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    order = np.argsort(np.abs(values))
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    colors = ["green" if v >= 0 else "red" for v in values]
    plt.barh(names, values, color=colors)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_lime_explanations(models: Dict[str, object], X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: str | Path, sample_index: int = 0, num_features: int = 10) -> None:
    from lime.lime_tabular import LimeTabularExplainer

    out_dir = Path(out_dir)
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=["benign", "attack"],
        mode="classification",
        discretize_continuous=True,
    )

    x_instance = X_test.iloc[sample_index].values
    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        exp = explainer.explain_instance(x_instance, model.predict_proba, num_features=num_features)
        weights = dict(exp.as_list())
        feat_names = list(weights.keys())
        feat_vals = list(weights.values())
        _save_bar_plot(feat_names, feat_vals, f"LIME - {name}", out_dir / f"lime_{name}.png")


def generate_shap_tree_explanations(models: Dict[str, object], X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: str | Path, eval_size: int = 300) -> None:
    import shap

    out_dir = Path(out_dir)
    for name in ["XGB", "GB", "CatBoost"]:
        if name not in models:
            continue
        model = models[name]
        X_eval = X_test.iloc[: min(eval_size, len(X_test))]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)

        # Global importance bar
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[1])
        else:
            sv = np.asarray(shap_values)
        importance = np.abs(sv).mean(axis=0)
        order = np.argsort(importance)
        plt.figure(figsize=(7, 4))
        plt.barh(np.array(X_eval.columns)[order], importance[order])
        plt.title(f"SHAP Global Importance - {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_bar_{name}.png", dpi=200)
        plt.close()

        # Summary plot
        shap.summary_plot(sv, X_eval, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_{name}.png", dpi=200, bbox_inches="tight")
        plt.close()


def generate_eli5_explanations(models: Dict[str, object], X_test: pd.DataFrame, out_dir: str | Path, sample_index: int = 0) -> None:
    import eli5

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = X_test.iloc[[sample_index]]
    for name in ["DT", "RF", "LR", "GB", "XGB"]:
        if name not in models:
            continue
        try:
            explanation = eli5.format_as_text(
                eli5.explain_prediction(models[name], sample, top=10)
            )
        except Exception as e:
            explanation = f"ELI5 explanation failed for {name}: {e}"
        (out_dir / f"eli5_{name}.txt").write_text(explanation, encoding="utf-8")
