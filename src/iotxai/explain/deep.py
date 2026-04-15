from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def save_rnn_weight_visualizations(model: torch.nn.Module, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (name, param) in enumerate(model.named_parameters()):
        if "weight_ih" in name or "weight_hh" in name or "bias" in name:
            arr = param.detach().cpu().numpy()
            plt.figure(figsize=(6, 4))
            if arr.ndim == 2:
                plt.imshow(arr, aspect="auto")
                plt.colorbar()
            else:
                plt.bar(np.arange(len(arr)), arr)
            plt.title(name)
            plt.tight_layout()
            plt.savefig(out_dir / f"{name}.png", dpi=200)
            plt.close()


def extract_hidden_states(model: torch.nn.Module, X: np.ndarray, out_path: str | Path, device: str = "cpu") -> None:
    out_path = Path(out_path)
    model.eval()
    xb = torch.tensor(X[:1], dtype=torch.float32).reshape(1, -1, 1).to(device)
    with torch.no_grad():
        if hasattr(model, "rnn"):
            out, hidden = model.rnn(xb)
            arr = out.squeeze(0).detach().cpu().numpy().T
            plt.figure(figsize=(8, 4))
            plt.imshow(arr, aspect="auto")
            plt.xlabel("Sequence step")
            plt.ylabel("Hidden unit")
            plt.title("RNN hidden states")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()


def run_lime_for_rnn(model: torch.nn.Module, X_train: pd.DataFrame, X_test: pd.DataFrame, out_path: str | Path, device: str = "cpu", sample_index: int = 0, num_features: int = 10) -> None:
    from lime.lime_tabular import LimeTabularExplainer

    feature_names = list(X_train.columns)

    def predict_fn(x_np: np.ndarray):
        xb = torch.tensor(x_np, dtype=torch.float32).reshape(len(x_np), -1, 1).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(xb)).detach().cpu().numpy()
        return np.vstack([1 - probs, probs]).T

    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=["benign", "attack"],
        mode="classification",
        discretize_continuous=True,
    )
    exp = explainer.explain_instance(X_test.iloc[sample_index].values, predict_fn, num_features=num_features)
    weights = dict(exp.as_list())

    plt.figure(figsize=(7, 4))
    items = list(weights.items())
    feats = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["green" if v >= 0 else "red" for v in vals]
    plt.barh(feats, vals, color=colors)
    plt.title("LIME - RNN")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_shap_for_rnn(model: torch.nn.Module, X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: str | Path, background_size: int = 200, eval_size: int = 300, device: str = "cpu") -> None:
    import shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    background = X_train.iloc[: min(background_size, len(X_train))].values
    eval_x = X_test.iloc[: min(eval_size, len(X_test))].values

    def f(x_np):
        xb = torch.tensor(x_np, dtype=torch.float32).reshape(len(x_np), -1, 1).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(xb)).detach().cpu().numpy()
        return probs

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(eval_x, nsamples=min(100, len(eval_x) * 2))
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[0]

    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)
    plt.figure(figsize=(7, 4))
    plt.barh(np.array(X_test.columns)[order], importance[order])
    plt.title("Mean SHAP values - RNN")
    plt.tight_layout()
    plt.savefig(out_dir / "rnn_shap_mean.png", dpi=200)
    plt.close()

    shap.summary_plot(sv, eval_x, feature_names=list(X_test.columns), show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "rnn_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_integrated_gradients(model: torch.nn.Module, X_test: pd.DataFrame, out_path: str | Path, device: str = "cpu", sample_index: int = 0) -> None:
    from captum.attr import IntegratedGradients

    model.eval()
    ig = IntegratedGradients(model)
    sample = torch.tensor(X_test.iloc[[sample_index]].values, dtype=torch.float32).reshape(1, -1, 1).to(device)
    baseline = torch.zeros_like(sample)
    attr, _ = ig.attribute(sample, baselines=baseline, target=None, return_convergence_delta=True)
    vals = attr.detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(7, 4))
    plt.bar(X_test.columns, vals)
    plt.xticks(rotation=45, ha="right")
    plt.title("Integrated Gradients - RNN")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_anchor_explanation(model: torch.nn.Module, X_train: pd.DataFrame, X_test: pd.DataFrame, out_path: str | Path, device: str = "cpu") -> None:
    from alibi.explainers import AnchorTabular

    feature_names = list(X_train.columns)

    def predict_fn(x_np: np.ndarray):
        xb = torch.tensor(x_np, dtype=torch.float32).reshape(len(x_np), -1, 1).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(xb)).detach().cpu().numpy()
        return (probs >= 0.5).astype(int)

    explainer = AnchorTabular(predict_fn, feature_names)
    explainer.fit(X_train.values, disc_perc=(25, 50, 75))
    exp = explainer.explain(X_test.iloc[0].values, threshold=0.95)
    coverage = np.asarray(exp.raw.get("coverage", []), dtype=float)
    precision = np.asarray(exp.raw.get("precision", []), dtype=float)

    if coverage.size == 0 or precision.size == 0:
        coverage = np.array([exp.coverage])
        precision = np.array([exp.precision])

    plt.figure(figsize=(5, 4))
    plt.scatter(coverage, precision)
    plt.xlabel("Coverage")
    plt.ylabel("Precision")
    plt.title("Anchor precision vs coverage")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
