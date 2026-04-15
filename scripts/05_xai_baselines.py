#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from iotxai.config import load_config
from iotxai.data import holdout_split
from iotxai.explain.classical import (
    generate_eli5_explanations,
    generate_lime_explanations,
    generate_shap_tree_explanations,
)
from iotxai.models.classical import build_classical_models
from iotxai.training import fit_classical_model
from iotxai.utils import ensure_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)

    out_dir = ensure_dir(cfg.output_root / "figures" / "classical_xai")
    df = pd.read_csv(cfg["data"]["encoded_dataset_path"])
    label_col = cfg["data"].get("label_column", "label")

    X = df.drop(columns=[label_col, "AminoSequence"], errors="ignore")
    y = df[label_col]
    X_train, X_test, y_train, y_test = holdout_split(X, y, cfg["data"]["test_size"], cfg.random_state)

    models = build_classical_models(cfg.random_state)
    enabled = set(cfg["classical_models"]["enabled"])
    fitted = {}
    for name, model in models.items():
        if name not in enabled:
            continue
        fitted[name] = fit_classical_model(model, X_train, y_train)

    generate_lime_explanations(
        fitted,
        X_train,
        X_test,
        out_dir / "lime",
        sample_index=cfg["xai"]["baseline_sample_index"],
        num_features=cfg["xai"]["lime_num_features"],
    )
    generate_shap_tree_explanations(
        fitted,
        X_train,
        X_test,
        out_dir / "shap",
        eval_size=cfg["xai"]["shap_eval_size"],
    )
    generate_eli5_explanations(
        fitted,
        X_test,
        out_dir / "eli5",
        sample_index=cfg["xai"]["baseline_sample_index"],
    )
    print("Saved baseline XAI outputs.")


if __name__ == "__main__":
    main()
