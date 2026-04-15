#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from iotxai.config import load_config
from iotxai.data import holdout_split
from iotxai.metrics import compute_metrics
from iotxai.models.classical import build_classical_models
from iotxai.training import fit_classical_model, predict_classical
from iotxai.utils import ensure_dir, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)
    out_root = ensure_dir(cfg.output_root)
    models_dir = ensure_dir(out_root / "models" / "classical")

    df = pd.read_csv(cfg["data"]["encoded_dataset_path"])
    label_col = cfg["data"].get("label_column", "label")
    X = df.drop(columns=[label_col, "AminoSequence"], errors="ignore")
    y = df[label_col]

    X_train, X_test, y_train, y_test = holdout_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg.random_state
    )

    models = build_classical_models(cfg.random_state)
    enabled = set(cfg["classical_models"]["enabled"])
    results = {}

    for name, model in models.items():
        if name not in enabled:
            continue
        print(f"Training {name}...")
        model = fit_classical_model(model, X_train, y_train)
        y_pred, y_score = predict_classical(model, X_test)
        results[name] = compute_metrics(y_test, y_pred, y_score)
        joblib.dump(model, models_dir / f"{name}.joblib")

    save_json(results, out_root / "metrics_holdout_classical.json")
    print("Saved classical metrics.")


if __name__ == "__main__":
    main()
