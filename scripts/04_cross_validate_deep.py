#!/usr/bin/env python
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from iotxai.config import load_config
from iotxai.metrics import aggregate_metric_dicts
from iotxai.models.deep import build_deep_model
from iotxai.training import get_device, train_torch_model
from iotxai.utils import save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)
    device = get_device(cfg.device)

    df = pd.read_csv(cfg["data"]["encoded_dataset_path"])
    label_col = cfg["data"].get("label_column", "label")
    X = df.drop(columns=[label_col, "AminoSequence"], errors="ignore")
    y = df[label_col].to_numpy()

    skf = StratifiedKFold(
        n_splits=cfg["cross_validation"]["folds"],
        shuffle=True,
        random_state=cfg.random_state,
    )

    all_results = {}
    for name in cfg["deep_models"]["enabled"]:
        if name == "GNN":
            print("Skipping GNN in CV script unless torch_geometric pipeline is added.")
            continue

        fold_metrics = []
        print(f"Cross-validating {name}...")
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = df.iloc[train_idx][label_col], df.iloc[test_idx][label_col]

            model = build_deep_model(
                name=name,
                hidden_dim=cfg["deep_models"]["hidden_dim"],
                num_layers=cfg["deep_models"]["num_layers"],
                dropout=cfg["deep_models"]["dropout"],
            )
            tr = train_torch_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size=cfg["deep_models"]["batch_size"],
                epochs=cfg["deep_models"]["epochs"],
                learning_rate=cfg["deep_models"]["learning_rate"],
                weight_decay=cfg["deep_models"]["weight_decay"],
                device=device,
            )
            fold_metrics.append(tr.metrics)
            print(f"  fold {fold}/{cfg['cross_validation']['folds']}: acc={tr.metrics['Accuracy']:.4f} f1={tr.metrics['F1Score']:.4f}")

        all_results[name] = aggregate_metric_dicts(fold_metrics)

    save_json(all_results, cfg.output_root / "metrics_cv_deep.json")
    print("Saved CV metrics.")


if __name__ == "__main__":
    main()
