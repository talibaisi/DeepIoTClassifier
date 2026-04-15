#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch

from iotxai.config import load_config
from iotxai.data import holdout_split
from iotxai.models.deep import build_deep_model
from iotxai.training import get_device, train_torch_model
from iotxai.utils import ensure_dir, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)
    out_root = ensure_dir(cfg.output_root)
    models_dir = ensure_dir(out_root / "models" / "deep")
    device = get_device(cfg.device)

    df = pd.read_csv(cfg["data"]["encoded_dataset_path"])
    label_col = cfg["data"].get("label_column", "label")
    X = df.drop(columns=[label_col, "AminoSequence"], errors="ignore")
    y = df[label_col]

    X_train, X_test, y_train, y_test = holdout_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg.random_state
    )

    results = {}
    scalers = {}
    for name in cfg["deep_models"]["enabled"]:
        if name == "GNN":
            print("Skipping GNN in holdout training script unless torch_geometric pipeline is added.")
            continue

        print(f"Training {name} on {device}...")
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
        results[name] = tr.metrics
        torch.save(tr.model.state_dict(), models_dir / f"{name}.pt")
        joblib.dump(tr.scaler, models_dir / f"{name}_scaler.joblib")

    save_json(results, out_root / "metrics_holdout_deep.json")
    print("Saved deep metrics.")


if __name__ == "__main__":
    main()
