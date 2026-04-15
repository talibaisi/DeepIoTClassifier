#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd

from iotxai.benchmarking import benchmark_torch_model
from iotxai.config import load_config
from iotxai.models.deep import build_deep_model
from iotxai.training import get_device
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
    X = df.drop(columns=[label_col, "AminoSequence"], errors="ignore").to_numpy(dtype="float32")

    results = {}
    for name in cfg["deep_models"]["enabled"]:
        if name == "GNN":
            continue
        model = build_deep_model(
            name=name,
            hidden_dim=cfg["deep_models"]["hidden_dim"],
            num_layers=cfg["deep_models"]["num_layers"],
            dropout=cfg["deep_models"]["dropout"],
        )
        results[name] = benchmark_torch_model(model, X, device=device)

    save_json(results, cfg.output_root / "metrics_benchmark_deep.json")
    print("Saved benchmark metrics.")


if __name__ == "__main__":
    main()
