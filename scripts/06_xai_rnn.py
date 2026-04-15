#!/usr/bin/env python
from __future__ import annotations

import argparse

import joblib
import pandas as pd
import torch

from iotxai.config import load_config
from iotxai.data import holdout_split
from iotxai.explain.deep import (
    extract_hidden_states,
    run_anchor_explanation,
    run_integrated_gradients,
    run_lime_for_rnn,
    run_shap_for_rnn,
    save_rnn_weight_visualizations,
)
from iotxai.models.deep import build_deep_model
from iotxai.training import get_device, prepare_deep_arrays, train_torch_model
from iotxai.utils import ensure_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)
    device = get_device(cfg.device)

    out_dir = ensure_dir(cfg.output_root / "figures" / "rnn_xai")

    df = pd.read_csv(cfg["data"]["encoded_dataset_path"])
    label_col = cfg["data"].get("label_column", "label")
    X = df.drop(columns=[label_col, "AminoSequence"], errors="ignore")
    y = df[label_col]
    X_train, X_test, y_train, y_test = holdout_split(X, y, cfg["data"]["test_size"], cfg.random_state)

    model = build_deep_model(
        name="RNN",
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
    model = tr.model

    X_train_np = tr.scaler.transform(X_train)
    X_test_np = tr.scaler.transform(X_test)

    run_shap_for_rnn(
        model,
        pd.DataFrame(X_train_np, columns=X_train.columns),
        pd.DataFrame(X_test_np, columns=X_test.columns),
        out_dir / "shap",
        background_size=cfg["xai"]["shap_background_size"],
        eval_size=cfg["xai"]["shap_eval_size"],
        device=device,
    )
    run_lime_for_rnn(
        model,
        pd.DataFrame(X_train_np, columns=X_train.columns),
        pd.DataFrame(X_test_np, columns=X_test.columns),
        out_dir / "lime_rnn.png",
        device=device,
        sample_index=cfg["xai"]["baseline_sample_index"],
        num_features=cfg["xai"]["lime_num_features"],
    )
    run_integrated_gradients(
        model,
        pd.DataFrame(X_test_np, columns=X_test.columns),
        out_dir / "integrated_gradients_rnn.png",
        device=device,
        sample_index=cfg["xai"]["baseline_sample_index"],
    )
    run_anchor_explanation(
        model,
        pd.DataFrame(X_train_np, columns=X_train.columns),
        pd.DataFrame(X_test_np, columns=X_test.columns),
        out_dir / "anchor_precision_coverage.png",
        device=device,
    )
    save_rnn_weight_visualizations(model, out_dir / "weights")
    extract_hidden_states(model, X_test_np, out_dir / "hidden_states.png", device=device)
    print("Saved RNN XAI outputs.")


if __name__ == "__main__":
    main()
