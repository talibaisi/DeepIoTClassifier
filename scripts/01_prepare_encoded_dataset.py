#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iotxai.config import load_config
from iotxai.data import (
    basic_clean,
    infer_feature_columns,
    infer_label_column,
    load_csvs,
    normalize_label_series,
    stratified_subset,
)
from iotxai.encoding import encode_dataframe
from iotxai.selector import hybrid_feature_selector
from iotxai.utils import ensure_dir, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)

    out_root = ensure_dir(cfg.output_root)
    raw_path = cfg["data"]["raw_path"]
    df = load_csvs(raw_path)
    df = basic_clean(df)

    label_col = infer_label_column(df, cfg["data"].get("label_column"))
    df[label_col] = normalize_label_series(df[label_col])
    df = stratified_subset(df, label_col, cfg["data"]["n_samples"], cfg.random_state)

    id_cols = cfg["data"].get("id_columns", []) or []
    feature_columns = cfg["data"].get("feature_columns")
    if not feature_columns:
        feature_columns = infer_feature_columns(df, label_col, id_cols)

    selection_meta = {"selected_feature_columns": feature_columns}
    if cfg["feature_selection"]["enabled"]:
        sel = hybrid_feature_selector(
            df[feature_columns],
            df[label_col],
            chi2_top_k=cfg["feature_selection"]["chi2_top_k"],
            cnn_top_k=cfg["feature_selection"]["cnn_top_k"],
            epochs=cfg["feature_selection"]["epochs"],
            batch_size=cfg["feature_selection"]["batch_size"],
            random_state=cfg.random_state,
        )
        feature_columns = sel.cnn_guided_features
        selection_meta = {
            "chi2_features": sel.chi2_features,
            "selected_feature_columns": sel.cnn_guided_features,
        }

    encoded = encode_dataframe(df[feature_columns], feature_columns)
    encoded[label_col] = df[label_col].values

    encoded_path = Path(cfg["data"]["encoded_dataset_path"])
    encoded_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(encoded_path, index=False)
    save_json(selection_meta, out_root / "feature_selection.json")
    print(f"Saved encoded dataset to {encoded_path}")


if __name__ == "__main__":
    main()
