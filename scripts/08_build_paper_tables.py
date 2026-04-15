#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd

from iotxai.config import load_config
from iotxai.reporting import cv_json_to_table, metrics_json_to_table
from iotxai.utils import ensure_dir, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = ensure_dir(cfg.output_root / "tables")

    classical_path = cfg.output_root / "metrics_holdout_classical.json"
    deep_path = cfg.output_root / "metrics_holdout_deep.json"
    cv_path = cfg.output_root / "metrics_cv_deep.json"
    bench_path = cfg.output_root / "metrics_benchmark_deep.json"

    if classical_path.exists():
        classical_df = metrics_json_to_table(classical_path, tables_dir / "table_holdout_classical.csv")
    else:
        classical_df = pd.DataFrame()

    if deep_path.exists():
        deep_df = metrics_json_to_table(deep_path, tables_dir / "table_holdout_deep.csv")
    else:
        deep_df = pd.DataFrame()

    if not classical_df.empty or not deep_df.empty:
        combined = pd.concat([classical_df, deep_df], axis=0, ignore_index=True)
        combined.to_csv(tables_dir / "table_all_models_holdout.csv", index=False)

    if cv_path.exists():
        cv_json_to_table(cv_path, tables_dir / "table_cv_deep.csv")

    if bench_path.exists():
        bench = load_json(bench_path)
        rows = []
        for model, vals in bench.items():
            rows.append(
                {
                    "Model": model,
                    "Params (M)": vals["Params_M"],
                    "Memory (MB)": vals["Memory_MB"],
                    "FLOPs (M)": vals["FLOPs_M"],
                    "Latency (ms)": vals["Latency_ms"],
                    "Throughput (flows/s)": vals["Throughput_flows_per_s"],
                }
            )
        pd.DataFrame(rows).to_csv(tables_dir / "table_benchmark.csv", index=False)

    print("Saved manuscript-style tables.")


if __name__ == "__main__":
    main()
