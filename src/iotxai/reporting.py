from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .utils import load_json


def metrics_json_to_table(metrics_json_path: str | Path, out_csv_path: str | Path) -> pd.DataFrame:
    metrics = load_json(metrics_json_path)
    rows = []
    for model_name, m in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": m.get("Accuracy"),
                "Precision": m.get("Precision"),
                "Recall (DR)": m.get("Recall"),
                "F1-Score": m.get("F1Score"),
                "FPR (FAR)": m.get("FPR"),
                "ROC AUC": m.get("ROCAUC"),
                "MCC": m.get("MCC"),
                "G-Mean": m.get("GMean"),
                "Confusion Matrix": str(m.get("ConfusionMatrix")),
            }
        )
    df = pd.DataFrame(rows)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return df


def cv_json_to_table(metrics_json_path: str | Path, out_csv_path: str | Path) -> pd.DataFrame:
    metrics = load_json(metrics_json_path)
    rows = []
    for model_name, m in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": f'{m["Accuracy"]["mean"]*100:.2f} ± {m["Accuracy"]["std"]*100:.2f}',
                "F1-Score": f'{m["F1Score"]["mean"]*100:.2f} ± {m["F1Score"]["std"]*100:.2f}',
                "FPR": f'{m["FPR"]["mean"]*100:.2f} ± {m["FPR"]["std"]*100:.2f}',
                "Recall": f'{m["Recall"]["mean"]*100:.2f} ± {m["Recall"]["std"]*100:.2f}',
            }
        )
    df = pd.DataFrame(rows)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return df
