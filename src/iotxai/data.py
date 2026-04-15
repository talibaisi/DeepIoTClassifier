from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


LABEL_CANDIDATES = ["label", "Label", "attack_type", "Attack", "Class", "class"]


def discover_csv_files(path: str | Path) -> List[Path]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return [p]
    if p.is_dir():
        return sorted([x for x in p.rglob("*.csv") if x.is_file()])
    raise FileNotFoundError(f"No CSV file or directory found at: {path}")


def load_csvs(path: str | Path, max_files: int | None = None) -> pd.DataFrame:
    files = discover_csv_files(path)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {path}")
    frames = []
    for fp in files:
        frames.append(pd.read_csv(fp))
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df


def infer_label_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(f"Could not infer label column. Available columns: {list(df.columns)}")


def normalize_label_series(y: pd.Series) -> pd.Series:
    if y.dtype == object:
        y = y.astype(str).str.strip()
        if set(y.unique()) <= {"0", "1"}:
            return y.astype(int)
        benign_tokens = {"benign", "normal", "0", "false"}
        return y.str.lower().map(lambda v: 0 if v in benign_tokens else 1).astype(int)
    return y.astype(int)


def infer_feature_columns(
    df: pd.DataFrame,
    label_column: str,
    id_columns: Iterable[str] | None = None,
) -> List[str]:
    id_columns = set(id_columns or [])
    cols = []
    for col in df.columns:
        if col == label_column or col in id_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns inferred.")
    return cols


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(axis=0, how="any")
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def stratified_subset(
    df: pd.DataFrame,
    label_column: str,
    n_samples: int,
    random_state: int,
) -> pd.DataFrame:
    if len(df) <= n_samples:
        return df.copy().reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    y = df[label_column]
    idx, _ = next(splitter.split(df, y))
    return df.iloc[idx].reset_index(drop=True)


def holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
