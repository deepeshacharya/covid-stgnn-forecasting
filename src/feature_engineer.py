"""
feature_engineer.py
-------------------
Leakage-safe feature engineering and temporal train/val/test splitting.

All lag features are built from t-1 values so the model never sees
future data. The z-score scaler is fit ONLY on training dates and
applied identically to val/test to prevent distribution leakage.

Functions:
    build_lag_features(panel)        -- create lag features + log target
    temporal_split(model_df)         -- 70/15/15 chronological split
    fit_scaler(train_df, feat_cols)  -- fit train-only z-score scaler
    apply_scaler(df, means, stds, feat_cols) -- apply scaler to any split
"""

import numpy as np
import pandas as pd
from .config import (TARGET_COL, LOOKBACK_LAGS, CANDIDATE_FEATURES,
                     TRAIN_RATIO, VAL_RATIO)


def build_lag_features(panel: pd.DataFrame):
    """
    Build lag features and log-transformed prediction target.

    For each candidate feature: create lag1 (value at t-1).
    For TARGET_COL: additionally create lag1, lag2, lag3, lag7.
    Target: log1p(TARGET_COL shifted by -1) = next time step's value.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel with columns including TARGET_COL and CANDIDATE_FEATURES.

    Returns
    -------
    model_df : pd.DataFrame
        Panel with lag feature columns and target_log column added.
    lag_feature_cols : list of str
        Sorted list of all lag feature column names (18 total).
    """
    df = panel.copy().sort_values(["fips", "date"]).reset_index(drop=True)

    # Lag1 of all candidate features
    for col in CANDIDATE_FEATURES:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("fips")[col].shift(1)

    # Additional lags of the target column
    for lag in LOOKBACK_LAGS:
        col_name = f"{TARGET_COL}_lag{lag}"
        if col_name not in df.columns:
            df[col_name] = df.groupby("fips")[TARGET_COL].shift(lag)

    # Log-transformed next-step target (shift -1 = next date)
    df["target_log"] = df.groupby("fips")[TARGET_COL].transform(
        lambda x: np.log1p(x.shift(-1).clip(lower=0))
    )
    df["target_raw"] = df.groupby("fips")[TARGET_COL].shift(-1).clip(lower=0)

    # Collect lag feature columns
    lag_cols = sorted(set(
        [c for c in df.columns if c.endswith("_lag1") and
         c.replace("_lag1", "") in CANDIDATE_FEATURES] +
        [f"{TARGET_COL}_lag{l}" for l in LOOKBACK_LAGS]
    ))

    # Drop rows without complete lag features or target
    df = df.dropna(subset=lag_cols + ["target_log", "target_raw"]).copy()
    df = df.reset_index(drop=True)

    return df, lag_cols


def temporal_split(model_df: pd.DataFrame):
    """
    Chronological 70/15/15 split on unique dates.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    train_dates, val_dates, test_dates : list of timestamps
    """
    dates = sorted(model_df["date"].unique())
    n = len(dates)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_dates = dates[:n_train]
    val_dates   = dates[n_train: n_train + n_val]
    test_dates  = dates[n_train + n_val:]

    train_df = model_df[model_df["date"].isin(set(train_dates))].copy()
    val_df   = model_df[model_df["date"].isin(set(val_dates))].copy()
    test_df  = model_df[model_df["date"].isin(set(test_dates))].copy()

    print(f"Split: {len(train_dates)} train | "
          f"{len(val_dates)} val | {len(test_dates)} test dates")
    return train_df, val_df, test_df, train_dates, val_dates, test_dates


def fit_scaler(train_df: pd.DataFrame, feat_cols: list):
    """
    Compute per-feature mean and std from training data only.

    Returns
    -------
    means : pd.Series  -- feature means indexed by feature name
    stds  : pd.Series  -- feature stds  indexed by feature name
    """
    means = train_df[feat_cols].mean()
    stds  = train_df[feat_cols].std().replace(0, 1)
    return means, stds


def apply_scaler(df: pd.DataFrame, means: pd.Series,
                 stds: pd.Series, feat_cols: list) -> pd.DataFrame:
    """Apply train-only z-score scaler to any DataFrame split."""
    df = df.copy()
    for col in feat_cols:
        if col in df.columns:
            df[col] = (df[col] - float(means[col])) / float(stds[col])
    return df
