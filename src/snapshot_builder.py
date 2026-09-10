"""
snapshot_builder.py
-------------------
Builds per-date graph snapshots consumed by the ST-GNN training loop.

Each snapshot is a dict:
    x      : Tensor [N, F]  -- scaled lag feature matrix (zeroed for inactive nodes)
    y_log  : Tensor [N]     -- log1p target for each node
    y_raw  : Tensor [N]     -- raw target (cases/100k) for evaluation
    mask   : BoolTensor [N] -- True for nodes with valid target on this date
    active : Tensor [N, 1]  -- float mask for nodes with valid features

Function:
    build_snapshot(df_day, county_to_idx, lag_feature_cols, n_nodes, device)
    build_all_snapshots(model_df, dates, ...)
"""

import numpy as np
import torch


def build_snapshot(df_day: "pd.DataFrame",
                   county_to_idx: dict,
                   lag_feature_cols: list,
                   n_nodes: int,
                   device: torch.device) -> dict:
    """
    Build a single graph snapshot for one date.

    Parameters
    ----------
    df_day          : rows of model_df filtered to one date
    county_to_idx   : {fips -> node_index} mapping
    lag_feature_cols: ordered list of feature column names
    n_nodes         : total number of nodes in the graph
    device          : torch device

    Returns
    -------
    dict with keys: x, y_log, y_raw, mask, active
    """
    x      = np.zeros((n_nodes, len(lag_feature_cols)), dtype=np.float32)
    y_log  = np.zeros(n_nodes, dtype=np.float32)
    y_raw  = np.zeros(n_nodes, dtype=np.float32)
    mask   = np.zeros(n_nodes, dtype=bool)
    active = np.zeros(n_nodes, dtype=np.float32)

    for _, row in df_day.iterrows():
        fips = row["fips"]
        if fips not in county_to_idx:
            continue
        nid = county_to_idx[fips]

        feat_vals = row[lag_feature_cols].values.astype(np.float32)
        if np.any(np.isnan(feat_vals)):
            continue

        x[nid]      = feat_vals
        y_log[nid]  = float(row["target_log"])
        y_raw[nid]  = float(row["target_raw"])
        mask[nid]   = True
        active[nid] = 1.0

    return {
        "x":      torch.tensor(x,      device=device),
        "y_log":  torch.tensor(y_log,  device=device),
        "y_raw":  torch.tensor(y_raw,  device=device),
        "mask":   torch.tensor(mask,   device=device),
        "active": torch.tensor(active, device=device).unsqueeze(1),
    }


def build_all_snapshots(model_df, dates: list,
                        county_to_idx: dict,
                        lag_feature_cols: list,
                        n_nodes: int,
                        device: torch.device) -> dict:
    """
    Build snapshots for all dates in a list.

    Returns
    -------
    dict: {date -> snapshot_dict}
    """
    snaps = {}
    for d in sorted(dates):
        df_day = model_df[model_df["date"] == d]
        if len(df_day) == 0:
            continue
        snaps[d] = build_snapshot(
            df_day, county_to_idx, lag_feature_cols, n_nodes, device
        )
    print(f"Built {len(snaps)} snapshots for {len(dates)} dates.")
    return snaps
