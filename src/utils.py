"""
utils.py
--------
Shared utility functions: metrics, gravity scoring, node metadata helpers.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_gravity_score(baseline_mean: float,
                           intra_degree: float,
                           inter_degree: float) -> float:
    """
    Compound gravity score for county intervention targeting.
    Higher score = higher priority for treatment.

        gravity = 0.60 * baseline_burden
                + 0.25 * intra_degree
                + 0.15 * inter_degree
    """
    return 0.60 * baseline_mean + 0.25 * intra_degree + 0.15 * inter_degree


def get_node_degrees(edge_index_intra: torch.Tensor,
                     edge_index_inter: torch.Tensor,
                     n_nodes: int):
    """
    Compute per-node intra-state and inter-state degree from edge tensors.

    Returns
    -------
    intra_deg : np.ndarray [n_nodes]
    inter_deg : np.ndarray [n_nodes]
    """
    intra_deg = np.zeros(n_nodes, dtype=float)
    inter_deg = np.zeros(n_nodes, dtype=float)

    for s in edge_index_intra[0].detach().cpu().numpy():
        intra_deg[int(s)] += 1.0
    for s in edge_index_inter[0].detach().cpu().numpy():
        inter_deg[int(s)] += 1.0

    return intra_deg, inter_deg


def format_delta(delta: float) -> str:
    """Format a delta fraction as a readable percentage string."""
    return f"{delta:+.0%}"
