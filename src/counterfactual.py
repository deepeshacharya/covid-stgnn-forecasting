"""
counterfactual.py
-----------------
Gravity-targeted counterfactual policy simulation engine.

Uses the trained ST-GNN as a structural simulator: by modifying
input features of high-priority counties during the epidemic peak
window, the model produces a counterfactual trajectory showing
what would have happened under the hypothetical intervention.

Key concepts:
    GRAVITY SCORE: compound priority score for county targeting
        gravity = 0.60 * baseline_burden
                + 0.25 * intra_degree
                + 0.15 * inter_degree

    AUTOREGRESSIVE FEEDBACK: simulated predictions at step t are
        fed back as lag features at step t+1, propagating the
        intervention effect forward in time.

    SPILLOVER ZONES:
        - Treated  : counties receiving the direct intervention
        - Neighbor : counties adjacent to treated counties
        - Far-field: all remaining counties

IMPORTANT LIMITATION:
    This model was trained as a forecaster, not a causal model.
    Counterfactual validity requires the assumption that learned
    correlations reflect structural causal relationships between
    epidemiological inputs and case outcomes. Treat simulation
    results as directional estimates, not causal proof.

Main function:
    run_counterfactual(model, all_snaps, baseline_preds,
                       edge_intra, edge_inter, n_nodes,
                       target_state, feature_col, delta,
                       feat_idx, lag_feature_cols,
                       county_to_idx, idx_to_county,
                       county_state_map, intra_deg, inter_deg)
"""

import numpy as np
import torch
from .config import DEVICE, HIDDEN_DIM
from .utils import compute_gravity_score


def select_treated_counties(target_state: str,
                             baseline_preds: dict,
                             county_to_idx: dict,
                             county_state_map: dict,
                             intra_deg: np.ndarray,
                             inter_deg: np.ndarray,
                             top_k: int = 10) -> list:
    """
    Select top-k counties in target_state by gravity score.

    Parameters
    ----------
    target_state   : 2-letter state abbreviation (e.g. "CA")
    baseline_preds : {fips -> mean_pred} from unmodified model run
    top_k          : number of counties to treat

    Returns
    -------
    List of (fips, gravity_score) tuples, sorted descending.
    """
    state_counties = [
        fips for fips, st in county_state_map.items()
        if st == target_state and fips in county_to_idx
    ]

    scored = []
    for fips in state_counties:
        nid   = county_to_idx[fips]
        burden = baseline_preds.get(fips, 0.0)
        g      = compute_gravity_score(burden, intra_deg[nid], inter_deg[nid])
        scored.append((fips, g))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def run_counterfactual(model,
                       all_snaps: dict,
                       baseline_detail,
                       edge_intra, edge_inter,
                       n_nodes: int,
                       target_state: str,
                       feature_col: str,
                       delta: float,
                       feat_idx: int,
                       lag_feature_cols: list,
                       county_to_idx: dict,
                       idx_to_county: dict,
                       county_state_map: dict,
                       intra_deg: np.ndarray,
                       inter_deg: np.ndarray,
                       top_k: int = 10) -> dict:
    """
    Run a gravity-targeted counterfactual simulation.

    Parameters
    ----------
    model          : trained FixedThreePartSTGNN (frozen, eval mode)
    all_snaps      : {date -> snapshot} for all test dates
    baseline_detail: DataFrame with [fips, date, ytrue, ypred] from normal eval
    edge_intra/inter: graph edge tensors
    n_nodes        : total node count
    target_state   : state to intervene in (e.g. "CA")
    feature_col    : name of the epidemiological feature to modify
    delta          : fractional change to apply (e.g. -0.20 = reduce 20%)
    feat_idx       : index of feature_col in lag_feature_cols
    lag_feature_cols: ordered feature list
    county_to_idx  : {fips -> node_index}
    idx_to_county  : {node_index -> fips}
    county_state_map: {fips -> state}
    intra_deg/inter_deg: per-node degree arrays from utils.get_node_degrees()
    top_k          : number of counties to treat

    Returns
    -------
    dict with keys:
        treated_fips      : list of treated county FIPS codes
        baseline_treated  : mean baseline pred for treated counties
        cf_treated        : mean counterfactual pred for treated counties
        baseline_neighbor : mean baseline pred for neighbor counties
        cf_neighbor       : mean counterfactual pred for neighbor counties
        baseline_farfield : mean baseline pred for far-field counties
        cf_farfield       : mean counterfactual pred for far-field counties
        pct_change_treated: percent change in treated zone
    """
    model.eval()

    # Compute baseline mean per county from baseline_detail
    baseline_mean_per_county = (
        baseline_detail.groupby("fips")["ypred"].mean().to_dict()
    )

    # Select treated counties
    treated_list  = select_treated_counties(
        target_state, baseline_mean_per_county,
        county_to_idx, county_state_map,
        intra_deg, inter_deg, top_k
    )
    treated_fips  = set(f for f, _ in treated_list)
    treated_nodes = set(county_to_idx[f] for f in treated_fips)

    # Identify neighbor nodes (adjacent to any treated node, not treated)
    adj_set = set()
    ei = edge_intra.cpu().numpy()
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u in treated_nodes:
            adj_set.add(v)
        if v in treated_nodes:
            adj_set.add(u)
    neighbor_nodes  = adj_set - treated_nodes
    farfield_nodes  = set(range(n_nodes)) - treated_nodes - neighbor_nodes

    # Run counterfactual forward pass
    h = torch.zeros((n_nodes, HIDDEN_DIM), device=DEVICE)
    cf_preds = {nid: [] for nid in range(n_nodes)}
    bl_preds = {nid: [] for nid in range(n_nodes)}

    with torch.no_grad():
        for d in sorted(all_snaps.keys()):
            snap = all_snaps[d]

            # Baseline prediction (unmodified)
            y_bl, _ = model.forward_step(
                snap["x"], h, snap["active"], edge_intra, edge_inter
            )

            # Counterfactual: modify treated counties' feature
            x_cf = snap["x"].clone()
            for nid in treated_nodes:
                x_cf[nid, feat_idx] = x_cf[nid, feat_idx] * (1.0 + delta)

            y_cf, h = model.forward_step(
                x_cf, h, snap["active"], edge_intra, edge_inter
            )
            h = h.detach()

            for nid in range(n_nodes):
                if snap["mask"][nid]:
                    bl_preds[nid].append(
                        float(torch.expm1(y_bl[nid]).clamp(min=0))
                    )
                    cf_preds[nid].append(
                        float(torch.expm1(y_cf[nid]).clamp(min=0))
                    )

    def zone_mean(node_set, pred_dict):
        vals = [np.mean(pred_dict[n]) for n in node_set if pred_dict[n]]
        return float(np.mean(vals)) if vals else 0.0

    bl_t  = zone_mean(treated_nodes,  bl_preds)
    cf_t  = zone_mean(treated_nodes,  cf_preds)
    bl_nb = zone_mean(neighbor_nodes, bl_preds)
    cf_nb = zone_mean(neighbor_nodes, cf_preds)
    bl_ff = zone_mean(farfield_nodes, bl_preds)
    cf_ff = zone_mean(farfield_nodes, cf_preds)

    pct_change = 100.0 * (cf_t - bl_t) / bl_t if bl_t > 0 else 0.0

    return {
        "treated_fips":       list(treated_fips),
        "baseline_treated":   bl_t,
        "cf_treated":         cf_t,
        "baseline_neighbor":  bl_nb,
        "cf_neighbor":        cf_nb,
        "baseline_farfield":  bl_ff,
        "cf_farfield":        cf_ff,
        "pct_change_treated": pct_change,
    }
