"""
trainer.py
----------
Training loop and evaluation for the ST-GNN model.

Training uses TBPTT (Truncated Backpropagation Through Time):
the GRU hidden state carries forward across snapshots within each
epoch but is detached between steps to prevent gradient explosion.

Functions:
    train_model(...)       -- train with early stopping on val RMSE
    evaluate_snapshots(...) -- evaluate RMSE/MAE with expm1 back-transform
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .config import DEVICE, HIDDEN_DIM, EPOCHS, PATIENCE, GRAD_CLIP, HUBER_DELTA


def evaluate_snapshots(model, snapshots: dict,
                       edge_intra, edge_inter,
                       n_nodes: int,
                       idx_to_county: dict,
                       county_state_map: dict,
                       split_name: str = "eval") -> dict:
    """
    Evaluate model predictions on a snapshot dict.

    Returns dict with: split, rmse, mae, detail (DataFrame),
                       n_rows, n_unique_pred_values_rounded_2dp
    """
    model.eval()
    preds, trues, rows = [], [], []
    h = torch.zeros((n_nodes, HIDDEN_DIM), device=DEVICE)

    with torch.no_grad():
        for d in sorted(snapshots.keys()):
            snap = snapshots[d]
            y_hat_log, h = model.forward_step(
                snap["x"], h, snap["active"], edge_intra, edge_inter
            )
            y_pred = torch.expm1(y_hat_log).clamp(min=0)
            mask   = snap["mask"]

            yt = snap["y_raw"][mask].cpu().numpy()
            yp = y_pred[mask].cpu().numpy()
            ni = torch.where(mask)[0].cpu().numpy()

            preds.extend(yp.tolist())
            trues.extend(yt.tolist())

            for idx, yt_i, yp_i in zip(ni, yt, yp):
                fips = idx_to_county[int(idx)]
                rows.append({
                    "date":  pd.Timestamp(d),
                    "fips":  fips,
                    "state": county_state_map.get(fips),
                    "ytrue": float(yt_i),
                    "ypred": float(max(0.0, yp_i)),
                })

    detail = (
        pd.DataFrame(rows)
        .sort_values(["date", "state", "fips"])
        .reset_index(drop=True)
    )
    return {
        "split":  split_name,
        "rmse":   float(np.sqrt(mean_squared_error(trues, preds))),
        "mae":    float(mean_absolute_error(trues, preds)),
        "detail": detail,
        "n_rows": len(detail),
        "n_unique_pred_values_rounded_2dp": int(
            detail["ypred"].round(2).nunique()) if len(detail) else 0,
    }


def train_model(model, optimizer,
                train_snaps: dict, val_snaps: dict,
                edge_intra, edge_inter,
                n_nodes: int,
                idx_to_county: dict,
                county_state_map: dict):
    """
    Train ST-GNN with early stopping on validation RMSE.

    Returns model with best validation weights restored.
    """
    best_val   = float("inf")
    best_state = None
    patience   = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        h = torch.zeros((n_nodes, HIDDEN_DIM), device=DEVICE)
        losses = []

        for d in sorted(train_snaps.keys()):
            snap = train_snaps[d]
            optimizer.zero_grad()

            y_hat_log, h = model.forward_step(
                snap["x"], h, snap["active"], edge_intra, edge_inter
            )
            loss = F.huber_loss(
                y_hat_log[snap["mask"]],
                snap["y_log"][snap["mask"]],
                delta=HUBER_DELTA
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            h = h.detach()
            losses.append(loss.item())

        val_m = evaluate_snapshots(
            model, val_snaps, edge_intra, edge_inter,
            n_nodes, idx_to_county, county_state_map, "VAL"
        )
        print(f"Epoch {epoch:03d} | Loss {np.mean(losses):.4f} | "
              f"ValRMSE {val_m['rmse']:.4f} | ValMAE {val_m['mae']:.4f}")

        if val_m["rmse"] < best_val:
            best_val   = val_m["rmse"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience   = PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    print(f"Best ValRMSE: {best_val:.4f}")
    return model
