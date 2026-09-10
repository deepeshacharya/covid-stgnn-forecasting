"""
model.py
--------
FixedThreePartSTGNN: three-pathway Spatio-Temporal GNN.

SELF PATH:     Linear(in_dim->hidden) + ReLU -> s_i
SPATIAL PATH:  GCNConv(intra) + GCNConv(inter) -> concat -> Linear -> n_i
TEMPORAL PATH: GRU(0.5*s_i + n_i, h_prev) -> m_i
FUSION:        softmax(alpha_s, alpha_n, alpha_m) * [s_i || n_i || m_i]
               -> Linear(192->64) + ReLU + Dropout -> h_t
OUTPUT:        Linear(64->1) -> log(cases/100k) at t+1

Loss:      Huber(delta=1.0) in log-space
Inference: expm1(y_hat).clamp(min=0) -> raw cases/100k
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class FixedThreePartSTGNN(nn.Module):
    """
    Three-pathway ST-GNN for county-level epidemic forecasting.

    Parameters
    ----------
    in_dim : int      -- number of lag features (default 18)
    hidden_dim : int  -- shared hidden dimension (default 64)
    gcn_hidden : int  -- GCNConv output dim, x2 after concat (default 32)
    dropout : float   -- dropout after fusion (default 0.15)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 gcn_hidden: int = 32, dropout: float = 0.15):
        super().__init__()

        # Self path
        self.self_linear  = nn.Linear(in_dim, hidden_dim)

        # Spatial path
        self.gcn_intra    = GCNConv(in_dim, gcn_hidden)
        self.gcn_inter    = GCNConv(in_dim, gcn_hidden)
        self.neigh_linear = nn.Linear(2 * gcn_hidden, hidden_dim)

        # Temporal path
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Adaptive fusion
        self.fuse    = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable pathway weights (softmax-normalized in forward)
        self.alpha_self  = nn.Parameter(torch.tensor(0.55))
        self.alpha_neigh = nn.Parameter(torch.tensor(0.25))
        self.alpha_mem   = nn.Parameter(torch.tensor(0.20))

        # Predictor head
        self.out = nn.Linear(hidden_dim, 1)

    def forward_step(self, x_t, h_prev, active_t, edge_intra, edge_inter):
        """
        Process one temporal snapshot.

        Parameters
        ----------
        x_t       : Tensor [N, in_dim]    node features at time t
        h_prev    : Tensor [N, hidden]    GRU hidden state from t-1
        active_t  : Tensor [N, 1]         binary county activity mask
        edge_intra: Tensor [2, E_intra]   intra-state edge index
        edge_inter: Tensor [2, E_inter]   inter-state edge index

        Returns
        -------
        y_hat_log : Tensor [N]            predicted log1p(cases/100k) at t+1
        h_t       : Tensor [N, hidden]    updated GRU hidden state
        """
        x_eff = x_t * active_t

        # Self path
        self_part  = torch.relu(self.self_linear(x_eff))

        # Spatial path
        intra_part = torch.relu(self.gcn_intra(x_eff, edge_intra))
        inter_part = torch.relu(self.gcn_inter(x_eff, edge_inter))
        neigh_part = torch.relu(
            self.neigh_linear(torch.cat([intra_part, inter_part], dim=-1))
        )

        # Temporal path
        z_t      = self_part + 0.5 * neigh_part
        mem_part = self.gru(z_t, h_prev)

        # Adaptive fusion
        a = torch.softmax(
            torch.stack([self.alpha_self, self.alpha_neigh, self.alpha_mem]),
            dim=0
        )
        fused    = torch.cat(
            [a[0] * self_part, a[1] * neigh_part, a[2] * mem_part], dim=-1
        )
        h_t      = torch.relu(self.fuse(fused))
        h_t      = self.dropout(h_t)

        y_hat_log = self.out(h_t).squeeze(-1)
        return y_hat_log, h_t

    def get_pathway_weights(self) -> dict:
        """Return current softmax-normalized pathway attribution weights."""
        with torch.no_grad():
            a = torch.softmax(
                torch.stack([self.alpha_self, self.alpha_neigh, self.alpha_mem]),
                dim=0
            )
        return {
            "alpha_self":  round(float(a[0]), 4),
            "alpha_neigh": round(float(a[1]), 4),
            "alpha_mem":   round(float(a[2]), 4),
        }
