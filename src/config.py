"""
config.py
---------
Central configuration for the ST-GNN COVID-19 forecasting pipeline.
Edit DATA_FOLDER to point to your local CDC CSV directory.
"""

import os
import random
import numpy as np
import torch

# ── Reproducibility ───────────────────────────────────────────
RND_SEED = 42
random.seed(RND_SEED)
np.random.seed(RND_SEED)
torch.manual_seed(RND_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RND_SEED)

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths — EDIT THIS ─────────────────────────────────────────
DATA_FOLDER = r"/Users/test/Documents/Covid/Covid 19 different timeline"
ADJ_FILE    = "https://www2.census.gov/geo/docs/reference/county_adjacency.txt"
OUT_DIR     = os.path.join(DATA_FOLDER, "gnn_outputs_fixed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Target ────────────────────────────────────────────────────
TARGET_COL    = "casesper100klast7days"
LOOKBACK_LAGS = [1, 2, 3, 7]

CANDIDATE_FEATURES = [
    "caseslast7days",
    "casesper100klast7days",
    "totalcases",
    "deathslast7days",
    "deathsper100klast7days",
    "totaldeaths",
    "testpositivityratelast7days",
    "confirmedcovidhosplast7days",
    "confirmedcovidhospper100bedslast7days",
    "suspectedcovidhosplast7days",
    "suspectedcovidhospper100bedslast7days",
    "pctinpatientbedsusedavglast7days",
    "pctinpatientbedsusedcovidavglast7days",
    "pcticubedsusedavglast7days",
    "pcticubedsusedcovidavglast7days",
]

# ── Split ratios ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# ── Model hyperparameters ─────────────────────────────────────
HIDDEN_DIM = 64
GCN_HIDDEN = 32
DROPOUT    = 0.15

# ── Training hyperparameters ──────────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 60
PATIENCE     = 10
GRAD_CLIP    = 2.0
HUBER_DELTA  = 1.0

# ── Isolated states ───────────────────────────────────────────
ISOLATED_STATES = {"PR", "AK", "HI", "GU", "VI", "MP", "AS"}
