# COVID-19 Spatio-Temporal Graph Neural Network (ST-GNN) Forecasting

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3+-red.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A county-level COVID-19 forecasting model using a three-pathway **Spatio-Temporal Graph Neural Network (ST-GNN)** trained on CDC Community Profile Report data across **3,221 US counties** (February 2021 – May 2023).

The model simultaneously captures:
- **Local epidemic dynamics** — Self Path: linear projection of each county's own 18 lagged epidemiological features
- **Geographic spillover** — Spatial Path: dual GCNConv for intra-state and inter-state county adjacency
- **Temporal epidemic memory** — Temporal Path: GRU cell with truncated backpropagation through time (TBPTT)

After training, the frozen model serves as a **counterfactual policy simulator** — quantifying what would have happened under hypothetical public health interventions, with autoregressive feedback and gravity-targeted county selection.

---

## Architecture

cat > "/Users/test/Documents/STGNN modeling/covid-stgnn-forecasting/README.md" << 'ENDOFFILE'
# COVID-19 Spatio-Temporal Graph Neural Network (ST-GNN) Forecasting

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3+-red.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A county-level COVID-19 forecasting model using a three-pathway **Spatio-Temporal Graph Neural Network (ST-GNN)** trained on CDC Community Profile Report data across **3,221 US counties** (February 2021 – May 2023).

The model simultaneously captures:
- **Local epidemic dynamics** — Self Path: linear projection of each county's own 18 lagged epidemiological features
- **Geographic spillover** — Spatial Path: dual GCNConv for intra-state and inter-state county adjacency
- **Temporal epidemic memory** — Temporal Path: GRU cell with truncated backpropagation through time (TBPTT)

After training, the frozen model serves as a **counterfactual policy simulator** — quantifying what would have happened under hypothetical public health interventions, with autoregressive feedback and gravity-targeted county selection.

---

## Architecture
