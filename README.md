# FedRLR

Reference implementation of the core optimization components for **FedRLR**.

FedRLR models client-specific multivariate responses through

\[
Y_m = f_{\theta_m}(X_m\Theta_m) + E_m,
\qquad
\Theta_m = U_m D_m V_m^\top,
\]

with sparse low-rank structural factors, client-specific nonlinear response models, and optional fusion across related client structures.

This repository is intentionally limited to the core method. It contains the structured client model, proximal--Riemannian structural updates, SCAD-based fusion surrogate, local nonlinear-response updates, and the no-fusion ablation.

The following experiment-specific components are not included:

- synthetic data generation;
- real-data preprocessing;
- competing baseline implementations;
- experiment launchers and parameter-search utilities;
- automatic tuning/calibration rules;
- evaluation, aggregation, and plotting scripts;
- paper-specific checkpoints and result files.

The nonlinear response map is exposed as a generic PyTorch `nn.Module`. Users should construct their own client datasets, response models, initialization scheme, and hyperparameter configuration before calling `train_fedrlr`.

## Core API

- `FedRLRClient`: low-rank client structure \(U_m D_m V_m^\top\) plus a local nonlinear response module.
- `FedRLRConfig`: optimization and regularization configuration.
- `ClientBatch`: local client tensors.
- `train_fedrlr`: optimization loop for `FedRLR` or `FedRLR-NoFusion`.

## Installation

```bash
pip install -r requirements.txt
```

## Scope

This code is a compact method-level reference rather than an end-to-end reproduction package for the experimental tables and figures in the paper.
