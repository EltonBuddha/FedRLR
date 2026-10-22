# FedRLR

Official implementation of **FedRLR** for the paper:

**Reduce, Link, Respond: Learning Shared Low-Rank Structures with Personalized Nonlinear Responses**

FedRLR is a structured personalized federated regression framework for multivariate prediction. For client $m$, it models

$$
Y_m = f_{\theta_m}(X_m\Theta_m) + E_m,
\qquad
\Theta_m = U_m D_m V_m^\top,
$$

where $\Theta_m$ captures a sparse low-rank predictor--response structure and $f_{\theta_m}$ represents a client-specific nonlinear response mapping. Structural information is shared across related clients through fusion regularization while the nonlinear response component remains personalized.

## Method

FedRLR combines three main components:

- **Low-rank structural learning:** $\Theta_m = U_m D_m V_m^\top$ represents the multivariate predictor--response relationship through a compact latent structure.
- **Cross-client structural fusion:** fusion regularization encourages structurally related clients to share information and supports latent client grouping.
- **Personalized nonlinear responses:** each client retains its own nonlinear response model $f_{\theta_m}$.

The optimization procedure alternates between proximal--Riemannian updates of the structural factors and local gradient updates of the nonlinear response parameters.

## Repository structure

```text
fedrlr/
├── __init__.py
├── model.py
├── penalties.py
└── trainer.py
```

- `model.py`: client-specific sparse low-rank structural model and nonlinear response interface.
- `penalties.py`: sparsity and fusion regularization components.
- `trainer.py`: alternating optimization for FedRLR and FedRLR-NoFusion.

## Installation

```bash
pip install -r requirements.txt
```

## Core API

```python
from fedrlr import FedRLRClient, ClientBatch, FedRLRConfig, train_fedrlr
```

- `FedRLRClient`: defines the client-specific low-rank structure and nonlinear response model.
- `ClientBatch`: stores local client data.
- `FedRLRConfig`: specifies optimization and regularization parameters.
- `train_fedrlr`: runs the FedRLR optimization procedure.

## Methods

The implementation supports:

- `FedRLR`
- `FedRLR-NoFusion`
