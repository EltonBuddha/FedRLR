from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _orthonormal(rows: int, cols: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    if cols > rows:
        raise ValueError("rank must not exceed matrix dimension")
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    a = torch.randn(rows, cols, generator=g, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q


class FedRLRClient(nn.Module):
    def __init__(
        self,
        p: int,
        q: int,
        rank: int,
        response_model: nn.Module,
        U_init: Optional[torch.Tensor] = None,
        D_init: Optional[torch.Tensor] = None,
        V_init: Optional[torch.Tensor] = None,
        seed: int = 0,
    ):
        super().__init__()
        if rank > min(p, q):
            raise ValueError("rank must be <= min(p, q)")

        params = list(response_model.parameters())
        if params:
            device = params[0].device
            dtype = params[0].dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        if U_init is None:
            U_init = _orthonormal(p, rank, device, dtype, seed)
        if V_init is None:
            V_init = _orthonormal(q, rank, device, dtype, seed + 1)
        if D_init is None:
            D_init = torch.ones(rank, device=device, dtype=dtype)

        self.U = nn.Parameter(U_init.detach().clone())
        self.D = nn.Parameter(D_init.detach().clone())
        self.V = nn.Parameter(V_init.detach().clone())
        self.response_model = response_model

    def theta_matrix(self) -> torch.Tensor:
        return (self.U * self.D.unsqueeze(0)) @ self.V.T

    def latent(self, X: torch.Tensor) -> torch.Tensor:
        return ((X @ self.U) * self.D.unsqueeze(0)) @ self.V.T

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.response_model(self.latent(X))
