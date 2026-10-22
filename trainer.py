from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from .model import FedRLRClient
from .penalties import fusion_surrogates, smooth_l1, soft_threshold_nonnegative


Method = Literal["FedRLR", "FedRLR-NoFusion"]


@dataclass
class ClientBatch:
    X: torch.Tensor
    Y: torch.Tensor


@dataclass
class FedRLRConfig:
    rounds: int
    local_steps: int
    lr_uv: float
    lr_d: float
    lr_response: float
    lambda_d: float
    lambda_a: float
    lambda_b: float
    lambda_c: float
    scad_a: float = 3.7
    smooth_eps: float = 1e-4
    fusion_eps: float = 1e-3
    max_fusion_weight: float = 10.0


def _project_stiefel(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    sym = 0.5 * (x.T @ g + g.T @ x)
    return g - x @ sym


def _retract(x: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(x, mode="reduced")
    d = torch.diagonal(r)
    s = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    return q * s.unsqueeze(0)


def _set_response_grad(model: FedRLRClient, enabled: bool) -> None:
    for p in model.response_model.parameters():
        p.requires_grad_(enabled)


def _zero(params) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def _structure_step(
    model: FedRLRClient,
    batch: ClientBatch,
    cfg: FedRLRConfig,
    fusion_target: torch.Tensor | None,
    fusion_weight: float,
) -> None:
    _set_response_grad(model, False)
    for p in (model.U, model.D, model.V):
        p.requires_grad_(True)
    _zero((model.U, model.D, model.V))

    pred = model(batch.X)
    n = batch.X.shape[0]
    loss = 0.5 * (pred - batch.Y).square().sum() / n
    loss = loss + cfg.lambda_a * smooth_l1(model.U * model.D.unsqueeze(0), cfg.smooth_eps)
    loss = loss + cfg.lambda_b * smooth_l1(model.V * model.D.unsqueeze(0), cfg.smooth_eps)

    if fusion_target is not None and fusion_weight > 0.0:
        loss = loss + 0.5 * fusion_weight * (model.theta_matrix() - fusion_target).square().sum()

    loss.backward()

    with torch.no_grad():
        gu = _project_stiefel(model.U, model.U.grad)
        gv = _project_stiefel(model.V, model.V.grad)
        model.U.copy_(_retract(model.U - cfg.lr_uv * gu))
        model.V.copy_(_retract(model.V - cfg.lr_uv * gv))
        d = model.D - cfg.lr_d * model.D.grad
        model.D.copy_(soft_threshold_nonnegative(d, cfg.lr_d * cfg.lambda_d))

    _set_response_grad(model, True)


def _response_step(model: FedRLRClient, batch: ClientBatch, cfg: FedRLRConfig) -> None:
    for p in (model.U, model.D, model.V):
        p.requires_grad_(False)
    response_params = list(model.response_model.parameters())
    _set_response_grad(model, True)
    _zero(response_params)

    pred = model(batch.X)
    n = batch.X.shape[0]
    loss = 0.5 * (pred - batch.Y).square().sum() / n
    loss.backward()

    with torch.no_grad():
        for p in response_params:
            if p.grad is not None:
                p.add_(-cfg.lr_response * p.grad)

    for p in (model.U, model.D, model.V):
        p.requires_grad_(True)


def train_fedrlr(
    models: Sequence[FedRLRClient],
    clients: Sequence[ClientBatch],
    config: FedRLRConfig,
    method: Method = "FedRLR",
) -> list[FedRLRClient]:
    if len(models) != len(clients):
        raise ValueError("models and clients must have the same length")
    if method not in {"FedRLR", "FedRLR-NoFusion"}:
        raise ValueError("method must be FedRLR or FedRLR-NoFusion")

    use_fusion = method == "FedRLR" and config.lambda_c > 0
    models = list(models)

    for _ in range(config.rounds):
        targets: list[torch.Tensor | None] = [None] * len(models)
        weights = [0.0] * len(models)

        if use_fusion:
            snapshot = [m.theta_matrix().detach().clone() for m in models]
            targets, weights = fusion_surrogates(
                snapshot,
                config.lambda_c,
                config.scad_a,
                config.fusion_eps,
                config.max_fusion_weight,
            )

        for i, (model, batch) in enumerate(zip(models, clients)):
            for _ in range(config.local_steps):
                _structure_step(model, batch, config, targets[i], weights[i])
                _response_step(model, batch, config)

    return models
