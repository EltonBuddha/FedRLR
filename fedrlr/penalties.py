from __future__ import annotations

import torch


def smooth_l1(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(x.square() + eps * eps).sum()


def soft_threshold_nonnegative(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.clamp(x - threshold, min=0.0)


def scad_derivative(distance: torch.Tensor, lam: float, a: float) -> torch.Tensor:
    if lam <= 0:
        return torch.zeros_like(distance)
    l = torch.as_tensor(lam, dtype=distance.dtype, device=distance.device)
    aa = torch.as_tensor(a, dtype=distance.dtype, device=distance.device)
    middle = torch.clamp((aa * l - distance) / (aa - 1), min=0.0)
    return torch.where(distance <= l, l, torch.where(distance <= aa * l, middle, torch.zeros_like(distance)))


def fusion_surrogates(
    structures: list[torch.Tensor],
    lam: float,
    a: float,
    eps: float,
    max_weight: float,
) -> tuple[list[torch.Tensor | None], list[float]]:
    m = len(structures)
    targets: list[torch.Tensor | None] = []
    sums: list[float] = []

    for i in range(m):
        weighted = []
        weights = []
        for j in range(m):
            if i == j:
                continue
            d = torch.linalg.vector_norm(structures[i] - structures[j])
            w = scad_derivative(d, lam, a) / (d + eps)
            w = torch.clamp(w, max=max_weight)
            if float(w.detach()) > 0.0:
                weights.append(w)
                weighted.append(structures[j])

        if not weights:
            targets.append(None)
            sums.append(0.0)
            continue

        w = torch.stack(weights)
        total = w.sum().clamp_min(1e-12)
        target = sum(wi * sj for wi, sj in zip(weights, weighted)) / total
        targets.append(target.detach())
        sums.append(float(total.detach().cpu()))

    return targets, sums
