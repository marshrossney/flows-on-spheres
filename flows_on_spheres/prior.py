"""
Classes defining von Mises-Fisher and uniform distributions on the D-spheres.
"""
from math import lgamma, log, pi as π
from functools import partial
from typing import Callable, TypeAlias

import torch

from torchlft.linalg import norm_keepdim

Tensor: TypeAlias = torch.Tensor


def _circular_uniform(
    batch_size: int, device: str | torch.device, dtype: torch.dtype
):
    ϕ = torch.empty(batch_size, device=device, dtype=dtype).uniform_(0, 2 * π)
    x = torch.stack([ϕ.cos(), ϕ.sin()], dim=1)
    log_p = torch.full(
        (batch_size,), fill_value=-log(2 * π), device=device, dtype=dtype
    )
    return x, log_p


def _spherical_uniform(
    batch_size: int, device: torch.device, dtype: torch.dtype
):
    θ = (
        1 - torch.empty(batch_size, device=device, dtype=dtype).uniform_(0, 2)
    ).acos()
    ϕ = torch.empty(batch_size, device=device, dtype=dtype).uniform_(0, 2 * π)

    sinθ, cosθ, sinϕ, cosϕ = θ.sin(), θ.cos(), ϕ.sin(), ϕ.cos()
    x = torch.stack(
        [
            sinθ * cosϕ,
            sinθ * sinϕ,
            cosθ,
        ],
        dim=1,
    )
    log_p = torch.full(
        (batch_size,), fill_value=-log(4 * π), device=device, dtype=dtype
    )

    return x, log_p


def _hyperspherical_uniform(
    D: int,
    batch_size: int,
    device: str | torch.device,
    dtype: torch.dtype,
):
    x = torch.empty(
        batch_size, D, dtype=torch.float64, device=device
    ).normal_()
    x = x / norm_keepdim(x)

    # Drop nans and infs
    isfinite = x.isfinite().flatten(start_dim=1).all(dim=1)
    if not isfinite.all():
        x = x[isfinite]

    x = x.to(dtype)

    log_surface_area = log(2) + (D / 2) * log(π) - lgamma(D / 2)
    log_p = torch.full(
        (batch_size,), fill_value=-log_surface_area, device=device, dtype=dtype
    )
    return x, log_p


def uniform_prior(
    dim: int, device: str | torch.device, dtype: torch.dtype
) -> Callable[int, tuple[Tensor, Tensor]]:
    assert isinstance(dim, int)
    assert dim > 0

    if dim == 1:
        return partial(_circular_uniform, device=device, dtype=dtype)
    elif dim == 2:
        return partial(_spherical_uniform, device=device, dtype=dtype)
    else:
        return partial(
            _hyperspherical_uniform, D=dim, device=device, dtype=dtype
        )
