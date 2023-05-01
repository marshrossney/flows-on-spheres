"""
Classes defining von Mises-Fisher and uniform distributions on the D-spheres.
"""
from math import lgamma, log, pi as π
from typing import Optional, TypeAlias

import torch
import torch.linalg as LA

from flows_on_spheres.abc import Flow

Tensor: TypeAlias = torch.Tensor
IterableDataset: TypeAlias = torch.utils.data.IterableDataset


class CircularUniformPrior(IterableDataset):
    """
    Uniform density on the unit circle.
    """

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        ϕ = torch.empty(self.batch_size).uniform_(0, 2 * π)
        x = torch.stack([ϕ.cos(), ϕ.sin()], dim=1)
        log_p = torch.full((self.batch_size,), fill_value=-log(2 * π))
        return x, log_p


class SphericalUniformPrior(IterableDataset):
    """
    Uniform density on the unit sphere.
    """

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        θ = (1 - torch.empty(self.batch_size).uniform_(0, 2)).acos()
        ϕ = torch.empty(self.batch_size).uniform_(0, 2 * π)

        sinθ, cosθ, sinϕ, cosϕ = θ.sin(), θ.cos(), ϕ.sin(), ϕ.cos()
        x = torch.stack(
            [
                sinθ * cosϕ,
                sinθ * sinϕ,
                cosθ,
            ],
            dim=1,
        )
        log_p = torch.full((self.batch_size,), fill_value=-log(4 * π))

        return x, log_p


def marsaglia(
    D: int,
    size: torch.Size = torch.Size([1]),
    device: Optional[torch.device] = None,
) -> Tensor:
    device = torch.device("cpu") if device is None else device

    x = torch.empty(*size, D, dtype=torch.float64, device=device).normal_()
    x.divide_(LA.vector_norm(x, dim=-1, keepdim=True))

    # Drop nans and infs
    isfinite = x.isfinite().flatten(start_dim=1).all(dim=1)
    if not isfinite.all():
        x = x[isfinite]

    return x.to(torch.float32)


class HyperSphericalUniformPrior(torch.utils.data.IterableDataset):
    def __init__(self, dim: int, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

        assert isinstance(dim, int)
        assert dim > 2

        self.dim = dim
        self._log_surface_area = log(2) + (dim / 2) * log(π) - lgamma(dim / 2)

    def __iter__(self):
        return self

    def __next__(self):
        x = marsaglia(self.dim, self.shape)
        log_p = torch.full(
            (self.batch_size,), fill_value=-self.log_surface_area
        )
        return x, log_p


def uniform_prior(dim: int, batch_size: int) -> IterableDataset:
    """Returns an IterableDataset that generates uniform samples on the sphere."""
    if dim == 1:
        return CircularUniformPrior(batch_size)
    elif dim == 2:
        return SphericalUniformPrior(batch_size)
    elif dim > 2:
        return HyperSphericalUniformPrior(dim, batch_size)
    else:
        raise ValueError("dim must be an integer greater than or equal to 1")


class FlowedPrior(IterableDataset):
    def __init__(self, flow: Flow, prior: IterableDataset):
        super().__init__()
        self.flow = flow
        self.prior = prior

    def __iter__(self):
        return self

    def __next__(self):
        x, log_p = next(self.prior)
        fx, ldj = self.flow(x)
        return fx, log_p - ldj
