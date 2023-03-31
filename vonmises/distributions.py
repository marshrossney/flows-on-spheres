"""
Classes defining von Mises-Fisher and uniform distributions on the D-spheres.
"""
from abc import ABC, abstractmethod
from math import exp, isclose, lgamma, log, pi as π, sqrt
from typing import Optional, TypeAlias

import torch
import torch.linalg as LA
from scipy.special import iv

Tensor: TypeAlias = torch.Tensor
IterableDataset: TypeAlias = torch.utils.data.IterableDataset


class Density(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def density(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def log_density(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def grad_density(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def grad_log_density(self, x: Tensor) -> Tensor:
        ...


class MixtureDensity(Density):
    def __init__(
        self,
        components: list[Density],
        weights: Optional[list[float]] = None,
    ):
        self.components = components
        if weights is None:
            self.weights = [1 / len(components) for _ in components]
        else:
            exp_weights = [exp(weight) for weight in weights]
            sum_exp_weights = sum(exp_weights)
            self.weights = [
                exp_weight / sum_exp_weights for exp_weight in exp_weights
            ]

        assert len(set([f.dim for f in components])) == 1
        assert len(self.weights) == len(self.components)
        assert isclose(sum(self.weights), 1)

    @property
    def dim(self) -> int:
        return self.components[0].dim

    def density(self, x: Tensor) -> Tensor:
        total = torch.zeros(x.shape[0], device=x.device)
        for ρ, f in zip(self.weights, self.components):
            total += ρ * f.density(x)
        return total

    def log_density(self, x: Tensor) -> Tensor:
        return self.density(x).log()

    def grad_density(self, x: Tensor) -> Tensor:
        total = torch.zeros(x.shape[0], self.dim, device=x.device)
        for ρ, f in zip(self.weights, self.components):
            total += ρ * f.grad_density(x)
        return total

    def grad_log_density(self, x: Tensor) -> Tensor:
        return self.grad_density(x).divide(self.density(x).unsqueeze(1))


class VonMisesFisherDensity(Density):
    def __init__(self, κ: float, μ: list[float]):
        if κ == 0:
            raise ValueError(
                "κ must be nonzero. For a uniform density please use `CircularUniformDistribuion`"
            )
        elif κ < 0:
            raise ValueError("κ must be positive.")

        mod_μ = sqrt(sum([pow(μi, 2) for μi in μ]))
        μ = [μi / mod_μ for μi in μ]

        D = len(μ) - 1
        v = (D - 1) / 2

        self.κ = κ
        self.μ = μ
        self._dim = D
        self._norm = pow(self.κ, -v) * pow(2 * π, v + 1) * iv(v, self.κ)
        self._log_norm = log(self._norm)

    @property
    def dim(self) -> int:
        return self._dim

    def density(self, x: Tensor) -> Tensor:
        μ = torch.tensor(self.μ, device=x.device)
        return x.mv(μ).mul(self.κ).exp().divide(self._norm)

    def log_density(self, x: Tensor) -> Tensor:
        μ = torch.tensor(self.μ, device=x.device)
        return x.mv(μ).mul(self.κ).subtract(self._log_norm)

    def grad_density(self, x: Tensor) -> Tensor:
        κμ = torch.tensor(self.μ, device=x.device).mul(self.κ)
        return torch.outer(self.density(x), κμ)

    def grad_log_density(self, x: Tensor) -> Tensor:
        κμ = torch.tensor(self.μ, device=x.device).mul(self.κ)
        return κμ


class VonMisesFisherMixtureDensity(MixtureDensity):
    def __init__(
        self,
        κ: list[int],
        μ: list[list[float]],
        weights: Optional[list[float]] = None,
    ):
        components = [VonMisesFisherDensity(κi, μi) for κi, μi in zip(κ, μ)]
        super().__init__(components, weights)


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
